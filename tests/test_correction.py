import copy
import unittest

import numpy as np

from originshare.catalog import Catalog
from originshare.correction import (CorrectionError, build_report, em_batch,
                                    estimate_confusion, lp_bounds, load_inputs,
                                    matrix_diagnostics, parse_as_of, VersionData)
from originshare.service import Application


class FakeCatalog:
    def __init__(self, tables, fingerprint="fake"):
        self.tables = tables
        self.fingerprint = fingerprint

    def rows(self, name):
        return copy.deepcopy(self.tables[name])


def make_box(identifier, day, predicted, scores, line="L0", species="对虾",
             weight=20.0, version="v1"):
    return {"id": identifier, "day": day, "line": line, "species": species,
            "weight_kg": weight, "model_version": version,
            "scores": scores, "predicted": predicted}


def make_catalog(boxes, reviews, classes=None, sampling=None):
    if classes is None:
        classes = [{"id": "a", "label": "甲", "model_version": "v1"},
                   {"id": "b", "label": "乙", "model_version": "v1"}]
    if sampling is None:
        sampling = [{"id": "D" + box["id"], "box_id": box["id"],
                     "selection_probability": 0.5, "stratum": "s0"} for box in boxes]
    return FakeCatalog({"classes": classes, "boxes": boxes,
                        "reviews": reviews, "sampling": sampling})


class EmTests(unittest.TestCase):
    def test_recovers_proportions_without_negative(self):
        matrix = np.array([[[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]]])
        truth = np.array([0.5, 0.3, 0.2])
        counts = (truth @ matrix[0]) * 20000
        estimate, _ = em_batch(matrix, counts[None, :])
        self.assertTrue(np.all(estimate >= 0))
        self.assertAlmostEqual(float(estimate.sum()), 1.0)
        self.assertTrue(np.allclose(estimate[0], truth, atol=0.01))

    def test_unidentifiable_direction_stays_in_simplex(self):
        matrix = np.array([[[0.8, 0.2], [0.8, 0.2], [0.1, 0.9]]])
        counts = np.array([[800.0, 200.0]])
        estimate, _ = em_batch(matrix, counts)
        self.assertTrue(np.all(estimate >= 0))
        self.assertAlmostEqual(float(estimate.sum()), 1.0)
        self.assertAlmostEqual(float(estimate[0, 2]), 0.0, delta=0.05)

    def test_zero_counts_keep_uniform(self):
        matrix = np.array([[[0.5, 0.5], [0.5, 0.5]]])
        estimate, _ = em_batch(matrix, np.array([[0.0, 0.0]]))
        self.assertTrue(np.allclose(estimate, 0.5))


class IpWTests(unittest.TestCase):
    def test_design_weights_correct_stratum_bias(self):
        boxes, reviews, sampling = [], [], []
        # 真值 a：低入选概率层 80 箱全部误判为 b，高概率层 20 箱全部判对
        for i in range(80):
            boxes.append(make_box(f"lo{i}", "2026-07-01", "b", {"a": 0.4, "b": 0.6}))
        for i in range(20):
            boxes.append(make_box(f"hi{i}", "2026-07-01", "a", {"a": 0.9, "b": 0.1}))
        for i in range(100):
            probability = 0.25 if i < 80 else 1.0
            sampling.append({"id": f"D{i}", "box_id": boxes[i]["id"],
                             "selection_probability": probability, "stratum": "s"})
        # 每层恰好抽中期望数量：lo 层 20 条、hi 层 20 条复核
        for i in range(0, 80, 4):
            reviews.append({"id": f"V{i}", "box_id": f"lo{i}", "confirmed_class": "a",
                            "confirmed_at": "2026-07-05T00:00:00+08:00"})
        for i in range(20):
            reviews.append({"id": f"W{i}", "box_id": f"hi{i}", "confirmed_class": "a",
                            "confirmed_at": "2026-07-05T00:00:00+08:00"})
        catalog = make_catalog(boxes, reviews, sampling=sampling)
        data = load_inputs(catalog, None)
        vd = VersionData("v1", data["known"]["v1"], list(data["boxes"].values()),
                         data["reviews"], "week")
        estimate = estimate_confusion(vd, np.ones(vd.n_reviews, dtype=int), 1e-9)
        row = estimate["raw"][0, 0]
        # 未加权会看到 20:20（0.5），设计加权应还原 80:20（误判率 0.8）
        self.assertAlmostEqual(float(row[1]), 0.8, places=6)


class AsOfTests(unittest.TestCase):
    def test_later_confirmations_excluded(self):
        boxes = [make_box("B0", "2026-07-01", "a", {"a": 0.8, "b": 0.2}),
                 make_box("B1", "2026-07-02", "b", {"a": 0.3, "b": 0.7})]
        reviews = [{"id": "V0", "box_id": "B0", "confirmed_class": "a",
                    "confirmed_at": "2026-07-05T00:00:00+08:00"},
                   {"id": "V1", "box_id": "B1", "confirmed_class": "b",
                    "confirmed_at": "2026-08-05T00:00:00+08:00"}]
        catalog = make_catalog(boxes, reviews)
        report = build_report(catalog, as_of="2026-08-01T00:00:00%2B08:00".replace("%2B", "+"),
                              replicates=0)
        used = report["provenance"]["reviews"]
        self.assertEqual(used["used"], 1)
        self.assertEqual(used["excluded_after_as_of"], 1)
        self.assertEqual(used["ids"], ["V0"])

    def test_no_evidence_before_cutoff(self):
        boxes = [make_box("B0", "2026-07-01", "a", {"a": 0.8, "b": 0.2})]
        reviews = [{"id": "V0", "box_id": "B0", "confirmed_class": "a",
                    "confirmed_at": "2026-07-05T00:00:00+08:00"}]
        catalog = make_catalog(boxes, reviews)
        report = build_report(catalog, as_of="2026-07-01T00:00:00+08:00", replicates=0)
        self.assertEqual(report["status"], "evidence_insufficient")
        self.assertEqual(report["versions"]["v1"]["status"], "evidence_insufficient")

    def test_as_of_requires_timezone(self):
        with self.assertRaises(CorrectionError):
            parse_as_of("2026-08-01T00:00:00")
        with self.assertRaises(CorrectionError):
            parse_as_of("not-a-date")
        self.assertIsNone(parse_as_of(None))
        self.assertIsNotNone(parse_as_of("2026-08-01T00:00:00+08:00"))


class OtherClassTests(unittest.TestCase):
    def test_unknown_origin_becomes_other(self):
        boxes = [make_box("B0", "2026-07-01", "a", {"a": 0.8, "b": 0.2}),
                 make_box("B1", "2026-07-02", "b", {"a": 0.3, "b": 0.7})]
        reviews = [{"id": "V0", "box_id": "B0", "confirmed_class": "a",
                    "confirmed_at": "2026-07-05T00:00:00+08:00"},
                   {"id": "V1", "box_id": "B1", "confirmed_class": "新产地",
                    "confirmed_at": "2026-07-06T00:00:00+08:00"}]
        catalog = make_catalog(boxes, reviews)
        report = build_report(catalog, replicates=0)
        version = report["versions"]["v1"]
        self.assertTrue(version["other"]["included"])
        self.assertEqual(version["other"]["sources"], {"新产地": 1})
        self.assertIn("other", version["classes"])
        period = next(iter(version["periods"].values()))
        self.assertIn("known_only_shares", period["count"])
        self.assertIn("other", period["count"]["shares"])
        self.assertIn("other", period["count"]["bounds"])

    def test_versions_are_analyzed_separately(self):
        classes = [{"id": "a", "label": "甲", "model_version": "v1"},
                   {"id": "b", "label": "乙", "model_version": "v1"},
                   {"id": "x", "label": "丙", "model_version": "v2"},
                   {"id": "y", "label": "丁", "model_version": "v2"}]
        boxes = [make_box("B0", "2026-07-01", "a", {"a": 0.8, "b": 0.2}, version="v1"),
                 make_box("B1", "2026-07-02", "x", {"x": 0.7, "y": 0.3}, version="v2")]
        reviews = [{"id": "V0", "box_id": "B0", "confirmed_class": "a",
                    "confirmed_at": "2026-07-05T00:00:00+08:00"},
                   {"id": "V1", "box_id": "B1", "confirmed_class": "y",
                    "confirmed_at": "2026-07-06T00:00:00+08:00"}]
        report = build_report(make_catalog(boxes, reviews, classes=classes), replicates=0)
        self.assertEqual(set(report["versions"]), {"v1", "v2"})
        self.assertEqual(report["versions"]["v1"]["classes"], ["a", "b"])
        self.assertEqual(report["versions"]["v2"]["classes"], ["x", "y"])
        self.assertEqual(report["provenance"]["models"]["v2"]["boxes"], 1)


class IdentifiabilityTests(unittest.TestCase):
    def test_identical_rows_merge_and_bounds_widen(self):
        labels = ["x", "y", "z"]
        smoothed = np.array([[0.8, 0.2], [0.8, 0.2], [0.1, 0.9]])
        raw = smoothed.copy()
        diag = matrix_diagnostics(smoothed, raw, labels, 0.25)
        self.assertEqual(diag["merged_groups"], [["x", "y"]])
        self.assertGreater(diag["null_dimension"], 0)
        q = np.array([0.45, 0.55])
        fit = np.array([0.25, 0.25, 0.5])
        bounds = lp_bounds(smoothed, q, fit)
        # x 与 y 无法区分：各自范围很宽，但 z 的份额被 q 约束
        self.assertGreater(bounds[0][1] - bounds[0][0], 0.3)
        self.assertGreater(bounds[1][1] - bounds[1][0], 0.3)
        self.assertAlmostEqual(bounds[2][0], 0.5, delta=0.05)
        self.assertAlmostEqual(bounds[2][1], 0.5, delta=0.05)


class RealDataReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report(Catalog("data"), period="week", replicates=60, seed=11)

    def test_structure_and_provenance(self):
        report = self.report
        self.assertEqual(report["status"], "ok")
        provenance = report["provenance"]
        self.assertEqual(provenance["reviews"]["used"], 183)
        self.assertEqual(len(provenance["reviews"]["ids"]), 183)
        self.assertIn("v1", provenance["models"])
        self.assertEqual(provenance["estimator"].startswith("originshare-correction"), True)
        version = report["versions"]["v1"]
        self.assertEqual(version["classes"], ["north", "south", "inland", "other"])
        self.assertEqual(version["other"]["sources"], {"new_origin": 1})
        self.assertEqual(set(version["checks"]), {"mechanism_stability", "score_moment_consistency"})

    def test_shares_are_valid_and_bounds_ordered(self):
        version = self.report["versions"]["v1"]
        for key, period in version["periods"].items():
            for denominator in ("count", "weight"):
                block = period[denominator]
                shares = block["shares"]
                self.assertAlmostEqual(sum(shares.values()), 1.0, places=6)
                self.assertTrue(all(value >= 0 for value in shares.values()))
                for label, (lo, hi) in block["bounds"].items():
                    self.assertLessEqual(lo, hi + 1e-9)
                    self.assertGreaterEqual(lo, -1e-9)
                self.assertIn("ci95", block)
                for label, (lo, hi) in block["ci95"].items():
                    self.assertLessEqual(lo, hi + 1e-9)
                order = block["covariance"]["order"]
                size = len(order)
                self.assertEqual(len(block["covariance"]["matrix"]), size)

    def test_joint_uncertainty_is_reported(self):
        version = self.report["versions"]["v1"]
        period = version["periods"][sorted(version["periods"])[0]]
        count = period["count"]
        self.assertIn("joint95_halfwidth", count)
        self.assertIn("se", count)
        for label in count["shares"]:
            self.assertGreaterEqual(count["joint95_halfwidth"][label], count["se"][label])

    def test_new_origin_merged_with_similar_class(self):
        version = self.report["versions"]["v1"]
        groups = None
        for period in version["periods"].values():
            groups = period["merged"]["groups"]
        self.assertIn(["north", "other"], groups)
        for cell, block in version["cells"].items():
            self.assertEqual(block["diagnostics"]["null_dimension"], 1)
            self.assertTrue(block["notes"])

    def test_checks_have_verdicts(self):
        checks = self.report["versions"]["v1"]["checks"]
        self.assertIn(checks["mechanism_stability"]["verdict"], ("consistent", "flag"))
        self.assertIsNotNone(checks["mechanism_stability"]["p_value"])
        # replicates=60 低于矩检验下限，应明确标记不可用
        self.assertEqual(checks["score_moment_consistency"]["verdict"], "unavailable")

    def test_reproducible_with_same_seed(self):
        again = build_report(Catalog("data"), period="week", replicates=60, seed=11)
        first = dict(self.report)
        second = dict(again)
        for side in (first, second):
            side["provenance"] = dict(side["provenance"])
            side["provenance"].pop("generated_at")
        self.assertEqual(first, second)

    def test_as_of_removes_later_reviews(self):
        early = build_report(Catalog("data"), period="week", replicates=0,
                             as_of="2026-08-01T00:00:00+08:00", seed=11)
        used = early["provenance"]["reviews"]
        self.assertLess(used["used"], used["total"])
        self.assertEqual(used["used"] + used["excluded_after_as_of"], used["total"])
        self.assertLessEqual(used["latest_confirmation"], "2026-08-01T00:00:00+08:00")


class DriftDetectionTests(unittest.TestCase):
    def test_mechanism_check_flags_confusion_drift(self):
        catalog = Catalog("data")
        tables = {name: catalog.rows(name) for name in catalog.tables}
        for box in tables["boxes"]:
            if box["day"] >= "2026-09-15":
                scores = box["scores"]
                scores["north"], scores["south"] = scores["south"], scores["north"]
                box["predicted"] = max(scores, key=scores.get)
        report = build_report(FakeCatalog(tables), period="week", replicates=1, seed=5)
        check = report["versions"]["v1"]["checks"]["mechanism_stability"]
        self.assertEqual(check["verdict"], "flag")
        self.assertLess(check["p_value"], 0.05)


class EndpointTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.application = Application("data", self.workspace.name)

    def test_correction_endpoint(self):
        status, result = self.application.dispatch(
            "GET", "/correction?period=month&replicates=20&seed=1")
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "ok")
        self.assertIn("v1", result["versions"])
        self.assertEqual(result["provenance"]["parameters"]["replicates"], 20)

    def test_endpoint_rejects_bad_params(self):
        self.assertEqual(self.application.dispatch("GET", "/correction?period=year")[0], 400)
        self.assertEqual(self.application.dispatch("GET", "/correction?as_of=2026-08-01")[0], 400)
        self.assertEqual(self.application.dispatch("GET", "/correction?replicates=-1")[0], 400)
        self.assertEqual(self.application.dispatch("GET", "/correction?replicates=abc")[0], 400)

    def test_endpoint_as_of_filters_reviews(self):
        status, result = self.application.dispatch(
            "GET", "/correction?replicates=0&as_of=2026-07-01T00%3A00%3A00%2B08%3A00")
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "evidence_insufficient")


if __name__ == "__main__":
    unittest.main()
