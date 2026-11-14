import unittest
from originshare.catalog import Catalog
from originshare.domain import observed_labels


class DomainTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog("data")

    def test_labels_count_boxes_not_reviews(self):
        self.assertEqual(sum(observed_labels(self.catalog).values()), len(self.catalog.rows("boxes")))
        self.assertLess(len(self.catalog.rows("reviews")), len(self.catalog.rows("boxes")))


    def test_hard_label_counts_do_not_claim_truth(self):
        records = [{"predicted": "north", "weight_kg": 100}, {"predicted": "south", "weight_kg": 1}]
        self.assertEqual(observed_labels(FakeCatalog({"boxes": records})), {"north": 1, "south": 1})

    def test_empty_boxes(self):
        self.assertEqual(observed_labels(FakeCatalog({"boxes": []})), {})

    def test_score_mass_is_defined_per_box(self):
        for record in self.catalog.rows("boxes"):
            self.assertAlmostEqual(sum(record["scores"].values()), 1)

    def test_reviewed_unknown_class_is_preserved(self):
        known = {record["id"] for record in self.catalog.rows("classes")}
        self.assertTrue(any(record["confirmed_class"] not in known for record in self.catalog.rows("reviews")))


class FakeCatalog:
    def __init__(self, tables):
        self.tables = tables

    def rows(self, name):
        return self.tables[name]
