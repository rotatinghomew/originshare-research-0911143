"""面向总体产地占比的校正分析。

逐箱汇总识别器标签会把系统性混淆当成产地构成的真实变化，逐箱复核又太慢。
本模块只在总体占比（quantification）层面做校正，不替任何单箱给出更肯定的标签。

方法要点：

* 复核样本按设计入选概率做 Horvitz-Thompson 加权，分包装线×品种单元估计混淆结构；
  证据不足的混淆行按 单元→包装线→品种→全局 逐级回退，并记录每一行的来源层级。
* 每个时期在单纯形内用 EM 解混，分别给出箱数与重量两种口径的产地构成，
  占比不会出现负值。
* 分层自助法同时重抽复核样本与到货箱，给出占比向量的联合不确定性
  （协方差、边际区间、同时置信半宽）。
* 两项诊断检验“只有产地比例变化、识别机制不变”的前提：混淆结构的时间稳定性
  （置换检验）与分数矩一致性（参数化自助）。
* 混淆矩阵接近奇异、真值类数多于标签数（例如识别器标签集外的新产地）或类别缺少
  复核证据时，报告可识别的合并比例、代数可行范围或无法判断的原因，不强行求逆。

分析按识别器版本分段。as_of 之后确认的复核不参与估计；报告记录所用识别器版本、
复核样本、参数与数据指纹，保证周报可追溯到当时可用的证据。
"""

import math
from datetime import date, datetime, timezone

import numpy as np
from scipy.optimize import linprog

ESTIMATOR_VERSION = "originshare-correction/1.0"
OTHER_CLASS = "other"
PERIOD_KINDS = ("week", "month")
DEFAULT_SEED = 20261115
MIN_CHECK_REPLICATES = 100
MAX_REPLICATES = 5000


class CorrectionError(ValueError):
    """校正分析的参数或输入资料无效。"""


def parse_as_of(text):
    """解析证据截点；要求带时区的 ISO 时间，查询串中的加号需百分号编码。"""
    if text is None:
        return None
    candidates = [text]
    if " " in text:
        candidates.append(text.replace(" ", "+"))
    for candidate in candidates:
        try:
            moment = datetime.fromisoformat(candidate)
        except (ValueError, TypeError):
            continue
        if moment.tzinfo is not None:
            return moment
    raise CorrectionError("as_of 应为带时区的 ISO 时间，例如 2026-10-26T00:00:00%2B08:00")


def period_key(day_text, kind):
    """到货业务日归入 ISO 周或月份。"""
    if kind == "month":
        return day_text[:7]
    day = date.fromisoformat(day_text)
    iso = day.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def load_inputs(catalog, as_of):
    """装配箱、复核与设计权重；as_of 之后确认的复核直接排除。"""
    known_by_version = {}
    for row in catalog.rows("classes"):
        known_by_version.setdefault(row["model_version"], []).append(row["id"])

    design = {}
    for row in catalog.rows("sampling"):
        probability = row["selection_probability"]
        if probability is None or probability <= 0:
            raise CorrectionError("复核设计入选概率必须为正数")
        design[row["box_id"]] = {"probability": probability, "stratum": row["stratum"]}

    boxes = {}
    for row in catalog.rows("boxes"):
        known = known_by_version.get(row["model_version"])
        if not known:
            raise CorrectionError("模型版本缺少已知类定义：" + row["model_version"])
        if row["predicted"] not in known:
            raise CorrectionError("预测标签不在已知类中：" + row["id"])
        missing = [label for label in known if label not in row["scores"]]
        if missing:
            raise CorrectionError("识别分数缺少已知类：" + row["id"])
        if row["weight_kg"] is None or row["weight_kg"] < 0:
            raise CorrectionError("箱重量缺失或为负：" + row["id"])
        boxes[row["id"]] = {
            "id": row["id"], "day": row["day"], "line": row["line"], "species": row["species"],
            "weight": float(row["weight_kg"]), "version": row["model_version"],
            "predicted": row["predicted"],
            "scores": [float(row["scores"][label]) for label in known],
        }

    reviews = []
    excluded = 0
    total = 0
    for row in catalog.rows("reviews"):
        total += 1
        box = boxes.get(row["box_id"])
        if box is None:
            raise CorrectionError("复核指向不存在的箱：" + row["id"])
        if row["box_id"] not in design:
            raise CorrectionError("复核箱缺少设计入选概率：" + row["id"])
        moment = datetime.fromisoformat(row["confirmed_at"])
        if as_of is not None and moment > as_of:
            excluded += 1
            continue
        reviews.append({
            "id": row["id"], "box_id": row["box_id"], "confirmed": row["confirmed_class"],
            "confirmed_at": moment, "design_weight": 1.0 / design[row["box_id"]]["probability"],
            "stratum": design[row["box_id"]]["stratum"], "box": box,
        })
    return {"known": known_by_version, "boxes": boxes, "reviews": reviews,
            "review_total": total, "review_excluded": excluded}


class VersionData:
    """单个识别器版本下的箱、复核与设计信息（数组化）。"""

    def __init__(self, version, known, boxes, reviews, period_kind):
        self.version = version
        self.known = list(known)
        self.n_known = len(known)
        other = [record for record in reviews if record["confirmed"] not in self.known]
        self.has_other = bool(other)
        self.labels = self.known + ([OTHER_CLASS] if self.has_other else [])
        self.n_classes = len(self.labels)
        self.class_index = {label: index for index, label in enumerate(self.labels)}
        self.other_sources = {}
        for record in other:
            self.other_sources[record["confirmed"]] = self.other_sources.get(record["confirmed"], 0) + 1

        self.cells = sorted({box["line"] + "|" + box["species"] for box in boxes})
        self.lines = sorted({box["line"] for box in boxes})
        self.species = sorted({box["species"] for box in boxes})
        self.periods = sorted({period_key(box["day"], period_kind) for box in boxes})
        cell_index = {cell: i for i, cell in enumerate(self.cells)}
        line_index = {line: i for i, line in enumerate(self.lines)}
        species_index = {name: i for i, name in enumerate(self.species)}
        period_index = {key: i for i, key in enumerate(self.periods)}
        self.cell_line = np.array([line_index[cell.split("|")[0]] for cell in self.cells])
        self.cell_species = np.array([species_index[cell.split("|")[1]] for cell in self.cells])

        self.n_boxes = len(boxes)
        self.box_cell = np.array([cell_index[box["line"] + "|" + box["species"]] for box in boxes])
        self.box_period = np.array([period_index[period_key(box["day"], period_kind)] for box in boxes])
        self.box_pred = np.array([self.known.index(box["predicted"]) for box in boxes])
        self.box_weight = np.array([box["weight"] for box in boxes])
        self.box_score = np.array([box["scores"] for box in boxes]) if boxes else np.zeros((0, self.n_known))

        self.n_reviews = len(reviews)
        self.review_ids = sorted(record["id"] for record in reviews)
        self.latest_confirmation = max((record["confirmed_at"] for record in reviews), default=None)
        if reviews:
            self.rev_cell = np.array([cell_index[record["box"]["line"] + "|" + record["box"]["species"]]
                                      for record in reviews])
            self.rev_line = np.array([line_index[record["box"]["line"]] for record in reviews])
            self.rev_species = np.array([species_index[record["box"]["species"]] for record in reviews])
            self.rev_true = np.array([self.class_index[record["confirmed"]]
                                      if record["confirmed"] in self.class_index
                                      else self.class_index[OTHER_CLASS] for record in reviews])
            self.rev_pred = np.array([self.known.index(record["box"]["predicted"]) for record in reviews])
            self.rev_design = np.array([record["design_weight"] for record in reviews])
            self.rev_score = np.array([record["box"]["scores"] for record in reviews])
            self.rev_day = np.array([date.fromisoformat(record["box"]["day"]).toordinal()
                                     for record in reviews])
            self.rev_stratum = np.array([record["stratum"] for record in reviews])
            self.rev_period = np.array([period_index[period_key(record["box"]["day"], period_kind)]
                                        for record in reviews])
        else:
            self.rev_cell = np.zeros(0, dtype=int)
            self.rev_line = np.zeros(0, dtype=int)
            self.rev_species = np.zeros(0, dtype=int)
            self.rev_true = np.zeros(0, dtype=int)
            self.rev_pred = np.zeros(0, dtype=int)
            self.rev_design = np.zeros(0)
            self.rev_score = np.zeros((0, self.n_known))
            self.rev_day = np.zeros(0, dtype=int)
            self.rev_stratum = np.zeros(0, dtype=object)
            self.rev_period = np.zeros(0, dtype=int)


def estimate_confusion(vd, multiplicities, alpha):
    """按设计权重估计各单元混淆矩阵与分数均值，证据不足逐级回退。

    返回每单元的平滑矩阵（估计用）、未平滑行（诊断用）、分数均值、来源层级与
    设计加权证据量。完全无证据的类标记为 unidentified，平滑行退化为均匀分布。
    """
    n_cells = len(vd.cells)
    K, J = vd.n_classes, vd.n_known
    w = multiplicities * vd.rev_design
    cell_tab = np.zeros((n_cells, K, J))
    line_tab = np.zeros((len(vd.lines), K, J))
    species_tab = np.zeros((len(vd.species), K, J))
    cell_score = np.zeros((n_cells, K, J))
    line_score = np.zeros((len(vd.lines), K, J))
    species_score = np.zeros((len(vd.species), K, J))
    if vd.n_reviews:
        np.add.at(cell_tab, (vd.rev_cell, vd.rev_true, vd.rev_pred), w)
        np.add.at(line_tab, (vd.rev_line, vd.rev_true, vd.rev_pred), w)
        np.add.at(species_tab, (vd.rev_species, vd.rev_true, vd.rev_pred), w)
        weighted_scores = w[:, None] * vd.rev_score
        np.add.at(cell_score, (vd.rev_cell, vd.rev_true), weighted_scores)
        np.add.at(line_score, (vd.rev_line, vd.rev_true), weighted_scores)
        np.add.at(species_score, (vd.rev_species, vd.rev_true), weighted_scores)
    global_tab = cell_tab.sum(axis=0)
    global_score = cell_score.sum(axis=0)

    smoothed = np.zeros((n_cells, K, J))
    raw = np.full((n_cells, K, J), np.nan)
    means = np.full((n_cells, K, J), np.nan)
    levels = np.empty((n_cells, K), dtype=object)
    evidence = np.zeros((n_cells, K))
    for c in range(n_cells):
        line_i, species_i = vd.cell_line[c], vd.cell_species[c]
        for k in range(K):
            if cell_tab[c, k].sum() > 0:
                level, tab_row, score_row = "cell", cell_tab[c, k], cell_score[c, k]
            elif line_tab[line_i, k].sum() > 0:
                level, tab_row, score_row = "line", line_tab[line_i, k], line_score[line_i, k]
            elif species_tab[species_i, k].sum() > 0:
                level, tab_row, score_row = "species", species_tab[species_i, k], species_score[species_i, k]
            elif global_tab[k].sum() > 0:
                level, tab_row, score_row = "global", global_tab[k], global_score[k]
            else:
                level, tab_row, score_row = "unidentified", None, None
            levels[c, k] = level
            if tab_row is None:
                smoothed[c, k] = 1.0 / J
                continue
            mass = float(tab_row.sum())
            evidence[c, k] = mass
            raw[c, k] = tab_row / mass
            smoothed[c, k] = (tab_row + alpha) / (mass + alpha * J)
            means[c, k] = score_row / mass
    return {"smoothed": smoothed, "raw": raw, "means": means, "levels": levels, "evidence": evidence}


def em_batch(matrices, counts, tol=1e-9, max_iter=1000):
    """批量 EM 解混：matrices (S,K,J) 行和为一，counts (S,J) 非负。

    每行独立在单纯形内迭代，结果占比不会为负；零计数行保持均匀初值。
    混淆结构接近不可识别时沿零空间方向收敛缓慢，该方向的点估计本就不可解释，
    迭代上限截断不影响可识别方向上的结果。
    """
    n_rows, K = matrices.shape[0], matrices.shape[1]
    totals = counts.sum(axis=1)
    active = totals > 0
    safe_totals = np.where(active, totals, 1.0)
    p = np.full((n_rows, K), 1.0 / K)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        product = p[:, :, None] * matrices
        mix = product.sum(axis=1)
        np.maximum(mix, 1e-300, out=mix)
        update = (product / mix[:, None, :] * counts[:, None, :]).sum(axis=2) / safe_totals[:, None]
        update[~active] = p[~active]
        delta = float(np.abs(update - p).max()) if active.any() else 0.0
        p = update
        if delta < tol:
            break
    return p, iterations


def matrix_diagnostics(smoothed, raw, labels, merge_tau):
    """混淆矩阵的可识别性诊断：奇异值、零空间维数、行全变差距离与合并组。"""
    K = smoothed.shape[0]
    singular = np.linalg.svd(smoothed, compute_uv=False)
    if singular.size and singular[0] > 0:
        rank = int((singular > singular[0] * 1e-10).sum())
    else:
        rank = 0
    condition = None
    if singular.size and singular[-1] > 1e-12:
        condition = float(singular[0] / singular[-1])
    pairs = []
    parent = list(range(K))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(K):
        for j in range(i + 1, K):
            if np.isnan(raw[i]).any() or np.isnan(raw[j]).any():
                pairs.append((labels[i], labels[j], None))
                continue
            tv = float(0.5 * np.abs(raw[i] - raw[j]).sum())
            pairs.append((labels[i], labels[j], tv))
            if tv < merge_tau:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pi] = pj
    groups = {}
    for k in range(K):
        groups.setdefault(find(k), []).append(labels[k])
    merged = sorted(sorted(members) for members in groups.values() if len(members) > 1)
    return {"singular_values": [float(v) for v in singular],
            "rank": rank, "null_dimension": K - rank, "condition": condition,
            "row_total_variation": [[a, b, tv] for a, b, tv in pairs],
            "merged_groups": merged}


def lp_bounds(matrix, q, p_fit):
    """给定观测标签分布 q 与混淆矩阵，每类占比的代数可行范围。

    可行集为 {p ≥ 0, Σp = 1, |Mᵀp − q| ≤ r}，r 为 EM 拟合残差加数值余量。
    这是“给定观测与结构下的代数范围”，不含抽样不确定性（由自助法区间给出）。
    """
    K = matrix.shape[0]
    residual = np.abs(p_fit @ matrix - q) + 1e-9
    a_ub = np.vstack([matrix.T, -matrix.T])
    b_ub = np.concatenate([q + residual, -(q - residual)])
    a_eq = np.ones((1, K))
    bounds = [(0.0, 1.0)] * K
    out = []
    for k in range(K):
        direction = np.zeros(K)
        direction[k] = 1.0
        low = linprog(direction, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0],
                      bounds=bounds, method="highs")
        high = linprog(-direction, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0],
                       bounds=bounds, method="highs")
        lo = float(low.fun) if low.success else 0.0
        hi = float(-high.fun) if high.success else 1.0
        out.append((max(lo, 0.0), min(hi, 1.0)))
    return out


def _summarize_replicates(replicates, point):
    """自助复制的联合不确定性：协方差、边际区间与同时置信半宽。"""
    finite = replicates[np.isfinite(replicates).all(axis=1)]
    if finite.shape[0] < 2:
        return None
    cov = np.atleast_2d(np.cov(finite, rowvar=False))
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    ci_lo = np.percentile(finite, 2.5, axis=0)
    ci_hi = np.percentile(finite, 97.5, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(finite - point) / np.where(se > 0, se, np.nan)
    max_z = np.nanmax(np.where(np.isnan(z), -np.inf, z), axis=1)
    max_z = max_z[np.isfinite(max_z)]
    halfwidth = np.percentile(max_z, 95) * se if max_z.size else np.zeros(replicates.shape[1])
    return {"se": se, "ci_lo": ci_lo, "ci_hi": ci_hi, "halfwidth": halfwidth, "cov": cov}


def _check_mechanism_stability(vd, replicates, rng):
    """混淆结构时间稳定性：按复核箱到货日中位分前后两窗，单元内置换检验。"""
    if replicates < 50:
        return {"verdict": "unavailable", "p_value": None, "note": "重抽样次数不足，未运行稳定性检验"}
    if vd.n_reviews < 10:
        return {"verdict": "unavailable", "p_value": None, "note": "复核样本过少，无法检验"}
    median = float(np.median(vd.rev_day))
    window = (vd.rev_day > median).astype(int)
    if window.sum() == 0 or window.sum() == vd.n_reviews:
        return {"verdict": "unavailable", "p_value": None, "note": "复核到货日集中在同一时刻，无法分窗"}
    split_day = date.fromordinal(int(median)).isoformat()
    K, J = vd.n_classes, vd.n_known

    def statistic(labels):
        total = 0.0
        for c in range(len(vd.cells)):
            mask = vd.rev_cell == c
            if not mask.any():
                continue
            table = np.zeros((K, J, 2))
            np.add.at(table, (vd.rev_true[mask], vd.rev_pred[mask], labels[mask]), 1)
            by_true = table.sum(axis=(1, 2))
            by_true_window = table.sum(axis=1)
            by_true_pred = table.sum(axis=2)
            expected = (by_true_pred[:, :, None] * by_true_window[:, None, :]
                        / np.maximum(by_true, 1e-300)[:, None, None])
            valid = expected > 0
            total += float((((table - expected) ** 2) / np.maximum(expected, 1e-300))[valid].sum())
        return total

    observed = statistic(window)
    hits = 0
    for _ in range(replicates):
        permuted = window.copy()
        for c in range(len(vd.cells)):
            idx = np.where(vd.rev_cell == c)[0]
            permuted[idx] = rng.permutation(permuted[idx])
        if statistic(permuted) >= observed - 1e-12:
            hits += 1
    p_value = (1 + hits) / (replicates + 1)
    return {"verdict": "flag" if p_value < 0.05 else "consistent",
            "window_split": split_day, "statistic": observed,
            "p_value": p_value, "replicates": replicates,
            "note": "p 值小于 0.05 提示混淆结构可能随时间变化；不显著不等于证明稳定"}


def _score_pools(vd, levels):
    """分数矩检验的经验分数池：按观测数据选定的回退层级取复核分数向量。"""
    pools = {}
    for c in range(len(vd.cells)):
        for k in range(vd.n_classes):
            level = levels[c, k]
            if level == "unidentified":
                pools[(c, k)] = None
                continue
            if level == "cell":
                mask = (vd.rev_cell == c) & (vd.rev_true == k)
            elif level == "line":
                mask = (vd.rev_line == vd.cell_line[c]) & (vd.rev_true == k)
            elif level == "species":
                mask = (vd.rev_species == vd.cell_species[c]) & (vd.rev_true == k)
            else:
                mask = vd.rev_true == k
            idx = np.where(mask)[0]
            weights = vd.rev_design[idx]
            pools[(c, k)] = (vd.rev_score[idx], weights / weights.sum())
    return pools


def _check_score_moments(vd, observed_est, p_observed, cell_of_s, n_tc, replicates, rng, alpha):
    """分数矩一致性：各期平均识别分数能否由固定机制按估计比例重新混合解释。

    原假设为机制稳定（仅已知类，避免新产地零空间使统计量依赖初始化）；参数化自助
    在原假设下重抽复核（估计均值）并按观测拟合比例从经验分数池模拟未复核箱的分数，
    给出各期与全局的 p 值。显著既可能意味着识别机制漂移，也可能意味着出现了
    模型标签集外的产地。
    """
    note = ("p 值小于 0.05 提示识别机制可能漂移，或出现了模型标签集外的产地；"
            "不显著不等于证明稳定")
    if replicates < MIN_CHECK_REPLICATES:
        return {"verdict": "unavailable", "p_value": None, "periods": {},
                "note": "重抽样次数不足 " + str(MIN_CHECK_REPLICATES) + "，未运行矩检验"}
    means = observed_est["means"]
    if np.isnan(means).any():
        note = "部分类别缺少复核分数证据，相应类不参与矩检验；" + note
    pools = _score_pools(vd, observed_est["levels"])
    n_classes_fit = p_observed.shape[1]
    n_cells = len(vd.cells)
    n_periods = len(vd.periods)
    n_s = len(cell_of_s)
    s_of_box = vd.box_period * n_cells + vd.box_cell

    mean_score = np.zeros((n_s, vd.n_known))
    for s in range(n_s):
        mask = s_of_box == s
        if mask.any():
            mean_score[s] = vd.box_score[mask].mean(axis=0)

    # 复核箱是总体中的已观测成员：模拟时保持其分数与标签不变，仅重抽未复核箱，
    # 使原假设下的复制与观测统计量共享同一批复核证据，避免校准失真。
    rev_s = vd.rev_period * n_cells + vd.rev_cell
    rev_pred_counts = np.zeros((n_s, vd.n_known))
    np.add.at(rev_pred_counts, (rev_s, vd.rev_pred), 1)
    rev_score_sum = np.zeros((n_s, vd.n_known))
    np.add.at(rev_score_sum, rev_s, vd.rev_score)
    rev_in_s = rev_pred_counts.sum(axis=1)

    def residual_stat(p_fit, mu, means_s):
        stat = np.zeros(n_periods)
        for s in range(n_s):
            if n_tc[s] <= 0:
                continue
            c = cell_of_s[s]
            fitted = np.zeros(vd.n_known)
            for k in range(p_fit.shape[1]):
                if not np.isnan(mu[c, k]).any():
                    fitted += p_fit[s, k] * mu[c, k]
            diff = means_s[s] - fitted
            stat[s // n_cells] += n_tc[s] * float(diff @ diff)
        return stat

    observed_stat = residual_stat(p_observed, means, mean_score)
    strata = [np.where(vd.rev_stratum == name)[0] for name in sorted(set(vd.rev_stratum))]
    exceed = np.zeros(n_periods, dtype=int)
    exceed_global = 0
    for _ in range(replicates):
        multiplicities = np.zeros(vd.n_reviews, dtype=int)
        for idx in strata:
            if len(idx):
                multiplicities[idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)))
        est_b = estimate_confusion(vd, multiplicities, alpha)
        sim_means = np.zeros((n_s, vd.n_known))
        sim_counts = np.zeros((n_s, vd.n_known))
        for s in range(n_s):
            n = int(n_tc[s])
            if n == 0:
                continue
            c = cell_of_s[s]
            n_sim = n - int(rev_in_s[s])
            sim_score_sum = np.zeros(vd.n_known)
            sim_pred_counts = np.zeros(vd.n_known)
            if n_sim > 0:
                classes = [k for k in range(n_classes_fit)
                           if not np.isnan(means[c, k]).any() and pools[(c, k)] is not None]
                weights = np.array([p_observed[s, k] for k in classes])
                if weights.sum() <= 0:
                    weights = np.full(len(classes), 1.0 / len(classes))
                else:
                    weights = weights / weights.sum()
                drawn = rng.choice(len(classes), size=n_sim, p=weights)
                scores = np.zeros((n_sim, vd.n_known))
                for pos, k in enumerate(classes):
                    take = drawn == pos
                    if not take.any():
                        continue
                    pool_scores, pool_p = pools[(c, k)]
                    chosen = rng.choice(len(pool_scores), size=int(take.sum()), p=pool_p)
                    scores[take] = pool_scores[chosen]
                sim_score_sum = scores.sum(axis=0)
                sim_pred_counts = np.bincount(scores.argmax(axis=1), minlength=vd.n_known)
            sim_means[s] = (rev_score_sum[s] + sim_score_sum) / n
            sim_counts[s] = rev_pred_counts[s] + sim_pred_counts
        p_b, _ = em_batch(est_b["smoothed"][cell_of_s][:, :n_classes_fit, :], sim_counts)
        stat_b = residual_stat(p_b, est_b["means"], sim_means)
        exceed += (stat_b >= observed_stat - 1e-12)
        if stat_b.sum() >= observed_stat.sum() - 1e-12:
            exceed_global += 1
    p_values = (1 + exceed) / (replicates + 1)
    global_p = (1 + exceed_global) / (replicates + 1)
    periods = {vd.periods[t]: {"statistic": float(observed_stat[t]), "p_value": float(p_values[t])}
               for t in range(n_periods)}
    return {"verdict": "flag" if global_p < 0.05 else "consistent",
            "statistic": float(observed_stat.sum()), "p_value": float(global_p),
            "periods": periods, "note": note}


def _groups_from_pairs(labels, pairs):
    """由合并产地对构造连通分量（时期级合并组）。"""
    parent = {label: label for label in labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups = {}
    for label in labels:
        groups.setdefault(find(label), []).append(label)
    return sorted(sorted(members) for members in groups.values() if len(members) > 1)


def _share_block(shares, naive, known_only, bounds, replicates, vd):
    labels = vd.labels
    block = {"shares": {label: float(shares[k]) for k, label in enumerate(labels)},
             "naive_shares": {label: float(naive[k]) for k, label in enumerate(vd.known)},
             "bounds": {label: [float(bounds[k, 0]), float(bounds[k, 1])] for k, label in enumerate(labels)}}
    if vd.has_other:
        block["known_only_shares"] = {label: float(known_only[k]) for k, label in enumerate(vd.known)}
    if replicates is not None:
        summary = _summarize_replicates(replicates, shares)
        if summary is not None:
            block["se"] = {label: float(summary["se"][k]) for k, label in enumerate(labels)}
            block["ci95"] = {label: [float(summary["ci_lo"][k]), float(summary["ci_hi"][k])]
                             for k, label in enumerate(labels)}
            block["joint95_halfwidth"] = {label: float(summary["halfwidth"][k])
                                          for k, label in enumerate(labels)}
            block["covariance"] = {"order": list(labels), "matrix": summary["cov"].tolist()}
    return block


def _merged_block(groups, share_count, share_weight, reps_count, reps_weight, vd):
    if not groups:
        return {"groups": [], "note": "各单元未发现需要合并报告的产地对"}
    index = vd.class_index
    out = {"groups": groups,
           "note": "列出的产地组在至少一个包装线×品种单元中几乎无法区分，合并比例可识别，拆分不可信"}
    for name, shares, reps in (("count", share_count, reps_count), ("weight", share_weight, reps_weight)):
        entry = {}
        for group in groups:
            members = [index[label] for label in group]
            key = "+".join(group)
            entry[key] = {"share": float(sum(shares[m] for m in members))}
            if reps is not None:
                merged_reps = reps[:, members].sum(axis=1)
                merged_reps = merged_reps[np.isfinite(merged_reps)]
                if merged_reps.size >= 2:
                    entry[key]["ci95"] = [float(np.percentile(merged_reps, 2.5)),
                                          float(np.percentile(merged_reps, 97.5))]
        out[name] = entry
    return out


def analyze_version(vd, replicates, rng, alpha, merge_tau):
    """单个识别器版本的完整校正分析。"""
    if vd.n_reviews == 0:
        return {"status": "evidence_insufficient",
                "reason": "证据截点内没有该识别器版本的复核样本，无法校正"}
    n_cells = len(vd.cells)
    n_periods = len(vd.periods)
    K, J = vd.n_classes, vd.n_known

    observed = estimate_confusion(vd, np.ones(vd.n_reviews, dtype=int), alpha)

    cell_of_s = np.tile(np.arange(n_cells), n_periods)
    t_of_s = np.repeat(np.arange(n_periods), n_cells)
    n_s = len(cell_of_s)
    s_of_box = vd.box_period * n_cells + vd.box_cell
    flat = s_of_box * J + vd.box_pred
    counts = np.bincount(flat, minlength=n_s * J).reshape(n_s, J).astype(float)
    wcounts = np.bincount(flat, weights=vd.box_weight, minlength=n_s * J).reshape(n_s, J)
    n_tc = counts.sum(axis=1)
    w_tc = wcounts.sum(axis=1)
    n_t = np.bincount(t_of_s, weights=n_tc, minlength=n_periods)
    w_t = np.bincount(t_of_s, weights=w_tc, minlength=n_periods)

    matrices = observed["smoothed"][cell_of_s]
    both = em_batch(np.concatenate([matrices, matrices]), np.concatenate([counts, wcounts]))[0]
    p_count, p_weight = both[:n_s], both[n_s:]
    p_known_count, _ = em_batch(matrices[:, :vd.n_known, :], counts)
    p_known_weight, _ = em_batch(matrices[:, :vd.n_known, :], wcounts)

    def period_shares(p_s, cell_sizes, period_sizes):
        out = np.full((n_periods, p_s.shape[1]), np.nan)
        for t in range(n_periods):
            if period_sizes[t] > 0:
                rows = t_of_s == t
                out[t] = (p_s[rows] * cell_sizes[rows, None]).sum(axis=0) / period_sizes[t]
        return out

    share_count = period_shares(p_count, n_tc, n_t)
    share_weight = period_shares(p_weight, w_tc, w_t)
    share_known_count = period_shares(p_known_count, n_tc, n_t)
    share_known_weight = period_shares(p_known_weight, w_tc, w_t)
    naive_count = np.zeros((n_periods, J))
    np.add.at(naive_count, t_of_s, counts)
    naive_count /= np.maximum(n_t[:, None], 1.0)
    naive_weight = np.zeros((n_periods, J))
    np.add.at(naive_weight, t_of_s, wcounts)
    naive_weight /= np.maximum(w_t[:, None], 1e-300)
    naive_weight[w_t <= 0] = np.nan

    # 代数可行范围（按单元求解后按单元规模加权汇总）
    bounds_count = np.zeros((n_periods, K, 2))
    bounds_weight = np.zeros((n_periods, K, 2))
    for t in range(n_periods):
        for c in range(n_cells):
            s = t * n_cells + c
            if n_tc[s] > 0:
                cell_bounds = lp_bounds(observed["smoothed"][c], counts[s] / n_tc[s], p_count[s])
                for k, (lo, hi) in enumerate(cell_bounds):
                    bounds_count[t, k, 0] += n_tc[s] / n_t[t] * lo
                    bounds_count[t, k, 1] += n_tc[s] / n_t[t] * hi
            if w_tc[s] > 0 and w_t[t] > 0:
                cell_bounds = lp_bounds(observed["smoothed"][c], wcounts[s] / w_tc[s], p_weight[s])
                for k, (lo, hi) in enumerate(cell_bounds):
                    bounds_weight[t, k, 0] += w_tc[s] / w_t[t] * lo
                    bounds_weight[t, k, 1] += w_tc[s] / w_t[t] * hi

    # 自助法：重抽复核（按设计层）与到货箱（按时期×单元）
    reps_count = np.full((replicates, n_periods, K), np.nan)
    reps_weight = np.full((replicates, n_periods, K), np.nan)
    if replicates > 0:
        strata = [np.where(vd.rev_stratum == name)[0] for name in sorted(set(vd.rev_stratum))]
        box_groups = [np.where(s_of_box == s)[0] for s in range(n_s)]
        for b in range(replicates):
            multiplicities = np.zeros(vd.n_reviews, dtype=int)
            for idx in strata:
                if len(idx):
                    multiplicities[idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)))
            est_b = estimate_confusion(vd, multiplicities, alpha)
            mat_b = est_b["smoothed"][cell_of_s]
            box_mult = np.zeros(vd.n_boxes, dtype=int)
            for idx in box_groups:
                if len(idx):
                    box_mult[idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)))
            counts_b = np.bincount(flat, weights=box_mult, minlength=n_s * J).reshape(n_s, J)
            wcounts_b = np.bincount(flat, weights=box_mult * vd.box_weight,
                                    minlength=n_s * J).reshape(n_s, J)
            n_b = counts_b.sum(axis=1)
            w_b = wcounts_b.sum(axis=1)
            n_bt = np.bincount(t_of_s, weights=n_b, minlength=n_periods)
            w_bt = np.bincount(t_of_s, weights=w_b, minlength=n_periods)
            both_b = em_batch(np.concatenate([mat_b, mat_b]),
                              np.concatenate([counts_b, wcounts_b]))[0]
            reps_count[b] = period_shares(both_b[:n_s], n_b, n_bt)
            reps_weight[b] = period_shares(both_b[n_s:], w_b, w_bt)

    # 可识别性：单元诊断与时期级合并组（各单元合并组的并集，保守口径）
    cell_diagnostics = {}
    union_pairs = set()
    for c in range(n_cells):
        diag = matrix_diagnostics(observed["smoothed"][c], observed["raw"][c], vd.labels, merge_tau)
        cell_diagnostics[vd.cells[c]] = diag
        for group in diag["merged_groups"]:
            for a in group:
                for b_ in group:
                    if a < b_:
                        union_pairs.add((a, b_))
    merged_groups = _groups_from_pairs(vd.labels, union_pairs)

    ridge_note = None
    if any(cell_diagnostics[cell]["null_dimension"] > 0 for cell in vd.cells):
        involved = "、".join("+".join(group) for group in merged_groups) or "、".join(vd.labels)
        ridge_note = ("存在代数上不可拆分的方向（" + involved +
                      "），点估计沿该方向由初始化决定，请以合并比例与可行范围为准")

    periods = {}
    for t in range(n_periods):
        key = vd.periods[t]
        warnings = []
        if n_t[t] < 30:
            warnings.append("该期到货箱数较少，区间较宽")
        if not w_t[t] > 0:
            warnings.append("该期重量为零，重量口径不可用")
        if ridge_note:
            warnings.append(ridge_note)
        periods[key] = {
            "boxes": int(n_t[t]), "weight_kg": float(w_t[t]),
            "count": _share_block(share_count[t], naive_count[t], share_known_count[t],
                                  bounds_count[t], reps_count[:, t, :] if replicates else None, vd),
            "weight": _share_block(share_weight[t], naive_weight[t], share_known_weight[t],
                                   bounds_weight[t], reps_weight[:, t, :] if replicates else None, vd),
            "merged": _merged_block(merged_groups, share_count[t], share_weight[t],
                                    reps_count[:, t, :] if replicates else None,
                                    reps_weight[:, t, :] if replicates else None, vd),
            "warnings": warnings}

    if replicates > 0:
        mechanism = _check_mechanism_stability(vd, max(replicates, 1000), rng)
    else:
        mechanism = {"verdict": "unavailable", "p_value": None, "note": "重抽样次数为零，未运行"}
    moments = _check_score_moments(vd, observed, p_known_count, cell_of_s, n_tc, replicates, rng, alpha)
    checks = {"mechanism_stability": mechanism, "score_moment_consistency": moments}

    cells_block = {}
    for c in range(n_cells):
        diag = cell_diagnostics[vd.cells[c]]
        notes = []
        if diag["null_dimension"] > 0:
            notes.append("真值类数超过可区分方向，点估计沿零空间方向不可识别，以可行范围与合并比例为准")
        if diag["condition"] is None or (diag["condition"] and diag["condition"] > 30):
            notes.append("混淆矩阵接近奇异，部分产地拆分依赖强假设")
        for a, b_, tv in diag["row_total_variation"]:
            if tv is not None and tv < merge_tau:
                notes.append(f"{a} 与 {b_} 的识别行为几乎相同（全变差距离 {tv:.3f}），建议仅报告合并比例")
        for k, label in enumerate(vd.labels):
            if observed["levels"][c, k] == "unidentified":
                notes.append(f"类别 {label} 无复核证据，其混淆行不可识别，点估计依赖均匀行假设")
        cells_block[vd.cells[c]] = {
            "boxes": int((vd.box_cell == c).sum()),
            "reviews": int((vd.rev_cell == c).sum()),
            "confusion": {"rows": list(vd.labels), "columns": list(vd.known),
                          "matrix": observed["smoothed"][c].tolist(),
                          "row_level": {vd.labels[k]: observed["levels"][c, k] for k in range(K)},
                          "row_evidence": {vd.labels[k]: float(observed["evidence"][c, k]) for k in range(K)}},
            "diagnostics": diag, "notes": notes}

    other_note = None
    if vd.has_other:
        other_note = ("复核中出现识别器标签集外的产地，已并入 other 类；其混淆行由 "
                      f"{sum(vd.other_sources.values())} 条复核证据全局合并估计，"
                      "other 与相似已知类的拆分不可识别时以合并比例与可行范围为准")
    version_notes = []
    if ridge_note:
        version_notes.append(ridge_note)
    if other_note:
        version_notes.append(other_note)
    return {"status": "ok",
            "classes": list(vd.labels),
            "other": {"included": vd.has_other, "evidence_reviews": sum(vd.other_sources.values()),
                      "sources": dict(vd.other_sources), "note": other_note},
            "cells": cells_block,
            "periods": periods,
            "checks": checks,
            "notes": version_notes}


def build_report(catalog, period="week", as_of=None, replicates=500, seed=DEFAULT_SEED,
                 alpha=0.5, merge_tau=0.25):
    """生成总体产地占比校正报告。"""
    if period not in PERIOD_KINDS:
        raise CorrectionError("period 仅支持 week 或 month")
    if not 0 <= replicates <= MAX_REPLICATES:
        raise CorrectionError(f"replicates 应在 0 至 {MAX_REPLICATES} 之间")
    if alpha <= 0:
        raise CorrectionError("平滑参数 alpha 应为正数")
    if not 0 < merge_tau < 1:
        raise CorrectionError("merge_tau 应在 (0, 1) 之间")
    moment = parse_as_of(as_of)
    data = load_inputs(catalog, moment)
    rng = np.random.default_rng(seed)

    boxes_by_version = {}
    for box in data["boxes"].values():
        boxes_by_version.setdefault(box["version"], []).append(box)
    reviews_by_version = {}
    for record in data["reviews"]:
        reviews_by_version.setdefault(record["box"]["version"], []).append(record)

    versions = {}
    for version in sorted(boxes_by_version):
        vd = VersionData(version, data["known"][version], boxes_by_version[version],
                         reviews_by_version.get(version, []), period)
        versions[version] = analyze_version(vd, replicates, rng, alpha, merge_tau)

    status = "ok" if any(block.get("status") == "ok" for block in versions.values()) else "evidence_insufficient"
    days = [box["day"] for box in data["boxes"].values()]
    latest = max((record["confirmed_at"] for record in data["reviews"]), default=None)
    report = {
        "status": status,
        "provenance": {
            "source_fingerprint": catalog.fingerprint,
            "estimator": ESTIMATOR_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "as_of": moment.isoformat() if moment else None,
            "period": period,
            "parameters": {"replicates": replicates, "seed": seed, "smoothing_alpha": alpha,
                           "merge_threshold": merge_tau},
            "reviews": {"total": data["review_total"], "used": len(data["reviews"]),
                        "excluded_after_as_of": data["review_excluded"],
                        "ids": sorted(record["id"] for record in data["reviews"]),
                        "latest_confirmation": latest.isoformat() if latest else None},
            "models": {version: {"known_classes": data["known"][version],
                                 "boxes": len(boxes_by_version[version]),
                                 "reviews": len(reviews_by_version.get(version, []))}
                       for version in sorted(boxes_by_version)},
            "arrival_days": {"first": min(days) if days else None, "last": max(days) if days else None},
        },
        "versions": versions,
    }
    return _jsonable(report)


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
