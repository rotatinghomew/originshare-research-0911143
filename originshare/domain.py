from collections import Counter


def observed_labels(catalog):
    return dict(Counter(record["predicted"] for record in catalog.rows("boxes")))


def overview(catalog):
    return {"识别标签箱数": observed_labels(catalog), "人工复核箱数": len(catalog.rows("reviews")), "模型版本": sorted({record["model_version"] for record in catalog.rows("boxes")})}
