import numpy as np
import pytest

from src.compare_models import select_model
from src.evaluate import compute_metrics

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]


def test_compute_metrics_known_values():
    y_true = [0, 0, 1, 1, 2, 2, 3, 3]
    y_pred = [0, 0, 1, 0, 2, 2, 3, 1]
    m = compute_metrics(y_true, y_pred, CLASSES)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["per_class"]["glioma"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["glioma"]["recall"] == pytest.approx(1.0)
    assert m["per_class"]["pituitary"]["recall"] == pytest.approx(0.5)
    assert m["macro_recall"] == pytest.approx(np.mean([1, 0.5, 1, 0.5]))
    assert list(m["per_class"]) == CLASSES


def test_confusion_matrix_dims_and_order_with_missing_class():
    m = compute_metrics([0, 1, 1], [0, 1, 0], CLASSES)  # classes 2,3 absent
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (4, 4)
    assert cm[1, 0] == 1 and cm[1, 1] == 1 and cm[0, 0] == 1 and cm[2:].sum() == 0


def _res(val_f1, worst_recall, test_f1, ms):
    pc = {c: {"recall": 1.0} for c in CLASSES}
    pc["glioma"]["recall"] = worst_recall
    return {"val_metrics": {"macro_f1": val_f1, "per_class": pc}, "macro_f1": test_f1,
            "per_class": pc, "inference": {"batch1_ms_median": ms}}


def test_select_model_uses_validation_not_test():
    res = {"custom_cnn": _res(0.80, 0.7, 0.95, 5), "efficientnet": _res(0.90, 0.8, 0.85, 20)}
    best, why = select_model(res)
    assert best == "efficientnet" and "test-set selection bias" in why


def test_select_model_prefers_better_minority_recall_when_close():
    res = {"custom_cnn": _res(0.905, 0.60, 0.9, 5), "efficientnet": _res(0.900, 0.85, 0.9, 20)}
    best, _ = select_model(res)
    assert best == "efficientnet"
