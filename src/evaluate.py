"""Test-set evaluation: metrics, classification report, confusion matrix, timing, size.

Usage:  python -m src.evaluate --model custom_cnn
        python -m src.evaluate --model efficientnet

The test set is only used here, after training and model selection are finished.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from . import config as C
from . import data_loader as D
from .utils import get_logger, load_json, save_json

log = get_logger("evaluate")


def predict_dataset(model, ds) -> tuple[np.ndarray, np.ndarray]:
    ys, ps = [], []
    for x, y in ds:
        ps.append(np.asarray(model(x, training=False)))
        ys.append(np.asarray(y))
    return np.concatenate(ys), np.concatenate(ps)


def compute_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    """All metrics with explicit, fixed label order ``0..K-1`` == ``class_names``."""
    labels = list(range(len(class_names)))
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {"accuracy": float(accuracy_score(y_true, y_pred)), "n_samples": int(len(y_true)),
           "class_names": class_names}
    for avg in ("macro", "weighted"):
        ap, ar, af, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        out.update({f"{avg}_precision": float(ap), f"{avg}_recall": float(ar), f"{avg}_f1": float(af)})
    out["per_class"] = {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                        for i, c in enumerate(class_names)}
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    out["classification_report"] = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                                         digits=4, zero_division=0)
    return out


def model_size_mb(path: Path) -> float:
    return Path(path).stat().st_size / 2**20


def measure_inference(model, x: np.ndarray, warmup: int = 5, repeats: int = 3, single_n: int = 50) -> dict:
    """Latency on pre-loaded, pre-resized float32 arrays (excludes file IO / decoding).

    * ``batch1_ms``: median per-image latency over ``single_n`` single-image calls
    * ``batched_ms_per_image``: best-of-``repeats`` full pass in batches of 16
    Warm-up calls are made first so graph tracing / GPU init is excluded.
    """
    import tensorflow as tf

    infer = tf.function(lambda t: model(t, training=False), reduce_retracing=True)
    one = tf.constant(x[:1])
    for _ in range(warmup):
        infer(one).numpy()
    lat = []
    for i in range(min(single_n, len(x))):
        t = tf.constant(x[i:i + 1])
        t0 = time.perf_counter()
        infer(t).numpy()
        lat.append(time.perf_counter() - t0)
    batches = [tf.constant(x[i:i + 16]) for i in range(0, len(x), 16)]
    for b in batches[:2]:
        infer(b).numpy()
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for b in batches:
            infer(b).numpy()
        runs.append(time.perf_counter() - t0)
    return {"batch1_ms_median": float(np.median(lat) * 1e3), "batch1_ms_p90": float(np.percentile(lat, 90) * 1e3),
            "batched_ms_per_image": float(min(runs) / len(x) * 1e3), "n_images": int(len(x)),
            "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
            "includes_preprocessing": False, "warmup_calls": warmup}


def evaluate_model(model_name: str, split: str = "test") -> dict:
    import keras

    from .plots import plot_confusion_matrix
    from .preprocessing import make_dataset

    model_dir = C.MODELS_DIR / model_name
    path = model_dir / "model.keras"
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Train it first: python -m src.train --model {model_name}")
    try:
        model = keras.models.load_model(path)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to load model {path}: {e}") from e

    class_names, _ = D.load_class_mapping()
    saved = load_json(model_dir / "class_mapping.json")["class_names"]
    if saved != class_names:
        raise ValueError(f"Class order mismatch: model {saved} vs manifest {class_names}")

    df = D.load_split(split)
    ds = make_dataset(df, C.DataConfig().image_size, 16, training=False, cache=False)
    y_true, y_prob = predict_dataset(model, ds)
    y_pred = y_prob.argmax(1)
    m = compute_metrics(y_true, y_pred, class_names)

    from .preprocessing import prepare_batch

    x = prepare_batch([C.PROJECT_ROOT / f for f in df["filepath"]])
    m["inference"] = measure_inference(model, x)
    m["model_size_mb"] = model_size_mb(path)
    m["params"] = int(model.count_params())
    m["split"] = split
    summ = load_json(model_dir / "training_summary.json")
    m["train_time_s"] = summ.get("train_time_s")
    m["selected_stage"] = summ.get("selected_stage")

    out = C.METRICS_DIR / model_name
    out.mkdir(parents=True, exist_ok=True)
    save_json({k: v for k, v in m.items() if k != "classification_report"}, out / f"{split}_metrics.json")
    (out / f"{split}_classification_report.txt").write_text(m["classification_report"], encoding="utf-8")
    pd.DataFrame({"filepath": df["filepath"], "true": [class_names[i] for i in y_true],
                  "pred": [class_names[i] for i in y_pred], "confidence": y_prob.max(1),
                  **{f"p_{c}": y_prob[:, i] for i, c in enumerate(class_names)}}).to_csv(
        out / f"{split}_predictions.csv", index=False)
    plot_confusion_matrix(np.array(m["confusion_matrix"]), class_names,
                          C.PLOTS_DIR / model_name / f"{split}_confusion_matrix.png",
                          f"{model_name} - {split} confusion matrix (n={len(y_true)})")
    log.info("%s on %s:\n%s", model_name, split, m["classification_report"])
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="folder name under artifacts/models (custom_cnn | efficientnet)")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    evaluate_model(a.model, a.split)
