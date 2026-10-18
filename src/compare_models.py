"""Compare evaluated models and write the final comparison report.

Usage:  python -m src.compare_models [--models custom_cnn efficientnet]

All numbers are read from ``artifacts/metrics/<model>/test_metrics.json`` and the
validation results stored at training time — nothing is hardcoded.
"""
from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from . import config as C
from .plots import plot_model_comparison, plot_per_class_comparison
from .utils import get_logger, load_json, save_json

log = get_logger("compare")

DISPLAY = {"custom_cnn": "Custom CNN", "efficientnet": "EfficientNetB0"}


def load_results(models: list[str]) -> dict:
    res = {}
    for m in models:
        p = C.METRICS_DIR / m / "test_metrics.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run: python -m src.evaluate --model {m}")
        res[m] = load_json(p)
        summ = load_json(C.MODELS_DIR / m / "training_summary.json")
        res[m]["val_metrics"] = summ["stages"][summ["selected_stage"]]["val_metrics"]
        res[m]["stages"] = summ["stages"]
    return res


def comparison_table(res: dict) -> pd.DataFrame:
    rows = {}
    for m, r in res.items():
        min_rec_cls = min(r["per_class"], key=lambda c: r["per_class"][c]["recall"])
        rows[DISPLAY.get(m, m)] = {
            "Validation Macro F1": r["val_metrics"]["macro_f1"],
            "Test Accuracy": r["accuracy"],
            "Macro Precision": r["macro_precision"],
            "Macro Recall": r["macro_recall"],
            "Macro F1-score": r["macro_f1"],
            "Weighted F1-score": r["weighted_f1"],
            "Lowest class recall": f"{r['per_class'][min_rec_cls]['recall']:.4f} ({min_rec_cls})",
            "Training Time (min)": (r.get("train_time_s") or float("nan")) / 60,
            "Inference ms/image (batch 1, median)": r["inference"]["batch1_ms_median"],
            "Inference ms/image (batch 16)": r["inference"]["batched_ms_per_image"],
            "Model Size (MB)": r["model_size_mb"],
            "Parameters": r["params"],
        }
    return pd.DataFrame(rows)


def select_model(res: dict, f1_tolerance: float = 0.01, recall_gap: float = 0.05) -> tuple[str, str]:
    """Choose on validation macro-F1; consult test only to flag disagreement.

    Rule: highest validation macro-F1. If two models are within ``f1_tolerance``,
    prefer the one with better worst-class validation recall; if still tied, the
    smaller/faster model. A model whose worst-class recall is ``recall_gap`` lower
    than the other's is not preferred on accuracy alone.
    """
    cands = sorted(res, key=lambda m: res[m]["val_metrics"]["macro_f1"], reverse=True)
    best, why = cands[0], ["highest validation macro-F1"]
    if len(cands) > 1:
        other = cands[1]
        dv = res[best]["val_metrics"]["macro_f1"] - res[other]["val_metrics"]["macro_f1"]
        wb = min(v["recall"] for v in res[best]["val_metrics"]["per_class"].values())
        wo = min(v["recall"] for v in res[other]["val_metrics"]["per_class"].values())
        if dv <= f1_tolerance:
            if wo > wb + 1e-9:
                best, why = other, [f"validation macro-F1 within {f1_tolerance}; better worst-class recall"]
            elif abs(wo - wb) < 1e-9 and res[other]["inference"]["batch1_ms_median"] < res[best]["inference"]["batch1_ms_median"]:
                best, why = other, ["tied on validation; faster inference"]
        elif wo - wb > recall_gap:
            why.append(f"WARNING: worst-class recall is {wo - wb:.3f} lower than {DISPLAY.get(other, other)}")
        test_best = max(res, key=lambda m: res[m]["macro_f1"])
        if test_best != best:
            why.append(f"note: {DISPLAY.get(test_best, test_best)} has higher test macro-F1; selection is "
                       f"kept on validation to avoid test-set selection bias")
    return best, "; ".join(why)


def write_report(res: dict, table: pd.DataFrame, best: str, why: str) -> str:
    split = pd.read_csv(C.SPLITS_DIR / "all_images.csv")
    dist = pd.crosstab(split["class_name"], split["split"], margins=True)
    ds = load_json(C.EXPLORATION_DIR / "dataset_summary.json")
    fmt = table.copy().astype(object)
    for col in fmt.columns:
        fmt[col] = [f"{v:.4f}" if isinstance(v, float) else (f"{v:,}" if isinstance(v, int) else v) for v in fmt[col]]
    lines = [
        "# NeuroScan AI - Model Comparison Report",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} from saved metric files (no hardcoded values)._",
        "",
        "## Data",
        f"- Classes: {', '.join(ds['class_names'])}",
        f"- {ds['total_files']} images, {ds['unreadable_files']} unreadable, "
        f"{ds['duplicate_groups']} duplicate/near-duplicate groups",
        f"- Original Roboflow split had leakage: {ds['original_split_leakage']['group_id']} groups spanning splits; "
        "the data were re-split (grouped by source image + pHash, stratified 70/15/15).",
        f"- Max/min class ratio: {ds['imbalance_ratio_max_min']} (class weights used in training)",
        "",
        dist.to_markdown(),
        "",
        "## Test-set comparison",
        fmt.to_markdown(),
        "",
        f"Inference timing: {res[next(iter(res))]['inference']['device']}, pre-loaded 224x224 arrays "
        "(decoding/resizing excluded), 5 warm-up calls, median of 50 single-image calls; batched = best of 3 passes.",
        "",
        "## Per-class test results",
    ]
    for m, r in res.items():
        lines += [f"### {DISPLAY.get(m, m)} (stage: {r.get('selected_stage')})", "```",
                  (C.METRICS_DIR / m / "test_classification_report.txt").read_text(encoding="utf-8"), "```",
                  f"![cm](../plots/{m}/test_confusion_matrix.png)", ""]
    if "efficientnet" in res and len(res["efficientnet"]["stages"]) > 1:
        st = res["efficientnet"]["stages"]
        lines += ["## EfficientNetB0: feature extraction vs fine-tuning (validation)",
                  "| Stage | Val macro-F1 | Val accuracy | Best val loss | Epochs | Time (min) |", "|---|---|---|---|---|---|"]
        for k, v in st.items():
            lines.append(f"| {k} | {v['val_metrics']['macro_f1']:.4f} | {v['val_metrics']['accuracy']:.4f} | "
                         f"{v['val_loss']:.4f} | {v['epochs_run']} | {v['train_time_s'] / 60:.1f} |")
        lines.append("")
    lines += ["## Selection", f"**Preferred model: {DISPLAY.get(best, best)}** - {why}.", "",
              "## Limitations",
              "- No patient identifiers: grouping by source filename and perceptual hash approximates, but does not "
              "guarantee, patient-level separation (different slices of one patient may still fall in different splits).",
              "- Single dataset of 2D slices from one public source; no external validation set.",
              "- Results are from a single seed; small test set -> per-class metrics have wide confidence intervals.",
              "- This is a research prototype, not a medical device."]
    text = "\n".join(lines)
    (C.REPORTS_DIR / "model_comparison_report.md").write_text(text, encoding="utf-8")
    return text


def main(models: list[str]) -> None:
    C.ensure_dirs()
    res = load_results(models)
    table = comparison_table(res)
    table.to_csv(C.REPORTS_DIR / "model_comparison.csv")
    per_class = pd.DataFrame([{"model": DISPLAY.get(m, m), "class_name": c, **v}
                              for m, r in res.items() for c, v in r["per_class"].items()])
    per_class.to_csv(C.REPORTS_DIR / "per_class_comparison.csv", index=False)
    plot_per_class_comparison(per_class, C.PLOTS_DIR / "per_class_comparison.png")
    num = table.loc[["Test Accuracy", "Macro Precision", "Macro Recall", "Macro F1-score", "Weighted F1-score"]].astype(float)
    plot_model_comparison(num, C.PLOTS_DIR / "model_comparison.png")
    best, why = select_model(res)
    save_json({"selected_model": best, "reason": why}, C.REPORTS_DIR / "selected_model.json")
    write_report(res, table, best, why)
    log.info("\n%s\nSelected: %s (%s)", table.to_string(), best, why)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["custom_cnn", "efficientnet"])
    main(ap.parse_args().models)
