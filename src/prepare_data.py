"""Stage 1+2 CLI: explore the dataset, detect duplicates, build leakage-safe split manifests.

Usage:  python -m src.prepare_data [--phash-threshold 4]
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import config as C
from . import data_loader as D
from . import explore as E
from .utils import get_logger, save_json, set_global_seed

log = get_logger("prepare_data")


def main(phash_threshold: int | None = None) -> pd.DataFrame:
    cfg = C.get_config()
    if phash_threshold is not None:
        cfg.split.phash_threshold = phash_threshold
    set_global_seed(cfg.split.seed)
    C.ensure_dirs()
    out = C.EXPLORATION_DIR

    # ---- discovery & label cross-check
    df = D.discover_images()
    class_names = D.get_class_names(df)
    log.info("Detected classes: %s", class_names)
    mismatch = D.check_csv_labels(df)
    mismatch.to_csv(out / "csv_label_mismatches.csv", index=False)
    log.info("Folder vs _classes.csv label mismatches: %d", len(mismatch))

    # ---- integrity
    df = D.inspect_all(df)
    invalid = df[~df["readable"]][["filepath", "class_name", "original_split", "error"]]
    invalid.to_csv(out / "invalid_images.csv", index=False)
    if len(invalid):
        log.warning("Excluding %d unreadable images (listed in invalid_images.csv)", len(invalid))
    valid = df[df["readable"]].copy()

    # ---- duplicates & grouping
    valid, rep = D.assign_groups(valid, cfg.split.phash_threshold, cfg.split.cross_class_phash_threshold)
    rep.exact_pairs.to_csv(out / "exact_duplicate_pairs.csv", index=False)
    rep.near_pairs.to_csv(out / "near_duplicate_pairs.csv", index=False)
    rep.label_conflict_groups.to_csv(out / "label_conflict_groups.csv", index=False)
    E.plot_duplicate_pairs(rep.near_pairs[rep.near_pairs["grouped"]], out / "near_duplicate_examples.png")

    # leakage present in the ORIGINAL Roboflow split
    orig_leak = {k: int((valid.groupby(k)["original_split"].nunique() > 1).sum())
                 for k in ("source_stem", "md5", "group_id")}
    log.info("Original-split leakage (values spanning >1 split): %s", orig_leak)

    # ---- conflicting labels: never let one image carry two labels
    n_conflict = int(valid["label_conflict"].sum())
    if n_conflict:
        log.warning("Excluding %d images in %d groups with conflicting labels (see label_conflict_groups.csv)",
                    n_conflict, valid.loc[valid["label_conflict"], "group_id"].nunique())
    clean = valid[~valid["label_conflict"]].copy()

    # ---- split
    clean = D.grouped_stratified_split(clean, cfg.split)
    D.save_manifests(clean, class_names)

    # ---- summaries & plots
    E.class_distribution(df, "original_split").to_csv(out / "class_distribution_original.csv")
    dist_new = E.class_distribution(clean, "split")
    dist_new.to_csv(out / "class_distribution_new_split.csv")
    E.dimension_summary(df).to_csv(out / "image_dimension_summary.csv", index=False)
    E.plot_class_distribution(df, "original_split", out / "class_distribution_original.png",
                              "Original Roboflow split (contains leakage)")
    E.plot_class_distribution(clean, "split", out / "class_distribution_new_split.png",
                              "New grouped stratified split (70/15/15)")
    E.plot_samples(clean, out / "sample_images.png")

    summary = {
        "dataset_dir": C.DATASET_DIR_NAME,
        "class_names": class_names,
        "total_files": len(df),
        "unreadable_files": len(invalid),
        "csv_label_mismatches": len(mismatch),
        "unique_source_stems": int(valid["source_stem"].nunique()),
        "unique_md5": int(valid["md5"].nunique()),
        "duplicate_groups": int(valid["group_id"].nunique()),
        "exact_duplicate_pairs": len(rep.exact_pairs),
        "near_duplicate_pairs": len(rep.near_pairs),
        "phash_threshold": cfg.split.phash_threshold,
        "original_split_leakage": orig_leak,
        "excluded_label_conflict_images": n_conflict,
        "images_in_new_split": len(clean),
        "new_split_counts": clean["split"].value_counts().to_dict(),
        "new_split_fraction": (clean["split"].value_counts(normalize=True).round(4)).to_dict(),
        "per_class_total": clean["class_name"].value_counts().to_dict(),
        "imbalance_ratio_max_min": round(clean["class_name"].value_counts().max()
                                         / clean["class_name"].value_counts().min(), 3),
    }
    save_json(summary, out / "dataset_summary.json")
    pd.DataFrame([{k: v for k, v in summary.items() if not isinstance(v, (dict, list))}]).to_csv(
        out / "dataset_summary.csv", index=False)
    log.info("New split:\n%s", dist_new.to_string())
    return clean


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phash-threshold", type=int, default=None)
    main(ap.parse_args().phash_threshold)
