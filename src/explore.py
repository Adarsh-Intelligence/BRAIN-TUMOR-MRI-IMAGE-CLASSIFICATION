"""Dataset exploration outputs: summaries, charts, sample grids, integrity reports."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from . import config as C
from .utils import get_logger

log = get_logger("explore")


def class_distribution(df: pd.DataFrame, split_col: str) -> pd.DataFrame:
    t = pd.crosstab(df["class_name"], df[split_col], margins=True, margins_name="total")
    return t


def plot_class_distribution(df: pd.DataFrame, split_col: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.countplot(data=df, x="class_name", hue=split_col, ax=ax,
                  order=sorted(df["class_name"].unique()), palette="viridis")
    for c in ax.containers:
        ax.bar_label(c, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Images")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_samples(df: pd.DataFrame, path: Path, per_class: int = 6, seed: int = C.SEED) -> None:
    classes = sorted(df["class_name"].unique())
    fig, axes = plt.subplots(len(classes), per_class, figsize=(per_class * 2, len(classes) * 2.1))
    for r, cls in enumerate(classes):
        sub = df[df["class_name"] == cls].sample(per_class, random_state=seed)
        for c, fp in enumerate(sub["filepath"]):
            ax = axes[r, c]
            ax.imshow(Image.open(C.PROJECT_ROOT / fp).convert("RGB"))
            ax.axis("off")
            if c == 0:
                ax.set_title(cls, loc="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def dimension_summary(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["readable"]]
    return (ok.groupby(["width", "height", "mode", "format"]).size()
              .rename("count").reset_index().sort_values("count", ascending=False))


def plot_duplicate_pairs(pairs: pd.DataFrame, path: Path, n: int = 8) -> None:
    if pairs.empty:
        return
    sub = pairs.sort_values("hamming", ascending=False).head(n) if "hamming" in pairs else pairs.head(n)
    fig, axes = plt.subplots(len(sub), 2, figsize=(5, 2.4 * len(sub)))
    axes = np.atleast_2d(axes)
    for i, (_, r) in enumerate(sub.iterrows()):
        for k, side in enumerate("ab"):
            ax = axes[i, k]
            ax.imshow(Image.open(C.PROJECT_ROOT / r[f"filepath_{side}"]).convert("RGB"))
            ax.axis("off")
            label = f"{r[f'class_{side}']} / {r[f'split_{side}']}"
            if side == "a" and "hamming" in r:
                label += f"  (d={r['hamming']})"
            ax.set_title(label, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
