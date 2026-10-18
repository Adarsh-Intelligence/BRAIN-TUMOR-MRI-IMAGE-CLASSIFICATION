"""Plotting helpers for training curves, confusion matrices and comparisons."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_history(history: dict, out_prefix: Path, title: str, stage_boundary: int | None = None) -> None:
    """Save ``<prefix>_accuracy.png`` and ``<prefix>_loss.png``."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["loss"]) + 1)
    for metric, ylabel in (("accuracy", "Accuracy"), ("loss", "Loss")):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(epochs, history[metric], "o-", ms=3, label=f"train {metric}")
        ax.plot(epochs, history[f"val_{metric}"], "o-", ms=3, label=f"val {metric}")
        best = int(np.argmin(history["val_loss"])) + 1
        ax.axvline(best, color="grey", ls=":", lw=1, label=f"best val_loss (ep {best})")
        if stage_boundary:
            ax.axvline(stage_boundary + 0.5, color="red", ls="--", lw=1, label="start fine-tuning")
        ax.set(title=f"{title} - {ylabel}", xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_{metric}.png", dpi=130)
        plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], path: Path, title: str) -> None:
    cm = np.asarray(cm)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    annot = np.array([[f"{cm[i, j]}\n{norm[i, j]:.0%}" for j in range(cm.shape[1])] for i in range(cm.shape[0])])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(norm, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "row-normalized"},
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_per_class_comparison(per_class: pd.DataFrame, path: Path) -> None:
    """``per_class`` columns: model, class_name, precision, recall, f1."""
    m = per_class.melt(id_vars=["model", "class_name"], value_vars=["precision", "recall", "f1"], var_name="metric")
    g = sns.catplot(data=m, x="class_name", y="value", hue="model", col="metric", kind="bar",
                    height=3.6, aspect=1.1, palette="Set2")
    g.set(ylim=(0, 1.05))
    for ax in g.axes.flat:
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", fontsize=7)
        ax.tick_params(axis="x", rotation=20)
    g.figure.suptitle("Per-class test performance", y=1.03)
    g.savefig(path, dpi=130)
    plt.close(g.figure)


def plot_model_comparison(summary: pd.DataFrame, path: Path) -> None:
    """``summary`` index: metric names, columns: models (values in [0,1])."""
    ax = summary.plot.bar(figsize=(8, 4.2), rot=0, color=["#4C72B0", "#DD8452", "#55A868"][: summary.shape[1]])
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=7)
    ax.set(ylim=(0, 1.08), title="Test-set comparison", ylabel="Score")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
