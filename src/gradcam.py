"""Grad-CAM visualisation (model interpretation, optional stage).

Usage:  python -m src.gradcam --model efficientnet --n 8

What the heatmap means: regions of the last convolutional feature map whose
activation most increases the score of the explained class (gradient-weighted).
It is a coarse (7x7 or 14x14) attribution, **not** a tumor segmentation and not
proof that the model "found" a tumor; it can highlight shortcuts or artefacts.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C
from .utils import get_logger, load_json

log = get_logger("gradcam")

# last spatial feature layer (top-level layer name) per architecture
TARGET_LAYERS = {"custom_cnn": "b4_relu2", "efficientnetb0_transfer": "efficientnetb0"}


def gradcam_heatmap(model, image: np.ndarray, class_index: int | None = None,
                    target_layer: str | None = None) -> tuple[np.ndarray, int, np.ndarray]:
    """Return (heatmap in [0,1] at feature resolution, explained class, probabilities).

    Both supported models are linear chains of top-level layers, so we run the
    layers sequentially and capture the target layer's output on the tape.
    """
    import tensorflow as tf

    target_layer = target_layer or TARGET_LAYERS.get(model.name)
    if target_layer is None:
        raise ValueError(f"No Grad-CAM target layer known for model '{model.name}'")
    x = tf.convert_to_tensor(image[None].astype("float32"))
    with tf.GradientTape() as tape:
        h, feats = x, None
        for layer in model.layers[1:]:  # skip InputLayer
            h = layer(h, training=False)
            if layer.name == target_layer:
                feats = h
                tape.watch(feats)
        probs = h
        if feats is None:
            raise ValueError(f"Layer '{target_layer}' not found in model")
        ci = int(tf.argmax(probs[0])) if class_index is None else class_index
        score = probs[:, ci]
    grads = tape.gradient(score, feats)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(feats[0] * weights, axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam, ci, probs[0].numpy()


def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    import cv2

    heat = cv2.resize(cam.astype("float32"), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)[..., ::-1]
    return np.uint8(np.clip((1 - alpha) * image + alpha * heat, 0, 255))


def run(model_name: str, n: int = 8, seed: int = C.SEED) -> None:
    import keras
    import pandas as pd

    from .preprocessing import load_image

    model = keras.models.load_model(C.MODELS_DIR / model_name / "model.keras")
    class_names = load_json(C.MODELS_DIR / model_name / "class_mapping.json")["class_names"]
    preds = pd.read_csv(C.METRICS_DIR / model_name / "test_predictions.csv")
    # half correct, half misclassified (if available), across classes
    wrong = preds[preds.true != preds.pred]
    right = preds[preds.true == preds.pred].groupby("true", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 2), random_state=seed))
    sel = pd.concat([right.head(n - min(len(wrong), n // 2)), wrong.sample(min(len(wrong), n // 2), random_state=seed)])
    fig, axes = plt.subplots(len(sel), 2, figsize=(5.5, 2.8 * len(sel)))
    axes = np.atleast_2d(axes)
    for i, (_, r) in enumerate(sel.iterrows()):
        img = load_image(C.PROJECT_ROOT / r.filepath)
        cam, ci, p = gradcam_heatmap(model, img.astype("float32"))
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"true: {r.true}", fontsize=8)
        axes[i, 1].imshow(overlay(img, cam))
        axes[i, 1].set_title(f"pred: {class_names[ci]} ({p[ci]:.2f})", fontsize=8,
                             color="green" if class_names[ci] == r.true else "red")
        for a in axes[i]:
            a.axis("off")
    fig.suptitle(f"Grad-CAM - {model_name}\n(attribution, not segmentation)", fontsize=9)
    fig.tight_layout()
    out = C.PLOTS_DIR / model_name / "gradcam_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    log.info("Saved %s", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    run(a.model, a.n)
