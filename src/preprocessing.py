"""Image loading and model-specific preprocessing.

Input contract for **every saved model**: float32 RGB tensor of shape
``(batch, 224, 224, 3)`` with raw pixel values in **[0, 255]**.

* Custom CNN: a ``Rescaling(1/255)`` layer inside the model maps to [0, 1].
* EfficientNetB0 (Keras 3): the official model contains its own
  rescaling/normalization layers and expects [0, 255] input. We therefore do
  **not** call ``efficientnet.preprocess_input`` (it is a pass-through in Keras 3)
  and never normalize twice.

Keeping normalization inside the models means inference code only has to
resize to 224x224 RGB.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from . import config as C

AUTOTUNE = None  # set lazily to avoid importing TF at module import time


def preprocessing_spec(model_name: str, image_size=(224, 224)) -> dict:
    """Serializable description of preprocessing, saved next to each model."""
    common = {
        "input_shape": [image_size[0], image_size[1], 3],
        "color_mode": "RGB",
        "resize": {"method": "bilinear", "antialias": True, "keep_aspect_ratio": False},
        "input_dtype": "float32",
        "input_value_range": [0, 255],
    }
    if model_name == "custom_cnn":
        common["in_model_normalization"] = "Rescaling(1/255) -> [0, 1]"
    elif model_name.startswith("efficientnet"):
        common["in_model_normalization"] = "Keras EfficientNetB0 built-in Rescaling + Normalization (ImageNet)"
    return common


# --------------------------------------------------------------------------- numpy / PIL path
def load_image(path: str | Path, image_size=(224, 224)) -> np.ndarray:
    """Load an image file safely -> uint8 array (H, W, 3) RGB, resized.

    Raises ``ValueError`` with the file path if the image cannot be decoded.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"Corrupted or unreadable image: {path} ({e})") from e
    validate_image_array(arr, image_size)
    return arr


def validate_image_array(arr: np.ndarray, image_size=(224, 224)) -> None:
    expected = (image_size[0], image_size[1], 3)
    if arr.shape != expected:
        raise ValueError(f"Unexpected image shape {arr.shape}, expected {expected}")


def prepare_batch(paths, image_size=(224, 224)) -> np.ndarray:
    """Stack images into a float32 [0,255] batch matching the model input contract."""
    return np.stack([load_image(p, image_size) for p in paths]).astype(np.float32)


# --------------------------------------------------------------------------- tf.data path
def _decode_resize(path, image_size):
    import tensorflow as tf

    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, image_size, method="bilinear", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)


def make_dataset(df: pd.DataFrame, image_size=(224, 224), batch_size: int = 16, training: bool = False,
                 augmenter=None, seed: int = C.SEED, cache: bool = True):
    """Build a ``tf.data.Dataset`` yielding ``(float32 [0,255] images, int labels)``.

    Augmentation is applied **only** when ``training=True`` and an augmenter is given.
    Decoded images are cached as uint8 (≈150 KB each) before augmentation.
    """
    import tensorflow as tf

    paths = [str(C.PROJECT_ROOT / fp) for fp in df["filepath"]]
    labels = df["label"].astype("int32").to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (_decode_resize(p, image_size), y), num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    if training and augmenter is not None:
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def compute_class_weights(labels: np.ndarray, n_classes: int) -> dict[int, float]:
    """'Balanced' weights n_samples / (n_classes * count_c), computed on TRAIN labels only."""
    counts = np.bincount(labels, minlength=n_classes)
    if (counts == 0).any():
        raise ValueError(f"Some classes have no training samples: {counts}")
    return {i: float(len(labels) / (n_classes * c)) for i, c in enumerate(counts)}
