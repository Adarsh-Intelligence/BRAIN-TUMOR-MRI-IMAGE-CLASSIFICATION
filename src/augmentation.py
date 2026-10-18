"""Training-only data augmentation (Keras preprocessing layers).

Applied in the ``tf.data`` pipeline, not inside the saved models, so exported
models contain no random layers and validation/test data are never augmented.
Transformations are deliberately mild to avoid unrealistic anatomy.
"""
from __future__ import annotations

from . import config as C


def build_augmenter(cfg: C.AugmentConfig | None = None, seed: int = C.SEED):
    import keras
    from keras import layers

    cfg = cfg or C.AugmentConfig()
    aug = [
        layers.RandomRotation(cfg.rotation_factor, fill_mode="constant", fill_value=0.0, seed=seed),
        layers.RandomTranslation(cfg.translation_factor, cfg.translation_factor,
                                 fill_mode="constant", fill_value=0.0, seed=seed + 1),
        layers.RandomZoom(cfg.zoom_factor, fill_mode="constant", fill_value=0.0, seed=seed + 2),
    ]
    if cfg.horizontal_flip:
        aug.append(layers.RandomFlip("horizontal", seed=seed + 3))
    if cfg.brightness_delta:
        aug.append(layers.RandomBrightness(cfg.brightness_delta, value_range=(0.0, 255.0), seed=seed + 4))
    if cfg.contrast_factor:
        aug.append(layers.RandomContrast(cfg.contrast_factor, value_range=(0.0, 255.0), seed=seed + 5))
    return keras.Sequential(aug, name="train_augmentation")
