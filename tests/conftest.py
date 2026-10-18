import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]


def _mri_like(rng, size=(64, 80), mode="L"):
    """Random blob image (distinct per call so pHashes differ)."""
    yy, xx = np.mgrid[: size[0], : size[1]]
    img = np.zeros(size)
    for _ in range(4):
        cy, cx, r = rng.integers(0, size[0]), rng.integers(0, size[1]), rng.integers(5, 20)
        img += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r**2)) * rng.uniform(80, 255)
    img = np.clip(img + rng.normal(0, 10, size), 0, 255).astype(np.uint8)
    im = Image.fromarray(img, "L")
    return im.convert(mode)


@pytest.fixture()
def fake_dataset(tmp_path):
    """<root>/train-xyz/train/<class>/Tr-xx_000N_jpg.rf.<hash>.jpg with one corrupt file."""
    rng = np.random.default_rng(0)
    root = tmp_path / "DATASET"
    for split, n in (("train", 6), ("valid", 2), ("test", 2)):
        for ci, c in enumerate(CLASSES):
            d = root / f"{split}-0001" / split / c
            d.mkdir(parents=True)
            for i in range(n):
                stem = f"Tr-{c[:2]}_{split}{i:03d}"
                _mri_like(rng, mode="RGB" if i % 2 else "L").save(d / f"{stem}_jpg.rf.{i:032x}.jpg")
    # same source stem appearing in two splits (Roboflow-style leakage)
    src = next((root / "train-0001" / "train" / "glioma").glob("*.jpg"))
    Image.open(src).transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(
        root / "test-0001" / "test" / "glioma" / (src.name.split("_jpg")[0] + "_jpg.rf." + "f" * 32 + ".jpg"))
    (root / "train-0001" / "train" / "glioma" / "corrupt_jpg.rf.abc.jpg").write_bytes(b"not an image")
    return root
