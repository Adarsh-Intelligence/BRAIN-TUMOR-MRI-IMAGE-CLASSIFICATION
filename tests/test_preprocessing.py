import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src import preprocessing as P
from src.augmentation import build_augmenter


def test_load_image_resizes_and_converts_grayscale(tmp_path):
    p = tmp_path / "g.png"
    Image.fromarray(np.random.randint(0, 255, (50, 70), np.uint8), "L").save(p)
    arr = P.load_image(p)
    assert arr.shape == (224, 224, 3) and arr.dtype == np.uint8
    assert np.array_equal(arr[..., 0], arr[..., 1])  # gray replicated to RGB


def test_load_image_corrupted_raises(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"\x00\x01garbage")
    with pytest.raises(ValueError, match="Corrupted or unreadable"):
        P.load_image(p)


def test_load_image_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        P.load_image(tmp_path / "missing.jpg")


def test_validate_shape():
    with pytest.raises(ValueError):
        P.validate_image_array(np.zeros((10, 10, 3)))


def test_preprocessing_spec():
    cnn, eff = P.preprocessing_spec("custom_cnn"), P.preprocessing_spec("efficientnet")
    assert cnn["input_shape"] == [224, 224, 3] and cnn["input_value_range"] == [0, 255]
    assert "1/255" in cnn["in_model_normalization"] and "EfficientNet" in eff["in_model_normalization"]


def test_class_weights_balanced():
    w = P.compute_class_weights(np.array([0, 0, 0, 1]), 2)
    assert w[0] == pytest.approx(4 / 6) and w[1] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        P.compute_class_weights(np.array([0, 0]), 2)


def _df(tmp_path, n=6):
    rows = []
    for i in range(n):
        p = tmp_path / f"{i}.jpg"
        Image.fromarray(np.full((40, 60, 3), i * 30, np.uint8)).save(p)
        rows.append({"filepath": p.as_posix(), "label": i % 3})
    return pd.DataFrame(rows)


def test_make_dataset_shapes_and_range(tmp_path):
    ds = P.make_dataset(_df(tmp_path), batch_size=4, training=False)
    x, y = next(iter(ds))
    assert tuple(x.shape) == (4, 224, 224, 3) and str(x.dtype) == "<dtype: 'float32'>"
    assert float(x.numpy().min()) >= 0 and float(x.numpy().max()) <= 255
    assert y.numpy().tolist() == [0, 1, 2, 0]  # eval order preserved


def test_augmentation_preserves_labels_and_shape(tmp_path):
    df = _df(tmp_path)
    plain = P.make_dataset(df, batch_size=6, training=False)
    aug = P.make_dataset(df, batch_size=6, training=True, augmenter=build_augmenter())
    xa, ya = next(iter(aug))
    xp, yp = next(iter(plain))
    assert xa.shape == xp.shape
    assert sorted(ya.numpy().tolist()) == sorted(yp.numpy().tolist())
    assert float(xa.numpy().min()) >= 0 and float(xa.numpy().max()) <= 255.0 + 1e-3


def test_eval_pipeline_is_deterministic(tmp_path):
    df = _df(tmp_path)
    a = next(iter(P.make_dataset(df, batch_size=6)))[0].numpy()
    b = next(iter(P.make_dataset(df, batch_size=6)))[0].numpy()
    assert np.array_equal(a, b)
