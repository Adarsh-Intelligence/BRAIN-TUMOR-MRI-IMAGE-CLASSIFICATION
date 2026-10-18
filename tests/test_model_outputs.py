import numpy as np
import pytest

from src.models.custom_cnn import build_custom_cnn
from src.models.efficientnet import build_efficientnet, get_backbone, unfreeze_top

N = 4


@pytest.fixture(scope="module")
def cnn():
    return build_custom_cnn(N)


@pytest.fixture(scope="module")
def eff():
    return build_efficientnet(N, weights=None)  # no download in tests


def test_cnn_shapes_and_softmax(cnn):
    assert cnn.input_shape == (None, 224, 224, 3)
    assert cnn.output_shape == (None, N)
    p = np.asarray(cnn(np.random.uniform(0, 255, (2, 224, 224, 3)).astype("float32"), training=False))
    assert p.shape == (2, N) and np.allclose(p.sum(1), 1, atol=1e-5) and (p >= 0).all()


def test_cnn_has_no_pretrained_and_rescales(cnn):
    assert cnn.layers[1].name == "rescale_0_1"


def test_cnn_rejects_bad_classes():
    with pytest.raises(ValueError):
        build_custom_cnn(1)


def test_efficientnet_shapes(eff):
    assert eff.output_shape == (None, N)
    assert not get_backbone(eff).trainable
    p = np.asarray(eff(np.random.uniform(0, 255, (1, 224, 224, 3)).astype("float32"), training=False))
    assert np.allclose(p.sum(), 1, atol=1e-5)


def test_unfreeze_keeps_batchnorm_frozen():
    from keras import layers

    m = build_efficientnet(N, weights=None)
    n = unfreeze_top(m, "block6a_expand_conv")
    base = get_backbone(m)
    assert n > 0
    assert all(not l.trainable for l in base.layers if isinstance(l, layers.BatchNormalization))
    assert not base.get_layer("block1a_project_conv").trainable
    assert base.get_layer("top_conv").trainable


def test_save_and_load_roundtrip(cnn, tmp_path):
    import keras

    x = np.random.uniform(0, 255, (2, 224, 224, 3)).astype("float32")
    path = tmp_path / "m.keras"
    cnn.save(path)
    loaded = keras.models.load_model(path)
    assert np.allclose(np.asarray(cnn(x, training=False)), np.asarray(loaded(x, training=False)), atol=1e-5)


@pytest.mark.parametrize("which", ["cnn", "eff"])
def test_gradcam_heatmap(which, cnn, eff):
    from src.gradcam import gradcam_heatmap

    model = cnn if which == "cnn" else eff
    cam, ci, probs = gradcam_heatmap(model, np.random.uniform(0, 255, (224, 224, 3)).astype("float32"))
    assert cam.ndim == 2 and cam.min() >= 0 and cam.max() <= 1 + 1e-6
    assert 0 <= ci < N and probs.shape == (N,)
