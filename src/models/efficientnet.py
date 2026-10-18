"""EfficientNetB0 transfer learning (ImageNet weights) with two-stage training."""
from __future__ import annotations

BACKBONE_NAME = "efficientnetb0"


def build_efficientnet(num_classes: int, input_shape=(224, 224, 3), dropout: float = 0.3,
                       weights: str | None = "imagenet"):
    """Frozen EfficientNetB0 backbone + GAP + Dropout + softmax head.

    Keras 3 EfficientNet includes its own input rescaling/normalization and expects
    raw [0,255] RGB, so no external preprocessing is applied.

    The backbone is called with ``training=False`` so its BatchNormalization layers
    always run in inference mode, which keeps ImageNet statistics intact during
    fine-tuning (recommended Keras practice).
    """
    import keras
    from keras import layers

    base = keras.applications.EfficientNetB0(include_top=False, weights=weights, input_shape=input_shape)
    base.name = BACKBONE_NAME
    base.trainable = False
    inputs = keras.Input(shape=input_shape, name="image")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_drop")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)
    return keras.Model(inputs, outputs, name="efficientnetb0_transfer")


def get_backbone(model):
    return model.get_layer(BACKBONE_NAME)


def unfreeze_top(model, from_layer: str = "block6a_expand_conv") -> int:
    """Unfreeze backbone layers from ``from_layer`` onward; BatchNorm layers stay frozen.

    Returns the number of trainable backbone layers.
    """
    from keras import layers

    base = get_backbone(model)
    base.trainable = True
    names = [l.name for l in base.layers]
    if from_layer not in names:
        raise ValueError(f"Layer '{from_layer}' not found in backbone")
    start = names.index(from_layer)
    n = 0
    for i, layer in enumerate(base.layers):
        trainable = i >= start and not isinstance(layer, layers.BatchNormalization)
        layer.trainable = trainable
        n += int(trainable)
    return n
