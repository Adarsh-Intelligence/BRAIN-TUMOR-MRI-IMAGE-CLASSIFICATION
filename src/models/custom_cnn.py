"""Custom CNN trained from scratch (no pretrained weights)."""
from __future__ import annotations


def build_custom_cnn(num_classes: int, input_shape=(224, 224, 3), filters=(32, 64, 128, 256),
                     block_dropout=(0.10, 0.15, 0.20, 0.25), head_dropout: float = 0.4,
                     dense_units: int = 128, l2: float = 1e-4):
    """VGG-style CNN: 4 blocks of [Conv-BN-ReLU] x2 -> MaxPool -> Dropout, then GAP head.

    Expects raw [0,255] RGB input; rescaling to [0,1] happens inside the model.
    The final softmax is kept in float32 for numerical stability under mixed precision.
    """
    import keras
    from keras import layers, regularizers

    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    reg = regularizers.l2(l2) if l2 else None
    inputs = keras.Input(shape=input_shape, name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale_0_1")(inputs)
    for b, (f, d) in enumerate(zip(filters, block_dropout), start=1):
        for c in (1, 2):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False, kernel_regularizer=reg, name=f"b{b}_conv{c}")(x)
            x = layers.BatchNormalization(name=f"b{b}_bn{c}")(x)
            x = layers.ReLU(name=f"b{b}_relu{c}")(x)
        x = layers.MaxPooling2D(2, name=f"b{b}_pool")(x)
        x = layers.Dropout(d, name=f"b{b}_drop")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=reg, name="fc")(x)
    x = layers.Dropout(head_dropout, name="head_drop")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)
    return keras.Model(inputs, outputs, name="custom_cnn")
