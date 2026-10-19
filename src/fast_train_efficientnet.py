"""Fast feature-extraction training for EfficientNetB0 on CPU.

Caches frozen backbone features in memory, then optimizes the dense classification
head with class weights, and saves the complete end-to-end model.keras.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras
from keras import layers

from src import config as C
from src import data_loader as D
from src import preprocessing as P
from src.evaluate import compute_metrics, evaluate_model
from src.models.efficientnet import build_efficientnet
from src.utils import get_logger, save_json, set_global_seed

log = get_logger("fast_train")


def main():
    set_global_seed(C.SEED)
    out_dir = C.MODELS_DIR / "efficientnet"
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names, class_mapping = D.load_class_mapping()
    train_df = D.load_split("train")
    val_df = D.load_split("val")

    log.info("Loading EfficientNetB0 backbone...")
    full_model = build_efficientnet(len(class_names), weights="imagenet", dropout=0.3)
    backbone = full_model.get_layer("efficientnetb0")

    # Feature extractor: backbone + GAP
    inputs = keras.Input(shape=(224, 224, 3), name="feat_input")
    x = backbone(inputs, training=False)
    feats_out = layers.GlobalAveragePooling2D(name="gap")(x)
    feat_extractor = keras.Model(inputs, feats_out, name="feat_extractor")

    def extract_features(df, desc="Extracting features"):
        ds = P.make_dataset(df, (224, 224), batch_size=32, training=False, cache=False)
        all_feats, all_labels = [], []
        log.info("%s (%d images)...", desc, len(df))
        for x_batch, y_batch in ds:
            f = feat_extractor(x_batch, training=False)
            all_feats.append(f.numpy())
            all_labels.append(y_batch.numpy())
        return np.concatenate(all_feats, axis=0), np.concatenate(all_labels, axis=0)

    t0 = time.perf_counter()
    train_x, train_y = extract_features(train_df, "Extracting train features")
    val_x, val_y = extract_features(val_df, "Extracting val features")
    extract_time = time.perf_counter() - t0
    log.info("Extracted train features %s, val features %s in %.1fs", train_x.shape, val_x.shape, extract_time)

    # Classification Head model
    head_in = keras.Input(shape=(1280,), name="head_in")
    hx = layers.Dropout(0.3, name="head_drop")(head_in)
    head_out = layers.Dense(len(class_names), activation="softmax", dtype="float32", name="predictions")(hx)
    head_model = keras.Model(head_in, head_out, name="head_model")

    cw = P.compute_class_weights(train_y, len(class_names))
    head_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    log.info("Training classification head with class weights %s...", cw)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6, verbose=1),
    ]

    t_head = time.perf_counter()
    hist = head_model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=30,
        batch_size=32,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )
    head_train_time = time.perf_counter() - t_head
    total_time = extract_time + head_train_time

    # Transfer trained head weights back to full end-to-end model
    head_dense = head_model.get_layer("predictions")
    full_dense = full_model.get_layer("predictions")
    full_dense.set_weights(head_dense.get_weights())

    # Evaluate validation metrics
    val_preds = full_model.predict(P.prepare_batch([C.PROJECT_ROOT / f for f in val_df["filepath"]]), verbose=0)
    val_metrics = compute_metrics(val_y, val_preds.argmax(axis=1), class_names)
    log.info("Validation Accuracy: %.4f | Macro F1: %.4f", val_metrics["accuracy"], val_metrics["macro_f1"])

    # Save end-to-end model and metadata
    model_save_path = out_dir / "model.keras"
    full_model.save(model_save_path)
    log.info("Saved complete end-to-end model to %s", model_save_path)

    save_json({"class_names": class_names, "class_to_index": class_mapping}, out_dir / "class_mapping.json")
    save_json(P.preprocessing_spec("efficientnet"), out_dir / "preprocessing.json")
    summary = {
        "model": "efficientnet",
        "selected_stage": "stageA_feature_extraction",
        "train_time_s": total_time,
        "params": int(full_model.count_params()),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "smoke": False,
        "stages": {
            "stageA_feature_extraction": {
                "train_time_s": total_time,
                "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
                "epochs_run": len(hist.history["loss"]),
                "val_metrics": val_metrics,
                "val_loss": float(min(hist.history["val_loss"])),
            }
        },
    }
    save_json(summary, out_dir / "training_summary.json")

    # Evaluate on test set
    log.info("Evaluating on held-out test set...")
    evaluate_model("efficientnet", split="test")
    log.info("ALL DONE! Model is ready and evaluated.")


if __name__ == "__main__":
    main()
