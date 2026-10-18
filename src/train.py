"""Training entry point for both models.

Examples
--------
    python -m src.train --model custom_cnn
    python -m src.train --model efficientnet
    python -m src.train --model custom_cnn --smoke      # tiny CPU sanity run

Model selection uses the **validation** set only. The test set is never read here.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from . import config as C
from . import data_loader as D
from .augmentation import build_augmenter
from .preprocessing import compute_class_weights, make_dataset, preprocessing_spec
from .utils import get_logger, save_json, set_global_seed

log = get_logger("train")


# --------------------------------------------------------------------------- hardware
def configure_hardware(require_gpu: bool = False, mixed_precision: bool = False) -> dict:
    import tensorflow as tf
    import keras

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError as e:  # already initialized
            log.warning("Could not set memory growth: %s", e)
    info = {"tensorflow": tf.__version__, "keras": keras.__version__,
            "gpus": [g.name for g in gpus], "cuda_build": tf.test.is_built_with_cuda()}
    if gpus:
        try:
            info["gpu_details"] = tf.config.experimental.get_device_details(gpus[0])
        except Exception:  # noqa: BLE001
            pass
        log.info("Training on GPU: %s", info.get("gpu_details", gpus[0].name))
    else:
        if require_gpu:
            raise RuntimeError("--require-gpu was set but TensorFlow detects no GPU. "
                               "Run `python -m src.gpu_check` inside WSL2 to diagnose.")
        log.warning("No GPU detected - training on CPU (slow).")
    if mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        log.info("Mixed precision enabled (mixed_float16)")
    info["mixed_precision"] = mixed_precision
    return info


# --------------------------------------------------------------------------- data
def load_data(cfg: C.PipelineConfig, smoke: bool = False):
    class_names, mapping = D.load_class_mapping()
    train_df, val_df = D.load_split("train"), D.load_split("val")
    if smoke:
        train_df = train_df.groupby("class_name", group_keys=False).apply(lambda g: g.head(12))
        val_df = val_df.groupby("class_name", group_keys=False).apply(lambda g: g.head(6))
    augmenter = build_augmenter(cfg.augment, seed=cfg.data.seed)
    train_ds = make_dataset(train_df, cfg.data.image_size, cfg.data.batch_size, training=True,
                            augmenter=augmenter, seed=cfg.data.seed)
    val_ds = make_dataset(val_df, cfg.data.image_size, cfg.data.batch_size, training=False)
    cw = compute_class_weights(train_df["label"].to_numpy(), len(class_names)) if cfg.train.use_class_weights else None
    log.info("Train %d | Val %d | classes %s | class weights %s", len(train_df), len(val_df), class_names, cw)
    return class_names, train_df, val_df, train_ds, val_ds, cw


# --------------------------------------------------------------------------- callbacks
class EpochTimer:
    def __new__(cls):
        import keras

        class _T(keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                self.times = []

            def on_epoch_begin(self, epoch, logs=None):
                self._t = time.perf_counter()

            def on_epoch_end(self, epoch, logs=None):
                self.times.append(time.perf_counter() - self._t)
                if logs is not None:
                    logs["epoch_time_s"] = self.times[-1]
                    logs["lr"] = float(self.model.optimizer.learning_rate.numpy())

        return _T()


def make_callbacks(ckpt_path: Path, csv_path: Path, patience: int, tcfg: C.TrainConfig):
    import keras

    return [
        EpochTimer(),
        keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor=tcfg.monitor, save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor=tcfg.monitor, patience=patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor=tcfg.monitor, factor=tcfg.reduce_lr_factor,
                                          patience=tcfg.reduce_lr_patience, min_lr=tcfg.min_lr, verbose=1),
        keras.callbacks.CSVLogger(str(csv_path)),
    ]


# --------------------------------------------------------------------------- validation metrics
def validation_metrics(model, val_ds, class_names) -> dict:
    from .evaluate import compute_metrics, predict_dataset

    y_true, y_prob = predict_dataset(model, val_ds)
    return compute_metrics(y_true, y_prob.argmax(1), class_names)


def _fit(model, train_ds, val_ds, epochs, lr, out_dir: Path, tag: str, patience: int, tcfg, cw, initial_epoch=0):
    import keras

    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    t0 = time.perf_counter()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=cw, verbose=2,
                     callbacks=make_callbacks(out_dir / f"{tag}_best.keras", out_dir / f"{tag}_log.csv", patience, tcfg))
    dt = time.perf_counter() - t0
    ckpt = out_dir / f"{tag}_best.keras"
    if ckpt.exists():  # make sure the best-validation weights are the ones kept
        model.set_weights(keras.models.load_model(ckpt).get_weights())
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    save_json(h, out_dir / f"{tag}_history.json")
    best_epoch = int(np.argmin(h[tcfg.monitor])) + 1
    log.info("[%s] finished %d epochs in %.1fs, best epoch %d (%s=%.4f)",
             tag, len(h["loss"]), dt, best_epoch, tcfg.monitor, min(h[tcfg.monitor]))
    return h, dt, best_epoch


# --------------------------------------------------------------------------- trainers
def train_custom_cnn(cfg, data, out_dir: Path, hw: dict) -> dict:
    from .models.custom_cnn import build_custom_cnn
    from .plots import plot_history

    class_names, train_df, val_df, train_ds, val_ds, cw = data
    model = build_custom_cnn(len(class_names), (*cfg.data.image_size, 3))
    model.summary(print_fn=lambda s: log.info(s))
    h, dt, best_epoch = _fit(model, train_ds, val_ds, cfg.train.epochs, cfg.train.learning_rate, out_dir,
                             "custom_cnn", cfg.train.early_stopping_patience, cfg.train, cw)
    plot_history(h, C.PLOTS_DIR / out_dir.name / "custom_cnn", "Custom CNN")
    val = validation_metrics(model, val_ds, class_names)
    final = out_dir / "model.keras"
    model.save(final)
    return {"stages": {"custom_cnn": {"train_time_s": dt, "best_epoch": best_epoch, "epochs_run": len(h["loss"]),
                                      "val_metrics": val}},
            "selected_stage": "custom_cnn", "train_time_s": dt, "params": int(model.count_params())}


def train_efficientnet(cfg, data, out_dir: Path, hw: dict) -> dict:
    from .models.efficientnet import build_efficientnet, unfreeze_top
    from .plots import plot_history

    import keras

    class_names, train_df, val_df, train_ds, val_ds, cw = data
    ecfg = cfg.efficientnet
    tcfg = cfg.train
    model = build_efficientnet(len(class_names), (*cfg.data.image_size, 3), ecfg.dropout)
    log.info("Stage A: feature extraction (backbone frozen), trainable params=%d",
             sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    hA, tA, beA = _fit(model, train_ds, val_ds, ecfg.head_epochs, ecfg.head_learning_rate, out_dir,
                       "stageA_feature_extraction", ecfg.early_stopping_patience, tcfg, cw)
    plot_history(hA, C.PLOTS_DIR / out_dir.name / "stageA_feature_extraction", "EfficientNetB0 - feature extraction")
    valA = validation_metrics(model, val_ds, class_names)
    model.save(out_dir / "stageA_model.keras")
    stages = {"stageA_feature_extraction": {"train_time_s": tA, "best_epoch": beA, "epochs_run": len(hA["loss"]),
                                            "val_metrics": valA, "val_loss": min(hA[tcfg.monitor])}}

    if ecfg.finetune_epochs > 0:
        n = unfreeze_top(model, ecfg.finetune_unfreeze_from)
        log.info("Stage B: fine-tuning %d backbone layers from '%s' (BatchNorm frozen), lr=%g",
                 n, ecfg.finetune_unfreeze_from, ecfg.finetune_learning_rate)
        hB, tB, beB = _fit(model, train_ds, val_ds, ecfg.finetune_epochs, ecfg.finetune_learning_rate, out_dir,
                           "stageB_fine_tuning", ecfg.early_stopping_patience, tcfg, cw)
        plot_history(hB, C.PLOTS_DIR / out_dir.name / "stageB_fine_tuning", "EfficientNetB0 - fine-tuning")
        plot_history({k: hA[k] + hB[k] for k in ("loss", "val_loss", "accuracy", "val_accuracy")},
                     C.PLOTS_DIR / out_dir.name / "efficientnet_combined", "EfficientNetB0 (A + B)",
                     stage_boundary=len(hA["loss"]))
        valB = validation_metrics(model, val_ds, class_names)
        model.save(out_dir / "stageB_model.keras")
        stages["stageB_fine_tuning"] = {"train_time_s": tB, "best_epoch": beB, "epochs_run": len(hB["loss"]),
                                        "val_metrics": valB, "val_loss": min(hB[tcfg.monitor]),
                                        "unfrozen_layers": n}

    # select stage on VALIDATION macro-F1 (tie-break: lower val loss); fine-tuning is not assumed better
    sel = max(stages, key=lambda s: (round(stages[s]["val_metrics"]["macro_f1"], 4), -stages[s]["val_loss"]))
    src = out_dir / ("stageB_model.keras" if sel == "stageB_fine_tuning" else "stageA_model.keras")
    shutil.copyfile(src, out_dir / "model.keras")
    log.info("Selected %s on validation macro-F1 (A=%.4f%s)", sel, valA["macro_f1"],
             f", B={stages['stageB_fine_tuning']['val_metrics']['macro_f1']:.4f}" if "stageB_fine_tuning" in stages else "")
    total = sum(s["train_time_s"] for s in stages.values())
    return {"stages": stages, "selected_stage": sel, "train_time_s": total,
            "params": int(keras.models.load_model(out_dir / "model.keras").count_params())}


# --------------------------------------------------------------------------- main
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["custom_cnn", "efficientnet"], required=True)
    ap.add_argument("--epochs", type=int, help="override max epochs (custom CNN / stage A)")
    ap.add_argument("--finetune-epochs", type=int, help="EfficientNet stage B epochs (0 disables)")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--mixed-precision", action="store_true")
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch, writes to artifacts/models/smoke_*")
    args = ap.parse_args(argv)

    cfg = C.get_config()
    if args.batch_size:
        cfg.data.batch_size = args.batch_size
    if args.epochs:
        cfg.train.epochs = cfg.efficientnet.head_epochs = args.epochs
    if args.finetune_epochs is not None:
        cfg.efficientnet.finetune_epochs = args.finetune_epochs
    if args.smoke:
        cfg.train.epochs = cfg.efficientnet.head_epochs = cfg.efficientnet.finetune_epochs = 1
    cfg.train.mixed_precision = args.mixed_precision

    set_global_seed(cfg.data.seed)
    C.ensure_dirs()
    hw = configure_hardware(args.require_gpu, args.mixed_precision)
    name = args.model if not args.smoke else f"smoke_{args.model}"
    out_dir = C.MODELS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(cfg, smoke=args.smoke)
    class_names = data[0]
    trainer = train_custom_cnn if args.model == "custom_cnn" else train_efficientnet
    result = trainer(cfg, data, out_dir, hw)

    save_json({"class_names": class_names, "class_to_index": {c: i for i, c in enumerate(class_names)}},
              out_dir / "class_mapping.json")
    save_json(preprocessing_spec(args.model, cfg.data.image_size), out_dir / "preprocessing.json")
    save_json({"model": args.model, "config": cfg.to_dict(), "hardware": hw, "smoke": args.smoke,
               "train_size": len(data[1]), "val_size": len(data[2]), **result}, out_dir / "training_summary.json")
    log.info("Artifacts saved to %s", out_dir)
    return result


if __name__ == "__main__":
    main()
