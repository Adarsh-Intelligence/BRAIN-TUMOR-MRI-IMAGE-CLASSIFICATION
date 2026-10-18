"""Verify TensorFlow GPU support (run inside WSL2 before training).

Usage:  python -m src.gpu_check
Exit code 0 = a GPU is visible and an op ran on it; 1 otherwise.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


def main() -> int:
    import tensorflow as tf

    print("TensorFlow", tf.__version__, "| built with CUDA:", tf.test.is_built_with_cuda())
    info = tf.sysconfig.get_build_info()
    print("CUDA", info.get("cuda_version"), "| cuDNN", info.get("cudnn_version"))
    gpus = tf.config.list_physical_devices("GPU")
    print("Physical GPUs:", gpus)
    if not gpus:
        print("FAIL: no GPU visible to TensorFlow.")
        return 1
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
        print("Details:", tf.config.experimental.get_device_details(g))

    tf.debugging.set_log_device_placement(False)
    with tf.device("/GPU:0"):
        a = tf.random.normal((2048, 2048))
        b = tf.random.normal((2048, 2048))
        c = tf.matmul(a, b)
        _ = c.numpy()
        print("matmul result device:", c.device)
        t0 = time.perf_counter()
        for _ in range(20):
            c = tf.matmul(a, b)
        _ = c.numpy()
        gpu_t = time.perf_counter() - t0
    with tf.device("/CPU:0"):
        a2, b2 = tf.identity(a), tf.identity(b)
        t0 = time.perf_counter()
        for _ in range(20):
            c2 = tf.matmul(a2, b2)
        _ = c2.numpy()
        cpu_t = time.perf_counter() - t0
    print(f"20x 2048^2 matmul: GPU {gpu_t:.3f}s | CPU {cpu_t:.3f}s | speed-up x{cpu_t / gpu_t:.1f}")

    # tiny conv training step on GPU (exercises cuDNN)
    import keras

    with tf.device("/GPU:0"):
        m = keras.Sequential([keras.Input((224, 224, 3)), keras.layers.Conv2D(16, 3, activation="relu"),
                              keras.layers.GlobalAveragePooling2D(), keras.layers.Dense(4, activation="softmax")])
        m.compile("adam", "sparse_categorical_crossentropy")
        m.fit(tf.random.uniform((32, 224, 224, 3)), tf.random.uniform((32,), 0, 4, tf.int32),
              batch_size=16, epochs=1, verbose=0)
    print("cuDNN conv training step: OK")
    try:
        mem = tf.config.experimental.get_memory_info("GPU:0")
        print(f"GPU memory current {mem['current'] / 2**20:.0f} MB, peak {mem['peak'] / 2**20:.0f} MB")
    except Exception:  # noqa: BLE001
        pass
    print("PASS: TensorFlow is using the GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
