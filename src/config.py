"""Central configuration for the NeuroScan AI model-development pipeline.

All paths are resolved relative to the project root so the same code runs on
native Windows (``D:\\...``) and inside WSL2 (``/mnt/d/...``).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Existing dataset folder (detected during inspection; never modified).
DATASET_DIR_NAME: str = "DATASET FOR NEROSCAN AI"
DATASET_DIR: Path = PROJECT_ROOT / DATASET_DIR_NAME

ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
SPLITS_DIR: Path = ARTIFACTS_DIR / "splits"
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
METRICS_DIR: Path = ARTIFACTS_DIR / "metrics"
PLOTS_DIR: Path = ARTIFACTS_DIR / "plots"
REPORTS_DIR: Path = ARTIFACTS_DIR / "reports"
EXPLORATION_DIR: Path = ARTIFACTS_DIR / "exploration"

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

SEED: int = 42


@dataclass
class SplitConfig:
    """Grouped, stratified 70/15/15 split built from pooled images."""

    n_folds: int = 20
    train_folds: int = 14
    val_folds: int = 3
    test_folds: int = 3
    phash_threshold: int = 4  # max Hamming distance (of 64 bits) for same-class near-duplicates
    cross_class_phash_threshold: int = 2  # stricter limit when labels differ (d=4 cross-class = false positives)
    seed: int = SEED


@dataclass
class DataConfig:
    image_size: tuple[int, int] = (224, 224)
    channels: int = 3
    batch_size: int = 16
    seed: int = SEED


@dataclass
class AugmentConfig:
    """Conservative, anatomy-preserving augmentation (training data only)."""

    rotation_factor: float = 10.0 / 360.0  # +/- 10 degrees
    translation_factor: float = 0.08       # +/- 8 % width/height
    zoom_factor: float = 0.10              # +/- 10 %
    horizontal_flip: bool = True           # brain is roughly left-right symmetric
    brightness_delta: float = 0.10         # mild, fraction of [0,1] range
    contrast_factor: float = 0.10


@dataclass
class TrainConfig:
    epochs: int = 40
    learning_rate: float = 1e-3
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.3
    min_lr: float = 1e-6
    monitor: str = "val_loss"
    use_class_weights: bool = True
    mixed_precision: bool = False


@dataclass
class EfficientNetConfig:
    dropout: float = 0.3
    head_epochs: int = 15
    head_learning_rate: float = 1e-3
    finetune_epochs: int = 20
    finetune_learning_rate: float = 1e-5
    finetune_unfreeze_from: str = "block6a_expand_conv"  # unfreeze block6, block7, top
    early_stopping_patience: int = 6


@dataclass
class PipelineConfig:
    split: SplitConfig = field(default_factory=SplitConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    efficientnet: EfficientNetConfig = field(default_factory=EfficientNetConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def get_config() -> PipelineConfig:
    return PipelineConfig()


def ensure_dirs() -> None:
    for d in (ARTIFACTS_DIR, SPLITS_DIR, MODELS_DIR, METRICS_DIR, PLOTS_DIR, REPORTS_DIR, EXPLORATION_DIR):
        d.mkdir(parents=True, exist_ok=True)
