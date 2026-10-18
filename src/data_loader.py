"""Dataset discovery, integrity checks, duplicate grouping and leakage-safe splitting.

The original dataset is treated as **read-only**. This module only reads image
files and writes CSV manifests under ``artifacts/``.

Expected layout (detected, not assumed): ``<dataset>/<...>/<split>/<class>/<image>``
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from scipy.fft import dctn
from sklearn.model_selection import StratifiedGroupKFold

from . import config as C
from .utils import get_logger, save_json, load_json

log = get_logger("data_loader")

# Roboflow export names: "<original_stem>_<ext>.rf.<hash>.<ext>"
_ROBOFLOW_RE = re.compile(r"^(?P<stem>.+?)_(?:jpg|jpeg|png)\.rf\.[0-9a-f]+$", re.IGNORECASE)
_SPLIT_ALIASES = {"train": "train", "training": "train", "valid": "val", "val": "val",
                  "validation": "val", "test": "test", "testing": "test"}


# --------------------------------------------------------------------------- discovery
def locate_dataset(dataset_dir: Path | None = None) -> Path:
    """Return the dataset directory or raise a clear error if it is missing."""
    path = Path(dataset_dir) if dataset_dir else C.DATASET_DIR
    if not path.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {path}. Expected the existing dataset folder "
            f"'{C.DATASET_DIR_NAME}' inside the project root {C.PROJECT_ROOT}."
        )
    return path


def source_stem(filename: str) -> str:
    """Original (pre-Roboflow) image stem, e.g. ``Tr-gl_0496``."""
    stem = Path(filename).stem
    m = _ROBOFLOW_RE.match(stem)
    return m.group("stem") if m else stem


def discover_images(dataset_dir: Path | None = None) -> pd.DataFrame:
    """List every image file with its folder-derived class and original split."""
    root = locate_dataset(dataset_dir)
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in C.IMAGE_EXTENSIONS:
            continue
        split_dir = p.parent.parent.name.lower()
        try:
            rel = p.relative_to(C.PROJECT_ROOT).as_posix()
        except ValueError:  # dataset outside project (e.g. tests) -> absolute path
            rel = p.as_posix()
        rows.append({
            "filepath": rel,
            "filename": p.name,
            "class_name": p.parent.name,
            "original_split": _SPLIT_ALIASES.get(split_dir, split_dir),
            "source_stem": source_stem(p.name),
        })
    if not rows:
        raise FileNotFoundError(f"No images with extensions {C.IMAGE_EXTENSIONS} found under {root}")
    df = pd.DataFrame(rows)
    log.info("Discovered %d image files in %d classes", len(df), df["class_name"].nunique())
    return df


def get_class_names(df: pd.DataFrame) -> list[str]:
    """Deterministic (alphabetical) class order used everywhere."""
    return sorted(df["class_name"].unique().tolist())


def build_class_mapping(class_names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(class_names)}


def check_csv_labels(df: pd.DataFrame, dataset_dir: Path | None = None) -> pd.DataFrame:
    """Cross-check folder labels against Roboflow ``_classes.csv`` one-hot files.

    Returns rows where the CSV label disagrees with the folder (empty if consistent).
    """
    root = locate_dataset(dataset_dir)
    norm = lambda s: re.sub(r"[^a-z]", "", s.lower())
    csv_labels: dict[str, str] = {}
    for csv in root.rglob("_classes.csv"):
        t = pd.read_csv(csv, skipinitialspace=True)
        t.columns = [c.strip() for c in t.columns]
        cls_cols = [c for c in t.columns if c != "filename"]
        for _, r in t.iterrows():
            active = [c for c in cls_cols if int(r[c]) == 1]
            csv_labels[r["filename"].strip()] = norm(active[0]) if len(active) == 1 else "|".join(active) or "none"
    out = df.assign(csv_label=df["filename"].map(csv_labels))
    mismatch = out[out["csv_label"].isna() | (out["csv_label"] != out["class_name"].map(norm))]
    return mismatch[["filepath", "class_name", "csv_label"]]


# --------------------------------------------------------------------------- integrity
def phash(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """64-bit perceptual hash (DCT-based, same idea as ``imagehash.phash``)."""
    size = hash_size * highfreq_factor
    g = np.asarray(img.convert("L").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float64)
    low = dctn(g, norm="ortho")[:hash_size, :hash_size]
    bits = (low > np.median(low)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def inspect_image(path: Path) -> dict:
    """Read one image safely and return metadata, hashes or the error message."""
    info = {"readable": False, "error": "", "width": np.nan, "height": np.nan, "mode": "",
            "format": "", "file_size": np.nan, "md5": "", "phash": np.nan}
    try:
        data = Path(path).read_bytes()
        info["file_size"] = len(data)
        info["md5"] = hashlib.md5(data).hexdigest()
        with Image.open(path) as im:
            im.verify()  # structural check
        with Image.open(path) as im:  # re-open: verify() invalidates the object
            im.load()
            info.update(width=im.width, height=im.height, mode=im.mode, format=im.format or "")
            info["phash"] = phash(im)
        info["readable"] = True
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def inspect_all(df: pd.DataFrame) -> pd.DataFrame:
    from tqdm import tqdm

    recs = [inspect_image(C.PROJECT_ROOT / fp) for fp in tqdm(df["filepath"], desc="Inspecting images")]
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    bad = out[~out["readable"]]
    if len(bad):
        log.warning("%d unreadable/corrupted images:\n%s", len(bad), bad[["filepath", "error"]].to_string())
    else:
        log.info("All %d images opened and decoded successfully", len(out))
    return out


# --------------------------------------------------------------------------- duplicates
class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    b = x.view(np.uint8).reshape(*x.shape, 8)
    return np.unpackbits(b, axis=-1).sum(-1)


def near_duplicate_pairs(hashes: np.ndarray, threshold: int) -> list[tuple[int, int, int]]:
    """All index pairs (i<j) whose 64-bit pHash Hamming distance <= threshold."""
    h = hashes.astype(np.uint64)
    pairs = []
    for i in range(len(h) - 1):
        d = _popcount64(np.bitwise_xor(h[i], h[i + 1:]))
        for j in np.nonzero(d <= threshold)[0]:
            pairs.append((i, i + 1 + int(j), int(d[j])))
    return pairs


@dataclass
class DuplicateReport:
    exact_pairs: pd.DataFrame
    near_pairs: pd.DataFrame
    label_conflict_groups: pd.DataFrame


def assign_groups(df: pd.DataFrame, phash_threshold: int,
                  cross_class_threshold: int = 2) -> tuple[pd.DataFrame, DuplicateReport]:
    """Union images sharing source stem, identical MD5, or near-identical pHash.

    Near-duplicate pairs within the same class are grouped up to ``phash_threshold``
    (these are typically adjacent slices of the same scan). Pairs with *different*
    labels are grouped only when nearly identical (``<= cross_class_threshold``);
    at larger distances they were visually verified to be unrelated scans, and
    grouping them would chain unrelated images together.
    """
    df = df.reset_index(drop=True).copy()
    uf = _UnionFind(len(df))
    for key in ("source_stem", "md5"):
        for _, idx in df.groupby(key).indices.items():
            for j in idx[1:]:
                uf.union(idx[0], j)

    exact = [(i, j) for _, idx in df.groupby("md5").indices.items() for i in idx for j in idx if i < j]
    near = near_duplicate_pairs(df["phash"].to_numpy(dtype=np.uint64), phash_threshold)
    cls = df["class_name"].to_numpy()
    used = []
    for i, j, d in near:
        if cls[i] == cls[j] or d <= cross_class_threshold:
            uf.union(i, j)
            used.append((i, j, d))
    rejected = len(near) - len(used)

    roots = np.array([uf.find(i) for i in range(len(df))])
    _, df["group_id"] = np.unique(roots, return_inverse=True)

    def pair_frame(pairs, with_dist):
        cols = ["filepath_a", "class_a", "split_a", "filepath_b", "class_b", "split_b"] + (["hamming"] if with_dist else [])
        rows = []
        for p in pairs:
            a, b = df.iloc[p[0]], df.iloc[p[1]]
            r = [a.filepath, a.class_name, a.original_split, b.filepath, b.class_name, b.original_split]
            rows.append(r + ([p[2]] if with_dist else []))
        return pd.DataFrame(rows, columns=cols)

    gstats = df.groupby("group_id")["class_name"].agg(lambda s: sorted(set(s)))
    conflicts = gstats[gstats.map(len) > 1]
    conflict_df = df[df["group_id"].isin(conflicts.index)][["group_id", "filepath", "class_name", "source_stem", "md5"]]
    df["group_size"] = df.groupby("group_id")["filepath"].transform("size")
    df["label_conflict"] = df["group_id"].isin(conflicts.index)

    near_df = pair_frame(near, True)
    near_df["grouped"] = [(cls[i] == cls[j] or d <= cross_class_threshold) for i, j, d in near]
    report = DuplicateReport(pair_frame(exact, False), near_df, conflict_df)
    log.info("Groups: %d for %d images | exact-dup pairs: %d | near-dup pairs (<=%d bits): %d grouped, "
             "%d cross-class rejected | label-conflict groups: %d",
             df["group_id"].nunique(), len(df), len(exact), phash_threshold, len(used), rejected, len(conflicts))
    return df, report


# --------------------------------------------------------------------------- splitting
def grouped_stratified_split(df: pd.DataFrame, cfg: C.SplitConfig) -> pd.DataFrame:
    """Assign train/val/test so that no group crosses splits, stratified by class.

    ``StratifiedGroupKFold`` partitions groups into ``n_folds`` class-balanced folds;
    folds are then allocated train/val/test (default 14/3/3 = 70/15/15).
    """
    assert cfg.train_folds + cfg.val_folds + cfg.test_folds == cfg.n_folds
    df = df.reset_index(drop=True).copy()
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    fold = np.full(len(df), -1)
    for k, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(df)), df["class_name"], df["group_id"])):
        fold[test_idx] = k
    assert (fold >= 0).all()
    df["fold"] = fold
    df["split"] = np.select(
        [fold < cfg.train_folds, fold < cfg.train_folds + cfg.val_folds], ["train", "val"], "test")
    verify_no_leakage(df)
    return df


def verify_no_leakage(df: pd.DataFrame) -> None:
    """Raise if any duplicate group, MD5 or source stem appears in more than one split."""
    for key in ("group_id", "md5", "source_stem"):
        spans = df.groupby(key)["split"].nunique()
        leaked = spans[spans > 1]
        if len(leaked):
            raise AssertionError(f"Leakage: {len(leaked)} values of '{key}' span multiple splits")


MANIFEST_COLUMNS = ["filepath", "class_name", "label", "split", "group_id", "group_size", "source_stem",
                    "original_split", "md5", "width", "height", "mode"]


def save_manifests(df: pd.DataFrame, class_names: list[str], out_dir: Path = C.SPLITS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = build_class_mapping(class_names)
    df = df.assign(label=df["class_name"].map(mapping))
    df[MANIFEST_COLUMNS].to_csv(out_dir / "all_images.csv", index=False)
    for s in ("train", "val", "test"):
        df[df["split"] == s][MANIFEST_COLUMNS].to_csv(out_dir / f"{s}.csv", index=False)
    save_json({"class_names": class_names, "class_to_index": mapping}, out_dir / "class_mapping.json")
    log.info("Saved manifests to %s", out_dir)


def load_split(split: str, splits_dir: Path = C.SPLITS_DIR) -> pd.DataFrame:
    path = Path(splits_dir) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split manifest {path} not found. Run `python -m src.prepare_data` first.")
    return pd.read_csv(path)


def load_class_mapping(splits_dir: Path = C.SPLITS_DIR) -> tuple[list[str], dict[str, int]]:
    m = load_json(Path(splits_dir) / "class_mapping.json")
    return m["class_names"], m["class_to_index"]
