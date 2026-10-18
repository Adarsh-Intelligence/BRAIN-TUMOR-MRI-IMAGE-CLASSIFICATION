import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import data_loader as D

from conftest import CLASSES


def test_locate_dataset_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset directory not found"):
        D.locate_dataset(tmp_path / "nope")


def test_real_dataset_path_detected():
    if not C.DATASET_DIR.exists():
        pytest.skip("real dataset not present")
    assert D.locate_dataset() == C.DATASET_DIR


@pytest.mark.parametrize("name,stem", [
    ("Tr-gl_0496_jpg.rf.010620fbbbaa509aa81d7ce5bdf75de5.jpg", "Tr-gl_0496"),
    ("Tr-no_0426_jpg.rf.0026b06f369c5d51aca4c4c9beba88f0.jpg", "Tr-no_0426"),
    ("plain_image.jpg", "plain_image"),
])
def test_source_stem(name, stem):
    assert D.source_stem(name) == stem


def test_discover_and_labels(fake_dataset):
    df = D.discover_images(fake_dataset)
    assert D.get_class_names(df) == CLASSES
    assert set(df["original_split"]) == {"train", "val", "test"}
    assert D.build_class_mapping(CLASSES) == {c: i for i, c in enumerate(CLASSES)}


def test_corrupted_image_reported(fake_dataset):
    df = D.inspect_all(D.discover_images(fake_dataset))
    bad = df[~df["readable"]]
    assert len(bad) == 1 and bad.iloc[0]["filename"].startswith("corrupt")
    assert bad.iloc[0]["error"]
    assert df[df["readable"]]["phash"].notna().all()


def test_grouping_merges_stem_and_md5():
    df = pd.DataFrame({
        "filepath": list("abcde"), "class_name": ["glioma"] * 5, "original_split": ["train"] * 5,
        "source_stem": ["s1", "s1", "s2", "s3", "s4"], "md5": ["m1", "m2", "m3", "m3", "m4"],
        "phash": [0, 2**63, 2**62, 2**61, (2**64 - 1)],
    })
    out, rep = D.assign_groups(df, phash_threshold=0)
    g = out.set_index("filepath")["group_id"]
    assert g["a"] == g["b"]          # same source stem
    assert g["c"] == g["d"]          # identical md5
    assert g["a"] != g["c"] != g["e"]
    assert len(rep.exact_pairs) == 1


def test_cross_class_near_duplicates_not_chained():
    df = pd.DataFrame({
        "filepath": list("abc"), "class_name": ["glioma", "glioma", "meningioma"], "original_split": ["train"] * 3,
        "source_stem": list("xyz"), "md5": list("123"),
        "phash": [0b0, 0b1111, 0b11110000],  # a-b d=4 same class; a-c d=4 different class
    })
    out, rep = D.assign_groups(df, phash_threshold=4, cross_class_threshold=2)
    g = out.set_index("filepath")["group_id"]
    assert g["a"] == g["b"] and g["a"] != g["c"]
    assert not out["label_conflict"].any()


def _synthetic(n_groups=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for gid in range(n_groups):
        cls = CLASSES[gid % 4]
        for k in range(rng.integers(1, 4)):
            rows.append({"filepath": f"{gid}_{k}", "class_name": cls, "group_id": gid,
                         "md5": f"{gid}_{k}", "source_stem": f"s{gid}"})
    return pd.DataFrame(rows)


def test_split_no_leakage_and_proportions():
    df = D.grouped_stratified_split(_synthetic(), C.SplitConfig())
    assert (df.groupby("group_id")["split"].nunique() == 1).all()
    frac = df["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.05 and abs(frac["val"] - 0.15) < 0.05 and abs(frac["test"] - 0.15) < 0.05
    for s in ("train", "val", "test"):
        assert set(df[df.split == s]["class_name"]) == set(CLASSES)


def test_split_is_reproducible():
    a = D.grouped_stratified_split(_synthetic(), C.SplitConfig())
    b = D.grouped_stratified_split(_synthetic(), C.SplitConfig())
    assert (a["split"] == b["split"]).all()


def test_verify_no_leakage_detects():
    df = pd.DataFrame({"group_id": [1, 1], "md5": ["a", "b"], "source_stem": ["x", "y"], "split": ["train", "test"]})
    with pytest.raises(AssertionError, match="Leakage"):
        D.verify_no_leakage(df)


def test_saved_manifests_consistent():
    if not (C.SPLITS_DIR / "all_images.csv").exists():
        pytest.skip("run `python -m src.prepare_data` first")
    names, mapping = D.load_class_mapping()
    assert names == sorted(names) and list(mapping.values()) == list(range(len(names)))
    df = pd.concat([D.load_split(s) for s in ("train", "val", "test")])
    D.verify_no_leakage(df)
    assert df["filepath"].is_unique
    assert (df["label"] == df["class_name"].map(mapping)).all()
    assert all((C.PROJECT_ROOT / p).exists() for p in df["filepath"].head(50))
