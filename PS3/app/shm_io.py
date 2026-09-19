"""I/O for the SHM dynamic-stress dataset.

Each ``*.csv`` is a single dynamic-stress time series (one column, ~581k samples
per file).  The column header is a numeric artefact of the export and carries no
information about the damage level, so only the values are used.

``Train_Labels.csv`` holds the reference Miner's-rule cumulative damage per
training file.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd

from common import DATA_ROOT as _DATASETS_ROOT

DATA_ROOT = os.path.join(_DATASETS_ROOT, "SHM")
TRAIN_DIR = os.path.join(DATA_ROOT, "Train")
TEST_DIR = os.path.join(DATA_ROOT, "Test")
LABELS_CSV = os.path.join(DATA_ROOT, "Train_Labels.csv")


def _numeric_key(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


def list_split(split: str) -> List[str]:
    d = {"train": TRAIN_DIR, "test": TEST_DIR}[split]
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".csv")]
    return sorted(files, key=_numeric_key)


def load_series(path: str) -> np.ndarray:
    """Return the dynamic-stress series in ``path`` as a 1-D float array."""
    return pd.read_csv(path).iloc[:, 0].to_numpy(dtype=float)


@lru_cache(maxsize=None)
def load_labels() -> Dict[str, float]:
    """Map ``filename -> damage`` for the training files."""
    df = pd.read_csv(LABELS_CSV)
    return dict(zip(df["filename"], df["damage"].astype(float)))


def load_split(split: str):
    """Return ``(file_ids, series_list, damage_or_None)`` for a split."""
    files = list_split(split)
    ids = [os.path.basename(f) for f in files]
    series = [load_series(f) for f in files]
    if split == "train":
        lab = load_labels()
        damage = np.array([lab[i] for i in ids])
        return ids, series, damage
    return ids, series, None
