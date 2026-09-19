"""Cached, reusable derived data for the SHM pipeline.

Rainflow counting (~0.5 s/file) and the pseudo-damage grid are the expensive
steps, so they are computed once and kept as pickles under ``shm/cache/``.
"""
from __future__ import annotations

import os
import pickle
from typing import List, Tuple

import numpy as np

from shm import damage, io

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

GRID = np.round(np.arange(3.0, 7.001, 0.05), 2)


def _pickle(name: str, builder):
    path = os.path.join(CACHE_DIR, name)
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    obj = builder()
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    return obj


def cycles(split: str) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """List of ``(range, mean, count)`` arrays per file, cached."""

    def build():
        _, series, _ = io.load_split(split)
        return [damage.rainflow_ranges(x) for x in series]

    return _pickle(f"cycles_v2_{split}.pkl", build)


def damage_grid(split: str, grid: np.ndarray = GRID) -> np.ndarray:
    """``(n_files, len(grid))`` matrix of ``S(m) = sum n * range**m``."""

    def build():
        cyc = cycles(split)
        return np.array(
            [[damage.pseudo_damage(r, c, float(m)) for m in grid] for r, _, c in cyc]
        )

    return _pickle(f"sgrid_v2_{split}_{len(grid)}.pkl", build)


def generic_features(split: str) -> np.ndarray:
    """Generic (physics-agnostic) time + spectral features, one row per file."""

    def build():
        _, series, _ = io.load_split(split)
        return np.array([_features(x) for x in series])

    return _pickle(f"generic_{split}.pkl", build)


def _features(x: np.ndarray) -> List[float]:
    x = np.asarray(x, dtype=float)
    n = x.size
    rms = float(np.sqrt(np.mean(x * x)))
    std = float(np.std(x))
    absx = np.abs(x)
    peak = float(absx.max())
    # time-domain
    feats = [
        np.log(rms + 1e-9),
        np.log(std + 1e-9),
        float(np.mean(x)),
        float(np.min(x)),
        float(np.max(x)),
        np.log(float(x.max() - x.min()) + 1e-9),
        peak / (rms + 1e-9),                       # crest factor
        float(np.mean((x - x.mean()) ** 3) / (std ** 3 + 1e-9)),  # skewness
        float(np.mean((x - x.mean()) ** 4) / (std ** 4 + 1e-9)),  # kurtosis
        float(np.mean(np.diff(np.sign(x)) != 0)),  # zero-crossing rate
    ]
    # spectral shape (normalised frequency - no sampling rate is given)
    xd = x - x.mean()
    spec = np.abs(np.fft.rfft(xd)) ** 2
    freq = np.linspace(0.0, 1.0, spec.size)        # normalised [0, Nyquist]
    tot = spec.sum() + 1e-12
    p = spec / tot
    centroid = float(np.sum(freq * p))
    spread = float(np.sqrt(np.sum((freq - centroid) ** 2 * p)))
    sk = float(np.sum((freq - centroid) ** 3 * p) / (spread ** 3 + 1e-12))
    ent = float(-np.sum(p * np.log(p + 1e-12)) / np.log(p.size))
    # band power fractions over 8 equal normalised bands
    bands = np.array_split(p, 8)
    band_feats = [float(np.log(b.sum() + 1e-12)) for b in bands]
    return feats + [centroid, spread, sk, ent] + band_feats
