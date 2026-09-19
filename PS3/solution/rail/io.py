"""I/O, file registry and wheel-geometry helpers for the Rail Corrugation task.

Physical conventions (from the Info Kit):
  * ``Rotating speed`` column is NOT a speed; it is the raw 0/1 output of a
    sensor watching a 90-tooth wheel.  ``v = (1 / tooth_period) * pi * D / 90``.
  * 64 axle boxes = 8 cars x 8 positions.  Odd positions -> Side I rail,
    even positions -> Side II rail.
  * Each axle box contributes a *vibration* and a *shock* channel, interleaved
    after the speed column: vib, shock, vib, shock, ...
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

FS = 10_000                 # sampling frequency [Hz]
WHEEL_DIAMETER = 0.85       # [m]
TEETH = 90                  # teeth per revolution
N_CARS = 8
N_POS = 8
N_AXLEBOXES = N_CARS * N_POS            # 64
SIDE1_POS = (0, 2, 4, 6)                # 0-based odd positions -> Side I
SIDE2_POS = (1, 3, 5, 7)                # 0-based even positions -> Side II
WHEEL_CIRC = np.pi * WHEEL_DIAMETER     # [m] per revolution

_THIS = os.path.dirname(os.path.abspath(__file__))
PS3_ROOT = os.path.dirname(os.path.dirname(_THIS))          # .../PS3
DATA_ROOT = os.path.join(PS3_ROOT, "02_Datasets", "Rail_Corrugation")
TRAIN_DIR = os.path.join(DATA_ROOT, "Train")
TEST_DIR = os.path.join(DATA_ROOT, "Test")
LABELS_CSV = os.path.join(DATA_ROOT, "Train_Labels.csv")

LABEL_NAMES = ("Normal", "Side I", "Side II")
LABEL_TO_INT = {name: i for i, name in enumerate(LABEL_NAMES)}


# --------------------------------------------------------------------------- #
# File registry / labels
# --------------------------------------------------------------------------- #
def _numeric_key(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


def list_split(split: str) -> List[str]:
    """Return the sorted absolute paths for ``split`` in {'train','test'}."""
    if split == "train":
        d = TRAIN_DIR
    elif split == "test":
        d = TEST_DIR
    else:
        raise ValueError(f"unknown split {split!r}")
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".csv")]
    return sorted(files, key=_numeric_key)


@lru_cache(maxsize=1)
def load_labels() -> Dict[str, str]:
    """filename -> label ('Normal', 'Side I', 'Side II')."""
    df = pd.read_csv(LABELS_CSV)
    return dict(zip(df["filename"], df["label"]))


def label_vector(paths: List[str]) -> np.ndarray:
    labels = load_labels()
    return np.array([LABEL_TO_INT[labels[os.path.basename(p)]] for p in paths], dtype=np.int64)


# --------------------------------------------------------------------------- #
# Low-level loading
# --------------------------------------------------------------------------- #
def read_raw(path: str) -> np.ndarray:
    """Read a rail CSV as float32 ``(10000, 129)`` (header skipped)."""
    df = pd.read_csv(path)
    return df.to_numpy(dtype=np.float32)


def split_channels(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vibration, shock) each shaped ``(T, 8 cars, 8 positions)``."""
    vib = raw[:, 1::2]
    shk = raw[:, 2::2]
    t = raw.shape[0]
    return vib.reshape(t, N_CARS, N_POS), shk.reshape(t, N_CARS, N_POS)


# --------------------------------------------------------------------------- #
# Speed / phase extraction from the tooth toggle
# --------------------------------------------------------------------------- #
def rising_edges(toggle: np.ndarray) -> np.ndarray:
    b = (toggle > 0.5).astype(np.int8)
    return np.flatnonzero((b[1:] == 1) & (b[:-1] == 0)) + 1


def speed_stats(toggle: np.ndarray) -> Dict[str, float]:
    """Estimate train speed and reliability from the tooth toggle."""
    edges = rising_edges(toggle)
    n = int(len(edges))
    if n < 2:
        return dict(speed_mps=0.0, speed_kmh=0.0, n_edges=n, revs=n / TEETH,
                    reliable=False, order_reliable=False, tooth_hz=0.0)
    periods = np.diff(edges) / FS
    period = float(np.median(periods))
    tooth_hz = 1.0 / period if period > 0 else 0.0
    v = tooth_hz * WHEEL_CIRC / TEETH
    revs = n / TEETH
    return dict(
        speed_mps=float(v),
        speed_kmh=float(v * 3.6),
        n_edges=n,
        revs=float(revs),
        tooth_hz=float(tooth_hz),
        reliable=bool(n >= 6),          # enough edges for a credible speed
        order_reliable=bool(revs >= 0.5),  # enough revolutions to order-track
    )


def theta_from_edges(edges: np.ndarray, n_samples: int) -> np.ndarray:
    """Wheel angle (rad) at every sample, piecewise-linear between tooth edges."""
    if len(edges) == 0:
        return np.zeros(n_samples, dtype=np.float64)
    t_edge = edges.astype(np.float64) / FS
    theta_edge = 2.0 * np.pi * np.arange(len(edges)) / TEETH
    all_t = np.arange(n_samples, dtype=np.float64) / FS
    theta = np.interp(all_t, t_edge, theta_edge)
    if len(edges) >= 2:
        rate = (theta_edge[-1] - theta_edge[0]) / (t_edge[-1] - t_edge[0])
        before = all_t < t_edge[0]
        after = all_t > t_edge[-1]
        theta[before] = theta_edge[0] - rate * (t_edge[0] - all_t[before])
        theta[after] = theta_edge[-1] + rate * (all_t[after] - t_edge[-1])
    # enforce strict monotonicity for interpolation
    theta = np.maximum.accumulate(theta + 1e-12 * np.arange(n_samples))
    return theta


# --------------------------------------------------------------------------- #
# Order tracking (angular resampling)
# --------------------------------------------------------------------------- #
def order_resample(x: np.ndarray, theta: np.ndarray, samples_per_rev: int = 256
                   ) -> Optional[np.ndarray]:
    """Resample signal(s) onto a uniform wheel-angle grid.

    ``x`` may be ``(T,)`` or ``(T, C)``.  Returns ``(N, ...)`` or ``None`` when
    there is not enough angular span to be useful.
    """
    span = float(theta[-1] - theta[0])
    if span <= 1e-6:
        return None
    n_out = int(samples_per_rev * span / (2.0 * np.pi))
    if n_out < 16:
        return None
    grid = theta[0] + np.linspace(0.0, span, n_out)
    if x.ndim == 1:
        return np.interp(grid, theta, x).astype(np.float32)
    out = np.empty((n_out, x.shape[1]), dtype=np.float32)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(grid, theta, x[:, c])
    return out


# --------------------------------------------------------------------------- #
# Convenience container
# --------------------------------------------------------------------------- #
@dataclass
class Recording:
    path: str
    raw: np.ndarray
    vibration: np.ndarray          # (T, 8, 8)
    shock: np.ndarray              # (T, 8, 8)
    theta: np.ndarray              # (T,) wheel angle
    speed: Dict[str, float]

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def load_recording(path: str) -> Recording:
    raw = read_raw(path)
    vib, shk = split_channels(raw)
    stats = speed_stats(raw[:, 0])
    theta = theta_from_edges(rising_edges(raw[:, 0]), raw.shape[0])
    return Recording(path=path, raw=raw, vibration=vib, shock=shk, theta=theta, speed=stats)


if __name__ == "__main__":  # tiny smoke test
    for split in ("train", "test"):
        files = list_split(split)
        print(f"{split}: {len(files)} files, e.g. {os.path.basename(files[0])}")
    rec = load_recording(list_split("train")[0])
    print("shape", rec.raw.shape, "speed", rec.speed)
