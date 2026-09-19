"""Miner's-rule cumulative fatigue damage from a stress time series.

Pipeline (the standard fatigue-assessment chain the reference labels were built
with, per the Info Kit):

  1. rainflow counting  -> independent (range, mean, count) stress cycles
  2. S-N curve          -> ``N_i = C / sigma_a,i ** m``
  3. Miner's rule       -> ``D = sum_i n_i / N_i = (1/C) * sum_i n_i * a_i ** m``

Because the S-N exponent ``m`` and the curve constant ``C`` are global
material/component properties, a single ``(m, k=1/C)`` pair links the measured
stress to the damage: ``D = k * sum_i n_i * (range_i / 2) ** m``.  The factor
``2 ** -m`` is absorbed into ``k``, so we work with stress *ranges* directly.
"""
from __future__ import annotations

from typing import Tuple

import fatpack
import numpy as np
import rainflow

Cycles = Tuple[np.ndarray, np.ndarray, np.ndarray]

# Number of load classes used by the rainflow counter.  ``k=64`` is both
# fatpack's default and a sharp optimum against the reference labels, i.e. the
# organisers almost certainly used a 64-class rainflow count.  See the README.
CYCLE_K = 64


def rainflow_ranges(x: np.ndarray, k: int = CYCLE_K) -> Cycles:
    """Primary counter: ``fatpack`` ranges with ``k`` load classes.

    Returns a ``(range, mean, count)`` triple for API compatibility with the
    :func:`pseudo_damage` helpers (fatpack yields one cycle per range, so the
    count is all-ones; the mean is unused by the damage model).
    """
    rng = np.asarray(fatpack.find_rainflow_ranges(np.asarray(x, dtype=float), k=k),
                     dtype=float)
    return rng, np.zeros_like(rng), np.ones_like(rng)


def rainflow_cycles(x: np.ndarray) -> Cycles:
    """Alternative counter (``rainflow`` package) kept for the ablation study."""
    c = np.array(list(rainflow.extract_cycles(np.asarray(x, dtype=float))))
    return c[:, 0], c[:, 1], c[:, 2]


def pseudo_damage(rng: np.ndarray, count: np.ndarray, m: float) -> float:
    """``sum n_i * range_i ** m`` — proportional to Miner damage for exponent m."""
    return float(np.sum(count * rng ** m))


def pseudo_damage_series(cycles_list, m: float) -> np.ndarray:
    """Vectorised pseudo-damage for a list of ``(range, mean, count)`` tuples."""
    return np.array([pseudo_damage(r, c, m) for r, _, c in cycles_list], dtype=float)


def cycle_stats(rng: np.ndarray) -> dict:
    """Descriptive rainflow statistics used for EDA / reporting."""
    tot = float(rng.size)
    return {
        "n_cycles": tot,
        "max_range": float(rng.max()) if tot else 0.0,
        "mean_range": float(rng.mean()) if tot else 0.0,
    }
