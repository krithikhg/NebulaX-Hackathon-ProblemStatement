"""Fit the two global constants of the Miner's-rule damage model.

Given rainflow cycles, the model is ``D_hat = k * S(m)`` with
``S(m) = sum n_i * range_i ** m``.  For any ``m`` the MAPE-optimal scale ``k``
has a closed form (a weighted median), so the only search is 1-D over ``m``.

Honest evaluation: ``cv()`` refits *both* ``m`` and ``k`` on each training fold
and scores the held-out files, so no label information leaks across the split.
"""
from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shm_damage import pseudo_damage_series

ALPHAS = np.logspace(-3, 3, 13)

EXP_GRID = np.round(np.arange(3.0, 7.001, 0.05), 2)
DEFAULT_M = 5.0


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)))


def score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return max(0.0, 1.0 - mape(y_true, y_pred))


def optimal_scale(S: np.ndarray, D: np.ndarray) -> float:
    """Scale minimising ``mean(|D - k*S| / D)`` — weighted median of ``D/S``.

    MAPE = (1/n) sum_i (S_i/D_i) * |D_i/S_i - k|, i.e. a weighted L1 problem in
    ``k`` whose minimiser is the weighted median of ``t_i = D_i/S_i`` with
    weights ``w_i = S_i/D_i``.
    """
    t = D / S
    w = S / D
    order = np.argsort(t)
    t, w = t[order], w[order]
    cw = np.cumsum(w)
    cw /= cw[-1]
    return float(t[np.searchsorted(cw, 0.5)])


def fit(cycles: Sequence, D: np.ndarray, grid=EXP_GRID) -> Dict:
    """Search ``m`` on ``grid``; return best exponent, scale and train MAPE."""
    best = {"m": DEFAULT_M, "k": 1.0, "mape": np.inf}
    for m in grid:
        S = pseudo_damage_series(cycles, m)
        k = optimal_scale(S, D)
        e = mape(D, k * S)
        if e < best["mape"]:
            best = {"m": float(m), "k": float(k), "mape": float(e)}
    return best


def predict(cycles: Sequence, model: Dict) -> np.ndarray:
    S = pseudo_damage_series(cycles, model["m"])
    return model["k"] * S


def fit_from_grid(Sgrid: np.ndarray, D: np.ndarray, grid=EXP_GRID) -> Dict:
    """Same as :func:`fit` but using a precomputed ``(n, len(grid))`` S matrix."""
    best = {"m": DEFAULT_M, "k": 1.0, "mape": np.inf}
    for j, m in enumerate(grid):
        k = optimal_scale(Sgrid[:, j], D)
        e = mape(D, k * Sgrid[:, j])
        if e < best["mape"]:
            best = {"m": float(m), "k": float(k), "mape": float(e), "j": int(j)}
    return best


def cv_from_grid(Sgrid: np.ndarray, D: np.ndarray, grid=EXP_GRID,
                 n_splits: int = 8, seed: int = 0) -> Dict:
    pred = np.zeros_like(D, dtype=float)
    for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(D):
        model = fit_from_grid(Sgrid[tr], D[tr], grid)
        pred[va] = model["k"] * Sgrid[va, model["j"]]
    return {"mape": mape(D, pred), "score": score(D, pred), "oof": pred}


def cv(cycles: Sequence, D: np.ndarray, n_splits: int = 8, seed: int = 0) -> Dict:
    """Out-of-fold MAPE, refitting ``(m, k)`` on every training fold."""
    pred = np.zeros_like(D, dtype=float)
    for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(D):
        model = fit([cycles[i] for i in tr], D[tr])
        pred[va] = predict([cycles[i] for i in va], model)
    return {"mape": mape(D, pred), "score": score(D, pred), "oof": pred}


def fit_bank(Sgrid: np.ndarray, D: np.ndarray):
    """Physics-informed Ridge: learn a weight profile over the exponent bank.

    Regressing ``log D`` on ``[log S(m) for m in grid]`` lets the model learn
    the S-N curve shape rather than assuming a single power law, while still
    using only the rainflow pseudo-damage features.  Standardised + RidgeCV.
    """
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    pipe.fit(np.log(Sgrid), np.log(D))
    return pipe


def predict_bank(pipe, Sgrid: np.ndarray) -> np.ndarray:
    return np.exp(pipe.predict(np.log(Sgrid)))


def cv_bank(Sgrid: np.ndarray, D: np.ndarray, n_splits: int = 8, seed: int = 0) -> Dict:
    pred = np.zeros_like(D, dtype=float)
    for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(D):
        pred[va] = predict_bank(fit_bank(Sgrid[tr], D[tr]), Sgrid[va])
    return {"mape": mape(D, pred), "score": score(D, pred), "oof": pred}


def save_constants(model: Dict, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(model, fh, indent=2)


def load_constants(path: str) -> Dict:
    with open(path) as fh:
        return json.load(fh)
