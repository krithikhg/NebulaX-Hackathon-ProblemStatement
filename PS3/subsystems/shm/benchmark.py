"""Prototype benchmark: physics inverse vs ML baselines vs residual correction.

Run:  cd PS3/subsystems && ../../.venv/bin/python -m shm.benchmark
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shm import cache, fit, io

TIME = slice(0, 10)
SPEC = slice(10, 22)
ALPHAS = np.logspace(-3, 3, 13)
N_SPLITS = 8
SEEDS = range(5)


def _ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def _phys_fit(Ssub, Dsub, grid):
    best = (np.inf, 0, 1.0)
    for j in range(Ssub.shape[1]):
        k = fit.optimal_scale(Ssub[:, j], Dsub)
        e = fit.mape(Dsub, k * Ssub[:, j])
        if e < best[0]:
            best = (e, j, k)
    return best[1], best[2]


def _ridge_log(Xtr, Dtr):
    m = _ridge()
    m.fit(Xtr, np.log(Dtr))
    return m


def _pred_log(m, X):
    return np.exp(m.predict(X))


def cv_physics(S, D, grid, seeds=SEEDS):
    out = []
    for s in seeds:
        pred = np.zeros_like(D)
        for tr, va in KFold(N_SPLITS, shuffle=True, random_state=s).split(D):
            j, k = _phys_fit(S[tr], D[tr], grid)
            pred[va] = k * S[va, j]
        out.append(fit.mape(D, pred))
    return out


def cv_ml(X, D, cols, seeds=SEEDS):
    out = []
    for s in seeds:
        pred = np.zeros_like(D)
        for tr, va in KFold(N_SPLITS, shuffle=True, random_state=s).split(D):
            m = _ridge_log(X[tr][:, cols], D[tr])
            pred[va] = _pred_log(m, X[va][:, cols])
        out.append(fit.mape(D, pred))
    return out


def cv_physics_bank(S, D, grid, seeds=SEEDS):
    L = np.log(S)
    out = []
    for s in seeds:
        pred = np.zeros_like(D)
        for tr, va in KFold(N_SPLITS, shuffle=True, random_state=s).split(D):
            m = _ridge_log(L[tr], D[tr])
            pred[va] = _pred_log(m, L[va])
        out.append(fit.mape(D, pred))
    return out


def cv_residual(S, X, D, grid, cols, seeds=SEEDS):
    out = []
    for s in seeds:
        pred = np.zeros_like(D)
        for tr, va in KFold(N_SPLITS, shuffle=True, random_state=s).split(D):
            j, k = _phys_fit(S[tr], D[tr], grid)
            target = np.log(D[tr]) - np.log(k * S[tr, j])
            g = _ridge()
            g.fit(X[tr][:, cols], target)
            corr = g.predict(X[va][:, cols])
            pred[va] = k * S[va, j] * np.exp(corr)
        out.append(fit.mape(D, pred))
    return out


def _row(name, errs):
    errs = np.array(errs)
    print(f"{name:<34} MAPE={errs.mean() * 100:6.2f}% +-{errs.std() * 100:4.2f}  score={1 - errs.mean():.4f}")


def main():
    ids, _, D = io.load_split("train")
    S = cache.damage_grid("train")
    X = np.nan_to_num(cache.generic_features("train"), nan=0.0, posinf=0.0, neginf=0.0)
    grid = cache.GRID
    print(f"train files={len(ids)}  Sgrid={S.shape}  generic={X.shape}\n")

    print("=== Tier 1: baselines ===")
    _row("constant median", [np.mean(np.abs(D - np.median(D)) / D)] * len(SEEDS))
    _row("physics inverse (m,k)", cv_physics(S, D, grid))
    _row("Ridge: log S(5) single", cv_ml(np.log(S), D, [int(np.argmin(np.abs(grid - 5.0)))]))
    _row("Ridge: physics bank log S(p)", cv_physics_bank(S, D, grid))
    _row("Ridge: generic time", cv_ml(X, D, list(range(X.shape[1]))[TIME]))
    _row("Ridge: generic time+spec", cv_ml(X, D, list(range(X.shape[1]))))
    _row("Ridge: generic spec only", cv_ml(X, D, list(range(X.shape[1]))[SPEC]))

    print("\n=== Tier 2: physics + residual Ridge ===")
    _row("phys + residual(generic all)", cv_residual(S, X, D, grid, list(range(X.shape[1]))))
    _row("phys + residual(time only)", cv_residual(S, X, D, grid, list(range(X.shape[1]))[TIME]))
    _row("phys + residual(spec only)", cv_residual(S, X, D, grid, list(range(X.shape[1]))[SPEC]))


if __name__ == "__main__":
    main()
