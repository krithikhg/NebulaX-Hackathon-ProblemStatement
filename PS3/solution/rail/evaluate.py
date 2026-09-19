"""CV evaluation harness for the Rail task.

Provides: feature matrix assembly, stratified CV producing out-of-fold
probabilities, and reporting (macro F1, per-class, confusion, per-speed-band).
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common.metrics import confusion, macro_f1, per_class_f1
from rail import features, io

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
SPEED_BANDS = [(0, 15), (15, 35), (35, 60), (60, 90)]


def feature_columns(df: pd.DataFrame) -> List[str]:
    drop = {"file", "label", "y"}
    return [c for c in df.columns if c not in drop]


def cv_oof(X: np.ndarray, y: np.ndarray, factory: Callable,
           n_splits: int = 5, seed: int = 0) -> np.ndarray:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), 3), dtype=np.float64)
    for tr, va in skf.split(X, y):
        model = factory()
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])
    return oof


# --------------------------------------------------------------------------- #
# Feature selection (importance-based) -- selection happens *inside* folds
# --------------------------------------------------------------------------- #
DEFAULT_K = 40


def select_top_idx(X, y, K=DEFAULT_K, seed=0) -> np.ndarray:
    from rail.models import LGBM3
    m = LGBM3(random_state=seed).fit(X, y)
    return np.argsort(m.feature_importances_)[::-1][:K]


def oof_select(X, y, K=DEFAULT_K, seeds=range(5)) -> np.ndarray:
    """Out-of-fold probabilities where top-K features are chosen per fold."""
    oof = np.zeros((len(y), 3))
    seeds = list(seeds)
    for s in seeds:
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            from rail.models import LGBM3
            idx = select_top_idx(X[tr], y[tr], K, seed=s)
            m = LGBM3(random_state=s).fit(X[tr][:, idx], y[tr])
            oof[va] += m.predict_proba(X[va][:, idx])
    return oof / len(seeds)


def fit_select_predict(X, y, Xte, K=DEFAULT_K, seeds=range(10)):
    """Select features on all training data, then seed-average the test probs."""
    from rail.models import LGBM3
    idx = select_top_idx(X, y, K, seed=0)
    P = np.mean([LGBM3(random_state=s).fit(X[:, idx], y).predict_proba(Xte[:, idx])
                 for s in seeds], axis=0)
    return P, idx


def report(y: np.ndarray, oof: np.ndarray, speed: np.ndarray,
           title: str = "") -> Dict:
    pred = oof.argmax(axis=1)
    out = {
        "title": title,
        "macro_f1": macro_f1(y, pred),
        "per_class_f1": dict(zip(io.LABEL_NAMES, np.round(per_class_f1(y, pred), 4).tolist())),
        "confusion": confusion(y, pred).tolist(),
    }
    bands = {}
    for lo, hi in SPEED_BANDS:
        m = (speed >= lo) & (speed < hi)
        if m.sum() == 0:
            continue
        present = np.unique(y[m])
        f1 = float(np.mean(per_class_f1(y[m], pred[m])[present]))
        bands[f"{lo}-{hi}"] = {"n": int(m.sum()),
                               "macro_f1_present": round(f1, 4),
                               "n_fault": int((y[m] > 0).sum())}
    out["speed_bands"] = bands
    return out


def print_report(rep: Dict):
    print(f"\n=== {rep['title']} ===")
    print(f"macro F1 = {rep['macro_f1']:.4f}   per-class = {rep['per_class_f1']}")
    cm = np.array(rep["confusion"])
    print("confusion (rows=true Normal/SideI/SideII, cols=pred):")
    print(cm)
    for band, d in rep["speed_bands"].items():
        print(f"  speed {band:>6} km/h: n={d['n']:3d} faults={d['n_fault']:2d} "
              f"macroF1(present)={d['macro_f1_present']:.3f}")
    return rep


def save_results(name: str, rep: Dict, oof: np.ndarray, importance=None, cols=None):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, f"result_{name}.json"), "w") as f:
        json.dump(rep, f, indent=2)
    np.save(os.path.join(CACHE_DIR, f"oof_{name}.npy"), oof)
    if importance is not None and cols is not None:
        imp = pd.DataFrame({"feature": cols, "importance": importance})
        imp.sort_values("importance", ascending=False).to_csv(
            os.path.join(CACHE_DIR, f"importance_{name}.csv"), index=False)
