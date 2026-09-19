"""Rail corrugation subsystem — Normal / Side I / Side II.

This is the **single** rail implementation: the web app (server.py) and the
submission CLI (predict.py) both call the functions here. The model is the
stronger LightGBM pipeline (out-of-fold macro F1 ~0.88) rather than the earlier
random forest.

Split across three modules:
  * rail_io.py        loading, tooth-toggle speed/phase, order resampling
  * rail_features.py  feature extraction (v5): physics + localisation + shape + T7
  * rail.py           model, in-fold top-K selection, inference (this file)

The saved bundle, models/rail_model.joblib, is 10 seed-averaged LightGBM models
over the top-40 features, plus `stationary_speed` for the UI's "not assessed"
rule. It is produced by PS3/subsystems/rail.
"""
from __future__ import annotations

import glob
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import rail_features
import rail_io
from common import MODEL_DIR

FS = rail_io.FS
WHEEL_DIAMETER_M = rail_io.WHEEL_DIAMETER
TEETH = rail_io.TEETH

# Frequency bands for the app's spectrum chart (mean log energy per rail side).
BANDS = [(50, 200), (200, 400), (400, 600), (600, 900), (900, 1300),
         (1300, 1800), (1800, 2500), (2500, 3500), (3500, 5000)]
STATIONARY_SPEED = 0.5  # m/s
CLASSES = ["Normal", "Side I", "Side II"]
MODEL_PATH = os.path.join(MODEL_DIR, "rail_model.joblib")
DEFAULT_K = 40
SEEDS = tuple(range(10))

GBM_BASE = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=8,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
    verbose=-1, random_state=0,
)


class LGBM3:
    """Flat 3-class LightGBM."""

    def __init__(self, **kw):
        params = dict(objective="multiclass", num_class=3,
                      class_weight="balanced", **GBM_BASE)
        params.update(kw)
        self.model = lgb.LGBMClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


# --------------------------------------------------------------- features ----
def extract_features(path_or_buffer) -> dict:
    """v5 model features, plus the 9-band side means the UI plots and `_speed`."""
    rec = rail_io.load_recording(path_or_buffer)
    feats = rail_features.extract_recording(rec)
    feats["_speed"] = float(rec.speed["speed_mps"])

    x2 = rec.vibration.reshape(rec.vibration.shape[0], -1).astype(np.float64)
    freqs = np.fft.rfftfreq(x2.shape[0], 1.0 / FS)
    power = np.abs(np.fft.rfft(x2, axis=0)) ** 2
    for i, (lo, hi) in enumerate(BANDS):
        m = (freqs >= lo) & (freqs < hi)
        e = np.log10(power[m].sum(axis=0) + 1e-12).reshape(8, 8)
        feats[f"s1vib_b{i}_mean"] = float(e[:, rail_io.SIDE1_POS].mean())
        feats[f"s2vib_b{i}_mean"] = float(e[:, rail_io.SIDE2_POS].mean())
    return feats


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("file", "label", "y")]


# --------------------------------------------------------------- inference ----
def predict_features(F: pd.DataFrame) -> pd.DataFrame:
    """One row per recording; each row carries a `filename` and feature keys."""
    bundle = joblib.load(MODEL_PATH)
    sel, models = bundle["selected"], bundle["models"]
    records = F.to_dict("records")
    X = np.asarray([[r.get(c, np.nan) for c in sel] for r in records], dtype=float)
    P = np.mean([m.predict_proba(X) for m in models], axis=0)
    out = pd.DataFrame({
        "file_id": F["filename"].to_numpy(),
        "prediction": [CLASSES[i] for i in P.argmax(axis=1)],
    })
    out["confidence"] = np.round(P.max(axis=1), 4)
    out["speed_m_s"] = np.round(F["_speed"].to_numpy(dtype=float), 2)
    return out


def predict(path_or_buffer, file_id: str | None = None) -> pd.DataFrame:
    feats = extract_features(path_or_buffer)
    raw = file_id or (path_or_buffer if isinstance(path_or_buffer, str)
                      else getattr(path_or_buffer, "name", "input.csv"))
    feats["filename"] = os.path.basename(str(raw))
    return predict_features(pd.DataFrame([feats]))


def predict_folder(folder: str) -> pd.DataFrame:
    paths = sorted(
        glob.glob(os.path.join(folder, "*.csv")),
        key=lambda p: int("".join(ch for ch in os.path.basename(p) if ch.isdigit()) or -1),
    )
    rows = []
    for p in paths:
        feats = extract_features(p)
        feats["filename"] = os.path.basename(p)
        rows.append(feats)
    return predict_features(pd.DataFrame(rows))


# ---------------------------------------------------------------- training ----
def _select_idx(X: np.ndarray, y: np.ndarray, K: int = DEFAULT_K, seed: int = 0) -> np.ndarray:
    m = LGBM3(random_state=seed).fit(X, y)
    return np.argsort(m.feature_importances_)[::-1][:K]


def _oof(X: np.ndarray, y: np.ndarray, K: int = DEFAULT_K, seeds=range(5)) -> np.ndarray:
    seeds = list(seeds)
    oof = np.zeros((len(y), 3))
    for s in seeds:
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            idx = _select_idx(X[tr], y[tr], K, seed=s)
            m = LGBM3(random_state=s).fit(X[tr][:, idx], y[tr])
            oof[va] += m.predict_proba(X[va][:, idx])
    return oof / len(seeds)


def train(verbose: bool = True) -> dict:
    tr = rail_features.build("train")
    cols = _feature_columns(tr)
    X, y = tr[cols].to_numpy(dtype=float), tr["y"].to_numpy()
    idx = _select_idx(X, y, DEFAULT_K, seed=0)
    sel = [cols[i] for i in idx]
    models = [LGBM3(random_state=s).fit(X[:, idx], y).model for s in SEEDS]
    joblib.dump({"models": models, "selected": sel,
                 "stationary_speed": STATIONARY_SPEED}, MODEL_PATH)

    macro = float(f1_score(y, _oof(X, y).argmax(axis=1), average="macro"))
    if verbose:
        print(f"[rail] {len(y)} files, {len(sel)} selected features, "
              f"oof macro F1 {macro:.3f} -> {MODEL_PATH}")
    return {"oof_macro_f1": macro}


if __name__ == "__main__":
    train()
