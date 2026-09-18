"""Rail corrugation subsystem — Normal / Side I / Side II from axle-box vibration.

Data
----
One file = 1 s at 10 kHz across 64 axle boxes (vibration + shock per box).
Odd positions (1,3,5,7) ride the Side I rail, even positions the Side II rail,
so the two sides are scored from the same recording.

Features
--------
Per channel: RMS, peak, kurtosis and log energy in 9 frequency bands.
Aggregated per side as mean / max / std across that side's 32 boxes, plus
side-difference features (Side I minus Side II) - corrugation is one-sided, so
the *asymmetry* between rails carries most of the signal.

Two deliberate choices, both stated in the write-up:

1. Raw speed is NOT a feature. Every fault file in the training set was recorded
   above 9.7 m/s while Normal files span 0-19.5 m/s including stationary ones,
   so a model given speed can learn a sampling artefact rather than physics.
2. Stationary files (speed ~ 0) are forced to Normal: with no wheel-rail
   excitation there is no corrugation signature to detect.

Class priors are tuned on grouped out-of-fold probabilities to maximise MACRO
F1, not accuracy - predicting Normal everywhere scores ~86% accuracy but macro
F1 of only 0.33.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

from common import MODEL_DIR, data_path

FS = 10_000
WHEEL_DIAMETER_M = 0.85
TEETH = 90
BANDS = [(50, 200), (200, 400), (400, 600), (600, 900), (900, 1300),
         (1300, 1800), (1800, 2500), (2500, 3500), (3500, 5000)]
STATIONARY_SPEED = 0.5  # m/s
MODEL_PATH = os.path.join(MODEL_DIR, "rail_rf.joblib")
CLASSES = ["Normal", "Side I", "Side II"]


def _channel_stats(x: np.ndarray, freqs: np.ndarray) -> list[float]:
    x = x - x.mean()
    power = np.abs(np.fft.rfft(x)) ** 2
    row = [
        float(np.sqrt((x ** 2).mean())),
        float(np.abs(x).max()),
        float(np.mean(x ** 4) / (np.mean(x ** 2) ** 2 + 1e-12)),
    ]
    for lo, hi in BANDS:
        row.append(float(np.log10(power[(freqs >= lo) & (freqs < hi)].sum() + 1e-12)))
    return row


STAT_NAMES = ["rms", "peak", "kurt"] + [f"b{i}" for i in range(len(BANDS))]


def extract_features(path_or_buffer) -> dict:
    d = pd.read_csv(path_or_buffer, dtype=np.float32)
    values = d.values
    columns = list(d.columns)[1:]

    # Speed from the 90-tooth wheel encoder over a 1 s window.
    toggles = np.abs(np.diff(values[:, 0])).sum()
    speed = float(toggles / 2 / TEETH * np.pi * WHEEL_DIAMETER_M)

    X = values[:, 1:]
    freqs = np.fft.rfftfreq(X.shape[0], 1 / FS)
    buckets: dict[tuple[int, str], list[list[float]]] = {
        (s, k): [] for s in (1, 2) for k in ("vib", "shk")
    }
    for j, c in enumerate(columns):
        pos = int(c.split("position ")[1].split(" ")[0])
        side = 1 if pos % 2 == 1 else 2
        kind = "vib" if c.lower().startswith("vibration") else "shk"
        buckets[(side, kind)].append(_channel_stats(X[:, j], freqs))

    out: dict[str, float] = {"_speed": speed}
    for side in (1, 2):
        for kind in ("vib", "shk"):
            A = np.array(buckets[(side, kind)])
            for i, nm in enumerate(STAT_NAMES):
                out[f"s{side}{kind}_{nm}_mean"] = A[:, i].mean()
                out[f"s{side}{kind}_{nm}_max"] = A[:, i].max()
                out[f"s{side}{kind}_{nm}_std"] = A[:, i].std()
    for kind in ("vib", "shk"):
        for nm in STAT_NAMES:
            out[f"d{kind}_{nm}"] = (
                out[f"s1{kind}_{nm}_mean"] - out[f"s2{kind}_{nm}_mean"]
            )
            out[f"dmax{kind}_{nm}"] = (
                out[f"s1{kind}_{nm}_max"] - out[f"s2{kind}_{nm}_max"]
            )
    return out


def extract_folder(folder: str, verbose: bool = True) -> pd.DataFrame:
    rows = []
    files = sorted(os.listdir(folder), key=lambda f: (len(f), f))
    for i, f in enumerate(files):
        r = extract_features(os.path.join(folder, f))
        r["filename"] = f
        rows.append(r)
        if verbose and i % 50 == 0:
            print(f"  [rail] {i}/{len(files)}", flush=True)
    return pd.DataFrame(rows)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {"filename", "label"}
    return [c for c in df.columns if not c.startswith("_") and c not in skip]


def _apply_priors(proba: np.ndarray, priors: np.ndarray) -> np.ndarray:
    scaled = proba * priors
    return scaled.argmax(axis=1)


def train(cache: str | None = None, verbose: bool = True) -> dict:
    cache = cache or os.path.join(MODEL_DIR, "rail_train_features.csv")
    if os.path.exists(cache):
        F = pd.read_csv(cache)
    else:
        F = extract_folder(data_path("Rail_Corrugation", "Train"), verbose)
        os.makedirs(MODEL_DIR, exist_ok=True)
        F.to_csv(cache, index=False)

    labels = pd.read_csv(data_path("Rail_Corrugation", "Train_Labels.csv"))
    D = F.merge(labels, on="filename")
    feats = _feature_columns(D)
    X = D[feats].replace([np.inf, -np.inf], np.nan).fillna(-12.0)
    y = D["label"].values

    # Grouped-by-file CV is automatic here (one row per file); stratify on class.
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    oof = np.zeros((len(D), 3))
    for tr, te in cv.split(X, y):
        m = RandomForestClassifier(
            n_estimators=600, class_weight="balanced", random_state=0, n_jobs=-1
        ).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, [list(m.classes_).index(c) for c in CLASSES]]

    stationary = D["_speed"].values < STATIONARY_SPEED
    best = (0.0, np.ones(3))
    grid = np.arange(0.6, 3.01, 0.1)
    for a in grid:
        for b in grid:
            priors = np.array([1.0, a, b])
            pred = np.array(CLASSES)[_apply_priors(oof, priors)]
            pred[stationary] = "Normal"
            s = f1_score(y, pred, average="macro")
            if s > best[0]:
                best = (s, priors)
    macro, priors = best
    pred = np.array(CLASSES)[_apply_priors(oof, priors)]
    pred[stationary] = "Normal"

    if verbose:
        print(f"[rail] {len(D)} files: " + str(pd.Series(y).value_counts().to_dict()))
        print(f"[rail] out-of-fold macro F1: {macro:.3f} (priors {priors.round(2)})")
        print(classification_report(y, pred, digits=3, zero_division=0))

    model = RandomForestClassifier(
        n_estimators=600, class_weight="balanced", random_state=0, n_jobs=-1
    ).fit(X, y)
    joblib.dump(
        {"model": model, "features": feats, "priors": priors,
         "classes": CLASSES, "stationary_speed": STATIONARY_SPEED},
        MODEL_PATH,
    )
    return {"oof_macro_f1": float(macro)}


def predict_features(F: pd.DataFrame) -> pd.DataFrame:
    bundle = joblib.load(MODEL_PATH)
    X = F[bundle["features"]].replace([np.inf, -np.inf], np.nan).fillna(-12.0)
    m = bundle["model"]
    order = [list(m.classes_).index(c) for c in bundle["classes"]]
    proba = m.predict_proba(X)[:, order]
    pred = np.array(bundle["classes"])[_apply_priors(proba, bundle["priors"])]
    pred = np.where(F["_speed"].values < bundle["stationary_speed"], "Normal", pred)
    out = pd.DataFrame({"file_id": F["filename"], "prediction": pred})
    out["confidence"] = proba.max(axis=1).round(4)
    out["speed_m_s"] = F["_speed"].round(2).values
    return out


def predict(path_or_buffer, file_id: str | None = None) -> pd.DataFrame:
    r = extract_features(path_or_buffer)
    r["filename"] = os.path.basename(
        str(file_id or getattr(path_or_buffer, "name", "input.csv"))
    )
    return predict_features(pd.DataFrame([r]))


def predict_folder(folder: str, cache: str | None = None) -> pd.DataFrame:
    if cache and os.path.exists(cache):
        F = pd.read_csv(cache)
    else:
        F = extract_folder(folder)
        if cache:
            F.to_csv(cache, index=False)
    return predict_features(F)


if __name__ == "__main__":
    train()
