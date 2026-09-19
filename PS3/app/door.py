"""Door subsystem — temporal segment detection + abnormal-resistance classification.

Segmentation
------------
The controller stream only contains rows recorded *during* a door cycle:
samples inside a cycle are exactly 20 ms apart, and consecutive cycles are
separated by gaps of 20 s or more. Splitting on a 0.1 s gap reproduced all 110
ground-truth boundaries in Train.csv exactly (IoU = 1.0), so segmentation is
deterministic and needs no model.

The opening/closing flags are deliberately NOT used: one of them is always 1,
so they carry no boundary information.

Classification
--------------
Per-cycle features from motor current / voltage / back-EMF (abnormal cycles draw
~720 mA mean current vs ~537 mA for normal ones) into a random forest.
Validation is a chronological split - segments come from one continuous stream,
so a random split would leak neighbouring-in-time cycles across the fold line.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

from common import MODEL_DIR, data_path, parse_door_times

GAP_SECONDS = 0.1
MODEL_PATH = os.path.join(MODEL_DIR, "door_rf.joblib")

SIGNALS = {
    "c": "Motor current(mA)",
    "v": "Motor Voltage(10mV)",
    "e": "Motor electrodynamic force",
}
ABNORMAL = "Abnormal resistance"
NORMAL = "Normal"


def load_stream(path_or_buffer) -> pd.DataFrame:
    d = pd.read_csv(path_or_buffer)
    d.columns = [c.strip() for c in d.columns]
    d["t"] = parse_door_times(d["Datetime"])
    return d


def segment(d: pd.DataFrame, gap_seconds: float = GAP_SECONDS) -> pd.DataFrame:
    """Add a `seg` id per door cycle by splitting on inter-sample time gaps."""
    dt = d["t"].diff().dt.total_seconds()
    d = d.copy()
    d["seg"] = (dt > gap_seconds).cumsum()
    return d


def segment_features(g: pd.DataFrame) -> dict:
    f = {"n": len(g)}
    f["dur"] = (g["t"].max() - g["t"].min()).total_seconds()
    for key, col in SIGNALS.items():
        a = g[col].to_numpy(dtype=float)
        f[f"{key}_mean"] = a.mean()
        f[f"{key}_max"] = a.max()
        f[f"{key}_std"] = a.std()
        f[f"{key}_med"] = np.median(a)
        f[f"{key}_q90"] = np.percentile(a, 90)
        f[f"{key}_auc"] = a.sum()
        for i, part in enumerate(np.array_split(a, 3)):  # per-phase means
            f[f"{key}_m{i}"] = part.mean()
    pos = g["Door leaf position"].to_numpy(dtype=float)
    f["p_range"] = pos.max() - pos.min()
    cur = g[SIGNALS["c"]].to_numpy(dtype=float)
    emf = g[SIGNALS["e"]].to_numpy(dtype=float)
    f["ratio_ce"] = cur.mean() / (emf.mean() + 1.0)  # current per unit back-EMF
    f["closing"] = int(g["Door is closing"].max())
    return f


def build_table(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (feature matrix, segment index with native-format timestamps)."""
    feats, index = [], []
    for seg, g in d.groupby("seg"):
        feats.append(segment_features(g))
        index.append(
            {
                "seg": seg,
                "start_time": g["Datetime"].iloc[0],
                "end_time": g["Datetime"].iloc[-1],
                "n_rows": len(g),
            }
        )
    return pd.DataFrame(feats), pd.DataFrame(index)


FEATURE_ORDER: list[str] | None = None


def train(verbose: bool = True):
    d = segment(load_stream(data_path("Door", "Train.csv")))
    truth = pd.read_csv(data_path("Door", "Train_Segments_Answer.csv"))
    X, idx = build_table(d)

    # Sanity check: segmentation must reproduce the published boundaries.
    exact = (
        len(idx) == len(truth)
        and (idx["start_time"].values == truth["start_time"].values).all()
        and (idx["end_time"].values == truth["end_time"].values).all()
    )
    y = (truth["status"].values == ABNORMAL).astype(int)

    # Chronological holdout: last 25% of the stream is unseen during fitting.
    cut = int(len(X) * 0.75)
    model = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=0
    )
    model.fit(X.iloc[:cut], y[:cut])
    pred = model.predict(X.iloc[cut:])
    report = classification_report(y[cut:], pred, digits=3, zero_division=0)
    macro = f1_score(y[cut:], pred, average="macro")

    if verbose:
        print(f"[door] segmentation reproduces ground truth exactly: {exact}")
        print(f"[door] {len(X)} segments, {y.sum()} abnormal")
        print(f"[door] chronological holdout macro F1: {macro:.3f}")
        print(report)

    model.fit(X, y)  # refit on everything for deployment
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": model, "features": list(X.columns)}, MODEL_PATH)
    return {"holdout_macro_f1": float(macro), "segmentation_exact": bool(exact)}


def predict(path_or_buffer) -> pd.DataFrame:
    bundle = joblib.load(MODEL_PATH)
    d = segment(load_stream(path_or_buffer))
    X, idx = build_table(d)
    X = X[bundle["features"]]
    proba = bundle["model"].predict_proba(X)[:, 1]
    out = idx[["start_time", "end_time"]].copy()
    out["prediction"] = np.where(proba >= 0.5, ABNORMAL, NORMAL)
    out["confidence"] = np.round(np.maximum(proba, 1 - proba), 4)
    return out


def predict_with_traces(path_or_buffer):
    """Predictions plus the raw current trace per segment, for the app."""
    bundle = joblib.load(MODEL_PATH)
    d = segment(load_stream(path_or_buffer))
    X, idx = build_table(d)
    proba = bundle["model"].predict_proba(X[bundle["features"]])[:, 1]
    idx = idx.copy()
    idx["prediction"] = np.where(proba >= 0.5, ABNORMAL, NORMAL)
    idx["confidence"] = np.round(np.maximum(proba, 1 - proba), 4)
    idx["mean_current_mA"] = X["c_mean"].round(1).values
    idx["peak_current_mA"] = X["c_max"].round(1).values
    return idx, d


if __name__ == "__main__":
    train()
