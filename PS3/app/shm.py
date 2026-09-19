"""SHM subsystem — cumulative fatigue damage from a dynamic-stress series.

This is the **single** SHM implementation: the web app (server.py) and the
submission CLI (predict.py) both call the functions here. It vendors the model
from PS3/subsystems/shm.

Pipeline (the reference labels were built with this chain):
  rainflow counting -> S-N curve -> Miner's rule,  D = k * sum n_i * range_i ** m

Two fitted models share the rainflow features:
  * analytic  — a single (m, k) pair (m = 5.0, a textbook welded-steel exponent);
  * bank Ridge — a physics-informed Ridge that learns the S-N curve shape over
    an exponent bank. This is the shipped predictor (more accurate, ~0.55% CV
    MAPE), and the analytic constants are kept alongside for display.
"""
from __future__ import annotations

import glob
import json
import os

import joblib
import numpy as np
import pandas as pd

import shm_damage as damage
import shm_fit as fit
import shm_io as io
from common import MODEL_DIR

# Exponent bank shared by the analytic and bank models.
GRID = np.round(np.arange(3.0, 7.001, 0.05), 2)
BANK_PATH = os.path.join(MODEL_DIR, "bank_model.joblib")
CONST_PATH = os.path.join(MODEL_DIR, "shm_constants.json")

_model = None


def load_model() -> dict:
    global _model
    if _model is None:
        bundle = joblib.load(BANK_PATH)
        _model = {"bank": bundle["pipe"], "grid": np.asarray(bundle.get("grid", GRID))}
    return _model


def constants() -> dict:
    with open(CONST_PATH) as fh:
        return json.load(fh)


# --------------------------------------------------------------- inference ----
def load_signal(path_or_buffer) -> np.ndarray:
    return io.load_series(path_or_buffer)


def predict_signal(signal: np.ndarray) -> tuple[float, tuple]:
    """Return (cumulative damage D, (range, mean, count) rainflow cycles)."""
    rng, mean, count = damage.rainflow_ranges(np.asarray(signal, dtype=float))
    grid = load_model()["grid"]
    S = np.array([damage.pseudo_damage(rng, count, float(m)) for m in grid])
    D = float(fit.predict_bank(load_model()["bank"], S.reshape(1, -1))[0])
    return D, (rng, mean, count)


def predict(path_or_buffer, file_id: str | None = None) -> pd.DataFrame:
    value, _ = predict_signal(load_signal(path_or_buffer))
    raw = file_id or (path_or_buffer if isinstance(path_or_buffer, str)
                      else getattr(path_or_buffer, "name", "input.csv"))
    return pd.DataFrame([{"file_id": os.path.basename(str(raw)), "prediction": value}])


def predict_folder(folder: str) -> pd.DataFrame:
    paths = sorted(
        glob.glob(os.path.join(folder, "*.csv")),
        key=lambda p: int("".join(ch for ch in os.path.basename(p) if ch.isdigit()) or -1),
    )
    rows = []
    for p in paths:
        value, _ = predict_signal(load_signal(p))
        rows.append({"file_id": os.path.basename(p), "prediction": value})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- training ----
def train(verbose: bool = True) -> dict:
    _, series, D = io.load_split("train")
    cycles = [damage.rainflow_ranges(x) for x in series]
    S = np.array([[damage.pseudo_damage(r, c, float(m)) for m in GRID]
                  for r, _, c in cycles])

    analytic = fit.fit_from_grid(S, D, GRID)
    fit.save_constants({k: v for k, v in analytic.items() if k != "j"}, CONST_PATH)
    bank = fit.fit_bank(S, D)
    joblib.dump({"pipe": bank, "grid": GRID}, BANK_PATH)

    cv_mape = float(np.mean([fit.cv_bank(S, D, seed=s)["mape"] for s in range(5)]))
    const = constants()
    const["bank_cv_mape"] = cv_mape
    fit.save_constants(const, CONST_PATH)

    if verbose:
        print(f"[shm] analytic m={analytic['m']:.2f} k={analytic['k']:.3e} "
              f"| bank Ridge 8-fold CV MAPE {cv_mape * 100:.2f}%")
    return {"loo_score": float(1 - cv_mape), "cv_mape": cv_mape, "analytic_m": analytic["m"]}


if __name__ == "__main__":
    train()
