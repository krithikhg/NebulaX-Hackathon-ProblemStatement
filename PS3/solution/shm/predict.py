"""Fit the SHM damage model and write the test predictions.

Two physically-equivalent models are fitted on the training labels:

  * **analytic**  ``D = k * sum(n * range^m)`` with m, k global constants
                  (interpretable baseline / the recovered Miner's-rule formula);
  * **bank Ridge** ``log D`` regressed on ``log S(m)`` over an exponent bank,
                  i.e. a learned S-N curve.  This is the shipped model because
                  it is robustly more accurate (see README).

CLI (matches the interface referenced by the Info Kit):
    python -m shm.predict --input <dir|file.csv> --output <out.csv>
    python -m shm.predict --retrain

Output schema: columns ``file_id`` (source filename incl. extension) and
``prediction`` (a single cumulative-damage number), one row per file.
"""
from __future__ import annotations

import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd

from shm import cache, damage, fit, io

HERE = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(HERE, "predictions")
CKPT_DIR = os.path.join(HERE, "checkpoints")
MODEL_PATH = os.path.join(CKPT_DIR, "constants.json")       # analytic
BANK_PATH = os.path.join(CKPT_DIR, "bank_model.joblib")     # physics-informed
DEFAULT_OUT = os.path.join(PRED_DIR, "shm_predictions.csv")


def _sgrid_from_ranges(cycles, grid):
    return np.array([[damage.pseudo_damage(r, c, float(m)) for m in grid]
                     for r, _, c in cycles])


def train_final(verbose: bool = True) -> dict:
    os.makedirs(CKPT_DIR, exist_ok=True)
    _, _, D = io.load_split("train")
    S = cache.damage_grid("train")
    analytic = fit.fit_from_grid(S, D, cache.GRID)
    fit.save_constants({k: v for k, v in analytic.items() if k != "j"}, MODEL_PATH)
    bank = fit.fit_bank(S, D)
    joblib.dump({"pipe": bank, "grid": cache.GRID}, BANK_PATH)
    if verbose:
        print(f"fitted analytic m={analytic['m']:.3f} k={analytic['k']:.4e} "
              f"(train MAPE {analytic['mape'] * 100:.2f}%)")
        print(f"fitted bank Ridge -> {BANK_PATH}")
    return {"analytic": analytic, "bank": bank}


def load_model() -> dict:
    if not (os.path.exists(MODEL_PATH) and os.path.exists(BANK_PATH)):
        return train_final()
    d = joblib.load(BANK_PATH)
    return {"analytic": fit.load_constants(MODEL_PATH), "bank": d["pipe"], "grid": d["grid"]}


def _resolve_paths(input_arg: str):
    if os.path.isdir(input_arg):
        return sorted(glob.glob(os.path.join(input_arg, "*.csv")),
                      key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or -1))
    return [input_arg]


def predict(input_arg: str, output_csv: str, model: dict | None = None):
    model = model or load_model()
    grid = model.get("grid", cache.GRID)
    paths = _resolve_paths(input_arg)
    rows, sgrid = [], []
    for p in paths:
        rng, _, cnt = damage.rainflow_ranges(io.load_series(p))
        rows.append(os.path.basename(p))
        sgrid.append([damage.pseudo_damage(rng, cnt, float(m)) for m in grid])
    preds = fit.predict_bank(model["bank"], np.asarray(sgrid, dtype=float))
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    pd.DataFrame({"file_id": rows, "prediction": preds}).to_csv(output_csv, index=False)
    print(f"wrote {len(rows)} predictions -> {output_csv}")
    return output_csv


def main():
    ap = argparse.ArgumentParser(description="SHM fatigue-damage inference.")
    ap.add_argument("--input", help="directory of CSVs or a single CSV")
    ap.add_argument("--output", default=DEFAULT_OUT, help="output CSV path")
    ap.add_argument("--retrain", action="store_true", help="refit the models")
    args = ap.parse_args()

    model = train_final() if args.retrain else load_model()
    if args.input:
        predict(args.input, args.output, model)
        return

    _, _, D = io.load_split("train")
    S = cache.damage_grid("train")
    a_cv = np.mean([fit.cv_from_grid(S, D, cache.GRID, seed=s)["mape"] for s in range(5)])
    b_cv = np.mean([fit.cv_bank(S, D, seed=s)["mape"] for s in range(5)])
    med = np.full_like(D, np.median(D))
    an = model["analytic"]
    print("\n=== Fitted models ===")
    print(f"  analytic : m={an['m']:.3f}  k={an['k']:.4e}")
    print("\n=== Honest evaluation (8-fold CV, 5 seeds) ===")
    print(f"  analytic physics : MAPE = {a_cv * 100:.3f}%   score = {1 - a_cv:.4f}")
    print(f"  bank Ridge       : MAPE = {b_cv * 100:.3f}%   score = {1 - b_cv:.4f}  <- shipped")
    print(f"  constant median  : MAPE = {fit.mape(D, med) * 100:.2f}%   score = {fit.score(D, med):.4f}")

    predict(os.path.join(io.DATA_ROOT, "Test"), DEFAULT_OUT, model)
    print(pd.read_csv(DEFAULT_OUT).head(8).to_string(index=False))


if __name__ == "__main__":
    main()
