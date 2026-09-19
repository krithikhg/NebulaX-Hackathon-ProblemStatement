"""Train / load the final Rail model and produce ``rail_predictions.csv``.

Two modes:

  * default (no args): train on the provided Train set, select features, save the
    model, and run the provided Test inputs through it.
  * inference: ``--input <dir-or-csv> --output <csv>`` loads the saved model and
    predicts any new recording(s) — this is the entry point the app should call.

CLI (matches the interface referenced by the Info Kit):
    python -m rail.predict --input <dir|file.csv> --output <out.csv>
    python -m rail.predict --retrain

Output schema (Info Kit Section 3): columns ``file_id`` (source filename incl.
extension) and ``prediction`` in {Normal, Side I, Side II}, one row per file.
"""
from __future__ import annotations

import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd

from rail import evaluate, features
from rail.models import LGBM3

PRED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions")
MODEL_PATH = os.path.join(PRED_DIR, "rail_model.joblib")
DEFAULT_OUT = os.path.join(PRED_DIR, "rail_predictions.csv")
FINAL_PARAMS = dict()          # regularised defaults (chosen by 10-seed means)
SEEDS = tuple(range(10))
LABELS = ["Normal", "Side I", "Side II"]


# --------------------------------------------------------------------------- #
def train_final(verbose=True):
    """Train the flat 3-class model on all Train data; save + return it."""
    tr = features.build("train")
    cols = evaluate.feature_columns(tr)
    X, y = tr[cols].to_numpy(), tr["y"].to_numpy()
    idx = evaluate.select_top_idx(X, y, evaluate.DEFAULT_K, seed=0)
    sel = [cols[i] for i in idx]
    models = [LGBM3(random_state=s, **FINAL_PARAMS).fit(X[:, idx], y) for s in SEEDS]
    os.makedirs(PRED_DIR, exist_ok=True)
    joblib.dump({"models": models, "selected": sel}, MODEL_PATH)
    if verbose:
        print(f"trained {len(models)} seeds on {len(sel)} selected features -> {MODEL_PATH}")
    return {"models": models, "selected": sel}


def load_model():
    if not os.path.exists(MODEL_PATH):
        return train_final()
    return joblib.load(MODEL_PATH)


def _resolve_paths(input_arg: str):
    if os.path.isdir(input_arg):
        return sorted(glob.glob(os.path.join(input_arg, "*.csv")),
                      key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or -1))
    return [input_arg]


def predict(input_arg: str, output_csv: str, bundle=None):
    """Run the saved model on files/dir and write the prediction CSV."""
    bundle = bundle or load_model()
    sel, models = bundle["selected"], bundle["models"]
    paths = _resolve_paths(input_arg)
    rows, X = [], []
    for p in paths:
        f = features.extract(p)
        X.append([f.get(c, np.nan) for c in sel])
        rows.append(os.path.basename(p))
    X = np.asarray(X, dtype=float)
    P = np.mean([m.predict_proba(X) for m in models], axis=0)
    labels = [LABELS[i] for i in P.argmax(1)]
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    pd.DataFrame({"file_id": rows, "prediction": labels}).to_csv(output_csv, index=False)
    print(f"wrote {len(rows)} predictions -> {output_csv}")
    print("distribution:", pd.Series(labels).value_counts().to_dict())
    return output_csv


def main():
    ap = argparse.ArgumentParser(description="Rail corrugation inference.")
    ap.add_argument("--input", help="directory of CSVs or a single CSV")
    ap.add_argument("--output", default=DEFAULT_OUT, help="output CSV path")
    ap.add_argument("--retrain", action="store_true", help="force retraining the model")
    args = ap.parse_args()

    bundle = train_final() if args.retrain else None
    if args.input:
        predict(args.input, args.output, bundle)
    else:
        # default: score the provided Test set
        bundle = bundle or load_model()
        # honest OOF report when training data is available
        try:
            tr = features.build("train")
            cols = evaluate.feature_columns(tr)
            oof = evaluate.oof_select(tr[cols].to_numpy(), tr["y"].to_numpy(),
                                      K=evaluate.DEFAULT_K, seeds=range(5))
            rep = evaluate.report(tr["y"].to_numpy(), oof,
                                  tr["speed_kmh"].to_numpy(),
                                  "Final top-40 selected (5-seed OOF)")
            evaluate.print_report(rep)
        except Exception as exc:  # pragma: no cover
            print("(skipped OOF report:", exc, ")")
        from rail import io
        predict(os.path.join(io.DATA_ROOT, "Test"), DEFAULT_OUT, bundle)


if __name__ == "__main__":
    main()
