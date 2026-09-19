"""Train/fit every subsystem and print the validation summary.

    python train_all.py

Door and SHM take seconds. Rail extracts features from 272 files (~50 s the
first time, then cached in models/rail_train_features.csv). ACV needs no
training - it is validated, not fitted.
"""
from __future__ import annotations

import time

import acv
import door
import rail
import shm


def main() -> None:
    t0 = time.time()
    results = {}

    print("=== Door ===")
    results["door"] = door.train()

    print("\n=== SHM ===")
    results["shm"] = shm.train()

    print("\n=== ACV ===")
    results["acv"] = {"rank_decay": acv.validate()["rank_decay"]}

    print("\n=== Rail corrugation ===")
    results["rail"] = rail.train()

    print(f"\nAll models ready in {time.time() - t0:.0f}s")
    print("\nValidation summary (out-of-sample):")
    print(f"  Door  macro F1 (chronological holdout) : {results['door']['holdout_macro_f1']:.3f}")
    print(f"  ACV   rank-decay (6 labelled cases)    : {results['acv']['rank_decay']:.3f}")
    print(f"  Rail  macro F1 (5-fold, tuned priors)  : {results['rail']['oof_macro_f1']:.3f}")
    print(f"  SHM   1 - MAPE (leave-one-out)         : {results['shm']['loo_score']:.3f}")


if __name__ == "__main__":
    main()
