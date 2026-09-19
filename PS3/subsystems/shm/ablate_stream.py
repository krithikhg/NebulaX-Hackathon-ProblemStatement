"""Ablation: online (streaming) rainflow vs the shipped batch counter.

Run:  cd PS3/subsystems && ../../.venv/bin/python -m shm.ablate_stream

Reports, per file, the relative difference between the streaming damage and the
batch ``fatpack`` damage:
  * online with each file's own min/max  -> isolates the cycle-extraction logic;
  * online with a single fixed grid      -> adds the effect of fixed bins.
The shipped submission keeps the batch counter, so this is a capability check.
"""
from __future__ import annotations

import os

import fatpack
import joblib
import numpy as np

from shm import damage, fit, io, streaming

HERE = os.path.dirname(os.path.abspath(__file__))
BANK_PATH = os.path.join(HERE, "checkpoints", "bank_model.joblib")


def batch_damage(pipe, grid, x):
    rng, _, cnt = damage.rainflow_ranges(x)
    S = np.array([[damage.pseudo_damage(rng, cnt, float(m)) for m in grid]])
    return float(np.exp(pipe.predict(np.log(S))[0]))


def rel(a, b):
    return abs(a - b) / b if b else float("nan")


def main():
    bundle = joblib.load(BANK_PATH)
    pipe, grid = bundle["pipe"], np.asarray(bundle["grid"])

    ids_tr, ser_tr, D = io.load_split("train")
    ids_te, ser_te, _ = io.load_split("test")
    lo_g = min(float(x.min()) for x in ser_tr)
    hi_g = max(float(x.max()) for x in ser_tr)
    print(f"fixed global grid: [{lo_g:.2f}, {hi_g:.2f}]  (k={streaming.K})")

    for tag, ids, series in [("train", ids_tr, ser_tr), ("test", ids_te, ser_te)]:
        e_self, e_fix, e_s5 = [], [], []
        for x in series:
            d_batch = batch_damage(pipe, grid, x)
            d_self = streaming.online_damage(x, float(x.min()), float(x.max()), pipe, grid)
            d_fix = streaming.online_damage(x, lo_g, hi_g, pipe, grid)
            e_self.append(rel(d_self, d_batch))
            e_fix.append(rel(d_fix, d_batch))
            # direct check of the counter itself (ignores the fitted model)
            s_batch = float(np.sum(fatpack.find_rainflow_ranges(x, k=64) ** 5))
            rf = streaming.OnlineRainflow(float(x.min()), float(x.max()), 64)
            rf.extend(x)
            rf.finish()
            e_s5.append(rel(rf.pseudo_damage(5.0), s_batch))
        e_self, e_fix, e_s5 = np.array(e_self), np.array(e_fix), np.array(e_s5)
        print(f"\n[{tag}] n={len(series)}")
        print(f"  counter S(5), online vs batch (per-file bins): mean={e_s5.mean()*100:7.4f}%  max={e_s5.max()*100:7.4f}%")
        print(f"  online (per-file bins) vs batch: mean={e_self.mean()*100:6.2f}%  max={e_self.max()*100:6.2f}%")
        print(f"  online (fixed bins)    vs batch: mean={e_fix.mean()*100:6.2f}%  max={e_fix.max()*100:6.2f}%")

    # what the fixed-bin stream would score on the training labels (in-sample ref)
    pred = np.array([streaming.online_damage(x, lo_g, hi_g, pipe, grid) for x in ser_tr])
    print(f"\n[reference] fixed-bin stream, in-sample train MAPE = {fit.mape(D, pred)*100:.2f}%")
    print(f"[reference] batch counter,    in-sample train MAPE = "
          f"{fit.mape(D, np.array([batch_damage(pipe, grid, x) for x in ser_tr]))*100:.2f}%")


if __name__ == "__main__":
    main()
