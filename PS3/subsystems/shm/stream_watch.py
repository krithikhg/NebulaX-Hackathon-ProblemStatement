"""Automated folder watcher for SHM stress segments.

Simulates the vehicle's periodic file drops: every new ``*.csv`` that appears in
``--input`` is rainflow-counted, its Miner damage added to a persistent
per-stream cumulative state, and appended to a predictions CSV. A live line
reports cumulative damage and remaining life, and threshold crossings raise an
alert. This is the "automated + incremental" capability in one command.

Usage (from PS3/subsystems):
    ../../.venv/bin/python -m shm.stream_watch --input ../02_Datasets/SHM/Test \
        --stream "Train 01 / bogie frame"

    # one pass over the files already present, then exit:
    ... --once

    # per-train subfolders (input/<stream>/*.csv):
    ... --input <folder of per-train subfolders>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import joblib
import numpy as np

from shm import damage, io

HERE = os.path.dirname(os.path.abspath(__file__))
BANK_PATH = os.path.join(HERE, "checkpoints", "bank_model.joblib")
DEFAULT_STATE = os.path.join(HERE, "state", "cumulative_state.json")
DEFAULT_CSV = os.path.join(HERE, "predictions", "shm_stream_predictions.csv")

# (threshold, label) — highest first.
BANDS = [(0.8, "ACT - schedule NDT inspection"), (0.5, "PLAN - add to next inspection"),
         (0.25, "MONITOR - keep watching")]


def load_model():
    bundle = joblib.load(BANK_PATH)
    return bundle["pipe"], np.asarray(bundle["grid"])


def segment_damage(path: str, pipe, grid) -> tuple[float, int]:
    x = io.load_series(path)
    rng, _, cnt = damage.rainflow_ranges(x)
    S = np.array([[damage.pseudo_damage(rng, cnt, float(m)) for m in grid]])
    d = float(np.exp(pipe.predict(np.log(S))[0]))
    return d, len(x)


def load_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {"streams": {}}


def save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, path)


def update_state(state: dict, stream: str, d: float, n_samples: int, file_id: str) -> dict:
    s = state["streams"].setdefault(stream, {"D": 0.0, "n_segments": 0, "n_samples": 0, "history": []})
    prev = s["D"]
    s["D"] = prev + d
    s["n_segments"] += 1
    s["n_samples"] += n_samples
    s["history"].append({"n": s["n_segments"], "file": file_id, "d": d, "D": s["D"]})
    s["history"] = s["history"][-1000:]
    return s


def remaining(s: dict) -> float:
    per = s["D"] / s["n_segments"] if s["n_segments"] else 0.0
    return (1 - s["D"]) / per if per > 0 else float("inf")


def alert(prev: float, now: float) -> str | None:
    for thr, msg in BANDS:
        if prev < thr <= now:
            return msg
    return None


def _iter_new(root: str, seen: set[str]):
    """Yield (stream, path) for every CSV not yet seen.

    If ``root`` has subdirectories, each is treated as a stream (per-train
    folders); otherwise the ``--stream`` label is used for the flat folder.
    """
    subdirs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    if subdirs:
        for d in sorted(subdirs):
            for p in sorted(glob.glob(os.path.join(d, "*.csv"))):
                if p not in seen:
                    yield os.path.basename(d), p
    else:
        for p in sorted(glob.glob(os.path.join(root, "*.csv"))):
            if p not in seen:
                yield None, p


def main():
    ap = argparse.ArgumentParser(description="Watch a folder and accumulate SHM damage.")
    ap.add_argument("--input", required=True, help="folder of segments (or per-stream subfolders)")
    ap.add_argument("--stream", default="Default stream", help="stream label for a flat folder")
    ap.add_argument("--state", default=DEFAULT_STATE, help="persistent state JSON")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="append per-segment predictions here")
    ap.add_argument("--interval", type=float, default=1.0, help="poll interval seconds")
    ap.add_argument("--once", action="store_true", help="process current files and exit")
    args = ap.parse_args()

    pipe, grid = load_model()
    state = load_state(args.state)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    if not os.path.exists(args.csv):
        with open(args.csv, "w") as fh:
            fh.write("file_id,prediction,stream,D_cumulative,n_segments\n")

    seen: set[str] = set()
    print(f"watching {args.input}  (state -> {args.state})")
    try:
        while True:
            found = list(_iter_new(args.input, seen))
            for stream_default, path in found:
                seen.add(path)
                stream = stream_default or args.stream
                d, n = segment_damage(path, pipe, grid)
                before = state["streams"].get(stream, {}).get("D", 0.0)
                s = update_state(state, stream, d, n, os.path.basename(path))
                save_state(args.state, state)
                with open(args.csv, "a") as fh:
                    fh.write(f"{os.path.basename(path)},{d},{stream},{s['D']},{s['n_segments']}\n")
                rem = remaining(s)
                print(f"[{stream}] {os.path.basename(path)}: d={d:.4f}  D_cum={s['D']:.4f}  "
                      f"segments={s['n_segments']}  to D=1: {rem:.1f} segments")
                msg = alert(before, s["D"])
                if msg:
                    print(f"    ALERT [{stream}] D crossed {[t for t, _ in BANDS if before < t <= s['D']][0]:.2f}: {msg}")
            if args.once:
                break
            time.sleep(max(0.1, args.interval))
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
