"""Cloud inference server + static host for the TrainWhisper web app.

One process serves the frontend and runs the *same* Python pipelines that write
the submission CSVs (door.py / acv.py / rail.py / shm.py). The browser only
uploads files and renders the JSON this returns, so there is no second
implementation to keep in sync.

    uvicorn server:app --host 0.0.0.0 --port 8080

Endpoints
    GET  /                     the web app (public/)
    POST /api/predict/{kind}   kind in door | acv | rail | shm, one file per request
"""
from __future__ import annotations

import datetime
import os
import re
import tempfile

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

import acv
import door
import rail
import shm
from common import HERE

PUBLIC = os.path.join(HERE, "public")
app = FastAPI(title="TrainWhisper", docs_url=None, redoc_url=None)


# --------------------------------------------------------------- helpers ----
def _num(x):
    """JSON-safe float: numpy scalars to float, NaN/inf to None (views treat null as non-finite)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _save(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[1]
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as fh:
        while True:
            chunk = upload.file.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    return path


def _decimate(xs, ys, buckets):
    """Min/max per bucket (same as the old browser worker) so peaks survive."""
    n = len(xs)
    if n <= buckets * 2:
        return list(zip(xs, ys))
    size = n / buckets
    out = []
    for b in range(buckets):
        a = int(b * size)
        e = min(n, int((b + 1) * size))
        lo = hi = a
        for i in range(a, e):
            if ys[i] < ys[lo]:
                lo = i
            if ys[i] > ys[hi]:
                hi = i
        for i in ([lo, hi] if lo < hi else [hi, lo]):
            out.append((xs[i], ys[i]))
    return out


def _histogram(cyc: np.ndarray, bins: int = 40):
    if len(cyc) == 0:
        return []
    amps = cyc[:, 0]
    lo, hi = float(amps.min()), float(amps.max())
    width = (hi - lo) / bins or 1.0
    counts = [0.0] * bins
    for amp, cnt in cyc:
        k = min(bins - 1, int((amp - lo) / width))
        counts[k] += float(cnt)
    return [{"lo": lo + k * width, "hi": lo + (k + 1) * width, "count": counts[k]} for k in range(bins)]


def _excel_serial(v):
    if isinstance(v, (pd.Timestamp, datetime.datetime, datetime.date)):
        return (pd.Timestamp(v) - pd.Timestamp("1899-12-30")).total_seconds() / 86400.0
    return _num(v)


# ------------------------------------------------------------- payloads ----
def door_payload(path: str, name: str) -> dict:
    idx, d = door.predict_with_traces(path)
    # pandas 3 may parse at us resolution, so convert to ms explicitly rather than assume ns.
    t_ms = d["t"].astype("datetime64[ms]").astype("int64").to_numpy()
    current = d[door.SIGNALS["c"]].to_numpy(dtype=float)
    closing = d["Door is closing"].to_numpy(dtype=float)

    result, cycles = [], []
    for (_, g), row in zip(d.groupby("seg", sort=True), idx.itertuples(index=False)):
        pos = g.index.to_numpy()
        a, b = int(pos[0]), int(pos[-1]) + 1
        pred = row.prediction
        result.append({
            "start_time": str(row.start_time),
            "end_time": str(row.end_time),
            "prediction": pred,
            "confidence": _num(row.confidence),
            "mean_current_mA": _num(row.mean_current_mA),
            "peak_current_mA": _num(row.peak_current_mA),
        })
        seg_idx = list(range(a, b))
        decimated = _decimate(seg_idx, [current[i] for i in seg_idx], 150)
        cycles.append({
            "points": [[int(i), _num(current[i]), int(t_ms[i])] for i, _ in decimated],
            "abnormal": pred == door.ABNORMAL,
            "operation": "Closing" if closing[a:b].max() >= 1 else "Opening",
            "t0": int(t_ms[a]),
            "duration_s": float(t_ms[b - 1] - t_ms[a]) / 1000.0,
        })
    return {"name": name, "result": result, "cycles": cycles}


_CAR_COL_RE = re.compile(r"^Car \d")


def acv_payload(path: str, name: str) -> dict:
    df = acv.load(path)
    ranked = acv.rank_cars(df)
    rows = []
    for r in ranked.itertuples(index=False):
        item = {
            "car": str(r.car),
            "score": _num(r.score),
            "cabin_temp_dev_C": _num(r.cabin_temp_dev_C),
        }
        if hasattr(r, "low_pressure_dev"):
            item["low_pressure_dev"] = _num(r.low_pressure_dev)
        rows.append(item)

    def context(keyword: str):
        return next(
            (h for h in df.columns if not _CAR_COL_RE.match(str(h)) and re.search(keyword, str(h), re.I)),
            None,
        )

    time_col, train_col = context("time"), context("train")
    window = None
    if time_col is not None:
        values = df[time_col].dropna()
        if len(values):
            window = [_excel_serial(values.iloc[0]), _excel_serial(values.iloc[-1])]
    train = None
    if train_col is not None:
        values = df[train_col].dropna()
        if len(values):
            first = values.iloc[0]
            train = _num(first) if isinstance(first, (int, float, np.number)) else str(first)

    return {"name": name, "ranked": rows, "samples": int(len(df)), "train": train, "window": window}


def rail_payload(path: str, name: str) -> dict:
    f = rail.extract_features(path)
    f["filename"] = name
    pred = rail.predict_features(pd.DataFrame([f])).iloc[0]
    bundle = joblib.load(rail.MODEL_PATH)
    stationary = bool(f["_speed"] < bundle["stationary_speed"])
    n = len(rail.BANDS)
    return {
        "result": [{
            "file_id": name,
            "prediction": str(pred["prediction"]),
            "confidence": _num(pred["confidence"]),
            "speed_m_s": _num(pred["speed_m_s"]),
        }],
        "bands": [{
            "file_id": name,
            "speed": _num(f["_speed"]),
            "stationary": stationary,
            "side1": [_num(f[f"s1vib_b{i}_mean"]) for i in range(n)],
            "side2": [_num(f[f"s2vib_b{i}_mean"]) for i in range(n)],
        }],
        "bandLabels": [f"{lo}\u2013{hi}" for lo, hi in rail.BANDS],
    }


def shm_payload(path: str, name: str) -> dict:
    signal = shm.load_signal(path)
    damage, cyc = shm.predict_signal(signal)
    fit = shm._fit()
    cycles = int(cyc[:, 1].sum()) if len(cyc) else 0
    peak = float(cyc[:, 0].max()) if len(cyc) else 0.0
    return {
        "result": [{"file_id": name, "prediction": _num(damage)}],
        "detail": [{
            "file_id": name,
            "samples": int(len(signal)),
            "cycles": cycles,
            "peak_amplitude": _num(peak),
            "histogram": _histogram(cyc, 40),
        }],
        "fit": {
            "m": fit["m"],
            "C": fit["C"],
            "amplitude_cutoff": fit["amplitude_cutoff"],
            "loo_mape": fit["loo_mape"],
            "loo_score": fit.get("loo_score"),
        },
    }


_PAYLOADS = {
    "door": door_payload,
    "acv": acv_payload,
    "rail": rail_payload,
    "shm": shm_payload,
}


# -------------------------------------------------------------- routes ----
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict/{kind}")
async def predict(kind: str, file: UploadFile = File(...)):
    builder = _PAYLOADS.get(kind)
    if builder is None:
        raise HTTPException(status_code=404, detail=f"Unknown subsystem: {kind}")
    name = os.path.basename(file.filename or "input")
    path = _save(file)
    try:
        return builder(path, name)
    except ValueError as exc:            # user-facing input problems
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:             # noqa: BLE001 - reported to the client
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        os.unlink(path)


app.mount("/", StaticFiles(directory=PUBLIC, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
