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
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles

import acv
import door
import rail
import shm
import state_store
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
    result, cycles = door.predict_with_traces(path)
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


def shm_payload(path: str, name: str, stream: str | None = None) -> dict:
    signal = shm.load_signal(path)
    value, (ranges, _, count) = shm.predict_signal(signal)
    const = shm.constants()
    amplitude = ranges / 2.0
    cycles = np.column_stack([amplitude, count]) if ranges.size else np.zeros((0, 2))
    payload = {
        "result": [{"file_id": name, "prediction": _num(value)}],
        "detail": [{
            "file_id": name,
            "samples": int(len(signal)),
            "cycles": int(count.sum()) if ranges.size else 0,
            "peak_amplitude": _num(amplitude.max()) if amplitude.size else 0.0,
            "histogram": _histogram(cycles, 40),
        }],
        "fit": {
            "m": const["m"],
            "C": 1.0 / const["k"],
            "amplitude_cutoff": 0.0,
            "loo_mape": const.get("bank_cv_mape", const.get("mape")),
        },
    }
    # Cumulative, per-stream Miner damage: add this segment to the running state.
    if stream:
        payload["state"] = state_store.update(stream, value, len(signal), name)
    return payload


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
async def predict(kind: str, file: UploadFile = File(...),
                  stream: str | None = Form(None)):
    builder = _PAYLOADS.get(kind)
    if builder is None:
        raise HTTPException(status_code=404, detail=f"Unknown subsystem: {kind}")
    name = os.path.basename(file.filename or "input")
    path = _save(file)
    try:
        if kind == "shm":
            return builder(path, name, stream)
        return builder(path, name)
    except ValueError as exc:            # user-facing input problems
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:             # noqa: BLE001 - reported to the client
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        os.unlink(path)


# ------------------------------------------------- cumulative SHM state ----
@app.get("/api/shm/state")
def shm_states():
    """Every stream's cumulative state (for the fleet view)."""
    return {"streams": state_store.list_states()}


@app.get("/api/shm/state/one")
def shm_state_one(stream: str = Query(...)):
    return state_store.load(stream)


@app.post("/api/shm/state/reset")
def shm_state_reset(stream: str | None = Form(None)):
    """Reset one stream (form field ``stream``) or all of them (omitted)."""
    return {"reset": state_store.reset(stream)}


app.mount("/", StaticFiles(directory=PUBLIC, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
