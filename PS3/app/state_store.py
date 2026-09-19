"""Persistent cumulative-damage state for the SHM subsystem.

Each monitored stream (a train / measurement point, free-text label) keeps a
tiny **Markov summary** of its fatigue history:

    D            cumulative Miner damage (sum of the per-segment damages)
    n_segments   segments ingested so far
    n_samples    stress samples covered
    rate         D per sample
    history      [{n, file, d, D}, ...] for the cumulative timeline

Because Miner's rule is additive, the next state depends only on the current
state and the incoming segment -- no other history is needed.

Backends, in priority order:

  1. **Google Cloud Storage** -- when ``PS3_STATE_BUCKET`` (or ``STATE_BUCKET``)
     is set. One JSON object per stream under ``shm-state/``. This survives
     restarts and is shared across Cloud Run instances.
  2. **Local JSON directory** (``PS3/app/state/``) -- for local development.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import threading

from common import HERE

BUCKET = os.environ.get("PS3_STATE_BUCKET") or os.environ.get("STATE_BUCKET")
LOCAL_DIR = os.path.join(HERE, "state")
PREFIX = "shm-state/"
MAX_HISTORY = 1000
_LOCK = threading.RLock()


def _safe(stream: str) -> str:
    """Filesystem/object-safe key for a free-text stream label."""
    key = re.sub(r"[^A-Za-z0-9._-]+", "_", str(stream).strip()).strip("_")
    return (key or "default")[:180]


def _bucket():
    if not BUCKET:
        return None
    try:
        from google.cloud import storage  # type: ignore
    except Exception:
        return None
    try:
        return storage.Client().bucket(BUCKET)
    except Exception:
        return None


def _path(stream: str) -> str:
    os.makedirs(LOCAL_DIR, exist_ok=True)
    return os.path.join(LOCAL_DIR, f"{_safe(stream)}.json")


def _read(stream: str) -> dict | None:
    b = _bucket()
    if b is not None:
        blob = b.blob(PREFIX + _safe(stream) + ".json")
        if not blob.exists():
            return None
        try:
            return json.loads(blob.download_as_text())
        except Exception:
            return None
    path = _path(stream)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def _write(stream: str, state: dict) -> None:
    b = _bucket()
    if b is not None:
        blob = b.blob(PREFIX + _safe(stream) + ".json")
        blob.upload_from_string(json.dumps(state), content_type="application/json")
        return
    path = _path(stream)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, path)


def _delete(stream: str) -> None:
    b = _bucket()
    if b is not None:
        blob = b.blob(PREFIX + _safe(stream) + ".json")
        if blob.exists():
            blob.delete()
        return
    path = _path(stream)
    if os.path.exists(path):
        os.remove(path)


def blank(stream: str) -> dict:
    return {"stream": stream, "D": 0.0, "n_segments": 0, "n_samples": 0,
            "rate": 0.0, "updated_at": None, "history": []}


def load(stream: str) -> dict:
    with _LOCK:
        state = _read(stream)
    return state if state else blank(stream)


def update(stream: str, d_seg: float, n_samples: int, file_id: str) -> dict:
    """Add one segment's damage to the stream and persist the new state."""
    with _LOCK:
        state = _read(stream) or blank(stream)
        state["D"] = float(state.get("D", 0.0)) + float(d_seg)
        state["n_segments"] = int(state.get("n_segments", 0)) + 1
        state["n_samples"] = int(state.get("n_samples", 0)) + int(n_samples)
        state["rate"] = state["D"] / state["n_samples"] if state["n_samples"] else 0.0
        state["updated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        state.setdefault("history", []).append(
            {"n": state["n_segments"], "file": file_id, "d": float(d_seg), "D": state["D"]})
        state["history"] = state["history"][-MAX_HISTORY:]
        _write(stream, state)
        return state


def reset(stream: str | None = None) -> int:
    """Clear one stream, or every stream when ``stream`` is None/falsey."""
    with _LOCK:
        if stream:
            _delete(stream)
            return 1
        n = 0
        for state in list_states():
            _delete(state.get("stream", ""))
            n += 1
        return n


def list_states() -> list[dict]:
    with _LOCK:
        b = _bucket()
        if b is not None:
            out = []
            for blob in b.list_blobs(prefix=PREFIX):
                try:
                    out.append(json.loads(blob.download_as_text()))
                except Exception:
                    continue
            return sorted(out, key=lambda s: str(s.get("stream", "")))
        if not os.path.isdir(LOCAL_DIR):
            return []
        out = []
        for name in sorted(os.listdir(LOCAL_DIR)):
            if name.endswith(".json"):
                try:
                    with open(os.path.join(LOCAL_DIR, name)) as fh:
                        out.append(json.load(fh))
                except Exception:
                    continue
        return sorted(out, key=lambda s: str(s.get("stream", "")))
