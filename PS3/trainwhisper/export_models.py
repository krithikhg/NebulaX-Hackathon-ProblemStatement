"""Export the fitted models to JSON for the browser app (public/models/).

    python export_models.py

Run after train_all.py. The web app evaluates the random forests itself, so it
needs no Python at request time and no scikit-learn version pinning: each tree
is written out as flat arrays (feature, threshold, left, right, leaf value).

Leaf values are stored already normalised to class probabilities, which is
what sklearn's predict_proba averages across trees. Features are compared as
float32 at inference, matching sklearn's own input casting.
"""
from __future__ import annotations

import json
import os
import shutil

import joblib
import numpy as np

import door
import rail
import shm
from common import HERE

OUT_DIR = os.path.join(HERE, "public", "models")


def _tree(est, class_order: list[int]) -> dict:
    t = est.tree_
    leaf = t.children_left == -1
    values = t.value[:, 0, :]
    proba = values / np.maximum(values.sum(axis=1, keepdims=True), 1e-300)
    proba = proba[:, class_order]
    return {
        "f": np.where(leaf, -1, t.feature).astype(int).tolist(),
        "t": np.where(leaf, 0.0, t.threshold).tolist(),
        "l": t.children_left.astype(int).tolist(),
        "r": t.children_right.astype(int).tolist(),
        # one probability row per node; zeros on internal nodes keep indices aligned
        "v": np.where(leaf[:, None], proba, 0.0).tolist(),
    }


def _forest(model, classes: list) -> list[dict]:
    order = [list(model.classes_).index(c) for c in classes]
    return [_tree(est, order) for est in model.estimators_]


def _write(name: str, payload: dict) -> None:
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB)")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    b = joblib.load(door.MODEL_PATH)
    _write("door_rf.json", {
        "features": b["features"],
        "classes": [0, 1],  # 1 = abnormal resistance
        "trees": _forest(b["model"], [0, 1]),
    })

    b = joblib.load(rail.MODEL_PATH)
    _write("rail_rf.json", {
        "features": b["features"],
        "classes": b["classes"],
        "priors": [float(p) for p in b["priors"]],
        "stationary_speed": float(b["stationary_speed"]),
        "trees": _forest(b["model"], b["classes"]),
    })

    shutil.copy(shm.MODEL_PATH, os.path.join(OUT_DIR, "shm_fit.json"))
    print(f"copied {shm.MODEL_PATH}")


if __name__ == "__main__":
    main()
