"""Local re-implementation of the organisers' four scoring metrics.

Lets you score a prediction file against labels you hold yourself (e.g. a
held-out slice of Train.csv) before submitting anything.

    from evaluate import door_iou_f1, acv_rank_decay, rail_macro_f1, shm_score
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import parse_door_time


def _to_epoch(x) -> float:
    try:
        return parse_door_time(x).value / 1e9
    except Exception:  # noqa: BLE001 - fall back to ISO parsing
        return pd.Timestamp(x).value / 1e9


def door_iou_f1(pred: pd.DataFrame, truth: pd.DataFrame,
                truth_label_col: str = "status") -> float:
    """IoU-weighted F1: greedy one-to-one matching on same-label pairs."""
    P = [(_to_epoch(r.start_time), _to_epoch(r.end_time), r.prediction)
         for r in pred.itertuples()]
    T = [(_to_epoch(r.start_time), _to_epoch(r.end_time), getattr(r, truth_label_col))
         for r in truth.itertuples()]
    if not P or not T:
        return 0.0

    candidates = []
    for i, (ps, pe, pl) in enumerate(P):
        for j, (ts, te, tl) in enumerate(T):
            if pl != tl:
                continue
            inter = max(0.0, min(te, pe) - max(ts, ps))
            union = (te - ts) + (pe - ps) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > 0:
                candidates.append((iou, i, j))

    candidates.sort(reverse=True)
    used_p, used_t, total = set(), set(), 0.0
    for iou, i, j in candidates:
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        total += iou

    soft_recall = total / len(T)
    soft_precision = total / len(P)
    if soft_recall + soft_precision == 0:
        return 0.0
    return 2 * soft_recall * soft_precision / (soft_recall + soft_precision)


def acv_rank_decay(ranked_cars: str, true_car: str) -> float:
    order = str(ranked_cars).split("|")
    n = len(order)
    if true_car not in order or n == 0:
        return 0.0
    r = order.index(true_car) + 1
    return (n - (r - 1)) / n


def rail_macro_f1(pred: pd.Series, truth: pd.Series) -> float:
    return float(f1_score(truth, pred, average="macro", zero_division=0))


def shm_score(pred: np.ndarray, truth: np.ndarray) -> float:
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    mape = float(np.mean(np.abs(truth - pred) / np.abs(truth)))
    return max(0.0, 1 - mape)


def combined(scores: dict[str, float]) -> dict[str, float]:
    """Overall (breadth, /4) and Average (depth, /attempted)."""
    vals = [scores.get(k, 0.0) for k in ("door", "acv", "rail", "shm")]
    attempted = [v for k, v in scores.items() if v is not None]
    return {
        "overall": sum(vals) / 4,
        "average": (sum(attempted) / len(attempted)) if attempted else 0.0,
    }


if __name__ == "__main__":
    # Sanity check on the Door training stream: our own pipeline vs ground truth.
    import door
    from common import data_path

    pred = door.predict(data_path("Door", "Train.csv"))
    truth = pd.read_csv(data_path("Door", "Train_Segments_Answer.csv"))
    print(f"Door IoU-weighted F1 on the training stream (fitted): "
          f"{door_iou_f1(pred, truth):.4f}")
