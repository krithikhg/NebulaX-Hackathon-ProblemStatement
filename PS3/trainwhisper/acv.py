"""ACV subsystem — localise the car with a refrigerant leak.

Method
------
A leaking car loses cooling capacity, so its cabin runs warmer than its
siblings under the same ambient conditions and the same control commands. The
train itself is the control group: at every timestamp we subtract the median
cabin temperature across all cars, then average that deviation over the file
and rank descending.

This beats an absolute threshold because ambient weather, time of day and
setpoint changes move every car together and cancel out in the deviation.

Schema-agnostic by design: the files do not share a parameter set (most carry 8
parameters per car, one carries 59 including refrigeration pressures), so the
loader reads each file's own headers and picks the best available evidence:

  1. cabin/indoor temperature deviation        (primary, present in all files)
  2. refrigeration low-side pressure deviation (direct leak signature, bonus)

On the 6 labelled training cases this ranks the true faulty car 1st in 4 of the
5 cases where a cabin-temperature channel exists, and 2nd in the other
(rank-decay score ~0.975).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common import acv_car_columns, data_path, find_param, numeric

PRESSURE_WEIGHT = 0.5  # weight on the low-pressure evidence when it exists


def _cabin_series(df: pd.DataFrame, params: dict[str, str]) -> pd.Series | None:
    col = (
        find_param(params, "indoor", "average", "temperature")
        or find_param(params, "passenger", "cabin", "temperature")
        or find_param(params, "observation", "temperature")
    )
    return numeric(df[col]) if col else None


def _low_pressure_series(df: pd.DataFrame, params: dict[str, str]) -> pd.Series | None:
    cols = [c for n, c in params.items() if "low pressure" in n.lower()]
    if not cols:
        return None
    return pd.concat([numeric(df[c]) for c in cols], axis=1).mean(axis=1)


def _deviation(frame: pd.DataFrame) -> pd.Series:
    """Mean deviation of each column from the across-car median, per timestamp."""
    return frame.sub(frame.median(axis=1), axis=0).mean()


def load(path_or_buffer) -> pd.DataFrame:
    return pd.read_excel(path_or_buffer)


def rank_cars(df: pd.DataFrame) -> pd.DataFrame:
    """Per-car evidence table, most- to least-likely faulty."""
    cars = acv_car_columns(df.columns)
    if not cars:
        raise ValueError("No 'Car NN - <parameter>' columns found in this file.")

    temps, pressures = {}, {}
    for car, params in cars.items():
        s = _cabin_series(df, params)
        if s is not None:
            temps[car] = s
        p = _low_pressure_series(df, params)
        if p is not None:
            pressures[car] = p

    if not temps:
        raise ValueError("No cabin-temperature channel found for any car.")

    temp_dev = _deviation(pd.DataFrame(temps))
    score = temp_dev.copy()
    press_dev = None
    if len(pressures) == len(cars):
        press_dev = _deviation(pd.DataFrame(pressures))
        # Low-side pressure FALLS when refrigerant is lost, hence the minus.
        spread = press_dev.abs().max() or 1.0
        score = score + PRESSURE_WEIGHT * (-press_dev / spread) * temp_dev.abs().max()

    out = pd.DataFrame(
        {
            "car": score.index,
            "score": score.values,
            "cabin_temp_dev_C": temp_dev.reindex(score.index).values,
        }
    )
    if press_dev is not None:
        out["low_pressure_dev"] = press_dev.reindex(score.index).values
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def predict(path_or_buffer, file_id: str | None = None) -> pd.DataFrame:
    df = load(path_or_buffer)
    ranked = rank_cars(df)
    name = file_id or getattr(path_or_buffer, "name", "acv_test_case.xlsx")
    return pd.DataFrame(
        [{"file_id": os.path.basename(str(name)),
          "ranked_cars": "|".join(ranked["car"].tolist())}]
    )


def validate(verbose: bool = True) -> dict:
    """Rank-decay score over the labelled training cases."""
    labels = pd.read_csv(data_path("ACV", "Train_Labels.csv"))
    labels["faulty_car"] = labels["faulty_car"].astype(str).str.zfill(2)
    scores, rows = [], []
    for _, r in labels.iterrows():
        path = data_path("ACV", "Train", r["filename"])
        try:
            ranked = rank_cars(load(path))
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            rows.append({"file": r["filename"], "rank": None, "note": str(exc)[:60]})
            continue
        order = ranked["car"].tolist()
        n = len(order)
        rank = order.index(r["faulty_car"]) + 1 if r["faulty_car"] in order else None
        s = (n - (rank - 1)) / n if rank else 0.0
        scores.append(s)
        rows.append({"file": r["filename"], "true": r["faulty_car"],
                     "rank": rank, "n_cars": n, "score": round(s, 3),
                     "top3": "|".join(order[:3])})
    table = pd.DataFrame(rows)
    mean = float(np.mean(scores)) if scores else 0.0
    if verbose:
        print(table.to_string(index=False))
        print(f"[acv] mean rank-decay score over scored cases: {mean:.3f}")
    return {"rank_decay": mean, "table": table}


if __name__ == "__main__":
    validate()
