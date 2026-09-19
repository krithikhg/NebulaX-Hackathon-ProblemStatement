"""ACV subsystem — localise the car with a refrigerant leak.

This is the **single** ACV implementation, vendored from
PS3/subsystems/acv/predict.py so the web app (server.py) and the submission CLI
(predict.py) share it.

Nothing is fitted: every car in a file sees the same weather, schedule and
setpoint, so the other seven cars are the control group for each one. Two
per-car indicators are turned into leave-one-out robust z-scores against the
peers and averaged:

  hot   mean Indoor Average Temperature            (higher = more suspicious)
  gap   mean(Indoor - Cooling setpoint) while in a
        cooling running mode                       (higher = more suspicious)

Literal 0 readings in the temperature/setpoint columns are treated as missing
(the sensor-dropout glitch). Rejected on evidence: the Invalid count, agreement
with the majority running mode, outdoor temperature, and case 04's extra
telemetry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import data_path

COOLING_MODES = {"Half Cooling", "Full Cooling", "Automatic Cooling"}

# Smallest peer spread a z-score is divided by (degrees C). The sensor steps in
# 0.5 C, so peers that agree to far less than a step would otherwise turn a
# quantisation difference into a huge z.
MIN_SCALE = 0.02

PARAM_ALIASES = {
    "Passenger Cabin Temperature Detected Value": "Indoor Average Temperature",
    "Target Temperature Value": "ACV Control Temperature (Cooling)",
}
ZERO_IS_MISSING = {
    "Indoor Average Temperature",
    "ACV Control Temperature (Cooling)",
}


# ----------------------------------------------------------------- loading ----
def load(path_or_buffer) -> pd.DataFrame:
    return pd.read_excel(path_or_buffer)


def cars_from(df: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    """{car id: {canonical parameter: column}} from the file's own headers."""
    cars: dict[str, dict[str, pd.Series]] = {}
    for col in df.columns:
        if not (str(col).startswith("Car ") and " - " in str(col)):
            continue
        car_part, param = str(col).split(" - ", 1)
        car = car_part[len("Car "):].strip()
        param = PARAM_ALIASES.get(param, param)
        series = df[col]
        if param in ZERO_IS_MISSING:
            series = pd.to_numeric(series, errors="coerce")
            series = series.mask(series == 0)
        cars.setdefault(car, {})[param] = series
    return dict(sorted(cars.items()))


# --------------------------------------------------------------- indicators ----
def indicators(cars: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
    """One row per car: mean indoor temperature (`hot`) and cooling gap (`gap`)."""
    rows = {}
    for car, p in cars.items():
        indoor = p.get("Indoor Average Temperature")
        if indoor is None:
            rows[car] = {"hot": np.nan, "gap": np.nan}
            continue
        indoor = pd.to_numeric(indoor, errors="coerce")
        ctrl = pd.to_numeric(p.get("ACV Control Temperature (Cooling)"), errors="coerce")
        mode = p.get("ACV Running Mode")
        gap = (indoor - ctrl).where(mode.isin(COOLING_MODES)) if mode is not None else None
        rows[car] = {"hot": indoor.mean(), "gap": gap.mean() if gap is not None else np.nan}
    out = pd.DataFrame.from_dict(rows, orient="index")
    # A car whose temperature sensor never reads has no data at all.
    out.loc[out["hot"].isna(), :] = np.nan
    return out


def peer_z(ind: pd.DataFrame) -> pd.Series:
    """Mean leave-one-out robust z-score per car (higher = more suspicious)."""
    z = pd.DataFrame(np.nan, index=ind.index, columns=ind.columns)
    for k in ind.columns:
        x = ind[k]
        for car in ind.index:
            if pd.isna(x[car]):
                continue
            peers = x.drop(car).dropna()
            if peers.empty:
                continue
            scale = 1.4826 * (peers - peers.median()).abs().median()
            if scale < 1e-9:
                scale = peers.std(ddof=0)
            z.loc[car, k] = (x[car] - peers.median()) / max(scale, MIN_SCALE)
    return z.mean(axis=1)


def _peer_median(x: pd.Series) -> pd.Series:
    return pd.Series({car: (x.drop(car).dropna().median() if not x.drop(car).dropna().empty else np.nan)
                      for car in x.index})


# ----------------------------------------------------------------- ranking ----
def _score(df: pd.DataFrame) -> pd.Series:
    cars = cars_from(df)
    if not cars:
        raise ValueError("No 'Car NN - <parameter>' columns found in this file.")
    ind = indicators(cars)
    if ind["hot"].isna().all():
        raise ValueError("No cabin-temperature channel found for any car.")
    return peer_z(ind)


def ranked_cars(score: pd.Series) -> str:
    """`|`-separated car ids, most to least suspicious; no-data cars last."""
    score = score.sort_index().round(9)
    order = score.dropna().sort_values(ascending=False, kind="stable")
    rest = [c for c in score.index if c not in order.index]
    return "|".join(list(order.index) + rest)


def rank_cars(df: pd.DataFrame) -> pd.DataFrame:
    """Per-car evidence table, most- to least-likely faulty (for the UI)."""
    cars = cars_from(df)
    if not cars:
        raise ValueError("No 'Car NN - <parameter>' columns found in this file.")
    ind = indicators(cars)
    if ind["hot"].isna().all():
        raise ValueError("No cabin-temperature channel found for any car.")
    score = peer_z(ind)
    deviation = ind["hot"] - _peer_median(ind["hot"])
    out = pd.DataFrame({
        "car": score.index,
        "score": score.to_numpy(),
        "cabin_temp_dev_C": deviation.reindex(score.index).to_numpy(),
    })
    return out.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)


# ------------------------------------------------------------------- public ----
def predict(path_or_buffer, file_id: str | None = None) -> pd.DataFrame:
    df = load(path_or_buffer)
    name = file_id or getattr(path_or_buffer, "name", "acv_test_case.xlsx")
    return pd.DataFrame([{"file_id": str(name).split("/")[-1],
                          "ranked_cars": ranked_cars(_score(df))}])


def validate(verbose: bool = True) -> dict:
    """Rank-decay score over the labelled training cases."""
    labels = pd.read_csv(data_path("ACV", "Train_Labels.csv"))
    scores, rows = [], []
    for _, r in labels.iterrows():
        try:
            order = ranked_cars(_score(load(data_path("ACV", "Train", r["filename"])))).split("|")
            faulty = str(r["faulty_car"]).zfill(2)
            n = len(order)
            rank = order.index(faulty) + 1 if faulty in order else None
            s = (n - (rank - 1)) / n if rank else 0.0
            scores.append(s)
            rows.append({"file": r["filename"], "true": faulty, "rank": rank, "n_cars": n,
                         "score": round(s, 3), "top3": "|".join(order[:3])})
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            rows.append({"file": r["filename"], "true": r["faulty_car"], "rank": None,
                         "n_cars": None, "score": None, "top3": str(exc)[:60]})
    mean = float(np.mean(scores)) if scores else 0.0
    if verbose:
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"[acv] mean rank-decay score over scored cases: {mean:.3f}")
    return {"rank_decay": mean, "table": pd.DataFrame(rows)}


if __name__ == "__main__":
    validate()
