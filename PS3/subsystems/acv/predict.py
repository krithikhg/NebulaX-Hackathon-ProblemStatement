"""ACV subsystem inference: refrigerant-leak localisation.

Given an ACV case file (`--input`), ranks the 8 cars from most to least likely
to carry the refrigerant leak and writes `acv_predictions.csv` (`file_id`,
`ranked_cars`) per the ACV Info Kit's required schema (Section 3).

Method (see 03_References/ACV/ACV_Subsystem_Info_Kit.md for the task, and
acv_composite_model.ipynb for how this design was chosen). Nothing is fitted:
every car in a file sees the same weather, schedule and setpoint, so the other
seven cars are the control group for each one.

1. **Indicators** -- per car, over the whole recording:
   - `hot`: mean Indoor Average Temperature (higher = more suspicious)
   - `gap`: mean of (Indoor - ACV Control Temperature (Cooling)) over the rows
     where that car's own running mode is a cooling state (higher = more
     suspicious) -- how well this car executes a command all 8 were given.
2. **Peer z-score** -- each indicator is turned into a leave-one-out robust
   z-score, z = (x - median(peers)) / max(1.4826 * MAD(peers), MIN_SCALE),
   where the peers are the other cars.
3. **Index** -- a car's mean z over its indicators. Highest = most suspicious.
   Cars with no data at all rank last; exact ties are ordered by car id.

Literal 0 readings in the temperature/setpoint columns are treated as missing:
they are the sensor-dropout glitch and coincide with the `Invalid` flags.

Deliberately not used (each was tested and eliminated -- see write_up_ACV.md):
the `Invalid` count, agreement with the majority running mode, outdoor
temperature, and the extra telemetry columns that only case 04 carries.

Usage:
    python predict.py --input path/to/acv_test_case.xlsx --output acv_predictions.csv

`--input` may also be a folder, in which case every .xlsx in it is scored and
one row per file is written.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

COOLING_MODES = {"Half Cooling", "Full Cooling", "Automatic Cooling"}

# Smallest peer spread a z-score is divided by (degrees C). The sensor steps in
# 0.5 C, so peers that agree to far less than a step would otherwise turn a
# quantisation difference into a huge z.
MIN_SCALE = 0.02

# The same measurement under different names in some case files.
PARAM_ALIASES = {
    "Passenger Cabin Temperature Detected Value": "Indoor Average Temperature",
    "Target Temperature Value": "ACV Control Temperature (Cooling)",
}
ZERO_IS_MISSING = {
    "Indoor Average Temperature",
    "ACV Control Temperature (Cooling)",
}


def load_cars(path: pathlib.Path) -> dict[str, dict[str, pd.Series]]:
    """{car id: {canonical parameter: column}} from the file's own headers.

    Car ids are exactly the strings `ranked_cars` must use (e.g. `03`).
    """
    df = pd.read_excel(path)
    cars: dict[str, dict[str, pd.Series]] = {}
    for col in df.columns:
        if not (col.startswith("Car ") and " - " in col):
            continue
        car_part, param = col.split(" - ", 1)
        car = car_part[len("Car "):].strip()
        param = PARAM_ALIASES.get(param, param)
        series = df[col]
        if param in ZERO_IS_MISSING:
            series = pd.to_numeric(series, errors="coerce")
            series = series.mask(series == 0)
        cars.setdefault(car, {})[param] = series
    return dict(sorted(cars.items()))


def indicators(cars: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
    """One row per car: mean indoor temperature and mean cooling gap."""
    rows = {}
    for car, p in cars.items():
        indoor = pd.to_numeric(p["Indoor Average Temperature"], errors="coerce")
        ctrl = pd.to_numeric(p["ACV Control Temperature (Cooling)"], errors="coerce")
        gap = (indoor - ctrl).where(p["ACV Running Mode"].isin(COOLING_MODES))
        rows[car] = {"hot": indoor.mean(), "gap": gap.mean()}
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


def ranked_cars(score: pd.Series) -> str:
    """`|`-separated car ids, most to least suspicious; no-data cars last."""
    score = score.sort_index().round(9)
    order = score.dropna().sort_values(ascending=False, kind="stable")
    rest = [c for c in score.index if c not in order.index]
    return "|".join(list(order.index) + rest)


def predict(path: pathlib.Path) -> str:
    return ranked_cars(peer_z(indicators(load_cars(path))))


def main() -> None:
    ap = argparse.ArgumentParser(description="ACV subsystem inference")
    ap.add_argument("--input", required=True, help="ACV case .xlsx, or a folder of them")
    ap.add_argument("--output", required=True, help="destination predictions CSV")
    args = ap.parse_args()

    src = pathlib.Path(args.input)
    # `~$*.xlsx` are Excel's temporary lock files, not data.
    files = (sorted(f for f in src.glob("*.xlsx") if not f.name.startswith("~$"))
             if src.is_dir() else [src])
    if not files:
        raise SystemExit(f"no .xlsx files found in {src}")

    out = pd.DataFrame({"file_id": [f.name for f in files],
                        "ranked_cars": [predict(f) for f in files]})

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out)} file(s) -> {out_path}")


if __name__ == "__main__":
    main()
