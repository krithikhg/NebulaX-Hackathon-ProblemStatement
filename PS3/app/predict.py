"""Unified inference CLI.

    python predict.py --subsystem door  --input .../Door/Test.csv        --output outputs/door_predictions.csv
    python predict.py --subsystem acv   --input .../Test/acv_test_case.xlsx --output outputs/acv_predictions.csv
    python predict.py --subsystem rail  --input .../Rail_Corrugation/Test  --output outputs/rail_predictions.csv
    python predict.py --subsystem shm   --input .../SHM/Test               --output outputs/shm_predictions.csv

--input takes a single file or a folder, depending on the subsystem:
Door expects one continuous stream, ACV one workbook, Rail and SHM a folder of
files (a single file also works).
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

import acv
import door
import rail
import shm

SUBMISSION_COLUMNS = {
    "door": ["start_time", "end_time", "prediction"],
    "acv": ["file_id", "ranked_cars"],
    "rail": ["file_id", "prediction"],
    "shm": ["file_id", "prediction"],
}


def run(subsystem: str, input_path: str, submission_only: bool = True) -> pd.DataFrame:
    subsystem = subsystem.lower()
    if subsystem == "door":
        out = door.predict(input_path)
    elif subsystem == "acv":
        if os.path.isdir(input_path):
            out = pd.concat(
                [acv.predict(os.path.join(input_path, f), f)
                 for f in sorted(os.listdir(input_path))
                 if f.lower().endswith((".xlsx", ".xls"))],
                ignore_index=True,
            )
        else:
            out = acv.predict(input_path, os.path.basename(input_path))
    elif subsystem == "rail":
        out = (rail.predict_folder(input_path) if os.path.isdir(input_path)
               else rail.predict(input_path, os.path.basename(input_path)))
    elif subsystem == "shm":
        out = (shm.predict_folder(input_path) if os.path.isdir(input_path)
               else shm.predict(input_path, os.path.basename(input_path)))
    else:
        raise SystemExit(f"Unknown subsystem: {subsystem}")

    return out[SUBMISSION_COLUMNS[subsystem]] if submission_only else out


def main() -> None:
    ap = argparse.ArgumentParser(description="MAVIS inference")
    ap.add_argument("--subsystem", required=True,
                    choices=["door", "acv", "rail", "shm"])
    ap.add_argument("--input", required=True, help="file or folder")
    ap.add_argument("--output", required=True, help="destination CSV")
    ap.add_argument("--keep-extras", action="store_true",
                    help="keep diagnostic columns (not submission schema)")
    args = ap.parse_args()

    out = run(args.subsystem, args.input, submission_only=not args.keep_extras)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
