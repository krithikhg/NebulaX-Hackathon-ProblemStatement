"""Validate ``rail_predictions.csv`` against the PS3 submission schema.

Checks (Info Kit Sections 3-4, PS3 spec Sections 4-5):
  * exactly two columns: ``file_id``, ``prediction``
  * one row per provided Test file, filenames matching exactly (incl. extension)
  * labels are only Normal / Side I / Side II
  * no missing/extra/duplicate rows

Usage:  cd PS3/solution && ../../.venv/bin/python -m rail.verify_submission
"""
from __future__ import annotations

import os
import sys

import pandas as pd

from rail import io

PRED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions",
                    "rail_predictions.csv")
VALID = {"Normal", "Side I", "Side II"}


def main(path: str = PRED) -> int:
    ok = True
    print(f"checking {path}")
    if not os.path.exists(path):
        print("  FAIL: file does not exist"); return 1
    df = pd.read_csv(path)

    if list(df.columns) != ["file_id", "prediction"]:
        print(f"  FAIL: columns are {list(df.columns)}, expected ['file_id','prediction']"); ok = False
    else:
        print("  OK  : columns = file_id, prediction")

    bad = set(df["prediction"].unique()) - VALID
    if bad:
        print(f"  FAIL: invalid labels {bad}"); ok = False
    else:
        print(f"  OK  : labels valid ({sorted(df['prediction'].unique())})")

    test = {os.path.basename(p) for p in io.list_split("test")}
    have = set(df["file_id"])
    missing, extra = test - have, have - test
    if missing:
        print(f"  FAIL: {len(missing)} test files missing, e.g. {sorted(missing)[:3]}"); ok = False
    if extra:
        print(f"  FAIL: {len(extra)} unexpected files, e.g. {sorted(extra)[:3]}"); ok = False
    if not missing and not extra:
        print(f"  OK  : all {len(test)} test files present, no extras")

    if df["file_id"].duplicated().any():
        print("  FAIL: duplicate file_id rows"); ok = False
    if df.isnull().any().any():
        print("  FAIL: null values present"); ok = False

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
