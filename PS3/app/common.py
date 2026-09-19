"""Shared helpers for the MAVIS condition-monitoring pipelines."""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
OUTPUT_DIR = os.path.join(HERE, "outputs")

# Defaults to the datasets that sit beside this folder in the repo (PS3/02_Datasets).
DATA_ROOT = os.environ.get("PS3_DATA_ROOT", os.path.join(HERE, "..", "02_Datasets"))


def data_path(*parts: str) -> str:
    return os.path.normpath(os.path.join(DATA_ROOT, *parts))


# --------------------------------------------------------------------------
# Door timestamps: "2023-7-5-0-0-3-760" -> Y-M-D-H-M-S-ms, not zero padded.
# --------------------------------------------------------------------------
def parse_door_time(s: str) -> pd.Timestamp:
    p = [int(x) for x in str(s).strip().split("-")]
    return pd.Timestamp(
        year=p[0], month=p[1], day=p[2], hour=p[3],
        minute=p[4], second=p[5], microsecond=p[6] * 1000,
    )


def parse_door_times(series: pd.Series) -> pd.Series:
    return series.map(parse_door_time)


# --------------------------------------------------------------------------
# ACV column parsing. Files differ in schema, so always read the real headers.
# NOTE: "Car model" is an identifying column, NOT a car. Matching on the
# two-digit pattern keeps it out of the ranking.
# --------------------------------------------------------------------------
CAR_COL_RE = re.compile(r"^Car (\d{2}) - (.+)$")


def acv_car_columns(columns) -> dict[str, dict[str, str]]:
    """{car_id: {parameter_name: column_name}} for every real car in the file."""
    out: dict[str, dict[str, str]] = {}
    for c in columns:
        m = CAR_COL_RE.match(str(c).strip())
        if m:
            out.setdefault(m.group(1), {})[m.group(2)] = c
    return out


def find_param(params: dict[str, str], *keywords: str) -> str | None:
    """First parameter whose name contains all keywords (case-insensitive)."""
    for name, col in params.items():
        low = name.lower()
        if all(k.lower() in low for k in keywords):
            return col
    return None


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_std(a: np.ndarray) -> float:
    return float(np.std(a)) if len(a) > 1 else 0.0
