"""Loaders for the four PS3 Train Condition Monitoring datasets.

Everything resolves through `data/` (see `data/README.md`); set the `PS3_DATA`
environment variable to point elsewhere.

The Door loader is the fleshed-out one, since that's the subsystem under active
work. Two facts about the raw Door stream drive its design:

1. **Cycle boundaries are time gaps, not flags.** The info kit warns against
   trusting the opening/closing columns. It turns out the stream is sampled at a
   steady 20 ms *within* a cycle and jumps by many seconds *between* cycles: on
   `Train.csv`, thresholding the inter-row gap at 1 s recovers 109 boundaries,
   which -- together with row 0 -- are exactly the 110 ground-truth segment
   starts, with no false splits. `segment_by_gap` is therefore the segmenter,
   and it is exact on Train rather than approximate.
2. **Time is not monotonic inside a cycle.** 497 of Train's inter-row diffs run
   backwards by up to ~0.9 s, so "first row" and "earliest timestamp" are
   different rows in 9 of the 110 cycles. The answer key resolves this
   positionally -- its `start_time`/`end_time` are the first and last *rows* of
   the cycle, which reproduce all 220 ground-truth boundaries exactly, whereas
   min/max of the timestamps misses 9 of them. `door_segment_table` therefore
   takes boundaries positionally, and predictions must do the same or they lose
   IoU on those cycles for no reason.

Both the segment-level view (one row per cycle) and the sample-level view (one
row per 20 ms reading) are returned -- classification happens per cycle, but the
features that separate Normal from Abnormal resistance live in the within-cycle
waveform.
"""

import os
import pathlib

import pandas as pd

# Resolved from this file's own location, not from the working directory, so
# imports behave the same from a notebook, a script or a subfolder. Set the
# PS3_DATA environment variable to point somewhere else entirely.
DATA_ROOT = pathlib.Path(
    os.environ.get("PS3_DATA", pathlib.Path(__file__).resolve().parent.parent.parent / "02_Datasets")
)

DOOR_DIR = DATA_ROOT / "Door"
ACV_DIR = DATA_ROOT / "ACV"
RAIL_DIR = DATA_ROOT / "Rail_Corrugation"
SHM_DIR = DATA_ROOT / "SHM"

DATETIME_FORMAT = "%Y-%m-%d-%H-%M-%S-%f"

# Inter-row gap above which a new door cycle is assumed to have started.
# Within-cycle sampling is 20 ms (with jitter); between-cycle gaps are seconds.
DOOR_GAP_SECONDS = 1.0

# The 16 sensor columns, i.e. everything but Datetime.
DOOR_SENSOR_COLS = [
    "Motor current(mA)", "Motor Voltage(10mV)", "Motor electrodynamic force",
    "Door opening time(.1s)", "Door closing time(.1s)",
    "Close command", "Open command",
    "DCSR", "DCSL", "DLSR", "DLSL",
    "Door Opened", "Door Locked", "Door is opening", "Door is closing",
    "Door leaf position",
]

# The continuously-varying signals the fault actually shows up in, as opposed to
# the binary switch/command flags.
DOOR_ANALOG_COLS = [
    "Motor current(mA)", "Motor Voltage(10mV)", "Motor electrodynamic force",
    "Door leaf position",
]


def parse_door_time(series):
    """Parse the dataset's native `2023-7-5-0-11-17-664` timestamps.

    The trailing field is milliseconds, and nothing is zero-padded, so the
    format has to be given explicitly -- inference gets it wrong on rows whose
    millisecond field has fewer than 3 digits.
    """
    return pd.to_datetime(series, format=DATETIME_FORMAT)


def format_door_time(ts):
    """Inverse of `parse_door_time`, for writing submissions back out in the
    dataset's own format. The scorer also accepts ISO, but round-tripping the
    native format keeps predictions visually diffable against the answer key.
    """
    ts = pd.Timestamp(ts)
    return (f"{ts.year}-{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}-"
            f"{ts.microsecond // 1000}")


def load_door_raw(split="Train"):
    """The door CSV exactly as it sits on disk -- no added or renamed columns.

    This is the entry point for any pipeline that does its own timestamp parsing
    and cycle assignment (the sequence-model notebook does both). Use it instead
    of `pd.read_csv` anywhere, so no caller has to know where the data lives.
    """
    return pd.read_csv(DOOR_DIR / f"{split}.csv")


def load_door_stream(split="Train"):
    """The raw continuous stream, one row per 20 ms reading.

    Adds `t` (parsed timestamp) and `dt` (seconds since the previous row); the
    original `Datetime` string is kept so predictions can be written back in the
    native format.
    """
    df = pd.read_csv(DOOR_DIR / f"{split}.csv")
    df["t"] = parse_door_time(df["Datetime"])
    df["dt"] = df["t"].diff().dt.total_seconds()
    return df


def segment_by_gap(stream, gap_seconds=DOOR_GAP_SECONDS):
    """Assign a 0-based `segment` index to every row, starting a new segment
    wherever the inter-row gap exceeds `gap_seconds`.

    Exact on Train (110/110 segments, no false splits) -- see the module
    docstring. Returns a copy with the `segment` column added.
    """
    stream = stream.copy()
    is_boundary = stream["dt"].isna() | (stream["dt"] > gap_seconds)
    stream["segment"] = is_boundary.cumsum() - 1
    return stream


def door_segment_table(stream):
    """Collapse a segmented stream to one row per cycle: `segment`, `start_time`,
    `end_time`, `n_rows`, plus that segment's row-index range.

    start/end are the segment's first and last *rows*, matching the answer key's
    own convention -- not the min/max timestamp, which disagrees on the 9 cycles
    where time runs backwards across the boundary (see the module docstring).
    """
    g = stream.groupby("segment")
    seg = pd.DataFrame({
        "start_time": g["t"].first(),
        "end_time": g["t"].last(),
        "n_rows": g.size(),
        "row_start": g["t"].apply(lambda s: s.index[0]),
        "row_end": g["t"].apply(lambda s: s.index[-1]),
    }).reset_index()
    seg["duration_s"] = (seg["end_time"] - seg["start_time"]).dt.total_seconds()
    return seg


def load_door_answer():
    """`Train_Segments_Answer.csv` with its timestamps parsed.

    `operation` (Open/Close) is informational only -- the submission asks for
    `status` alone.
    """
    ans = pd.read_csv(DOOR_DIR / "Train_Segments_Answer.csv")
    ans["start_time"] = parse_door_time(ans["start_time"])
    ans["end_time"] = parse_door_time(ans["end_time"])
    return ans


def load_door(split="Train", gap_seconds=DOOR_GAP_SECONDS, with_labels=None):
    """The usual starting point: `(stream, segments)`.

    `stream` is sample-level with a `segment` column; `segments` is cycle-level.
    For Train, ground truth is joined onto `segments` as `status` (and
    `operation`) by matching on start time, so a mis-segmentation shows up as a
    NaN rather than as a silently shifted label. `with_labels` defaults to True
    for Train and False for Test.
    """
    if with_labels is None:
        with_labels = split.lower() == "train"

    stream = segment_by_gap(load_door_stream(split), gap_seconds)
    segments = door_segment_table(stream)

    if with_labels:
        ans = load_door_answer()
        segments = segments.merge(
            ans[["segment_id", "start_time", "operation", "status"]],
            on="start_time", how="left",
        )
        segments["label"] = (segments["status"] == "Abnormal resistance").astype(float)
        segments.loc[segments["status"].isna(), "label"] = float("nan")
        stream = stream.merge(
            segments[["segment", "status", "label"]], on="segment", how="left")

    return stream, segments


def door_iou_f1(true_segments, pred_segments, label_col="status"):
    """The official Door metric: IoU-weighted F1 (info kit section 4).

    Matching is one-to-one, same-label-only, greedy by descending IoU; each
    match earns its IoU rather than a flat point. Reimplemented here so a
    candidate pipeline can be scored against a held-out slice of Train exactly
    the way the judges will score Test.

    Both frames need `start_time`, `end_time` and `label_col`.
    """
    def seconds(frame, col):
        return pd.to_datetime(frame[col]).astype("int64").to_numpy() / 1e9

    t0, t1 = seconds(true_segments, "start_time"), seconds(true_segments, "end_time")
    p0, p1 = seconds(pred_segments, "start_time"), seconds(pred_segments, "end_time")
    t_lab = true_segments[label_col].to_numpy()
    p_lab = pred_segments[label_col].to_numpy()

    candidates = []
    for i in range(len(t0)):
        for j in range(len(p0)):
            if t_lab[i] != p_lab[j]:
                continue
            inter = max(0.0, min(t1[i], p1[j]) - max(t0[i], p0[j]))
            union = (t1[i] - t0[i]) + (p1[j] - p0[j]) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > 0:
                candidates.append((iou, i, j))

    candidates.sort(reverse=True)
    used_true, used_pred, total_iou = set(), set(), 0.0
    matches = []
    for iou, i, j in candidates:
        if i in used_true or j in used_pred:
            continue
        used_true.add(i)
        used_pred.add(j)
        total_iou += iou
        matches.append((i, j, iou))

    n_true, n_pred = len(t0), len(p0)
    soft_recall = total_iou / n_true if n_true else 0.0
    soft_precision = total_iou / n_pred if n_pred else 0.0
    score = (2 * soft_recall * soft_precision / (soft_recall + soft_precision)
             if (soft_recall + soft_precision) > 0 else 0.0)

    return {
        "score": score,
        "soft_recall": soft_recall,
        "soft_precision": soft_precision,
        "n_true": n_true,
        "n_pred": n_pred,
        "n_matched": len(matches),
        "mean_iou": total_iou / len(matches) if matches else 0.0,
        "matches": matches,
    }


# --- The other three subsystems: enough to load and look at ------------------

def _resolve(name, base):
    """Accept either a bare filename or a path, searching Train/ then Test/."""
    path = pathlib.Path(name)
    if path.is_absolute() or path.exists():
        return path
    for folder in (base / "Train", base / "Test", base):
        if (folder / path.name).exists():
            return folder / path.name
    raise FileNotFoundError(f"{name} not found under {base}")


# Toggle: whether `load_acv` scrubs literal-0 readings in the ACV temperature
# and setpoint columns to NaN at load time. A handful of rows in several case
# files (and, in cases 05/06, ~86-90% of the Heating setpoint column) carry an
# exact 0 rather than a blank -- physically implausible for a cabin/outdoor
# temperature or an active setpoint, so it reads as a sensor/telemetry glitch
# rather than a real 0C reading. Whether that glitch itself correlates with
# the leaking car is an open question (it does in 3 of the 3 standard-schema
# labelled cases checked, but not in the Test case, where it's spread across
# several cars) -- scrubbing it to NaN throws that signal away along with the
# noise, which is exactly why this is a toggle and not just a fix: flip it to
# False to get the raw 0s back for that investigation.
ACV_TREAT_ZERO_AS_MISSING = True

# Raw column-name suffixes (after "Car NN - ") this toggle applies to -- the
# numeric temperature/setpoint parameters, under every name they appear as
# across the case files (see acv_features.PARAM_ALIASES for where these
# get folded onto one canonical name downstream).
ACV_ZERO_INVALID_PARAMS = {
    "Indoor Average Temperature",
    "Outdoor Average Temperature",
    "Outside Temperature Sensor Reading",
    "ACV Control Temperature (Cooling)",
    "ACV Control Temperature (Heating)",
    "Passenger Cabin Temperature Detected Value",
    "Fresh Air Temperature Detected Value",
    "Target Temperature Value",
}


def load_acv(case="acv_test_case.xlsx", treat_zero_as_missing=None):
    """One ACV case file (30 s telemetry for all 8 cars).

    `treat_zero_as_missing` defaults to `ACV_TREAT_ZERO_AS_MISSING`; pass it
    explicitly to override the module-level toggle for one call without
    flipping it globally.
    """
    if treat_zero_as_missing is None:
        treat_zero_as_missing = ACV_TREAT_ZERO_AS_MISSING

    df = pd.read_excel(_resolve(case, ACV_DIR))
    df["Time"] = pd.to_datetime(df["Time"])

    if treat_zero_as_missing:
        for col in df.columns:
            if not (col.startswith("Car ") and " - " in col):
                continue
            if col.split(" - ", 1)[1] not in ACV_ZERO_INVALID_PARAMS:
                continue
            df[col] = df[col].mask(df[col] == 0)

    return df


def acv_car_columns(df):
    """{car_id: [its columns]} parsed from the file's own headers.

    The parameter set differs between case files, so this reads the headers
    rather than assuming a fixed layout -- and the car ids it returns
    (`01`..`08`) are exactly the strings a `ranked_cars` submission must use.
    """
    cars = {}
    for col in df.columns:
        if col.startswith("Car ") and " - " in col:
            car = col.split(" - ", 1)[0][len("Car "):].strip()
            cars.setdefault(car, []).append(col)
    return dict(sorted(cars.items()))


def load_acv_labels():
    return pd.read_csv(ACV_DIR / "Train_Labels.csv", dtype={"faulty_car": str})


def load_rail(name):
    """One 1-second, 10 kHz axle-box recording: 10000 rows x 129 columns."""
    return pd.read_csv(_resolve(name, RAIL_DIR))


def load_rail_labels():
    return pd.read_csv(RAIL_DIR / "Train_Labels.csv")


def rail_side_columns(df):
    """{'Side I': [...], 'Side II': [...]} -- axle-box positions 1, 3, 5, 7 are
    Side I and 2, 4, 6, 8 are Side II, across all 8 cars.
    """
    sides = {"Side I": [], "Side II": []}
    for col in df.columns:
        if " position " not in col:
            continue
        position = int(col.split(" position ")[1].split()[0])
        sides["Side I" if position % 2 else "Side II"].append(col)
    return sides


def load_shm(name):
    """One dynamic-stress segment: a headerless single column of 581,120 values."""
    return pd.read_csv(_resolve(name, SHM_DIR), header=None).iloc[:, 0].to_numpy()


def load_shm_labels():
    return pd.read_csv(SHM_DIR / "Train_Labels.csv")
