"""Door subsystem inference: cycle-level anomaly detection.

Given a continuous door-controller stream (`--input`), finds every
door-open/close cycle in it and classifies each as Normal or Abnormal
resistance, writing `door_predictions.csv` (`start_time`, `end_time`,
`prediction`) per the Door Info Kit's required schema (Section 3).

Method (see 03_References/Door/Door_Subsystem_Info_Kit.md for the task, and
this folder's door_sequence_models.ipynb for how this design was chosen):

1. **Segmentation** -- a new cycle starts wherever the door's motion state
   flips (`Door is opening`/`Door is closing`) or its position jumps
   discontinuously (>100 units between consecutive rows). Both are read
   directly off the door's own state rather than a proxy for it (e.g. an
   idle gap between recordings), and together they reproduce every
   ground-truth boundary in Train.csv exactly.
2. **Per-operation normal profile** -- motor current is resampled onto a
   common fraction-of-cycle axis, and a median + IQR band is built from
   *normal* training cycles only, separately for Open and Close (their
   waveforms differ enough that a shared template would blur both).
3. **Cycle score** -- each cycle's current trace is turned into a z-profile
   against its operation's template, then reduced to one score by a phase
   statistic (e.g. the area of the largest contiguous run of z^2 above a
   percentile of that operation's own training z^2 distribution -- rewards a
   *sustained* excursion, which is how a mechanical jam actually shows up,
   over scattered noise).
4. **Threshold** -- one cut-off per operation, fit on held-out labelled
   Train cycles to maximise cycle accuracy.
5. **Hyperparameter search** -- the IQR floor and the phase statistic (which
   percentile level, and top-z^2 vs. the contiguous-run statistic) are not
   fixed choices: `fit()` grid-searches them on a tune split, selecting by
   tune accuracy and breaking ties by class-separation margin, with a
   further validation split scored only for reporting (never used to pick
   the design) -- the same discipline the notebook used to land on them.

Usage:
    python predict.py --input path/to/Test.csv --output door_predictions.csv

By default, the model is fit fresh from `Train.csv` / `Train_Segments_Answer.csv`
in `02_Datasets/Door/` (relative to this file) each run -- the profile-based
method trains in well under a second, so there is no separate model artifact
to ship or go stale. Point `--train`/`--train-answer` elsewhere to fit from a
different copy of the training data.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from common import MODEL_DIR, data_path

DATETIME_FORMAT = "%Y-%m-%d-%H-%M-%S-%f"
STATUS_NORMAL = "Normal"
STATUS_ABNORMAL = "Abnormal resistance"

POSITION_COL = "Door leaf position"
PROFILE_COLS = ["Motor current(mA)"]  # back-EMF measured and dropped: current
# alone gave a 15.7x train/tune separation margin vs ~1-1.2x for every design
# that let back-EMF vote too (see the notebook, section 5 / "what was tried").

N_PHASE = 200          # points each cycle is resampled onto (cycles run 137-187 rows)

# IQR floor (fraction of a channel's median IQR) and phase statistic are
# hyperparameters, grid-searched in `fit()` -- these are just the grid to
# search, not the chosen values.
FLOOR_GRID = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
STATISTIC_GRID = ["top05_z2", "scan_p90", "scan_p95", "scan_p99"]

DEFAULT_TRAIN = data_path("Door", "Train.csv")
DEFAULT_TRAIN_ANSWER = data_path("Door", "Train_Segments_Answer.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "door_model.joblib")


# ---- loading & segmentation -------------------------------------------------

def load_stream(path) -> pd.DataFrame:
    """Read a raw Door CSV and parse its native timestamp format."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["_t"] = pd.to_datetime(df["Datetime"], format=DATETIME_FORMAT)
    return df


def assign_cycles(df: pd.DataFrame, out_col: str = "cycle") -> pd.DataFrame:
    """Append `out_col`: the 1-based cycle number each row belongs to.

    A new cycle starts on a motion-state flip or a discontinuous door-position
    jump (>100 units between consecutive rows) -- see the module docstring.
    """
    df = df.copy()
    opening, closing = df["Door is opening"].to_numpy(), df["Door is closing"].to_numpy()
    position = df[POSITION_COL].to_numpy()

    motion_change = np.r_[True, (opening[1:] != opening[:-1]) | (closing[1:] != closing[:-1])]
    position_jump = np.r_[True, np.abs(np.diff(position)) > 100]
    df[out_col] = (motion_change | position_jump).cumsum().astype(int)
    return df


def cycle_operation(data: pd.DataFrame, cycle_col: str = "cycle") -> pd.Series:
    """Open/Close per cycle, from the sign of net door-leaf travel over the cycle."""
    position = data.groupby(cycle_col)[POSITION_COL]
    net_travel = position.last() - position.first()
    return pd.Series(np.where(net_travel > 0, "Open", "Close"),
                      index=net_travel.index, name="operation")


def cycle_bounds(data: pd.DataFrame, cycle_col: str = "cycle") -> pd.DataFrame:
    """One row per cycle: `start_time`/`end_time` as the cycle's first/last
    *rows* in stream order (not min/max timestamp -- a handful of cycles run
    backwards in time, and the row-order convention is what the answer key uses)."""
    g = data.groupby(cycle_col, sort=True)
    return pd.DataFrame({
        "start_time": g["_t"].first(),
        "end_time": g["_t"].last(),
        "n_rows": g.size(),
    })


# ---- per-operation normal profile -------------------------------------------

def _resample(x: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate `values` (n_rows, n_cols) from `x` onto `grid`, averaging
    duplicate x's first (door position/time can repeat while the door is held)."""
    order = np.argsort(x, kind="stable")
    x_sorted, values_sorted = x[order], values[order]
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    if len(unique_x) < 2:
        return np.repeat(values_sorted.mean(axis=0)[None, :], len(grid), axis=0)

    summed = np.zeros((len(unique_x), values.shape[1]))
    np.add.at(summed, inverse, values_sorted)
    averaged = summed / np.bincount(inverse, minlength=len(unique_x))[:, None]
    return np.stack([np.interp(grid, unique_x, averaged[:, c]) for c in range(values.shape[1])],
                     axis=1)


def cycle_profiles(data: pd.DataFrame, cycle_ids, grid: np.ndarray, cycle_col: str = "cycle"):
    """Resample each listed cycle's PROFILE_COLS onto the fraction-of-cycle `grid`."""
    cycle_ids = np.sort(np.asarray(cycle_ids))
    out = np.empty((len(cycle_ids), len(grid), len(PROFILE_COLS)))
    for row, cid in enumerate(cycle_ids):
        rows = data.loc[data[cycle_col] == cid]
        x = np.linspace(0, 1, len(rows))
        out[row] = _resample(x, rows[PROFILE_COLS].to_numpy(dtype=float), grid)
    return out, cycle_ids


def build_template(profiles: np.ndarray, iqr_floor_frac: float) -> dict:
    """Median profile + IQR band from a set of normal cycle profiles."""
    median = np.median(profiles, axis=0)
    q75, q25 = np.percentile(profiles, [75, 25], axis=0)
    iqr = q75 - q25
    channel_scale = np.median(iqr, axis=0)
    floor = iqr_floor_frac * np.where(channel_scale > 0, channel_scale, 1.0)
    template = {"median": median, "iqr": np.maximum(iqr, floor)}
    template["train_z2"] = (((profiles - template["median"]) / template["iqr"]) ** 2).ravel()
    return template


def build_templates(data: pd.DataFrame, normal_cycle_ids, iqr_floor_frac: float) -> dict:
    """One template per operation, fitted on the given (normal) training cycles."""
    operation = cycle_operation(data)
    grid = np.linspace(0, 1, N_PHASE)
    templates = {}
    for op in ["Close", "Open"]:
        ids = [c for c in np.sort(np.asarray(normal_cycle_ids)) if operation.loc[c] == op]
        profiles, _ = cycle_profiles(data, ids, grid)
        templates[op] = build_template(profiles, iqr_floor_frac)
        templates[op]["grid"] = grid
    return templates


def residuals_for(data: pd.DataFrame, cycle_ids, templates: dict, cycle_col: str = "cycle"):
    """Per-cycle z-profiles against each cycle's own operation's template.

    Returns (z, cycle_index, operation_per_cycle).
    """
    operation = cycle_operation(data, cycle_col)
    cycle_ids = np.sort(np.asarray(cycle_ids))
    grid = templates["Close"]["grid"]
    z = np.empty((len(cycle_ids), len(grid), len(PROFILE_COLS)))
    for row, cid in enumerate(cycle_ids):
        template = templates[operation.loc[cid]]
        profile, _ = cycle_profiles(data, [cid], grid, cycle_col)
        z[row] = (profile[0] - template["median"]) / template["iqr"]
    return z, cycle_ids, operation.loc[cycle_ids].to_numpy()


def _top_fraction(values: np.ndarray, fraction: float) -> np.ndarray:
    k = max(1, int(round(fraction * len(values))))
    return np.sort(values)[-k:]


# Label-free phase statistics: each reduces one cycle's z-profile to a single
# score without knowing which cycles are actually abnormal.
SCORE_STATISTICS = {
    "mean_z2":  lambda z: (z ** 2).mean(),
    "p95_abs":  lambda z: np.percentile(np.abs(z), 95),
    "top10_z2": lambda z: _top_fraction(z ** 2, 0.10).mean(),
    "top05_z2": lambda z: _top_fraction(z ** 2, 0.05).mean(),
}


def _scan_area(z_curve: np.ndarray, level: float) -> float:
    """Area of the largest contiguous run of z^2 above `level` -- rewards a
    sustained excursion over scattered spikes."""
    above = (z_curve ** 2) > level
    if not above.any():
        return 0.0
    best = current = 0.0
    for is_above, value in zip(above, z_curve ** 2):
        current = current + (value - level) if is_above else 0.0
        best = max(best, current)
    return best


def cycle_scores(z: np.ndarray, operations: np.ndarray, templates: dict, statistic: str) -> np.ndarray:
    """(n_cycles, n_phase, 1) residuals -> one anomaly score per cycle.

    `statistic` is one of `SCORE_STATISTICS`, or `"scan_pNN"` (e.g.
    `"scan_p99"`), which runs `_scan_area` at the NNth percentile of that
    cycle's own operation's training z^2 distribution as the level -- a
    data-driven threshold rather than a fixed one.
    """
    if statistic.startswith("scan_p"):
        percentile = float(statistic.removeprefix("scan_p"))
        return np.array([
            _scan_area(z[i, :, 0], np.percentile(templates[operations[i]]["train_z2"], percentile))
            for i in range(len(z))
        ])
    fn = SCORE_STATISTICS[statistic]
    return np.array([fn(z[i, :, 0]) for i in range(len(z))])


# ---- thresholding ------------------------------------------------------------

def fit_thresholds(scores: np.ndarray, cycles: np.ndarray, operations: np.ndarray,
                    truth: pd.Series) -> dict:
    """One threshold per operation, chosen to maximise cycle accuracy against
    `truth` (a Series of Normal/Abnormal resistance indexed by cycle id)."""
    labels = (truth.loc[cycles].to_numpy() == STATUS_ABNORMAL)
    thresholds = {}
    for op in np.unique(operations):
        m = operations == op
        s, y = scores[m], labels[m]
        order = np.sort(np.unique(s))
        candidates = np.concatenate([[order[0] - 1], (order[:-1] + order[1:]) / 2,
                                      [order[-1] + 1]])
        accuracy = [((s >= t) == y).mean() for t in candidates]
        thresholds[op] = float(candidates[int(np.argmax(accuracy))])
    return thresholds


def predict_status(scores: np.ndarray, operations: np.ndarray, thresholds: dict) -> np.ndarray:
    return np.where(scores >= np.array([thresholds[op] for op in operations]),
                     STATUS_ABNORMAL, STATUS_NORMAL)


def separation_margin(scores: np.ndarray, cycles: np.ndarray, operations: np.ndarray,
                       truth: pd.Series, trim: int = 1) -> float:
    """min over operations of (lowest abnormal score / highest normal score),
    each with the single most extreme point on its side dropped (`trim`).

    Used to break ties between designs that reach the same accuracy: accuracy
    saturates once a threshold separates every cycle in a small tune split,
    but margin keeps discriminating -- the larger it is, the more the
    threshold can drift on unseen data before anything is misclassified.
    Trimmed rather than raw min/max because with only a handful of cycles per
    operation/status group, the single most extreme score is itself a noisy
    statistic that can make a near-degenerate design look better than it is.
    """
    labels = truth.loc[cycles].to_numpy() == STATUS_ABNORMAL
    ratios = []
    for op in np.unique(operations):
        m = operations == op
        normal = np.sort(scores[m & ~labels])
        abnormal = np.sort(scores[m & labels])
        if not len(normal) or not len(abnormal):
            continue
        normal_edge = normal[-(trim + 1)] if len(normal) > trim else normal[-1]
        abnormal_edge = abnormal[trim] if len(abnormal) > trim else abnormal[0]
        ratios.append(abnormal_edge / normal_edge if normal_edge > 0 else np.inf)
    return min(ratios) if ratios else float("nan")


def split_cycles(answer: pd.DataFrame, normal_frac_train: float = 0.5, seed: int = 0) -> dict:
    """Split cycle numbers into train / tune / validation sets.

    - `train` is normal-only: it becomes the per-operation profile, and an
      abnormal cycle in it would teach the template to treat the fault as
      ordinary.
    - `tune` and `val` get matching class counts, so a design selected on
      tune doesn't just reflect a lucky class balance.

    Mirrors the split used to develop and validate this design in
    door_sequence_models.ipynb (same `normal_frac_train`/`seed`).
    """
    rng = np.random.default_rng(seed)
    status = answer["status"].to_numpy()
    cycles = np.arange(1, len(answer) + 1)

    normal = rng.permutation(cycles[status == STATUS_NORMAL])
    abnormal = rng.permutation(cycles[status != STATUS_NORMAL])

    n_train = int(round(normal_frac_train * len(normal)))
    train, rest = normal[:n_train], normal[n_train:]
    half_normal, half_abnormal = len(rest) // 2, len(abnormal) // 2
    return {
        "train": np.sort(train),
        "tune": np.sort(np.concatenate([rest[:half_normal], abnormal[:half_abnormal]])),
        "val": np.sort(np.concatenate([rest[half_normal:], abnormal[half_abnormal:]])),
    }


# ---- hyperparameter search ---------------------------------------------------

def evaluate_design(data: pd.DataFrame, splits: dict, truth: pd.Series,
                     iqr_floor_frac: float, statistic: str) -> dict:
    """Build templates at `iqr_floor_frac`, fit thresholds on tune under
    `statistic`, and report tune accuracy + margin plus val accuracy.

    Val is scored only for reporting -- it never enters the selection below,
    so it stays an honest estimate of how the selected design generalises.
    """
    templates = build_templates(data, splits["train"], iqr_floor_frac)

    z_t, c_t, op_t = residuals_for(data, splits["tune"], templates)
    s_t = cycle_scores(z_t, op_t, templates, statistic)
    thresholds = fit_thresholds(s_t, c_t, op_t, truth)
    tune_acc = (predict_status(s_t, op_t, thresholds) == truth.loc[c_t].to_numpy()).mean()
    margin = separation_margin(s_t, c_t, op_t, truth)

    z_v, c_v, op_v = residuals_for(data, splits["val"], templates)
    s_v = cycle_scores(z_v, op_v, templates, statistic)
    val_acc = (predict_status(s_v, op_v, thresholds) == truth.loc[c_v].to_numpy()).mean()

    return {"iqr_floor_frac": iqr_floor_frac, "statistic": statistic,
            "tune_acc": tune_acc, "margin": margin, "val_acc": val_acc}


def select_design(data: pd.DataFrame, splits: dict, truth: pd.Series) -> tuple[float, str]:
    """Grid-search the IQR floor and phase statistic, selecting by tune
    accuracy and breaking ties by separation margin (val is for information
    only -- see `evaluate_design`)."""
    grid = pd.DataFrame([evaluate_design(data, splits, truth, floor, statistic)
                         for floor in FLOOR_GRID for statistic in STATISTIC_GRID])
    best_tune = grid["tune_acc"].max()
    tied = grid[grid["tune_acc"] == best_tune]
    selected = tied.loc[tied["margin"].idxmax()]

    print(f"[door] hyperparameter search: {len(tied)}/{len(grid)} designs tie at "
          f"tune accuracy {best_tune:.3f}; selected floor={selected['iqr_floor_frac']} "
          f"statistic={selected['statistic']!r} "
          f"(margin={selected['margin']:.2f}x, val accuracy={selected['val_acc']:.3f})")
    return float(selected["iqr_floor_frac"]), str(selected["statistic"])


# ---- fit + predict -----------------------------------------------------------

def fit(train_path, train_answer_path) -> tuple[dict, dict, str]:
    """Fit templates + per-operation thresholds from labelled training data.

    Returns `(templates, thresholds, statistic)` -- `statistic` is needed
    alongside the templates/thresholds since scoring a new cycle must use
    the same phase statistic the thresholds were fit against.
    """
    data = assign_cycles(load_stream(train_path))
    answer = pd.read_csv(train_answer_path)
    answer["start_time"] = pd.to_datetime(answer["start_time"], format=DATETIME_FORMAT)
    truth = pd.Series(answer["status"].to_numpy(), index=np.arange(1, len(answer) + 1))

    splits = split_cycles(answer)
    iqr_floor_frac, statistic = select_design(data, splits, truth)

    templates = build_templates(data, splits["train"], iqr_floor_frac)
    z, cycles, ops = residuals_for(data, splits["tune"], templates)
    scores = cycle_scores(z, ops, templates, statistic)
    thresholds = fit_thresholds(scores, cycles, ops, truth)
    return templates, thresholds, statistic


def predict_stream(input_path, templates: dict, thresholds: dict, statistic: str) -> pd.DataFrame:
    """Segment `input_path` into cycles and classify each one."""
    data = assign_cycles(load_stream(input_path))
    bounds = cycle_bounds(data)

    z, cycles, ops = residuals_for(data, bounds.index.to_numpy(), templates)
    scores = cycle_scores(z, ops, templates, statistic)
    status = predict_status(scores, ops, thresholds)

    out = bounds.loc[cycles, ["start_time", "end_time"]].reset_index(drop=True)
    out["prediction"] = status
    return out


# ---- app-facing API ----------------------------------------------------------

_model = None


def get_model() -> tuple[dict, dict, str]:
    """Load the fitted templates/thresholds, else fit from the labelled stream."""
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            bundle = joblib.load(MODEL_PATH)
            _model = (bundle["templates"], bundle["thresholds"], bundle["statistic"])
        else:
            _model = fit(str(DEFAULT_TRAIN), str(DEFAULT_TRAIN_ANSWER))
    return _model


def predict(path_or_buffer, file_id: str | None = None) -> pd.DataFrame:
    """Submission schema: start_time, end_time, prediction (one row per cycle)."""
    templates, thresholds, statistic = get_model()
    return predict_stream(path_or_buffer, templates, thresholds, statistic)


def _decimate(xs, ys, buckets):
    """Min/max per bucket so a plotted trace keeps its peaks."""
    n = len(xs)
    if n <= buckets * 2:
        return list(zip(xs, ys))
    size = n / buckets
    out = []
    for b in range(buckets):
        a = int(b * size)
        e = min(n, int((b + 1) * size))
        lo = hi = a
        for i in range(a, e):
            if ys[i] < ys[lo]:
                lo = i
            if ys[i] > ys[hi]:
                hi = i
        for i in ([lo, hi] if lo < hi else [hi, lo]):
            out.append((xs[i], ys[i]))
    return out


def predict_with_traces(path_or_buffer):
    """Per-cycle predictions plus current traces / timing, for the web app."""
    templates, thresholds, statistic = get_model()
    data = assign_cycles(load_stream(path_or_buffer))
    current = data["Motor current(mA)"].to_numpy(dtype=float)
    t_ms = data["_t"].astype("datetime64[ms]").astype("int64").to_numpy()
    bounds = cycle_bounds(data)

    z, cycles, ops = residuals_for(data, bounds.index.to_numpy(), templates)
    status = predict_status(cycle_scores(z, ops, templates, statistic), ops, thresholds)
    op_name = {"Open": "Opening", "Close": "Closing"}

    result, traces = [], []
    for k, cid in enumerate(cycles):
        rows = np.flatnonzero(data["cycle"].to_numpy() == cid)
        a, b = int(rows[0]), int(rows[-1]) + 1
        seg = list(range(a, b))
        decimated = _decimate(seg, [current[i] for i in seg], 150)
        result.append({
            "start_time": str(bounds.loc[cid, "start_time"]),
            "end_time": str(bounds.loc[cid, "end_time"]),
            "prediction": status[k],
            "mean_current_mA": float(current[a:b].mean()),
            "peak_current_mA": float(current[a:b].max()),
        })
        traces.append({
            "points": [[int(i), float(current[i]), int(t_ms[i])] for i, _ in decimated],
            "abnormal": status[k] == STATUS_ABNORMAL,
            "operation": op_name.get(ops[k], str(ops[k])),
            "t0": int(t_ms[a]),
            "duration_s": float(t_ms[b - 1] - t_ms[a]) / 1000.0,
        })
    return result, traces


def train(verbose: bool = True) -> dict:
    """Fit from the labelled stream, save the model, and report val IoU-F1."""
    from evaluate import door_iou_f1

    templates, thresholds, statistic = fit(str(DEFAULT_TRAIN), str(DEFAULT_TRAIN_ANSWER))
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"templates": templates, "thresholds": thresholds, "statistic": statistic}, MODEL_PATH)

    answer = pd.read_csv(DEFAULT_TRAIN_ANSWER)
    val = np.asarray(split_cycles(answer)["val"]) - 1
    pred = predict_stream(str(DEFAULT_TRAIN), templates, thresholds, statistic)
    score = float(door_iou_f1(pred.iloc[val].reset_index(drop=True),
                              answer.iloc[val].reset_index(drop=True)))
    if verbose:
        print(f"[door] validation-split IoU-weighted F1: {score:.4f} -> {MODEL_PATH}")
    return {"holdout_macro_f1": score}


if __name__ == "__main__":
    train()
