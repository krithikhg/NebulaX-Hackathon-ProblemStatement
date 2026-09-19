"""Loading and per-car summary statistics for the PS3 ACV cases.

The ACV data is a *relative* problem: one row per 30 s timestamp, the same set
of parameters repeated for each of the 8 cars, and exactly one car per file
carrying a refrigerant leak. Every car on a file shares the same train, the same
ambient conditions and the same wall-clock window, so the other seven cars are
the control group for the one under test. Everything here is therefore built to
produce a **tidy (case, car, parameter) table** that can be compared within a
file, rather than absolute per-car numbers that would only be comparable if all
files were recorded under the same weather.

Three schema facts drive the design, all confirmed against the files on disk:

1. **The parameter set is not fixed.** Five of the six Train cases and the Test
   case carry 8 parameters per car (67 columns); `acv_case_04.xlsx` carries 63
   (483 columns) of much richer telemetry. Column *order* varies too -- most
   files start at `Car 08`, cases 05/06 start at `Car 04`. Everything here
   parses the file's own headers.
2. **Several parameters are the same measurement under a different name.**
   Cases 05/06 call the outdoor reading `Outside Temperature Sensor Reading`;
   every other file calls it `Outdoor Average Temperature`. Case 04 goes
   further and renames every one of the 8 shared parameters: its
   `Passenger Cabin Temperature Detected Value` is the same measurement as
   `Indoor Average Temperature`, `Fresh Air Temperature Detected Value` is
   `Outdoor Average Temperature`, `Target Temperature Value` is
   `ACV Control Temperature (Cooling)` (case 04 carries one active setpoint,
   not a separate cooling/heating pair -- see below), `ACV Control Mode` is
   `ACV Setting Mode`, and `Load Shedding` is `Load Halved` -- confirmed by
   matching value vocabularies (e.g. case 04's `ACV Control Mode` only ever
   reads `Centralized Control`, exactly the value every other file's
   `ACV Setting Mode` uses). `ACV Running Mode` itself needs no alias -- case
   04 uses that literal name and the same state vocabulary. `PARAM_ALIASES`
   folds all of this together, which is what makes 6 of the 8 shared
   parameters genuinely shared across all 7 files rather than 6 of 7 files.
   Case 04 has no equivalent of `ACV Information Valid`, and only one target
   temperature rather than separate cooling/heating setpoints -- those two
   stay legitimately absent for that file (NaN) rather than being invented.
3. **Mixed dtypes.** Four of the shared parameters are numeric (temperatures)
   and four are categorical/binary flags (modes, validity, load-halved), and
   they want different summaries -- means and spreads for the first, duty cycles
   and transition counts for the second.

Loading is cached to Parquet under `.cache/` (see `load_case`), because
`.xlsx` parsing is the actual bottleneck here -- case 04's 483 columns alone
take well over a minute through `openpyxl`, against a fraction of a second
once mirrored.
"""

import pathlib
import sys

import numpy as np
import pandas as pd

# ps3_data lives in the Door folder and already knows where 02_Datasets sits and
# how to parse `Car <NN> - <parameter>` headers; reuse it rather than keeping a
# second copy of the paths around to drift out of sync.
_PS3_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PS3_ROOT / "door"))

import ps3_data  # noqa: E402

ACV_DIR = ps3_data.ACV_DIR

TRAIN_CASES = [f"acv_case_{i:02d}.xlsx" for i in range(1, 7)]
TEST_CASES = ["acv_test_case.xlsx"]
ALL_CASES = TRAIN_CASES + TEST_CASES

# Every raw column name that is the same measurement as one of SHARED_PARAMS
# under a different name -- see the module docstring's schema note for how
# each of these was confirmed (matching units/ranges, or matching value
# vocabulary for the categorical ones).
PARAM_ALIASES = {
    "Outside Temperature Sensor Reading": "Outdoor Average Temperature",
    "Passenger Cabin Temperature Detected Value": "Indoor Average Temperature",
    "Fresh Air Temperature Detected Value": "Outdoor Average Temperature",
    "Target Temperature Value": "ACV Control Temperature (Cooling)",
    "ACV Control Mode": "ACV Setting Mode",
    "Load Shedding": "Load Halved",
}

# The 8 parameters present in every case file once aliases are applied (except
# `ACV Control Temperature (Heating)` and `ACV Information Valid`, which case
# 04 has no equivalent of -- see the module docstring), split by dtype since
# they want different summary statistics.
NUMERIC_PARAMS = [
    "Indoor Average Temperature",
    "Outdoor Average Temperature",
    "ACV Control Temperature (Cooling)",
    "ACV Control Temperature (Heating)",
]
CATEGORICAL_PARAMS = [
    "ACV Setting Mode",
    "ACV Running Mode",
    "ACV Information Valid",
    "Load Halved",
]
SHARED_PARAMS = NUMERIC_PARAMS + CATEGORICAL_PARAMS

SAMPLE_PERIOD_S = 30.0

_CACHE = {}

# `.xlsx` parsing (openpyxl) is the actual bottleneck here, not pandas --
# ~20s for a 67-column file and over a minute for case 04's 483 columns, and
# every fresh Python process pays it again since `_CACHE` is in-memory only.
# Each case is parsed once and mirrored to Parquet here; later calls in any
# process load the Parquet copy (column-oriented, no formula/style parsing)
# in a fraction of a second, and a stale copy re-derives itself automatically
# if the source `.xlsx` is ever replaced with a newer one.
PARQUET_CACHE_DIR = pathlib.Path(__file__).resolve().parent / ".cache"


def canonical_param(param):
    return PARAM_ALIASES.get(param, param)


def _parquet_cache_path(case, treat_zero_as_missing):
    # The zero-scrub toggle (see ps3_data.ACV_TREAT_ZERO_AS_MISSING) changes
    # the cell values themselves, so it gets its own cache file rather than
    # sharing one -- flipping the toggle must not silently serve a Parquet
    # mirror built under the other setting.
    suffix = "" if treat_zero_as_missing else "__rawzero"
    return PARQUET_CACHE_DIR / (pathlib.Path(case).stem + suffix + ".parquet")


def load_case(case, use_cache=True, treat_zero_as_missing=None):
    """One case file, with `Time` parsed. Columns are left exactly as on disk.

    `treat_zero_as_missing` is passed straight through to
    `ps3_data.load_acv` (defaults to its module-level toggle,
    `ps3_data.ACV_TREAT_ZERO_AS_MISSING`) -- see that toggle's docstring for
    why literal 0s in the temperature/setpoint columns are scrubbed to NaN by
    default. The in-memory and Parquet caches are both keyed on its resolved
    value, so calling with the toggle flipped never returns the other
    setting's cached data.

    Set `use_cache=False` to force a re-read straight from the source
    `.xlsx` (e.g. to confirm the Parquet mirror still matches it).
    """
    if treat_zero_as_missing is None:
        treat_zero_as_missing = ps3_data.ACV_TREAT_ZERO_AS_MISSING
    cache_key = (case, treat_zero_as_missing)

    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    xlsx_path = ps3_data._resolve(case, ACV_DIR)
    cache_path = _parquet_cache_path(case, treat_zero_as_missing)

    if use_cache and cache_path.exists() and \
            cache_path.stat().st_mtime >= xlsx_path.stat().st_mtime:
        df = pd.read_parquet(cache_path)
    else:
        df = ps3_data.load_acv(case, treat_zero_as_missing=treat_zero_as_missing)
        if use_cache:
            PARQUET_CACHE_DIR.mkdir(exist_ok=True)
            df.to_parquet(cache_path)

    if use_cache:
        _CACHE[cache_key] = df
    return df


def car_columns(df):
    """{car_id: [its columns]} straight from the file's own headers."""
    return ps3_data.acv_car_columns(df)


def load_labels():
    """`Train_Labels.csv`, car ids kept as the zero-padded strings a submission
    must use.
    """
    return ps3_data.load_acv_labels()


def label_map():
    """{filename: faulty car id} for the six labelled Train cases."""
    labels = load_labels()
    return dict(zip(labels["filename"], labels["faulty_car"]))


def to_long(case, params=None, use_cache=True):
    """A case reshaped to tidy long form: one row per (Time, car, parameter).

    This is the shape every plot and statistic here consumes, because it turns
    "compare car 03 against the other seven at the same instant" into a groupby
    rather than a column-name construction. Parameter names are canonicalised
    (see `PARAM_ALIASES`), and `params=None` keeps only the 7 shared ones so all
    seven files line up; pass an explicit list to reach case 04's extra
    telemetry.

    `value` is numeric-coerced only for `NUMERIC_PARAMS` -- everything else
    (the 4 categorical/flag parameters, and any of case 04's extra columns
    that are not in `NUMERIC_PARAMS`) is kept as its raw string, in a `raw`
    column, and `value` is left NaN for those rows rather than silently wiped
    out by `pd.to_numeric`. Numeric rows get `raw` too (the original cell,
    unparsed), so nothing is lost either way.
    """
    if params is None:
        params = SHARED_PARAMS
    params = set(params)

    df = load_case(case, use_cache=use_cache)
    cars = car_columns(df)

    frames = []
    for car, cols in cars.items():
        for col in cols:
            param = canonical_param(col.split(" - ", 1)[1])
            if param not in params:
                continue
            raw = df[col]
            is_numeric = param in NUMERIC_PARAMS
            frames.append(pd.DataFrame({
                "case": case,
                "Time": df["Time"],
                "car": car,
                "param": param,
                "value": pd.to_numeric(raw, errors="coerce") if is_numeric else np.nan,
                "raw": raw.astype(object).where(raw.notna(), None),
            }))

    long = pd.concat(frames, ignore_index=True)
    return long.sort_values(["param", "car", "Time"], ignore_index=True)


def _loo_median_matrix(matrix):
    """Leave-one-out median of each row of a (timestamps x cars) matrix.

    Vectorised across timestamps and looped over the 8 cars, rather than the
    other way round: a per-timestamp `groupby.apply` runs one Python call per
    row and takes minutes on a full case, while this is 8 array medians.
    """
    n_cars = matrix.shape[1]
    out = np.empty_like(matrix, dtype=float)
    for i in range(n_cars):
        out[:, i] = np.nanmedian(np.delete(matrix, i, axis=1), axis=1)
    return out


def with_train_reference(long):
    """Add the within-file, within-timestamp reference and residual columns.

    For every (parameter, timestamp), `ref` is the **median across the other
    cars** -- leave-one-out, so a genuinely anomalous car cannot drag its own
    baseline toward itself -- and `resid` is that car's deviation from it. This
    is the core transform: it strips out ambient temperature, time of day and
    the train-wide control schedule, all of which are shared by all 8 cars and
    none of which say anything about which car is leaking. `ref_mean` is the
    cheaper leave-one-out mean, kept for comparison.
    """
    long = long.copy()
    g = long.groupby(["param", "Time"])["value"]
    total = g.transform("sum")
    count = g.transform("count")
    long["ref_mean"] = (total - long["value"]) / (count - 1)

    # Per parameter, reshape to (timestamps x cars) so the leave-one-out median
    # is a handful of array operations, then melt the result back onto `long`.
    ref = pd.Series(np.nan, index=long.index, dtype=float)
    for param, sub in long.groupby("param", sort=False):
        wide = sub.pivot_table(index="Time", columns="car", values="value",
                               aggfunc="first")
        loo = pd.DataFrame(_loo_median_matrix(wide.to_numpy(dtype=float)),
                           index=wide.index, columns=wide.columns)
        stacked = loo.stack().rename("ref").reset_index()
        merged = sub.reset_index().merge(stacked, on=["Time", "car"], how="left")
        ref.loc[merged["index"].to_numpy()] = merged["ref"].to_numpy()

    long["ref"] = ref
    long["resid"] = long["value"] - long["ref"]
    return long


def with_majority_reference(long):
    """Add the categorical analogue of `with_train_reference`: for every
    (parameter, timestamp), `majority_raw` is the value most of the 8 cars
    report and `is_majority` flags whether this car agrees with it.

    This is what a residual is for a numeric parameter -- the train-wide
    command is shared (see the module docstring on centralized control), so a
    car sitting outside the majority state is the categorical equivalent of a
    car sitting off the sibling median. Ties keep pandas's own tie-break
    (lowest value in sort order); rows where every car is missing get
    `majority_raw = None` and `is_majority = NaN`.

    Vectorised the same way as `with_train_reference`: pivot each parameter to
    (timestamps x cars) and let `DataFrame.mode(axis=1)` do the row-wise mode
    in one C-level pass, instead of a Python-level `groupby.transform` calling
    `pd.Series.mode()` once per (parameter, timestamp) group -- which is what
    made this take minutes on the larger case files.
    """
    long = long.copy()
    maj = pd.Series(None, index=long.index, dtype=object)

    for param, sub in long.groupby("param", sort=False):
        wide = sub.pivot_table(index="Time", columns="car", values="raw",
                               aggfunc="first")
        mode_col = wide.mode(axis=1)[0] if not wide.empty else pd.Series(dtype=object)
        stacked = mode_col.reindex(wide.index).rename("majority_raw")
        merged = sub.reset_index().merge(
            stacked.rename_axis("Time").reset_index(), on="Time", how="left")
        maj.loc[merged["index"].to_numpy()] = merged["majority_raw"].to_numpy()

    long["majority_raw"] = maj
    long["is_majority"] = np.where(
        long["raw"].isna(), np.nan, (long["raw"] == maj).astype(float))
    return long


def car_stats(case, use_cache=True):
    """One row per (case, car, parameter) with that car's summary statistics.

    Numeric and categorical parameters both go through this; statistics that do
    not really apply to a given parameter come back anyway rather than being
    split across two tables, so one frame can be sorted and ranked uniformly.

    Raw columns describe the car on its own terms: `mean`/`std`/`p05`..`p95`
    for numeric parameters; `n_unique`/`n_switch` (value changes between
    consecutive samples) for both. Residual columns (`resid_*`) describe a
    numeric car relative to the other seven cars at the same instant, which is
    the comparison that survives differences in weather and schedule between
    files; `frac_majority` is the categorical equivalent -- the fraction of
    time this car agrees with the train-wide majority state (see
    `with_majority_reference`).
    """
    long = with_train_reference(to_long(case, use_cache=use_cache))
    long = with_majority_reference(long)

    rows = []
    for (param, car), sub in long.groupby(["param", "car"]):
        is_numeric = param in NUMERIC_PARAMS
        v = sub["value"].to_numpy(dtype=float)
        r = sub["resid"].to_numpy(dtype=float)
        raw = sub["raw"].to_numpy(dtype=object)
        is_maj = sub["is_majority"].to_numpy(dtype=float)
        ok = np.isfinite(v)
        rok = np.isfinite(r)
        raw_ok = np.array([x is not None for x in raw])
        maj_ok = np.isfinite(is_maj)

        row = {
            "case": case,
            "car": car,
            "param": param,
            "kind": "numeric" if is_numeric else "categorical",
            "n": len(sub),
            "n_missing": int((~ok).sum()) if is_numeric else int((~raw_ok).sum()),
        }
        if is_numeric:
            row.update({
                "mean": np.nanmean(v) if ok.any() else np.nan,
                "std": np.nanstd(v) if ok.any() else np.nan,
                "p05": np.nanpercentile(v[ok], 5) if ok.any() else np.nan,
                "p50": np.nanmedian(v) if ok.any() else np.nan,
                "p95": np.nanpercentile(v[ok], 95) if ok.any() else np.nan,
                "n_unique": int(pd.Series(v[ok]).nunique()) if ok.any() else 0,
                "duty": float(np.mean(v[ok] > np.nanmedian(v))) if ok.any() else np.nan,
                "n_switch": int((np.diff(v[ok]) != 0).sum()) if ok.sum() > 1 else 0,
                "resid_mean": np.nanmean(r) if rok.any() else np.nan,
                "resid_std": np.nanstd(r) if rok.any() else np.nan,
                "resid_absmean": np.nanmean(np.abs(r[rok])) if rok.any() else np.nan,
                "resid_p95": np.nanpercentile(np.abs(r[rok]), 95) if rok.any() else np.nan,
                "frac_majority": np.nan,
            })
        else:
            raw_seq = raw[raw_ok]
            row.update({
                "mean": np.nan, "std": np.nan, "p05": np.nan, "p50": np.nan,
                "p95": np.nan,
                "n_unique": int(pd.Series(raw_seq).nunique()) if raw_ok.any() else 0,
                "duty": np.nan,
                "n_switch": int((raw_seq[1:] != raw_seq[:-1]).sum()) if len(raw_seq) > 1 else 0,
                "resid_mean": np.nan, "resid_std": np.nan,
                "resid_absmean": np.nan, "resid_p95": np.nan,
                "frac_majority": float(np.mean(is_maj[maj_ok])) if maj_ok.any() else np.nan,
            })
        rows.append(row)

    return pd.DataFrame(rows)


# Running-mode states in which the compressor is actively commanded to cool.
# `Stopped` is the alternate spelling for `Stop` used interchangeably across
# files (see the module docstring); it never co-occurs with the cooling states
# so including it here is a no-op, kept for completeness.
COOLING_MODES = {"Half Cooling", "Full Cooling", "Automatic Cooling"}

# The two numeric + one categorical parameter this feature needs, in their
# canonical (post-alias) names -- all present in every case file, including
# 04, once PARAM_ALIASES is applied (see the module docstring). Checked
# defensively rather than assumed, so a genuinely incompatible file gets a
# clean all-NaN row instead of a KeyError.
_COOLING_GAP_REQUIRED = {
    "Indoor Average Temperature", "ACV Control Temperature (Cooling)",
    "ACV Running Mode",
}


def cooling_gap_stats(case, use_cache=True):
    """Per-car `Indoor Average Temperature - ACV Control Temperature (Cooling)`,
    restricted to rows where that car's own `ACV Running Mode` is a cooling
    state (`COOLING_MODES`).

    This is the strongest single ranking rule found in exploration (see the
    notebook, §5): because cooling is commanded train-wide (centralized
    control, ~89% of timestamps share one mode across all 8 cars -- see the
    module docstring), every car gets told to cool at the same moments as its
    siblings, so this gap measures how well *this car's own compressor*
    executes an identical, externally-imposed command -- exactly what a
    refrigerant leak degrades. It is not folded into `car_stats` because it
    needs both a numeric and a categorical parameter of the same car at once,
    which the one-parameter-at-a-time shape of `car_stats` cannot express.

    Works on every case file, including `acv_case_04.xlsx`'s different naming
    scheme, once `PARAM_ALIASES` maps its columns onto these three canonical
    names (see the module docstring). Returns one all-NaN row per car for a
    file that still doesn't carry them after aliasing, rather than raising, so
    a sweep over every case does not need its own special case.
    """
    df = load_case(case, use_cache=use_cache)
    cars = car_columns(df)

    rows = []
    for car, cols in cars.items():
        by_param = {canonical_param(c.split(" - ", 1)[1]): df[c] for c in cols}
        if not _COOLING_GAP_REQUIRED.issubset(by_param):
            rows.append({"case": case, "car": car, "n_cooling": 0,
                        "cooling_gap_mean": np.nan, "cooling_gap_std": np.nan,
                        "cooling_gap_max": np.nan})
            continue
        indoor = pd.to_numeric(by_param["Indoor Average Temperature"], errors="coerce")
        ctrl = pd.to_numeric(by_param["ACV Control Temperature (Cooling)"],
                             errors="coerce")
        mode = by_param["ACV Running Mode"]
        cooling = mode.isin(COOLING_MODES)
        gap = (indoor - ctrl)[cooling]
        rows.append({
            "case": case, "car": car,
            "n_cooling": int(cooling.sum()),
            "cooling_gap_mean": gap.mean(),
            "cooling_gap_std": gap.std(),
            "cooling_gap_max": gap.max(),
        })
    return pd.DataFrame(rows)


def all_cooling_gap_stats(cases=None, use_cache=True):
    """`cooling_gap_stats` stacked over cases, with ground truth joined on --
    the `car_stats`-shaped companion table for `cooling_gap_stats`, ready for
    `rank_by`/`rank_sweep` (pass `param=None`-style by calling `rank_by`
    directly on this table's `cooling_gap_mean` column).
    """
    if cases is None:
        cases = ALL_CASES
    stats = pd.concat([cooling_gap_stats(c, use_cache=use_cache) for c in cases],
                      ignore_index=True)
    labels = label_map()
    stats["param"] = "cooling_gap"
    stats["faulty_car"] = stats["case"].map(labels)
    stats["is_faulty"] = np.where(
        stats["faulty_car"].isna(), np.nan,
        (stats["car"] == stats["faulty_car"]).astype(float))
    return stats


def all_car_stats(cases=None, use_cache=True):
    """`car_stats` stacked over every case, with ground truth joined on.

    `is_faulty` is 1 for the labelled faulty car, 0 for its seven siblings, and
    NaN for the Test case (no published label).
    """
    if cases is None:
        cases = ALL_CASES
    stats = pd.concat([car_stats(c, use_cache=use_cache) for c in cases],
                      ignore_index=True)
    labels = label_map()
    stats["faulty_car"] = stats["case"].map(labels)
    stats["is_faulty"] = np.where(
        stats["faulty_car"].isna(), np.nan,
        (stats["car"] == stats["faulty_car"]).astype(float))
    return stats


def case_shapes(cases=None):
    """Per-file structural summary: rows, columns, cars, duration, parameters.

    Cheap to read and the first thing to look at, since the schema differences
    between files are the main trap in this dataset.
    """
    if cases is None:
        cases = ALL_CASES
    rows = []
    for case in cases:
        df = load_case(case)
        cars = car_columns(df)
        params = sorted({canonical_param(c.split(" - ", 1)[1])
                         for cols in cars.values() for c in cols})
        t = pd.to_datetime(df["Time"])
        rows.append({
            "case": case,
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "n_cars": len(cars),
            "params_per_car": len(params),
            "first_car_col": next(iter(cars)),
            "start": t.min(),
            "end": t.max(),
            "duration_h": (t.max() - t.min()).total_seconds() / 3600,
            "median_dt_s": t.diff().dt.total_seconds().median(),
            "shared_params_present": sum(p in params for p in SHARED_PARAMS),
        })
    return pd.DataFrame(rows)


def rank_by(stats, param, column, ascending=False):
    """Rank cars within each case by one statistic, and score that ranking the
    way the organisers will.

    Returns one row per case with the ordered `ranked_cars` string, the rank the
    true faulty car landed at, and the official linear rank-decay score
    `(n - (r - 1)) / n`. This is the harness for "is this statistic worth
    anything?" -- a useless feature averages about 0.56 (the expected score of a
    random 8-car ranking), so anything near that is noise.
    """
    sel = stats[stats["param"] == param].dropna(subset=[column])
    rows = []
    for case, sub in sel.groupby("case"):
        sub = sub.sort_values(column, ascending=ascending)
        order = list(sub["car"])
        faulty = sub["faulty_car"].iloc[0]
        n = len(order)
        if isinstance(faulty, str) and faulty in order:
            r = order.index(faulty) + 1
            score = (n - (r - 1)) / n
        else:
            r, score = np.nan, np.nan
        rows.append({
            "case": case, "param": param, "stat": column,
            "ranked_cars": "|".join(order),
            "faulty_car": faulty, "rank": r, "score": score,
        })
    return pd.DataFrame(rows)


def rank_sweep(stats, columns=None, params=None):
    """`rank_by` over every (parameter, statistic, direction) combination.

    A wide net, deliberately: with only six labelled cases there is no room for
    a careful one-feature-at-a-time search, so the sweep lays out every simple
    ranking rule and its mean score side by side. Returns `(summary, per_case)`.
    Treat the top of the summary as a shortlist to scrutinise, not as a result
    -- six cases is few enough that a feature can top it on luck alone.
    """
    if columns is None:
        columns = ["resid_mean", "resid_absmean", "resid_std", "resid_p95",
                   "std", "n_switch", "n_unique", "duty", "mean", "frac_majority"]
    if params is None:
        params = SHARED_PARAMS

    out = []
    for param in params:
        for column in columns:
            for ascending in (False, True):
                r = rank_by(stats, param, column, ascending=ascending)
                if r.empty:
                    continue
                r["ascending"] = ascending
                out.append(r)
    if not out:
        return pd.DataFrame(), pd.DataFrame()

    per_case = pd.concat(out, ignore_index=True)
    summary = (per_case.dropna(subset=["score"])
               .groupby(["param", "stat", "ascending"])
               .agg(mean_score=("score", "mean"),
                    n_top1=("rank", lambda s: int((s == 1).sum())),
                    mean_rank=("rank", "mean"),
                    n_cases=("score", "size"))
               .reset_index()
               .sort_values("mean_score", ascending=False, ignore_index=True))
    return summary, per_case
