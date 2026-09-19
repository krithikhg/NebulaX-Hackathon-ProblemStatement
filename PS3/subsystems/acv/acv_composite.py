"""Peer z-score index over per-car fault indicators (ACV).

Two indicators, each turned into a leave-one-out robust z-score against the
car's 7 peers (positive = more suspicious), then averaged with equal weights.
Higher mean z = more suspicious.

  hot    mean Indoor Average Temperature                  (higher = more suspicious)
  gap    mean(Indoor - Cooling setpoint) while in a
         cooling running mode                             (higher = more suspicious)

Two more indicators were dropped. `maj` (fraction of time the car's mode equals
the train-wide majority mode) ranks the faulty car mid-table (not near the top) in case 06 once
Invalid rows are excluded. `inval` (count of ACV Information Valid == "Invalid")
fires on five cars in the Test file, unlike the labelled files where it was
exclusive to the faulty car. `n_invalid` and `frac_majority` are still computed
by `indicators` (used by `overlap_table`).

`acv_composite_model.ipynb` is the walk-through; run as a script this module
prints the artifact checks (`verify`) that the notebook shows as tables.
"""

import numpy as np
import pandas as pd

import acv_features as af

# name -> (column in the indicator table, ascending flag; ascending=True means a
# LOWER value is more suspicious, so its z-score is negated)
INDICATORS = {
    "hot":   ("indoor_mean",      False),
    "gap":   ("cooling_gap_mean", False),
}

VARIANTS = {
    "A raw zeros":       dict(treat_zero_as_missing=False),
    "B zeros->NaN":      dict(treat_zero_as_missing=True),
    "C clean rows only": dict(treat_zero_as_missing=True, clean_rows_only=True),
}


def _wide(case, treat_zero_as_missing):
    """Per-parameter (timestamps x cars) frames, canonical parameter names."""
    df = af.load_case(case, treat_zero_as_missing=treat_zero_as_missing)
    per = {car: {af.canonical_param(c.split(" - ", 1)[1]): df[c] for c in cols}
           for car, cols in af.car_columns(df).items()}
    first = next(iter(per.values()))

    def grab(param, numeric):
        if param not in first:
            return None
        return pd.DataFrame({
            car: pd.to_numeric(p[param], errors="coerce") if numeric
            else p[param].astype(object)
            for car, p in per.items()})

    return {
        "indoor": grab("Indoor Average Temperature", True),
        "ctrl":   grab("ACV Control Temperature (Cooling)", True),
        "mode":   grab("ACV Running Mode", False),
        "valid":  grab("ACV Information Valid", False),
    }


def _row_mode(frame):
    """Row-wise most common value, ties -> lowest sorted value (as pandas'
    own `mode` does), all-missing rows -> NaN. Vectorised: `DataFrame.mode
    (axis=1)` is a Python call per row and slow on the 22k-row case."""
    vals = frame.to_numpy(dtype=object)
    if vals.size == 0:
        return pd.Series(np.nan, index=frame.index, dtype=object)
    codes, uniques = pd.factorize(vals.ravel(), sort=True)
    codes = codes.reshape(vals.shape)
    counts = np.stack([(codes == j).sum(axis=1) for j in range(len(uniques))], axis=1)
    best = counts.argmax(axis=1)
    out = np.where(counts.sum(axis=1) > 0, np.array(uniques, dtype=object)[best], np.nan)
    return pd.Series(out, index=frame.index, dtype=object)


def indicators(case, treat_zero_as_missing=True, clean_rows_only=False, block=None):
    """One row per car with the four raw indicator values.

    `clean_rows_only` keeps only timestamps where every car that has data is
    fully observed -- non-blank, non-zero, not flagged Invalid -- so nothing
    downstream depends on how a missing or glitched reading is handled.
    `block=(i, k)` keeps the i-th of k contiguous row blocks (persistence check).
    `n_invalid` is always counted on the block's raw rows, never on the
    clean-row subset, because the Invalid rows *are* the indicator.
    """
    w = _wide(case, treat_zero_as_missing)
    indoor, ctrl, mode, valid = w["indoor"], w["ctrl"], w["mode"], w["valid"]
    cars = list(indoor.columns)

    if block is not None:
        i, k = block
        a, b = len(indoor) * i // k, len(indoor) * (i + 1) // k
        indoor, ctrl, mode = indoor.iloc[a:b], ctrl.iloc[a:b], mode.iloc[a:b]
        valid = valid.iloc[a:b] if valid is not None else None

    if valid is not None:
        # An Invalid row already counts in `inval`; keep it out of `maj` too so
        # one glitch is not credited twice.
        mode = mode.where(valid != "Invalid")

    active = [c for c in cars if indoor[c].notna().any()]
    n_invalid = ((valid == "Invalid").sum() if valid is not None
                 else pd.Series(np.nan, index=cars))

    if clean_rows_only:
        ok = pd.Series(True, index=indoor.index)
        for frame in (indoor, ctrl, mode):
            ok &= frame[active].notna().all(axis=1)
        ok &= (indoor[active] != 0).all(axis=1) & (ctrl[active] != 0).all(axis=1)
        if valid is not None:
            ok &= (valid[active] != "Invalid").all(axis=1)
        indoor, ctrl, mode = indoor[ok], ctrl[ok], mode[ok]

    maj = _row_mode(mode[active])
    agree = mode[active].eq(maj, axis=0).astype(float).where(mode[active].notna())
    gap = (indoor - ctrl).where(mode.isin(af.COOLING_MODES))

    out = pd.DataFrame({
        "indoor_mean":      indoor.mean(),
        "n_invalid":        n_invalid,
        "cooling_gap_mean": gap.mean(),
        "frac_majority":    agree.mean().reindex(cars),
    })
    out.loc[[c for c in cars if c not in active], :] = np.nan
    return out


# Smallest peer spread a z-score is divided by (in the indicator's own units,
# degrees C). The sensor steps in 0.5 C, so peers that agree to well under a
# step (a short window, a near-constant indicator) would otherwise give a huge z
# from a difference that is only quantisation. 0.02 leaves every whole-file
# score untouched -- it only bites on short windows such as the quarters.
MIN_SCALE = 0.02


def peer_z(ind, use=tuple(INDICATORS)):
    """Leave-one-out robust z-score of each car against its 7 peers.

    z = (x - median(peers)) / (1.4826 * MAD(peers)), signed so that a positive z
    is always "more suspicious". Unlike a rank it says *how far* a car is from
    the rest. A peer group with zero MAD falls back to its standard deviation;
    the scale is never allowed below `MIN_SCALE`. Cars with no value get NaN.
    """
    z = pd.DataFrame(index=ind.index, columns=list(use), dtype=float)
    for k in use:
        col, ascending = INDICATORS[k]
        sign = 1.0 if not ascending else -1.0
        x = ind[col]
        for car in ind.index:
            if pd.isna(x[car]):
                continue
            peers = x.drop(car).dropna()
            if peers.empty:
                continue
            scale = 1.4826 * (peers - peers.median()).abs().median()
            if scale < 1e-9:
                scale = peers.std(ddof=0)
            z.loc[car, k] = sign * (x[car] - peers.median()) / max(scale, MIN_SCALE)
    return z, z.mean(axis=1)


def rank_case(case, **kw):
    """`(indicators, per-indicator z, mean z)` for one case file. Higher mean z
    = more suspicious."""
    ind = indicators(case, **kw)
    z, mz = peer_z(ind)
    return ind, z, mz


def ranked_cars(comp):
    """The `ranked_cars` submission string: most to least suspicious.

    `comp` is the mean z per car (higher = more suspicious). Cars with no score
    (no data at all) go last. Exact ties are ordered by
    car id, which is arbitrary -- callers who care should look at the tie group
    (`tie_aware` reports its size) rather than read meaning into that order.
    """
    comp = comp.sort_index().round(9)
    order = comp.dropna().sort_values(ascending=False, kind="stable")
    rest = [c for c in comp.index if c not in order.index]
    return "|".join(list(order.index) + rest)


def tie_aware(comp, faulty, n_total=8):
    """Where the faulty car lands when ties are NOT broken in its favour.

    `comp` is the mean z per car (higher = more suspicious). Sorting a tied
    score and taking the first hit silently breaks ties by
    row order (car id), which credits a rank the data never earned. Here a tie
    group shares its positions, so the faulty car's expected rank is the group
    midpoint. `n_total` is every car in the file: cars with no data rank last.
    """
    c = comp.dropna().round(9)
    f = c[faulty]
    better, ties = int((c > f).sum()), int((c == f).sum()) - 1
    r = better + 1 + ties / 2
    return dict(best=better + 1, worst=better + 1 + ties, ties=ties,
                score=(n_total - (r - 1)) / n_total)


def case_summary(cases=None):
    """One row per case: the ranking, and (when labelled) the faulty car's
    tie-aware position and expected score."""
    labels = af.label_map()
    rows = []
    for case in af.TRAIN_CASES if cases is None else cases:
        _, _, comp = rank_case(case)
        row = {"case": case, "ranked_cars": ranked_cars(comp)}
        if case in labels:
            t = tie_aware(comp, labels[case], n_total=len(comp))
            row.update(faulty=labels[case], best=t["best"], worst=t["worst"],
                       score=t["score"])
        rows.append(row)
    return pd.DataFrame(rows)


def variant_tables():
    """Mean-z index under the three data-handling variants, and the faulty car's
    z on each indicator under variants B and C.

    Returns `(composite_table, indicator_table)`. In the first, each cell is
    "best-worst rank (expected score)"; in the second, z near 0 means the
    indicator carried no information for that case and nan means it wasn't
    measured.
    """
    labels = af.label_map()
    t1, t2 = [], []
    for case in af.TRAIN_CASES:
        f = labels[case]
        r1, r2 = {"case": case[:-5], "faulty": f}, {"case": case[:-5], "faulty": f}
        by_variant = {}
        for vname, kw in VARIANTS.items():
            z, comp = peer_z(indicators(case, **kw))
            t = tie_aware(comp, f)
            r1[vname] = f'{t["best"]}-{t["worst"]} ({t["score"]:.3f})'
            by_variant[vname] = z.loc[f]
        for k in INDICATORS:
            r2[k] = f'{by_variant["B zeros->NaN"][k]:.2f} / {by_variant["C clean rows only"][k]:.2f}'
        t1.append(r1)
        t2.append(r2)
    return pd.DataFrame(t1), pd.DataFrame(t2)


def overlap_table():
    """For the faulty car in each file that has an Invalid flag: how many rows
    are Invalid, how many disagree with the majority running mode, how many are
    both, and what running mode the car reports on its Invalid rows."""
    labels = af.label_map()
    rows = []
    for case in af.TRAIN_CASES:
        w = _wide(case, True)
        if w["valid"] is None:
            continue
        f = labels[case]
        active = [c for c in w["indoor"].columns if w["indoor"][c].notna().any()]
        maj = _row_mode(w["mode"][active])
        inv = w["valid"][f] == "Invalid"
        mism = (w["mode"][f] != maj) & w["mode"][f].notna()
        seen = w["mode"][f][inv].value_counts(dropna=False)
        rows.append({"case": case[:-5], "faulty": f,
                     "invalid_rows": int(inv.sum()), "mismatch_rows": int(mism.sum()),
                     "both": int((inv & mism).sum()),
                     "mode_on_invalid_rows": {str(k): int(v) for k, v in seen.items()}})
    return pd.DataFrame(rows)


def persistence_table(indicators_to_check=("hot", "gap")):
    """The faulty car's z-score on an indicator in each quarter of the file
    (variant B). A chronic fault should stay high throughout."""
    labels = af.label_map()
    rows = []
    for case in af.TRAIN_CASES:
        f = labels[case]
        per_q = [peer_z(indicators(case, block=(b, 4)))[0].loc[f] for b in range(4)]
        for k in indicators_to_check:
            rows.append({"case": case[:-5], "indicator": k,
                         **{f"Q{b + 1}": per_q[b][k] for b in range(4)}})
    return pd.DataFrame(rows)


def verify():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    t_comp, t_ind = variant_tables()

    print("== 1. Composite, tie-aware, under three data-handling variants ==")
    print("   cell = best-worst rank of the faulty car (expected score).")
    print("   A: zeros kept as 0 | B: zeros -> NaN | C: only rows where every car is clean\n")
    print(t_comp.to_string(index=False))
    print("\n== 2. Faulty car's z per indicator, variant B / variant C ==\n")
    print(t_ind.to_string(index=False))
    print("\n== 3. Do the Invalid rows and the majority-mismatch rows overlap? (faulty car) ==\n")
    print(overlap_table().to_string(index=False))
    print("\n== 4. Persistence: faulty car's z in each quarter of the file ==\n")
    print(persistence_table().to_string(index=False))


if __name__ == "__main__":
    verify()
