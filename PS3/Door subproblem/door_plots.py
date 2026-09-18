"""Plotting helpers for the PS3 Door stream.

Shaped around what this dataset actually is: one physical door, one continuous
recording, and ~110 short cycles to compare against each other. That makes the
useful views different from the per-asset fleet plots in `archive/eda_plots.py`
(which were for the synthetic multi-door table):

- `plot_timeline` -- where the cycles sit in the stream, and how long the gaps
  between them are. Confirms segmentation at a glance.
- `plot_cycle` -- every channel of one cycle, stacked. The "read one example
  properly" plot.
- `plot_cycle_overlay` -- one channel, every cycle drawn on a common elapsed-time
  axis and coloured by status. This is the plot that shows whether Normal and
  Abnormal resistance separate at all, and where in the cycle they diverge.
- `plot_cycle_grid` -- small multiples, one panel per cycle, for reading
  individual shapes without occlusion.
- `plot_status_distributions` -- per-cycle summary statistics as Normal-vs-
  Abnormal histograms, for spotting features worth handing a classifier.

Open and Close cycles have fundamentally different waveforms, so every
cycle-comparison plot takes `operation=` to look at one of them at a time.
Comparing statuses while mixing operations mostly shows the operation.

Elapsed time within a cycle is computed from row position at the nominal 20 ms
sampling rate, not from the timestamps -- the timestamps jitter by up to ~0.9 s
inside a cycle, which would visibly smear an overlay.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SAMPLE_PERIOD_S = 0.02

STATUS_COLORS = {"Normal": "tab:blue", "Abnormal resistance": "tab:red"}


def _cycle_frame(stream, segment):
    """One segment's rows, in stream order, with an `elapsed` column."""
    sub = stream[stream["segment"] == segment]
    sub = sub.assign(elapsed=np.arange(len(sub)) * SAMPLE_PERIOD_S)
    return sub


def _select_segments(segments, operation=None, status=None, limit=None):
    sel = segments
    if operation is not None and "operation" in sel:
        sel = sel[sel["operation"] == operation]
    if status is not None and "status" in sel:
        sel = sel[sel["status"] == status]
    if limit is not None:
        sel = sel.head(limit)
    return sel


def plot_timeline(segments, ax=None, figsize=(12, 2.6)):
    """Each cycle as a horizontal bar on the stream's time axis, coloured by
    status (grey if unlabelled, i.e. Test). The white space between bars is the
    inter-cycle gap the segmenter keys on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for _, row in segments.iterrows():
        status = row.get("status")
        color = STATUS_COLORS.get(status, "0.5")
        width = row["end_time"] - row["start_time"]
        ax.barh(0, width, left=row["start_time"], height=0.6, color=color)

    ax.set_yticks([])
    ax.set_xlabel("time")
    ax.set_title(f"{len(segments)} detected cycles")
    handles = [plt.Line2D([], [], color=c, lw=6, label=s)
               for s, c in STATUS_COLORS.items() if s in set(segments.get("status", []))]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8)
    return ax


def plot_gap_distribution(stream, gap_seconds=1.0, ax=None, figsize=(7, 3)):
    """Histogram of inter-row time gaps on a log axis, with the segmentation
    threshold drawn on. A clean empty band around the threshold is what makes
    gap-based segmentation safe rather than a tuned guess.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    dt = stream["dt"].dropna()
    dt = dt[dt > 0]
    ax.hist(dt, bins=np.logspace(np.log10(dt.min()), np.log10(dt.max()), 60),
            color="tab:blue")
    ax.axvline(gap_seconds, color="tab:red", ls="--",
               label=f"threshold = {gap_seconds}s")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("inter-row gap (s)")
    ax.set_ylabel("count")
    ax.set_title("Within-cycle sampling vs. between-cycle gaps")
    ax.legend(fontsize=8)
    return ax


def plot_cycle(stream, segment, features, segments=None, figsize_per=(9, 1.3)):
    """All `features` of one cycle, one stacked subplot each. If `segments` is
    passed, the title reports that cycle's operation and status.
    """
    sub = _cycle_frame(stream, segment)
    fig, axes = plt.subplots(len(features), 1, sharex=True, squeeze=False,
                             figsize=(figsize_per[0], figsize_per[1] * len(features)))
    for ax, feature in zip(axes[:, 0], features):
        ax.plot(sub["elapsed"], sub[feature], lw=1, color="tab:blue")
        ax.set_ylabel(feature, fontsize=7, rotation=0, ha="right", va="center")
        ax.tick_params(labelsize=7)

    axes[-1, 0].set_xlabel("elapsed within cycle (s)")

    title = f"segment {segment}"
    if segments is not None:
        row = segments[segments["segment"] == segment]
        if len(row):
            row = row.iloc[0]
            title += f" -- {row.get('operation', '?')} / {row.get('status', 'unlabelled')}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_cycle_overlay(stream, segments, feature, operation=None, ax=None,
                       alpha=0.5, figsize=(10, 4), normalise_time=False):
    """Every selected cycle's `feature` trace on one axis, coloured by status.

    `normalise_time=True` rescales each cycle to 0-1 of its own duration, which
    separates "the shape differs" from "the cycle simply takes longer" -- worth
    flipping both ways, since a resistance fault could plausibly do either.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sel = _select_segments(segments, operation=operation)
    for _, row in sel.iterrows():
        sub = _cycle_frame(stream, row["segment"])
        x = sub["elapsed"] / sub["elapsed"].iloc[-1] if normalise_time else sub["elapsed"]
        ax.plot(x, sub[feature], lw=0.8, alpha=alpha,
                color=STATUS_COLORS.get(row.get("status"), "0.5"))

    ax.set_xlabel("fraction of cycle" if normalise_time else "elapsed within cycle (s)")
    ax.set_ylabel(feature)
    ax.set_title(f"{feature}" + (f" -- {operation} cycles" if operation else ""))
    handles = [plt.Line2D([], [], color=c, lw=2, label=s)
               for s, c in STATUS_COLORS.items() if s in set(sel.get("status", []))]
    if handles:
        ax.legend(handles=handles, fontsize=8)
    return ax


def plot_cycle_grid(stream, segments, feature, operation=None, status=None,
                    limit=None, ncols=8, cell_size=(2.2, 1.5), sharey=True):
    """Small multiples: one panel per cycle, each showing `feature` over that
    cycle's own elapsed time. Panel titles are `segment: status`.
    """
    sel = _select_segments(segments, operation=operation, status=status, limit=limit)
    n = len(sel)
    ncols = min(ncols, n) or 1
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, sharey=sharey, squeeze=False,
                             figsize=(cell_size[0] * ncols, cell_size[1] * nrows))
    axes_flat = axes.flatten()

    for ax, (_, row) in zip(axes_flat, sel.iterrows()):
        sub = _cycle_frame(stream, row["segment"])
        ax.plot(sub["elapsed"], sub[feature], lw=0.8,
                color=STATUS_COLORS.get(row.get("status"), "0.5"))
        ax.set_title(f"{row['segment']}: {str(row.get('status', ''))[:6]}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xticks([])

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(f"{feature}" + (f" -- {operation}" if operation else ""), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def cycle_summary(stream, segments, features, extra_stats=()):
    """One row per cycle of simple per-channel statistics (mean/max/std, plus the
    cycle's duration and row count), joined to its operation/status.

    Deliberately crude -- it isn't a feature set for a model, it's the table to
    eyeball with `plot_status_distributions` to see which channels separate the
    two statuses before committing to any feature engineering.
    """
    stats = {"mean": np.mean, "max": np.max, "min": np.min, "std": np.std}
    stats.update(dict(extra_stats))

    rows = []
    for _, seg in segments.iterrows():
        sub = _cycle_frame(stream, seg["segment"])
        row = {"segment": seg["segment"], "n_rows": seg["n_rows"],
               "duration_s": seg["duration_s"]}
        if "operation" in seg:
            row["operation"] = seg["operation"]
        if "status" in seg:
            row["status"] = seg["status"]
        for feature in features:
            values = sub[feature].to_numpy()
            for name, fn in stats.items():
                row[f"{feature}|{name}"] = fn(values)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_status_distributions(summary, columns=None, operation=None, ncols=4,
                              cell_size=(3.0, 2.0), bins=20):
    """Normal-vs-Abnormal histograms for each summary column.

    Columns whose two histograms barely overlap are the ones a classifier can
    actually use; heavily overlapping ones are noise no matter how physically
    plausible they sound.
    """
    sel = summary if operation is None else summary[summary["operation"] == operation]
    if columns is None:
        columns = [c for c in sel.columns
                   if "|" in c or c in ("duration_s", "n_rows")]

    n = len(columns)
    ncols = min(ncols, n) or 1
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(cell_size[0] * ncols, cell_size[1] * nrows))
    axes_flat = axes.flatten()

    for ax, column in zip(axes_flat, columns):
        values = sel[column].dropna()
        edges = np.histogram_bin_edges(values, bins=bins)
        for status, color in STATUS_COLORS.items():
            group = sel.loc[sel["status"] == status, column].dropna()
            if len(group):
                ax.hist(group, bins=edges, alpha=0.6, color=color, label=status,
                        density=True)
        ax.set_title(column, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_yticks([])

    for ax in axes_flat[n:]:
        ax.axis("off")

    axes_flat[0].legend(fontsize=7)
    fig.suptitle("Per-cycle statistics by status"
                 + (f" -- {operation}" if operation else ""), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def separation_ranking(summary, operation=None):
    """Rank summary columns by how well they separate the two statuses, using
    the absolute standardised mean difference (Cohen's d).

    A quick, assumption-light shortlist to point `plot_status_distributions` at
    when there are more candidate statistics than plots worth reading.
    """
    sel = summary if operation is None else summary[summary["operation"] == operation]
    normal = sel[sel["status"] == "Normal"]
    abnormal = sel[sel["status"] == "Abnormal resistance"]

    rows = []
    for column in sel.columns:
        if not pd.api.types.is_numeric_dtype(sel[column]) or column == "segment":
            continue
        a, b = normal[column].dropna(), abnormal[column].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        if pooled == 0:
            continue
        rows.append({"feature": column, "cohens_d": abs(a.mean() - b.mean()) / pooled,
                     "normal_mean": a.mean(), "abnormal_mean": b.mean()})

    return (pd.DataFrame(rows).sort_values("cohens_d", ascending=False)
            .reset_index(drop=True))
