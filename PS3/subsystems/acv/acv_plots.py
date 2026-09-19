"""Plotting helpers for the PS3 ACV cases.

Shaped around what the data is: 8 cars of one train, recorded side by side at
30 s under the same weather, with exactly one of them leaking. That makes almost
every useful view a *within-file comparison of the 8 cars*, so every plot here
draws all 8 together and highlights the labelled faulty one in red -- the point
of an ACV plot is "does this car stand out from its siblings?", never "what does
this car do on its own".

- `plot_case_timeline` -- one parameter, all 8 cars against wall-clock time.
  The literal raw view; everything else is a reduction of it.
- `plot_residual_timeline` -- the same, after subtracting the leave-one-out
  median across cars. Ambient temperature and the train-wide control schedule
  move all 8 traces together and vanish here, which is what makes a single
  deviating car visible at all.
- `plot_case_grid` -- small multiples, one panel per case, for one parameter.
  The "do the six labelled cases even look alike?" plot.
- `plot_car_distributions` -- per-car box plots of value and residual, the
  distributional summary behind the timelines.
- `plot_regime_strip` -- one categorical parameter as a car x time state strip:
  which of the 8 cars is in which regime, and when. A black tick marks every
  cell where that car's state differs from the train-wide majority state at
  that instant (see `acv_features.with_majority_reference`) -- the direct
  answer to "when do cars disagree, and which one is usually the odd one out".
- `plot_mode_heatmap` -- `plot_regime_strip` for all four categorical
  parameters of one case, stacked.
- `plot_regime_share` -- per-car bar of `frac_majority` (fraction of time
  agreeing with the train's shared command), one panel per case.
- `plot_stat_by_car` -- any column of `car_stats` as a per-car bar chart, faulty
  car highlighted. The direct "would ranking on this statistic work?" view.
- `plot_rank_summary` -- the rank sweep's mean score per candidate rule, against
  the 0.56 random-ranking baseline.

Faulty-car highlighting comes from `Train_Labels.csv` via `acv_features`; on the
Test case there is no label, so every car is drawn neutral.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import acv_features as af

FAULTY_COLOR = "tab:red"
NORMAL_COLOR = "0.65"

# Mean score of a uniformly random 8-car ranking under (n - (r - 1)) / n.
RANDOM_BASELINE = 0.5625


def _faulty_car(case):
    return af.label_map().get(case)


def _car_style(car, faulty):
    """Red and prominent for the faulty car, grey and receding for the rest."""
    if faulty is not None and car == faulty:
        return {"color": FAULTY_COLOR, "lw": 1.8, "zorder": 3, "alpha": 1.0}
    return {"color": NORMAL_COLOR, "lw": 0.9, "zorder": 1, "alpha": 0.8}


def _title(case, param, faulty):
    tag = f" -- faulty car {faulty}" if faulty else " -- unlabelled (Test)"
    return f"{case}: {param}{tag}"


def plot_case_timeline(case, param="Indoor Average Temperature", long=None,
                       ax=None, figsize=(13, 3.6), legend=True):
    """One parameter for all 8 cars of one case, against wall-clock time."""
    if long is None:
        long = af.to_long(case)
    faulty = _faulty_car(case)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sub = long[long["param"] == param]
    for car, s in sub.groupby("car"):
        ax.plot(s["Time"], s["value"], label=car, **_car_style(car, faulty))

    ax.set_xlabel("time")
    ax.set_ylabel(param)
    ax.set_title(_title(case, param, faulty))
    if legend:
        ax.legend(ncol=8, fontsize=7, loc="upper right")
    return ax


def plot_residual_timeline(case, param="Indoor Average Temperature", long=None,
                           ax=None, figsize=(13, 3.6), legend=True):
    """The same traces after subtracting the leave-one-out median across cars.

    A car that tracks its siblings sits on zero; a leaking one should drift off
    it. Zero is drawn in for reference, since "distance from the line" is the
    whole content of the plot.
    """
    if long is None:
        long = af.with_train_reference(af.to_long(case))
    if "resid" not in long:
        long = af.with_train_reference(long)
    faulty = _faulty_car(case)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sub = long[long["param"] == param]
    for car, s in sub.groupby("car"):
        ax.plot(s["Time"], s["resid"], label=car, **_car_style(car, faulty))

    ax.axhline(0, color="k", lw=0.8, ls="--", zorder=2)
    ax.set_xlabel("time")
    ax.set_ylabel(f"{param}\n(residual vs other cars)")
    ax.set_title(_title(case, param, faulty) + " -- residual")
    if legend:
        ax.legend(ncol=8, fontsize=7, loc="upper right")
    return ax


def plot_case_grid(param="Indoor Average Temperature", cases=None,
                   residual=False, ncols=2, figsize=(14, 3.0)):
    """Small multiples: one panel per case, all 8 cars drawn in each.

    Use this to ask whether the six labelled faults share a signature at all,
    before assuming any single rule will generalise to the held-out case.
    """
    if cases is None:
        cases = af.ALL_CASES
    nrows = math.ceil(len(cases) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize[0], figsize[1] * nrows),
                             squeeze=False)
    plot = plot_residual_timeline if residual else plot_case_timeline

    for ax, case in zip(axes.ravel(), cases):
        plot(case, param=param, ax=ax, legend=False)
        ax.set_title(ax.get_title(), fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")

    fig.suptitle(f"{param}{' (residual)' if residual else ''} -- all cases",
                 y=1.002)
    fig.tight_layout()
    return fig, axes


def plot_car_distributions(case, param="Indoor Average Temperature", long=None,
                           figsize=(12, 3.4)):
    """Per-car box plots of the raw value and of the residual, side by side.

    The raw panel shows how much of the spread is just the train-wide schedule;
    the residual panel shows what is left once the siblings are subtracted out.
    """
    if long is None:
        long = af.with_train_reference(af.to_long(case))
    faulty = _faulty_car(case)
    sub = long[long["param"] == param]
    cars = sorted(sub["car"].unique())

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, column, name in zip(axes, ["value", "resid"],
                                [param, "residual vs other cars"]):
        data = [sub.loc[sub["car"] == c, column].dropna().to_numpy() for c in cars]
        bp = ax.boxplot(data, tick_labels=cars, patch_artist=True, showfliers=False)
        for car, box in zip(cars, bp["boxes"]):
            box.set_facecolor(FAULTY_COLOR if car == faulty else NORMAL_COLOR)
            box.set_alpha(0.85)
        if column == "resid":
            ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("car")
        ax.set_ylabel(name)
        ax.tick_params(labelsize=8)

    fig.suptitle(_title(case, param, faulty))
    fig.tight_layout()
    return fig, axes


# Fixed colour per running-mode state, shared across every case so the same
# colour always means the same regime when panels are compared side by side.
# "Stopped" is folded visually next to "Stop" (same colour family) since the
# two names are used interchangeably across files -- see acv_features.py.
_RUNNING_MODE_COLORS = {
    "Stop": "#4c72b0", "Stopped": "#4c72b0",
    "Ventilation": "#64b5cd",
    "Half Cooling": "#dd8452",
    "Full Cooling": "#c44e52",
    "Automatic Cooling": "#55a868",
    "Emergency Ventilation": "#8172b2",
    "Self-Check": "#937860",
    "Invalid": "#7f7f7f",
}
_MISSING_COLOR = "#e8e8e8"


def _categorical_palette(categories):
    """A colour per raw category: the fixed running-mode palette where it
    applies, otherwise a fallback qualitative cycle so any parameter (not just
    Running Mode) can be plotted.
    """
    fallback = plt.get_cmap("tab10").colors
    colors = {}
    next_fallback = 0
    for c in categories:
        if c in _RUNNING_MODE_COLORS:
            colors[c] = _RUNNING_MODE_COLORS[c]
        else:
            colors[c] = fallback[next_fallback % len(fallback)]
            next_fallback += 1
    return colors


def plot_regime_strip(case, param="ACV Running Mode", long=None, ax=None,
                      figsize=(13, 2.6), mark_mismatch=True, legend=True):
    """Which regime every car is in, at every timestamp -- a car x time strip,
    one coloured band per state, in place of an unreadable overplotted
    staircase of 8 step lines.

    This is the direct answer to "when is each car under which regime, and
    when do they disagree": rows are cars (faulty car's label in red), colour
    is the raw state name (fixed palette, see `_RUNNING_MODE_COLORS`), grey is
    a missing reading. When `mark_mismatch` is set, a black tick is drawn atop
    every cell where that car's state differs from the train-wide majority
    state at that instant (see `with_majority_reference`) -- almost all of
    those ticks cluster on one car, and disproportionately on the labelled
    faulty car in most cases (see the module's rank-sweep discussion).
    """
    if long is None:
        long = af.with_majority_reference(af.to_long(case))
    if "is_majority" not in long:
        long = af.with_majority_reference(long)
    faulty = _faulty_car(case)
    sub = long[long["param"] == param].copy()

    cars = sorted(sub["car"].unique())
    times = sorted(sub["Time"].unique())
    categories = sorted(c for c in sub["raw"].dropna().unique())
    palette = _categorical_palette(categories)
    cat_index = {c: i for i, c in enumerate(categories)}

    grid = sub.pivot_table(index="car", columns="Time", values="raw",
                           aggfunc="first").reindex(index=cars, columns=times)
    codes = grid.map(lambda v: cat_index.get(v, -1)).to_numpy()

    cmap_colors = [_MISSING_COLOR] + [palette[c] for c in categories]
    cmap = plt.matplotlib.colors.ListedColormap(cmap_colors)
    bounds = list(range(-1, len(categories) + 1))
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(codes, aspect="auto", interpolation="nearest",
             cmap=cmap, norm=norm,
             extent=[0, len(times), len(cars), 0])

    if mark_mismatch:
        mismatch = sub.pivot_table(index="car", columns="Time", values="is_majority",
                                   aggfunc="first").reindex(index=cars, columns=times)
        rows, cols = np.where((mismatch.to_numpy() == 0))
        ax.scatter(cols + 0.5, rows + 0.5, marker="|", s=40, color="black",
                  linewidths=1.2, zorder=3,
                  label="differs from train majority")

    ax.set_yticks(np.arange(len(cars)) + 0.5, cars, fontsize=8)
    for tick, car in zip(ax.get_yticklabels(), cars):
        if car == faulty:
            tick.set_color(FAULTY_COLOR)
            tick.set_fontweight("bold")
    ax.set_xticks([])
    ax.set_ylabel("car", fontsize=8)
    ax.set_title(_title(case, param, faulty))

    if legend:
        handles = [plt.Rectangle((0, 0), 1, 1, color=palette[c]) for c in categories]
        if mark_mismatch:
            handles.append(plt.Line2D([], [], marker="|", color="black", lw=0,
                                      markersize=8, label="differs from majority"))
            categories = categories + ["differs from majority"]
        ax.legend(handles, categories, fontsize=6.5, ncol=min(len(categories), 5),
                 loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return ax


def plot_mode_heatmap(case, params=None, long=None, figsize=(13, 2.6)):
    """`plot_regime_strip` for every categorical parameter of one case, stacked."""
    if params is None:
        params = af.CATEGORICAL_PARAMS
    if long is None:
        long = af.with_majority_reference(af.to_long(case))

    fig, axes = plt.subplots(len(params), 1,
                             figsize=(figsize[0], figsize[1] * len(params)),
                             squeeze=False)
    axes = axes.ravel()
    for ax, param in zip(axes, params):
        plot_regime_strip(case, param=param, long=long, ax=ax)

    fig.suptitle(_title(case, "categorical parameters", _faulty_car(case)), y=1.01)
    fig.tight_layout()
    return fig, axes


def plot_regime_share(stats, param="ACV Running Mode", cases=None,
                      ncols=3, figsize=(4.2, 2.8)):
    """Per-car, per-case bar chart of `frac_majority` -- the fraction of time
    each car agrees with the train-wide majority regime.

    The direct summary behind `plot_regime_strip`'s mismatch ticks: if the
    faulty car's bar is consistently the shortest, disagreeing with the
    train's shared command is itself a usable ranking signal (see §3 for how
    consistently that holds).
    """
    if cases is None:
        cases = sorted(stats["case"].unique())
    sel = stats[stats["param"] == param]
    nrows = math.ceil(len(cases) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize[0] * ncols, figsize[1] * nrows),
                             squeeze=False)

    for ax, case in zip(axes.ravel(), cases):
        sub = sel[sel["case"] == case].sort_values("car")
        faulty = _faulty_car(case)
        colors = [FAULTY_COLOR if c == faulty else NORMAL_COLOR for c in sub["car"]]
        ax.bar(sub["car"], sub["frac_majority"], color=colors)
        ax.set_ylim(0, 1)
        ax.set_title(case.replace(".xlsx", ""), fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")

    fig.suptitle(f"fraction of time agreeing with train-majority {param}", y=1.002)
    fig.tight_layout()
    return fig, axes


def plot_stat_by_car(stats, column="resid_absmean",
                     param="Indoor Average Temperature", cases=None,
                     ncols=3, figsize=(4.2, 2.8)):
    """One `car_stats` column as a per-car bar chart, one panel per case.

    This is the direct read on a candidate ranking rule: if the red bar is the
    tallest (or shortest) in most panels, ranking on that statistic works.
    """
    if cases is None:
        cases = sorted(stats["case"].unique())
    sel = stats[stats["param"] == param]
    nrows = math.ceil(len(cases) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize[0] * ncols, figsize[1] * nrows),
                             squeeze=False)

    for ax, case in zip(axes.ravel(), cases):
        sub = sel[sel["case"] == case].sort_values("car")
        faulty = _faulty_car(case)
        colors = [FAULTY_COLOR if c == faulty else NORMAL_COLOR for c in sub["car"]]
        ax.bar(sub["car"], sub[column], color=colors)
        ax.set_title(case.replace(".xlsx", ""), fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")

    fig.suptitle(f"{column} -- {param}", y=1.002)
    fig.tight_layout()
    return fig, axes


def plot_rank_summary(summary, top=20, ax=None, figsize=(9, 6)):
    """Mean rank-decay score of each candidate ranking rule, best at the top.

    The dashed line is 0.5625, the expected score of a random 8-car ranking --
    the bar that matters is how far past it a rule gets, not its absolute
    height, since even a useless rule scores well over half.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sel = summary.head(top).iloc[::-1]
    labels = [f"{r.param} | {r.stat} | {'asc' if r.ascending else 'desc'}"
              for r in sel.itertuples()]
    ax.barh(range(len(sel)), sel["mean_score"], color="tab:blue")
    ax.axvline(RANDOM_BASELINE, color="k", ls="--", lw=1,
               label=f"random baseline ({RANDOM_BASELINE:.3f})")
    ax.set_yticks(range(len(sel)), labels, fontsize=7)
    ax.set_xlabel("mean rank-decay score over labelled cases")
    ax.set_title(f"top {len(sel)} single-statistic ranking rules")
    ax.legend(fontsize=8, loc="lower right")
    return ax


def plot_missingness(stats, figsize=(9, 4)):
    """Fraction of non-parsing samples per (case, parameter), as a heatmap.

    Worth one look before trusting any statistic: a parameter that is mostly
    blank in one file is a data artefact, not a signal.
    """
    frac = (stats.assign(frac=stats["n_missing"] / stats["n"])
            .pivot_table(index="case", columns="param", values="frac",
                         aggfunc="mean"))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(frac.to_numpy(), aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(frac.shape[1]), frac.columns, rotation=40,
                  ha="right", fontsize=7)
    ax.set_yticks(range(frac.shape[0]), frac.index, fontsize=7)
    ax.set_title("fraction of non-numeric / missing samples")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig, ax
