"""Figures for the SHM solution.

Run:  cd PS3/solution && ../../.venv/bin/python -m shm.report_figs
Out:  PS3/subsystems/shm/figures/*.png
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shm import benchmark, cache, fit, io

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def fig_exponent(S, D, grid):
    mape = []
    for j in range(S.shape[1]):
        k = fit.optimal_scale(S[:, j], D)
        mape.append(fit.mape(D, k * S[:, j]) * 100)
    mape = np.array(mape)
    best = grid[mape.argmin()]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, mape, lw=2, color="tab:blue")
    ax.axvline(best, color="tab:red", ls="--", label=f"best m = {best:.2f}")
    ax.axvline(5.0, color="tab:green", ls=":", label="m = 5 (textbook steel)")
    ax.set_xlabel("S-N exponent m")
    ax.set_ylabel("training MAPE [%]")
    ax.set_title("Damage exponent sweep (search over m)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "shm_01_exponent_sweep.png"), dpi=150)
    plt.close(fig)


def fig_parity(S, D, grid, ids):
    oof = fit.cv_bank(S, D, seed=0)["oof"]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(D, oof, s=28, alpha=0.8, color="tab:blue")
    lo, hi = min(D.min(), oof.min()) * 0.8, max(D.max(), oof.max()) * 1.25
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect")
    ax.plot([lo, hi], [lo * 0.9, hi * 0.9], color="gray", ls=":", lw=1, label="±10%")
    ax.plot([lo, hi], [lo * 1.1, hi * 1.1], color="gray", ls=":", lw=1)
    err = np.abs(D - oof) / D
    for i in np.argsort(err)[::-1][:4]:
        ax.annotate(ids[i].replace(".csv", ""), (D[i], oof[i]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points", color="tab:red")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("true damage"); ax.set_ylabel("predicted damage")
    ax.set_title(f"Out-of-fold parity (MAPE = {err.mean() * 100:.2f}%)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "shm_02_parity.png"), dpi=150)
    plt.close(fig)


def fig_methods(S, D, X, grid):
    names = ["constant\nmedian", "Ridge\ngeneric", "analytic\nphysics",
             "physics +\nresidual", "bank Ridge\n(shipped)"]
    med = np.full_like(D, np.median(D))
    errs = [
        fit.mape(D, med),
        np.mean(benchmark.cv_ml(X, D, list(range(X.shape[1])))),
        np.mean(benchmark.cv_physics(S, D, grid)),
        np.mean(benchmark.cv_residual(S, X, D, grid, list(range(X.shape[1])))),
        np.mean([fit.cv_bank(S, D, seed=s)["mape"] for s in range(5)]),
    ]
    colors = ["tab:gray", "tab:orange", "tab:blue", "tab:purple", "tab:green"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar(names, [e * 100 for e in errs], color=colors)
    for b, e in zip(bars, errs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.05 + 1,
                f"{e * 100:.2f}%", ha="center", fontsize=9)
    ax.set_ylabel("mean CV MAPE [%]  (lower is better)")
    ax.set_title("Method comparison (8-fold CV, 5 seeds)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "shm_03_methods.png"), dpi=150)
    plt.close(fig)


def fig_bank_weights(S, D, grid):
    pipe = fit.fit_bank(S, D)
    coef = pipe.named_steps["ridgecv"].coef_
    alpha = pipe.named_steps["ridgecv"].alpha_
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, coef, lw=2, color="tab:green")
    ax.axvline(5.0, color="tab:red", ls="--", label="m = 5 (textbook steel)")
    ax.set_xlabel("S-N exponent m")
    ax.set_ylabel("standardised Ridge weight")
    ax.set_title(f"Learned weight profile over the pseudo-damage bank (alpha={alpha:.3g})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "shm_05_bank_weights.png"), dpi=150)
    plt.close(fig)


def fig_rainflow(split="train"):
    cyc = cache.cycles(split)
    _, series, _ = io.load_split(split)
    rng = cyc[0][0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(series[0][:4000], lw=0.7, color="tab:blue")
    axes[0].set_title("Example dynamic-stress segment (train01)")
    axes[0].set_xlabel("sample"); axes[0].set_ylabel("stress")
    axes[0].grid(alpha=0.3)
    axes[1].hist(rng, bins=np.logspace(np.log10(max(rng.min(), 1e-3)), np.log10(rng.max()), 60),
                 color="tab:purple")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_title("Rainflow cycle-range distribution")
    axes[1].set_xlabel("stress range"); axes[1].set_ylabel("cycle count")
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "shm_04_rainflow.png"), dpi=150)
    plt.close(fig)


def main():
    grid = cache.GRID
    ids, _, D = io.load_split("train")
    S = cache.damage_grid("train")
    X = np.nan_to_num(cache.generic_features("train"))
    fig_exponent(S, D, grid)
    fig_parity(S, D, grid, ids)
    fig_methods(S, D, X, grid)
    fig_bank_weights(S, D, grid)
    fig_rainflow("train")
    print("Wrote SHM figures to", FIG_DIR)


if __name__ == "__main__":
    main()
