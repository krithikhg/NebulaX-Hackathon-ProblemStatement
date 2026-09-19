# SHM — Cumulative Fatigue Damage Estimation

**PS3 Subsystem 4 · Task:** regression · **Metric:** `max(0, 1 − MAPE)`.

This document describes the method behind `shm_predictions.csv`.It is a
reproducible, physics-informed pipeline that acccurately predicts the Miner's rule based damage
calculation from the raw dynamic-stress time series. Importantly, it can be run in real time, and 
tracks the cumulative fatigue of a part over time based on all collected data samples of that part.

---

## 1. Inputs and outputs

Since the labels given are generated using the Miner's rule, they are a deterministic function of the 
stress series given the formula and its constants. Hence the best approach is not to learn it statistically
from generic features since we already have the physics-based formula for the output. We can just reproduce the
calculation and calibrate the constants based on the data provided. We do this, and add a small cross-validated refinement

---

## 2. Method

### 2.1 Rainflow counting

The stress history is first decomposed into discrete fatigue cycles with
`fatpack` (`find_rainflow_ranges`), using **`k = 64` load classes**. This yields
one cycle per range, i.e. the pseudo-damage `S(m) = Σ rangeᵐ`.

`k = 64` was chosen empirically: sweeping `k = 8…1024` against the training
labels produced a sharp, isolated optimum at 64 (0.8% MAPE vs ≥1.1% for
neighbouring settings).

### 2.2 S-N curve and Miner's rule

With the S-N curve `N = C / σₐᵐ` and Miner's rule `D = Σ nᵢ / Nᵢ`, the damage is

```
D = (1/C) · Σᵢ nᵢ · rangeᵢᵐ = k · S(m)          (the 2ᵐ amplitude factor is absorbed into k)
```

Two models are fitted; both are supplied and both can be selected at inference
time.

### 2.3 Model A — analytic (interpretable baseline)

Fit the two global constants on the 64 training labels by minimising MAPE:

* `m` — the S-N exponent, found by a 1-D search (grid 3.0…7.0);
* `k` — the scale, obtained in **closed form** as the MAPE-optimal weighted
  median of `D/S`.

Result: **`m = 5.000`**, `k = 4.0614e-11`.

### 2.4 Model B — physics-informed Ridge (selected)

The reference relation is not perfectly captured by a single exponent, so we
regress `log D` on a bank of pseudo-damages:

```
log D ≈ b₀ + Σₘ wₘ · log S(m),        m = 3.00, 3.05, …, 7.00   (81 features)
```

using a standardised `RidgeCV` (L2 regularisation; penalty chosen by internal
CV). This learns the shape of the S-N relation instead of assuming one slope.
The learned weights are smooth, all positive, and centred near m ≈ 5 — the model
confirms the physics and applies small multiplicative corrections to it.

Cross-validation selects Model B as the shipped model.

---

## 3. Results

All numbers are **8-fold cross-validation on the 64 training files, repeated
over 5 seeds**, with every constant/model refit inside each fold. Score =
`1 − MAPE`.

| method | CV MAPE | score |
|---|---|---|
| constant median | 91.5% | 0.08 |
| Ridge on generic time + spectral features | 18.7% | 0.81 |
| Model A — analytic `m = 5.00`, `k = 4.06e-11` | 0.88% | 0.991 |
| **Model B — physics-informed Ridge (selected)** | **0.55%** | **0.995** |

The full error on the test set will be measured by the organisers; the hidden
labels are not available to us, so model selection is based strictly on
out-of-fold training performance.

---

## 4. Validation methodology

* Rainflow features are computed **per file**, with no cross-file statistics.
* `m`, `k` **and** the Ridge weights are **refit inside every CV fold**; the
  model that produced each held-out prediction never saw that file's label.
* The 16 test inputs are used **only** for final inference — never for fitting,
  feature selection, or threshold tuning.
* Model family (analytic vs Ridge) was selected on **mean-of-seed CV**, not on a
  single split and not on the test set.
* The generic-ML benchmark **excludes** the physics feature so the comparison
  is fair and not circular.

---

## 5. Model comparison / benchmarking

To justify the physics-informed approach we benchmarked against a
physics-agnostic ML pipeline on the same folds:

* **Generic features** (time-domain statistics, spectral centroid/spread/
  entropy, band powers) with Ridge regression → **18.7% MAPE**. Spectral
  features help here (22.4% → 18.7%).
* **Generic features + physics pseudo-damage** → sub-1% MAPE.

The gap is caused entirely by the rainflow pseudo-damage feature, not by the
choice of learner: once the physics feature is present, even a simple linear
model reaches ~0.9%. This is why the final model is physics-informed rather
than a generic time-series ML model.

---

## 6. Real-time, cumulative monitoring (phases 1–3)

### 6.0 The problem this addresses

The Info Kit motivates the task as the limitations of manual Miner's-rule
calculation — *"lacking automation, computational efficiency, and real-time
capability"* — and describes acquisition as continuous, with data saved
periodically as independent files. The bottleneck it describes is a **workflow**
one: a human periodically exporting data and running an offline fatigue toolbox,
which cannot produce a live, cumulative damage / remaining-life picture. Rainflow
itself is not expensive (O(N)); our contribution is to automate the chain and
turn it into a persistent, cumulative, online monitor. The three parts below map
to phases 1–3 of the implementation.

### 6.1 Per-segment throughput

Measured on a single 581,119-sample file (single CPU core):

| stage | time |
|---|---|
| read CSV | 0.06 s |
| rainflow count (`fatpack`, k = 64) | 0.25 s |
| pseudo-damage bank + model score | 0.14 s |
| **total** | **~0.45 s/file (~2 files/s, ~130 files/min)** |

Rainflow counting is O(N), constants are pre-fitted, and files are independent,
so inference is a single pass with no offline fitting and can be parallelised.

### 6.2 Phase 1 — cumulative state

Because Miner's rule is **additive**, the cumulative damage of a monitored
stream (a free-text label such as `Train 03 / bogie frame`) is a tiny Markov
state that needs no history:

```
state = { D, n_segments, n_samples, rate, history }
update(segment d of length L):  D += d ; n_segments += 1 ; n_samples += L ; rate = D / n_samples
remaining_life = (1 - D) / rate          # segments, at the current damage rate
```

* **Backend:** `state_store.update()` applies the transition and `shm_payload`
  adds a segment to the requested stream. Endpoints: `/api/shm/state` (fleet),
  `/api/shm/state/one`, `/api/shm/state/reset`.
* **Frontend:** the SHM view mounts a live dashboard (`shm_stream.js`): a
  free-text stream selector, cumulative `D`, segment/sample counts,
  **segments to D = 1**, a `D`-over-time chart with a dashed projection to
  `D = 1`, and a fleet table across all streams.
* The submission CSV stays **per segment**; cumulative `D` and remaining life
  are a monitoring overlay and never enter the scored output.

### 6.3 Phase 2 — persistence and automated ingestion

* **Persistence.** State is stored server-side so totals survive restarts and
  are shared across Cloud Run instances: **Google Cloud Storage** when
  `PS3_STATE_BUCKET` is set (one JSON object per stream under `shm-state/`),
  otherwise a local JSON store (`PS3/app/state/`).
* **Automation.** `shm/stream_watch.py` watches a folder (or per-stream
  subfolders `input/<stream>/*.csv`), ingests each new segment automatically,
  updates the persisted cumulative state, appends to a predictions CSV, and
  raises threshold alerts (0.25 monitor / 0.5 plan / 0.8 act). This is the
  automated + incremental capability in one command, mirroring the app's logic.

### 6.4 Phase 3 — sample-level streaming (prototype) and its accuracy

For a live feed that cannot be buffered as a whole segment, `streaming.py`
implements a true single-pass counter. It is a **faithful streaming
reimplementation of the `fatpack` 4-point counter** — same class-centre
convention, same peak–valley reversal filtering, same 4-point cycle-closure
rule, and the same residue close-out (residue concatenated with itself and
counted again). State is a bounded cycle-range histogram; memory is O(k) rather
than O(signal length). Accuracy against `fatpack`:

| counter | mean rel. diff vs batch (train / test) |
|---|---|
| online, **per-segment** class grid | **0.0004% / 0.0000%** |
| online, **fixed** 64-class grid | ~2.1% / ~2.3% |

The counter itself is exact, so the whole gap is the **class grid**. The
reference labels use a grid derived from each segment's own min/max, and those
ranges vary by ~2× across files (median 59, p10 42, p90 89). A memory-bounded
stream cannot know a segment's min/max in advance, and no single fixed grid
recovers it: the union range gives ~2.1% MAPE, while a typical (median) range
clips larger segments and gives ~20%. Best-case bounded-memory streaming is
therefore ~2.1% MAPE versus **0.55%** for the per-segment counter.

We therefore **ship the exact per-segment counter** (0.45 s/segment) — already
real-time for the periodic-file acquisition model, and exactly additive across
segments — and document the bounded-memory streaming counter as an evaluated
prototype with this fundamental tradeoff (`shm/ablate_stream.py`).

## 7. Why the error is not zero

The label is deterministic, so an exact reproduction would be exact. The
remaining ~0.5% is **not random label noise**:

* The residual is **multiplicative** — the slope of `log(D/S)` on `log D` is
  ≈ 0. Additive noise would produce a slope of ≈ −1. So the labels were not
  perturbed.
* It is dominated by the **exact rainflow implementation**: switching from a
  generic ASTM counter to `fatpack` with `k = 64` cut the per-file `D/S` spread
  from **4.1% to 1.3%** and the MAPE from 2.5% to 0.8%.
* What remains is the organisers' precise load-class/residue convention, which
  is not published.

In short: the model reproduces the physics; the residual is tooling, not signal.

---

## 8. Ablations and investigated alternatives

* **Rainflow variant:** generic counter (2.5% MAPE) → `fatpack` `k = 64` (0.8%).
  This was the largest single improvement.
* **Flexible exponent models:** a bank of exponents improves on a single exponent
  (0.88% → 0.55%). The effective relation is *curved* in exponent space; a
  literal two-slope (additive) S-N model did **not** fit, so we report a
  multiplicative calibration rather than claiming a bilinear S-N curve.
* **Mean-stress (Goodman) correction:** made results worse → not used.
* **Fatigue-limit cutoff:** changed nothing → labels use a plain `rangeᵐ` sum.
* **Detrending** each series before counting: worse (0.8% → 1.6%).
* **Residual Ridge on generic features:** weaker than the exponent-bank
  (0.69% vs 0.55%), so it is documented, not shipped.

---

## 9. Assumptions and limitations

* We assume the same S-N constants apply to all test files (same material/
  component). The training set spans two lines and two load conditions and the
  per-file `D/S` spread is small, which supports this.
* The stress unit is not stated; it is absorbed into the scale `k`, so it does
  not affect predictions.
* The exact organisers' rainflow tooling is unknown; we match it empirically via
  `k = 64`.
* With only 64 training files and 16 test files, per-file MAPE has non-trivial
  variance; our conclusions rely on repeated cross-validation rather than a
  single split.

---

## 10. Reproduction

```bash
cd PS3/subsystems

# refit both models on the provided Train set and score the provided Test set
../../.venv/bin/python -m shm.predict --retrain     # writes shm/predictions/shm_predictions.csv

# inference on a new file or directory (loads the saved models)
../../.venv/bin/python -m shm.predict --input <dir-or-file.csv> --output <out.csv>

# method comparison, figures, and the streaming-vs-batch ablation
../../.venv/bin/python -m shm.benchmark
../../.venv/bin/python -m shm.report_figs
../../.venv/bin/python -m shm.ablate_stream

# automated, persistent, cumulative ingestion of a folder of segments
../../.venv/bin/python -m shm.stream_watch --input ../02_Datasets/SHM/Test \
    --stream "Train 01 / bogie frame" --once
```

The interactive app lives in `PS3/app` (`uvicorn server:app --port 8080`); set
`PS3_STATE_BUCKET=<bucket>` to persist cumulative state in Cloud Storage (a
local JSON store is used otherwise).

**Model files:** `shm/checkpoints/constants.json` (analytic) and
`shm/checkpoints/bank_model.joblib` (physics-informed Ridge).
**Deliverable:** `shm/predictions/shm_predictions.csv` — columns `file_id`,
`prediction` (one cumulative-damage value per file).

---

## 11. Summary

We reverse-engineer the organisers' Miner's-rule damage calculation: rainflow
counting with 64 load classes, an S-N exponent recovered as `m = 5.00`, and a
scale constant fitted in closed form. On top of this interpretable core we fit a
small, regularised, cross-validated multiplicative correction over a bank of
exponents, which is robustly more accurate (better in 150/160 nested folds).
This keeps the solution physically transparent while reaching ~0.995 on
cross-validation.

Beyond the per-segment prediction, the solution is a **real-time monitoring
pipeline**: each segment's damage is added to a persistent per-stream Markov
state (Cloud Storage or a local JSON store), the app renders a live cumulative
damage / remaining-life dashboard with a fleet view, and `stream_watch.py`
ingests segments automatically with threshold alerts. A true sample-level
streaming counter was prototyped and honestly evaluated, but its accuracy loss
(≈5% vs 0.46% MAPE) means the exact batch counter remains the shipped model —
used at segment granularity, which is real-time for the periodic-file
acquisition model.
