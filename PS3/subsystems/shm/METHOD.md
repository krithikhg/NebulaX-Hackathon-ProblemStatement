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

## 4. Validation

Rainflow features are computed per file; nothing uses cross-file statistics. The
constants and the Ridge weights are refit inside every CV fold, so no held-out
label influences the model that predicts it, and the 16 test inputs are only
ever used for inference. The analytic-vs-Ridge choice was made on mean-of-seed
CV, not on one split or on the test set. The generic-ML benchmark leaves out the
physics feature, so the comparison is not circular.

## 5. Comparison with generic ML

On the same folds, Ridge on physics-agnostic features (time-domain stats,
spectral centroid/spread/entropy, band powers) reaches 18.7% MAPE. Spectral
features help it (22.4% → 18.7%), but it is far behind the physics-based models.
Adding the rainflow pseudo-damage feature to that same linear model brings it
under 1%. The physics feature is what matters, not the learner.

## 6. Real-time, cumulative monitoring

The Info Kit's complaint about manual Miner's-rule calculation is a workflow
problem, not an arithmetic one: someone exports data periodically and runs an
offline fatigue toolbox, so there is no live damage or remaining-life number.
Rainflow counting is cheap (O(N)). What we add is automation and a persistent
cumulative state.

Per-segment cost on one 581k-sample file, single core: about 0.06 s to read,
0.25 s for rainflow and 0.14 s for the model, so roughly 0.45 s per file
(~130 files/min). Constants are pre-fitted, files are independent, and the work
parallelises.

Miner's rule is additive, so cumulative damage is a small state with no history:

```
D += d ; n_segments += 1 ; n_samples += L ; rate = D / n_samples
remaining_life = (1 - D) / rate        # segments at the current damage rate
```

Each uploaded segment is filed under a free-text **component** label, chosen on
the SHM page before upload, and the state is keyed by that label. It is stored
server-side — Cloud Storage when `PS3_STATE_BUCKET` is set, otherwise a local
JSON file. The app shows a live dashboard from it: the component field,
cumulative D for the selected component, counts, segments to D = 1, a
D-over-time chart, and a table of every tracked component.
`stream_watch.py` does the same from the command line, ingesting a folder of
segments automatically and writing each prediction plus the running state. The
scored CSV stays per segment; the cumulative view is a monitoring overlay.

We also built a sample-level streaming counter (`streaming.py`) for a feed that
cannot be buffered as a whole segment. It reproduces `fatpack` exactly when it
uses the same per-segment grid (0.0004% mean difference), so the counter logic
is correct. The limitation is the grid: the labels use a grid from each
segment's own min/max, and those ranges vary by about 2x across files. A
bounded stream cannot know a segment's range in advance, and no fixed grid
matches — the union range gives ~2.1% MAPE and a typical range clips larger
segments (~20%). That is well behind the 0.55% of the per-segment counter, so
the per-segment counter is what we ship. The streaming version is kept as an
evaluated prototype (`ablate_stream.py`).

## 7. Why the error is not zero

The label is deterministic, so an exact reproduction would be exact. The
remaining ~0.5% is not label noise: the residual is multiplicative (the slope of
log(D/S) against log D is about 0), where additive noise would give about -1. It
comes from the rainflow tooling. Moving from a generic ASTM counter to `fatpack`
with k = 64 cut the per-file D/S spread from 4.1% to 1.3% and MAPE from 2.5% to
0.8%. What is left is the organisers' exact load-class and residue convention,
which they do not publish.

## 8. What we tried

- Rainflow variant: generic counter 2.5% → fatpack k = 64 0.8%. Biggest single gain.
- Exponent models: a bank of exponents (0.55%) beats a single exponent (0.88%).
  The relation is slightly curved in exponent space, but a two-slope (additive)
  S-N model did not fit, so we treat it as a multiplicative calibration rather
  than a bilinear curve.
- Mean-stress (Goodman) correction: worse, dropped.
- Fatigue-limit cutoff: no change, so the labels use a plain range^m sum.
- Detrending each series: worse (0.8% → 1.6%).
- Residual Ridge on generic features: 0.69%, weaker than the exponent bank, not
  shipped.

## 9. Assumptions

- The S-N constants are the same for all test files. Training spans two lines
  and two load conditions with a small D/S spread, which supports this.
- The stress unit is not given, so it is absorbed into the scale k.
- With 64 training and 16 test files, per-file MAPE has real variance, so we
  rely on repeated CV rather than one split.

## 10. Running it

```bash
cd PS3/subsystems

# refit and write the submission CSV
../../.venv/bin/python -m shm.predict --retrain

# inference on a file or folder
../../.venv/bin/python -m shm.predict --input <dir-or-file.csv> --output <out.csv>

# comparison, figures, streaming ablation
../../.venv/bin/python -m shm.benchmark
../../.venv/bin/python -m shm.report_figs
../../.venv/bin/python -m shm.ablate_stream

# automatic cumulative ingestion of a folder, filed per component
../../.venv/bin/python -m shm.stream_watch --input ../02_Datasets/SHM/Test \
    --stream "bogie frame A" --once
```

The app is in `PS3/app` (`uvicorn server:app --port 8080`); choose the component
in the field at the top of the SHM page, then upload. Set `PS3_STATE_BUCKET` to
keep cumulative state in Cloud Storage, otherwise it uses a local JSON file.

Model files: `shm/checkpoints/constants.json` (analytic) and
`bank_model.joblib` (physics-informed Ridge). Deliverable:
`shm/predictions/shm_predictions.csv` (`file_id`, `prediction`).

## 11. Summary

We reconstruct the organisers' Miner's-rule calculation: rainflow counting with
64 load classes, an S-N exponent that comes out at 5.00, and a scale constant
fitted in closed form. A small regularised correction over a bank of exponents
improves on the single-exponent fit (better in 150 of 160 nested folds), giving
about 0.995 on cross-validation.

Beyond the per-segment prediction, the pipeline is a cumulative monitor: each
segment's damage is added to a persistent per-stream state, the app shows a live
damage and remaining-life dashboard, and `stream_watch.py` ingests segments
automatically. A sample-level streaming counter was built and tested but costs
accuracy, so the exact per-segment counter remains the shipped model.
