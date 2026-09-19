# TrainWhisper — NebulaX PS3 (Train Condition Monitoring)

One app, four subsystems. Drop in telemetry, get back a **localised** maintenance
action: which door cycle, which car, which rail side, which measurement point.

The web app is a static site deployed on **Vercel**. The trained models are exported
to JSON and evaluated in the browser, so there is no server, no cold start, and
telemetry files never leave the user's machine. That matters here: a single rail
recording is ~17 MB and an SHM segment ~6 MB, well past the 4.5 MB request-body limit
of a Vercel serverless function.

## Results (all out-of-sample)

| Subsystem | Method | Validation | Score |
|---|---|---|---|
| Door | Gap segmentation + Random Forest on per-cycle features | Chronological holdout, scored with the real IoU-weighted F1 | **1.000** |
| ACV | Cabin-temperature deviation from the train median (+ low-pressure evidence) | 6 labelled cases, rank-decay metric | **0.958** |
| Rail | Spectral features per side + class priors tuned for macro F1 | Nested CV 0.743 ± 0.096 (0.808 with priors tuned in-sample) | **~0.74** |
| SHM | Rainflow counting + fitted Miner S-N constants | Leave-one-out, 64 files | **0.975** |

Implied Overall Score (sum ÷ 4) ≈ **0.92**. Treat Rail as the number most likely
to move on the held-out set; the other three are stable.

## Deploying on Vercel

1. Import `krithikhg/NebulaX-Hackathon-ProblemStatement` in Vercel.
2. Set **Root Directory** to `PS3/trainwhisper`.
3. Deploy. `vercel.json` sets framework *Other* and serves `public/`; there is no build step.

Or from the CLI: `cd PS3/trainwhisper && vercel --prod`.

To preview locally, run `python serve.py` from this folder and open
http://localhost:8000. It serves `public/` with caching disabled, so an edited JS
module is never mixed with a stale copy of another one (a blank page after
pulling changes is usually that; Ctrl+Shift+R fixes it). Opening `index.html`
from disk won't work: module workers need HTTP.

The **Try a sample** buttons fetch the published test files from this repo on
`raw.githubusercontent.com`, so they work on any deployment without bundling data.

## Web app layout

```
public/
  index.html, styles.css
  js/app.js          shell: routing, side menu, uploads, detail panel, zip, print
  js/views.js        one view per subsystem: payload -> stats, visual, actions, records, panels
  js/ui.js           shared UI pieces: status levels, names, info tips, tables, formatting
  js/benchmark.js    validation scores shown on the Benchmark page (edit when a model changes)
  js/charts.js       SVG charts with hover tooltips
  js/zip.js          builds predictions.zip in the browser (no dependency)
  js/worker.js       runs the models in a Web Worker so the page stays responsive
  js/engine/         JavaScript ports of door.py / acv.py / rail.py / shm.py
  models/            exported models (door_rf.json, rail_rf.json, shm_fit.json)
  vendor/xlsx.mjs    SheetJS 0.20.3 for reading ACV workbooks (Apache-2.0)
```

### Interface

Layout follows the MUI dashboard template (side menu, breadcrumbs, outlined stat
cards), ported to plain CSS so the site still needs no build step.

- **Overview**: one tile per subsystem with its status, key figure and a mini
  visual. Each tile has its own Upload and Sample buttons and accepts dropped
  files. A priority list ranks analysed subsystems by urgency.
- **Subsystem page**: status, headline and three tabs. *Summary* has stat cards,
  one interactive visual (cycle timeline, train consist, recording grid, damage
  gauges) and an action checklist. *Records* is a filterable table. *Signals*
  holds the engineering charts.
- **Detail panel**: selecting any cycle, car, recording or segment opens a side
  panel with that item's figures and chart.
- **Info icons (i)**: definitions and method notes are shown on hover or tap
  instead of on the page.
- **Side menu**: status dot per subsystem, plus `predictions.zip` and a printable
  report with sign-off lines.

Four statuses are used everywhere: Act now, Plan, Monitor, Normal. The Door and
SHM thresholds are team defaults set in `js/views.js`.

### UI payload contract (keep this when replacing a model)

`worker.js` returns these shapes and `views.js` renders them. A new model can do
anything internally as long as its handler still returns:

| Subsystem | Payload |
|---|---|
| Door | `{ name, result: [{ start_time, end_time, prediction, confidence, mean_current_mA, peak_current_mA }], cycles: [{ points: [[idx, mA, epochMs]], abnormal, operation, t0, duration_s }] }` |
| ACV | `{ name, ranked: [{ car, score, cabin_temp_dev_C, low_pressure_dev? }], samples, train, window }` (most likely first) |
| Rail | `{ result: [{ file_id, prediction, confidence, speed_m_s }], bands: [{ file_id, speed, stationary, side1[], side2[] }], bandLabels }` |
| SHM | `{ result: [{ file_id, prediction }], detail: [{ file_id, cycles, peak_amplitude, histogram }], fit: { m, C, loo_mape } }` |

`prediction` values and CSV columns must stay as the spec requires (`Normal` /
`Abnormal resistance`, `Normal` / `Side I` / `Side II`, numeric damage). `confidence`
is the model's probability for its chosen class (0–1). The UI shows it as
high (≥ 0.9), medium (≥ 0.7) or low certainty. After retraining, update the scores
in `js/benchmark.js`.

The engine mirrors the Python pipeline decision for decision: random forests are
walked tree by tree with float32 inputs (as scikit-learn casts them), the FFT is
exact for any length, and rainflow is a line-by-line port of `rainflow` 3.2.
`tests/parity.mjs` checks this against the Python outputs:

```bash
cd tests && npm install
PS3_DATA_ROOT=/path/to/PS3/02_Datasets npm test
```

It verifies all Door, ACV, Rail (68 files) and SHM test predictions against
`outputs/*.csv`, the ACV ranking on every labelled training case, and rail features
against the cached Python training features.

## Python pipeline (training and submission CSVs)

```bash
pip install -r requirements.txt
# defaults to ../02_Datasets; override if your data lives elsewhere
export PS3_DATA_ROOT=/path/to/NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets

python train_all.py          # fits everything, prints the validation summary
python export_models.py      # refreshes public/models/ for the web app
```

**Re-run `export_models.py` after any retraining**, or the web app keeps serving the
old models.

Generate submission CSVs (the web app's download buttons produce the same files):

```bash
python predict.py --subsystem door --input $PS3_DATA_ROOT/Door/Test.csv              --output outputs/door_predictions.csv
python predict.py --subsystem acv  --input $PS3_DATA_ROOT/ACV/Test/acv_test_case.xlsx --output outputs/acv_predictions.csv
python predict.py --subsystem rail --input $PS3_DATA_ROOT/Rail_Corrugation/Test       --output outputs/rail_predictions.csv
python predict.py --subsystem shm  --input $PS3_DATA_ROOT/SHM/Test                    --output outputs/shm_predictions.csv

cd outputs && zip -j ../submission/predictions.zip *_predictions.csv
```

`submission/predictions.zip` was built from the currently published test inputs and
its schema is verified against `04_Example_Submission/`. **Rebuild it from the
held-out inputs the organisers distribute before the deadline.**

The `.joblib` models were fitted with scikit-learn 1.8; loading them with another
version prints an `InconsistentVersionWarning` (predictions were verified unchanged
on 1.9.1). The web app is unaffected, since it reads the JSON export.

## Method notes

**Door.** The controller only logs rows *during* a cycle: 20 ms between samples
inside a cycle, 20+ s between cycles. Splitting on a 0.1 s gap reproduced all 110
ground-truth boundaries exactly, so segmentation needs no model and every matched
segment scores IoU = 1.0. Classification then runs on per-cycle current, voltage
and back-EMF features — abnormal cycles draw ~720 mA mean current against ~537 mA
for normal ones. Validated chronologically, because the segments come from one
continuous stream and a random split would leak neighbouring cycles.

**ACV.** The train is its own control group: each car's cabin temperature is
compared with the median across cars at every timestamp, so ambient conditions and
setpoint changes cancel out. The loader reads each file's own headers, which is
necessary — one training workbook carries 59 parameters per car (including
refrigeration pressures) while the rest carry 8. Car IDs are matched on `Car NN`
so the `Car model` column is never mistaken for a car.

**Rail corrugation.** Per-channel RMS, peak, kurtosis and 9-band log energy,
aggregated per rail side across its 32 axle boxes, plus side-difference features —
corrugation is one-sided, so asymmetry carries the signal. Two deliberate choices:
raw speed is excluded (every fault file was recorded above 9.7 m/s while Normal
files include stationary recordings, so speed is a sampling artefact), and
stationary files are forced to Normal. Class priors are tuned for macro F1, which
lifts it from 0.65 to ~0.74 nested.

Two approaches theory favoured were tested and **rejected on the evidence**:
wavelength-normalised spectra (0.556) and per-axle-box modelling aggregated to the
file (0.633). The per-box model did improve Side I recall, so an ensemble is the
obvious next step if there's time.

**SHM.** The labels were generated by rainflow counting plus Miner's rule, so the
model inverts that: `D = (1/C)·Σ nᵢ·σₐᵢᵐ`, with m grid-searched and C fitted in
closed form. The fit lands on m = 5.0, a textbook S-N exponent for welded steel —
good evidence the real generating process was recovered. Two parameters against 64
labels, confirmed by leave-one-out.

## Known gaps / next steps

- Rail Side I recall is the weak class (F1 ≈ 0.56). Ensemble the per-box model, or
  try a small CNN over per-box spectrograms for many more training samples.
- ACV pressure evidence is implemented but made no difference on the one case that
  carries it — worth revisiting if the held-out file has richer telemetry.
- Very large ACV workbooks (the 34 MB training case) parse slowly in the browser;
  the published test workbook (2 MB) takes ~3 s.
- The confirm / false-alarm feedback loop is designed but not wired to storage.
- Door segmentation assumes the held-out stream is built like the published one.
  Verify the 20 ms / 0.1 s structure on the real test input before trusting it.

## Python layout

```
common.py         shared parsing helpers (door timestamps, ACV schema discovery)
door.py           segmentation + classifier
acv.py            schema-agnostic loader + leak ranking
rail.py           spectral features + 3-class model with prior tuning
shm.py            rainflow + Miner fit
train_all.py      fit everything, print validation summary
export_models.py  write public/models/ for the web app
predict.py        CLI: --subsystem --input --output
evaluate.py       local copies of all four official metrics
```
