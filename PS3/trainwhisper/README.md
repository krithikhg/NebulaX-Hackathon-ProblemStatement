# TrainWhisper — NebulaX PS3 (Train Condition Monitoring)

One app, four subsystems. Drop in telemetry, get back a **localised** maintenance
action: which door cycle, which car, which rail side, which measurement point.

The app runs on **Google Cloud Run** as a single service: it serves the static
frontend and runs the Python pipelines behind `/api/predict/*`. The browser only
uploads files and renders the JSON it gets back, so there is exactly **one**
implementation of each model — the same `door.py` / `acv.py` / `rail.py` /
`shm.py` that `predict.py` uses to write the submission CSVs.

## Results (all out-of-sample)

| Subsystem | Method | Validation | Score |
|---|---|---|---|
| Door | Gap segmentation + Random Forest on per-cycle features | Chronological holdout, scored with the real IoU-weighted F1 | **1.000** |
| ACV | Cabin-temperature deviation from the train median (+ low-pressure evidence) | 6 labelled cases, rank-decay metric | **0.958** |
| Rail | Spectral features per side + class priors tuned for macro F1 | Nested CV 0.743 ± 0.096 (0.808 with priors tuned in-sample) | **~0.74** |
| SHM | Rainflow counting + fitted Miner S-N constants | Leave-one-out, 64 files | **0.975** |

Implied Overall Score (sum ÷ 4) ≈ **0.92**. Rail is the number most likely to
move on the held-out set; the other three are stable.

## Architecture

```
browser  ──POST /api/predict/{door|acv|rail|shm}──►  Cloud Run (FastAPI)
   │   one file per request (rail ≈ 17 MB)              │
   │  ◄──────────── payload JSON the views render ──────┤
   │                                                    ├── door.py  ──┐
   └── renders app.js / views.js / charts.js            ├── acv.py     ├── same code as
                                                        ├── rail.py    │   predict.py
                                                        └── shm.py   ──┘
```

All parsing, feature extraction and model inference happen server-side in
Python. `server.py` only turns each pipeline's output into the JSON shape the
views expect. There is no JavaScript port of any model to keep in sync.

## Deploying on Google Cloud Run

```bash
cd PS3/trainwhisper
gcloud run deploy trainwhisper \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 --memory 1Gi --cpu 1 --timeout 300
```

`--source .` builds the `Dockerfile` with Cloud Build and deploys it. The
service URL is printed on success. (Cloud Run's region is independent of the
project's default zone; change `--region` to move it.)

## Local run

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8080   # http://localhost:8080
```

## Layout

```
server.py         FastAPI: serves public/ and POST /api/predict/{kind}
Dockerfile        Cloud Run image
common.py         shared parsing helpers (door timestamps, ACV schema discovery)
door.py           segmentation + classifier
acv.py            schema-agnostic loader + leak ranking
rail.py           spectral features + 3-class model with prior tuning
shm.py            rainflow + Miner fit
train_all.py      fit everything, print validation summary
predict.py        CLI: --subsystem --input --output
evaluate.py       local copies of all four official metrics
public/
  index.html, styles.css
  js/app.js          shell: routing, side menu, uploads, detail panel, zip, print
  js/views.js        one view per subsystem: payload -> stats, visual, actions, records, panels
  js/ui.js           shared UI pieces: status levels, names, info tips, tables, formatting
  js/util.js         toCsv / pyFloat / remainingLife helpers used by the views
  js/benchmark.js    validation scores shown on the Benchmark page (edit when a model changes)
  js/charts.js       SVG charts with hover tooltips
  js/zip.js          builds predictions.zip in the browser (no dependency)
  js/worker.js       uploads files to /api/predict/* off the main thread
```

## API contract

`server.py` returns exactly what the views render. Keep these shapes when a
model changes; the UI is untouched as long as they hold. `prediction` values and
CSV columns must stay as the spec requires (`Normal` / `Abnormal resistance`,
`Normal` / `Side I` / `Side II`, numeric damage).

| Subsystem | Payload |
|---|---|
| Door | `{ name, result: [{ start_time, end_time, prediction, confidence, mean_current_mA, peak_current_mA }], cycles: [{ points: [[idx, mA, epochMs]], abnormal, operation, t0, duration_s }] }` |
| ACV | `{ name, ranked: [{ car, score, cabin_temp_dev_C, low_pressure_dev? }], samples, train, window }` (most likely first) |
| Rail | `{ result: [{ file_id, prediction, confidence, speed_m_s }], bands: [{ file_id, speed, stationary, side1[], side2[] }], bandLabels }` |
| SHM | `{ result: [{ file_id, prediction }], detail: [{ file_id, samples, cycles, peak_amplitude, histogram }], fit: { m, C, amplitude_cutoff, loo_mape } }` |

`confidence` is the model's probability for its chosen class (0–1); the UI shows
high (≥ 0.9), medium (≥ 0.7) or low certainty. After retraining, update the
scores in `js/benchmark.js`.

## Python pipeline (training and submission CSVs)

```bash
pip install -r requirements.txt
# defaults to ../02_Datasets; override if your data lives elsewhere
export PS3_DATA_ROOT=/path/to/NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets

python train_all.py          # fits everything, prints the validation summary
```

Generate submission CSVs (the web app's download buttons produce the same files
through the same code):

```bash
python predict.py --subsystem door --input $PS3_DATA_ROOT/Door/Test.csv              --output outputs/door_predictions.csv
python predict.py --subsystem acv  --input $PS3_DATA_ROOT/ACV/Test/acv_test_case.xlsx --output outputs/acv_predictions.csv
python predict.py --subsystem rail --input $PS3_DATA_ROOT/Rail_Corrugation/Test       --output outputs/rail_predictions.csv
python predict.py --subsystem shm  --input $PS3_DATA_ROOT/SHM/Test                    --output outputs/shm_predictions.csv

cd outputs && zip -j ../submission/predictions.zip *_predictions.csv
```

**Rebuild `submission/predictions.zip` from the held-out inputs the organisers
distribute before the deadline.**

The `.joblib` models were fitted with scikit-learn 1.8; loading them with another
version prints an `InconsistentVersionWarning` (predictions verified unchanged on
1.9.1, which `requirements.txt` pins).

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

- Rail Side I recall is the weak class (F1 ≈ 0.56). A stronger LightGBM pipeline
  (macro F1 ≈ 0.88) exists separately and is the obvious upgrade; it needs its
  feature/model code moved into `rail.py` so the app and the CSV both use it.
- ACV pressure evidence is implemented but made no difference on the one case that
  carries it — worth revisiting if the held-out file has richer telemetry.
- The confirm / false-alarm feedback loop is designed but not wired to storage.
- Door segmentation assumes the held-out stream is built like the published one.
  Verify the 20 ms / 0.1 s structure on the real test input before trusting it.
- Uploads are processed in memory/temp files and deleted after each response;
  nothing is persisted server-side.
