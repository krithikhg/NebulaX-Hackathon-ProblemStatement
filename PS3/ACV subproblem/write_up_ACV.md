# ACV refrigerant-leak localisation: how the model works and why

## Summary

The task is to rank the 8 cars of a train from most to least likely to carry a refrigerant
leak. The model is a **peer comparison**: two physically motivated indicators are measured
for every car, each car is compared with the other seven, and the car that stands furthest
above its peers ranks first. **Nothing is fitted.** There are no weights or thresholds
learned from the six labelled files, because six files with one faulty car each cannot
support that without memorising them.

| | |
|---|---|
| Indicators | `hot` (mean indoor temperature) and `gap` (indoor minus cooling setpoint while cooling) |
| Comparison | leave-one-out robust z-score against the 7 peer cars |
| Index | mean of the two z-scores; highest = most suspicious |
| Labelled files | 0.979 mean expected score (random ranking: 0.5625) |
| Test file | `01\|03\|04\|07\|08\|06\|02\|05` |

The headline number was measured on the same six files the indicators were chosen on. It
shows that the signal exists; it is not a forecast of the Test score. Section 9 lists what
could go wrong.

---

## 1. The problem and the data

- Each file holds 30-second telemetry for all 8 cars of one train. Exactly one car has a
  refrigerant leak. Six files are labelled (`Train_Labels.csv`); one Test file is not.
- Scoring is `(n - (r - 1)) / n` on the true faulty car's rank `r`, so 1st = 1.000,
  2nd = 0.875, 3rd = 0.750.
- Most files record 8 parameters per car (setting mode, running mode, cooling and heating
  control temperature, indoor and outdoor temperature, load-halved, information-valid). **Case 04**
  records 59 per car, sampled every 10 s, and only cars 01-04 report anything.

The two facts that shape the design: there are very few faulty examples, and every car in
a file experiences the same weather, schedule and train-wide setpoint.

## 2. Step 1: start from the physics, not the data

A refrigerant leak means undercharge, so the unit removes less heat than it should. In a
cabin that is meant to hold a setpoint, that produces two observable effects:

1. The cabin runs **warmer** than its siblings' under the same weather and schedule.
2. While the unit is **commanded to cool**, the cabin temperature stays further **above the
   setpoint** than it does in the healthy cars, because the compressor is not executing
   the command as well.

Because the setpoint and the weather are common to all 8 cars, the other seven cars are a
free control group. That is the same idea as fleet-based anomaly detection in condition
monitoring: judge each unit against its peers, with no labelled history required.

This also fixes what "explainable" means here: every indicator must have a stated physical
reason, and a result should be readable as "car X is N robust standard deviations warmer
than its siblings."

## 3. Step 2: clean the data

- **Zeros are missing readings.** A literal `0` in a temperature or setpoint column is a
  sensor dropout, not 0 degrees C. They line up with the `Invalid` flags (row for row in
  case 01, identical counts in cases 02-03 and in the Test file). They are turned into
  missing values at load time and skipped, never filled.
- **Column names are read from each file's own headers**, since the parameter set differs
  between files. Car ids are used exactly as they appear (`03`, not `Car 3`), as the
  submission format requires.
- **Case 04 uses different names for the same measurements** (for example "Passenger Cabin
  Temperature Detected Value" is the indoor temperature). Those few are mapped to the
  standard names; see section 6 for the rest of that file.
- **The check that this handling does not create the result:** the ranking was recomputed
  keeping zeros as 0, turning them into missing, and using only rows where every car is
  fully clean. The faulty car's rank was identical in all three variants in all six files.

## 4. Step 3: candidate indicators, each with a physical reason

Five candidates were considered. Three were eliminated (sections 5-6).

| Candidate | What it measures | Physical reason | Outcome |
|---|---|---|---|
| `hot` | mean indoor temperature over the file | a leak cuts cooling capacity, so the cabin runs warm | **kept** |
| `gap` | mean of (indoor - cooling setpoint) over rows where that car's own mode is a cooling state | how well this car executes a command all 8 were given | **kept** |
| `inval` | count of `ACV Information Valid = Invalid` rows | the unit's own flag for a dropped or implausible reading; plausibly a second symptom of one failing unit | eliminated |
| `maj` | fraction of time the car's running mode equals the train's majority mode | a struggling car crosses the thermostat threshold at different moments | eliminated |
| indoor vs outdoor temperature | how closely the cabin tracks the outside | a healthy cooled cabin follows the weather in a predictable way | eliminated |

`hot` uses the **mean**, not the median. The sensor steps in 0.5 degrees C, so a whole-file
median ties most cars (all eight in case 02) and carries no information, while the mean
still separates them.

## 5. Step 4: turn indicators into a ranking

**Peer z-score.** For each indicator and each car, the car's value is compared with the
median of the *other seven*, in units of their robust spread:

`z = (x_car - median(peers)) / max(1.4826 x MAD(peers), 0.02)`

The sign is arranged so a positive z is always "more suspicious". The final score is the
mean of the car's `hot` and `gap` z-scores, and cars are sorted from highest to lowest.

Why this and not something else:

- **Not a fitted model.** An XGBoost trial on 55 features scored a perfect 1.0
  leave-one-case-out, but the fitted trees were one signal (hotter than siblings, in three
  encodings) plus a few splits: a threshold on one signal, learned from 40 rows per fold.
  That is no evidence of anything, and absolute cut-offs transfer badly (the Test file's
  gaps are all near zero, unlike the labelled faulty cars').
- **Not ranks.** An earlier version averaged within-file ranks of four indicators. A rank
  says "first" but not by how much, so a clear outlier and a marginal leader look the
  same. The z-score keeps the size of the lead.
- **Leave-one-out peers.** A car is never part of its own reference group, so a strong
  outlier cannot inflate the spread it is measured against.
- **Spread floor 0.02 degrees C.** On short windows the peers can agree almost exactly,
  and a difference of one sensor step would give an absurd z. The floor changes no
  whole-file score; it only matters for the per-quarter checks.
- **Missing values are excluded, never defaulted.** A car with no reading for an
  indicator has no z for it; a car with no data at all ranks last.
- **Ties are scored honestly.** The faulty car's rank is the midpoint of its tie group,
  not the first hit, because sorting a tied score otherwise credits a rank the data never
  earned.

## 6. Step 5: test each indicator against the data, and eliminate

The rule was: keep an indicator only if it separates the faulty car, holds across files,
and is not an artifact of how missing values are handled.

**Results per indicator (mean expected score, six labelled files):**

| Setting | Score |
|---|---|
| `hot` alone | 0.979 |
| `gap` alone | 0.979 |
| `hot + gap` (final) | 0.979 |
| `inval` alone | 0.781 |
| `maj` alone | 0.833 |

`hot` and `gap` each put the faulty car first in five of six files and largely agree, so
the labelled result is **one signal measured twice**, not two independent confirmations.
The combination is kept because it costs nothing and averages out single-indicator quirks.

### `inval`: eliminated

- It marks the faulty car in cases 01-03 (25, 44 and 1 Invalid rows against a peer median
  of 0), but it is silent in cases 05 and 06 (0 rows) and **does not exist in case 04**.
- In the **Test file it fires on five cars**, not one (car 04 has 19 rows; cars 05 and 07
  have 2; cars 03 and 06 have 1). The labelled files' pattern of "exclusive to the faulty
  car" does not hold there, so its meaning on Test is unknown.
- Alone it scores 0.781, the worst of any indicator.

### `maj`: eliminated

- **It double-counts `inval`.** On an Invalid row the car reports mode `Invalid`, which is
  by definition a mismatch. In case 02, 44 of the faulty car's 45 mismatch rows are
  Invalid rows. Excluding those rows removes the double count.
- **After that fix it is unstable.** In case 03 it points the wrong way (the faulty car
  agrees with the majority *more* than its peers: 0.9999 against 0.9942). In case 06 it
  ranks the faulty car mid-table (tied 4th to 6th).
- It is strong in exactly one file, case 04 (0.574 against a peer median of 0.862), which
  has only four reporting cars.
- **This is a judgement call, and it costs something.** `hot + maj` scores 1.000 on all six
  files, against 0.979 for `hot + gap`; the entire difference is case 04. It was not
  adopted because the edge rests on one four-car file, it appears only when `gap` is left
  out, and it was found by searching 320 settings on six files, where some setting will
  score perfectly by chance. On the Test file it makes no difference: every car's majority
  agreement is between 0.994 and 1.000 (z between -0.21 and +0.20), and `hot + gap`,
  `hot + maj` and all three together give the same ranking.

### Indoor-vs-outdoor temperature: eliminated

Explored before this model was built. A cooled cabin should track the outside in a
predictable way, so the relationship (correlation over all rows, over cooling rows only,
and how the outdoor-minus-indoor difference widens once cooling starts) was tried as a
signal.

- **Where computable it was mediocre**, about 0.78-0.81.
- **It is often not computable.** Outdoor sensors exist on all 8 cars in cases 01-03, on
  cars 01-04 only in case 04, and on the two end cars (01 and 08) only in cases 05 and 06.
  In cases 05 and 06 the faulty car (04 and 06) has no outdoor reading to compare, so the
  indicator could be evaluated in only 4 of 6 labelled files.
- **It fails where it can be checked**: in case 02 the correlation variants rank the
  faulty car 7th of 8, and the transition-response variant ranks it 8th of 8.
- The Test file has an outdoor reading for every car, so coverage is not why it was
  dropped: it was dropped because it is unreliable on the labelled evidence.

The same family of ideas was also ruled out: `ACV Setting Mode` rules (near-constant field,
"perfect" scores off almost no transitions, not trusted), a per-car "desired temperature"
(the setpoint is one train-wide value per file, so there is nothing to compare), and `Load
Halved` (only ever reads `Normal`).

### Case 04's extra columns: deliberately not used

Case 04 records 59 parameters per car, 53 of them outside the standard set (operating
mode, time-staggered start, target-temperature offsets, self-check, grounding tests,
startup commands and more), at a 10-second rate.

- **The Test file follows the standard 8-parameter schema** (67 columns). Anything learned
  from those extra columns could not be computed on Test, so a model built on them could
  not be applied to the file it is meant to score.
- Using them for one file would also make the six training cases inconsistent with each
  other and with Test, weakening the only validation available.
- So only the standard parameters are read from case 04 (with the renamed columns mapped to
  standard names), and it is scored like every other file. It is the weakest labelled
  file: the faulty car (01) ranks 2nd behind car 04 (0.875), and its `hot` is
  indistinguishable from its siblings (z of 0.33).
- The extra telemetry was not explored further. It may well contain a more direct leak
  signal, but that is only useful for a file that also has it.

## 7. Step 6: tune against a train / validation split

There are only six labelled files, so any tuning has to be checked against files it did
not see. All candidate choices were treated as settings to tune and validated:

- **Search space (320 settings):** the four candidate indicators (`hot`, `gap`, `inval`,
  `maj`), each with weight 0, 0.5 or 1, crossed with four spread floors (0.005, 0.02, 0.05,
  0.1).
- **Validation:** every possible 4-train / 2-validation split (15 folds), and
  leave-one-case-out (6 folds) as a cross-check. In each fold the best setting on the
  training files is picked and scored on the held-out files. Ties are broken toward the
  simplest setting (fewest indicators, then a floor near 0.02).

| Method | Validation score |
|---|---|
| Setting picked on the training files (15 folds) | 0.896 |
| Setting picked on the training files (leave-one-case-out) | 0.896 |
| Fixed `hot + gap`, floor 0.02 | 0.979 |

What this shows:

- **There is nothing reliable to calibrate.** In every fold, between 61 and 293 of the 320
  settings score a perfect 1.0 on the four training files, so the training score cannot
  choose among them; the tie-break decides. Picks built around `maj` and `inval` then
  scored between 1.000 and 0.500 on the held-out files, depending on which files were held
  out. Selecting is worse than not selecting.
- **The floor does not matter** on whole-file scores: 0.005 to 0.1 all give 0.979.
- **`hot` and `gap` carry the result**: every combination that includes them at equal
  weight scores 0.979.
- **The elimination of `inval` and `maj` was tested, not assumed**, and the one setting
  family that beat 0.979 (all include `maj`) is documented in section 6.
- **This is the honest limit of the validation.** The indicators were shortlisted after
  looking at all six files, so a split of those six cannot make 0.979 a held-out number.

## 8. The final model and its output

```
python "ACV subproblem/predict.py" --input 02_Datasets/ACV/Test/acv_test_case.xlsx --output acv_predictions.csv
```

`predict.py` reads the file, computes `hot` and `gap` per car, applies the peer z-score, and
writes `acv_predictions.csv` with columns `file_id`, `ranked_cars` (ids separated by `|`,
most likely first). It reproduces the notebook's ranking on all six labelled files.

**Labelled files:**

| Case | Faulty car | Rank | Score | Faulty car's mean z | Best other car |
|---|---|---|---|---|---|
| 01 | 01 | 1 | 1.000 | 13.41 | 3.32 |
| 02 | 02 | 1 | 1.000 | 6.30 | 4.07 |
| 03 | 03 | 1 | 1.000 | 4.24 | 1.04 |
| 04 | 01 | 2 | 0.875 | 2.66 | 10.25 (car 04) |
| 05 | 04 | 1 | 1.000 | 3.68 | 3.30 |
| 06 | 06 | 1 | 1.000 | 15.49 | 1.73 |

The margins are informative: cases 01, 03 and 06 are decisive; cases 02 and 05 are narrow, so
their "1.000" scores are closer calls than a rank alone suggests.

**Test file:** `01|03|04|07|08|06|02|05`

| Car | z_hot | z_gap | Mean z |
|---|---|---|---|
| 01 | 1.29 | 5.73 | 3.51 |
| 03 | 0.82 | 0.95 | 0.89 |
| 04 | 0.74 | 0.20 | 0.47 |
| 07 | 0.18 | 0.24 | 0.21 |
| 08 | -0.21 | -0.20 | -0.20 |
| 06 | -0.29 | -0.95 | -0.62 |
| 02 | -1.07 | -2.56 | -1.81 |
| 05 | -1.66 | -2.87 | -2.27 |

A qualitative read of the Test data supports the top pick:

- Car 01 is the only car with a positive `gap` (+0.125; the others are between -0.05 and
  -0.19), roughly four peer-spreads clear.
- Its gap is the highest of the eight from the second quarter of the recording onward, and
  it leads by the most in the second half, as a car that has lost cooling capacity would
  as the load rises.
- Ranks 2 to 8 are mostly noise: cars 03, 04 and 07 are within about one z of the peer
  median and cannot be separated.

## 9. Limitations and risks

1. **The 0.979 is not held out.** Indicators were chosen after seeing all six files, and
   the validation in section 7 cannot fix that.
2. **The Test effect is smaller than in most of training.** In five of the six labelled
   files the faulty car's mean `gap` sat roughly 0.25 to 1.4 degrees above the mean of its
   siblings' (case 04: no difference); car 01 on Test is about 0.2 above. Confidence in
   first place is "likely", not "certain".
3. **Case 04 is the failure mode to watch.** A subtle fault that is not visible on `hot`
   or `gap` would rank badly, as it did there. If Test resembles case 04, dropping `maj`
   would cost about 0.125.
4. **Car 04 is the strongest alternative to car 01 on Test.** It has 19 rows where the car
   reports `Invalid`, against 0 to 2 for most other cars. In cases 01 and 02 a burst like
   that marked the faulty car. It was left out of the ranking because `inval` fires on five
   cars in Test, and car 04's temperature signals are unremarkable. Whether those rows are a
   failing unit or a communications fault cannot be settled from these columns; the raw rows
   around the bursts are where to look.
5. **One file, so the score is nearly all-or-nothing.** Test is a single file: first place
   scores 1.000, second 0.875, and so on.
6. **The signal detects weak cooling, not leaks specifically.** A failing compressor, a
   blocked filter or a bad sensor would also raise `hot` and `gap`. Nothing in the labelled
   data separates these.
7. **`Full Cooling` time falls steadily from car 01 to car 08** on Test (74, 80, 80, 31,
   35, 26, 17, 1 rows). That looks like a train-position effect, not a fault, and is not
   used.

## Appendix: files

| File | Role |
|---|---|
| `predict.py` | the submission script (`--input`, `--output`) |
| `acv_composite_model.ipynb` | full walk-through with all tables and plots |
| `acv_composite.py` | analysis module behind the notebook |
| `acv_features.py` | loaders and exploratory statistics used by the notebook |
