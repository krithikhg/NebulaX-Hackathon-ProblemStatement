// Door subsystem: gap segmentation + random forest. Port of door.py.
import { predictProba } from "./forest.js";
import { arraySplit, lines, max, mean, median, min, percentile, round, std, sum } from "./util.js";

export const GAP_SECONDS = 0.1;
export const SIGNALS = { c: "Motor current(mA)", v: "Motor Voltage(10mV)", e: "Motor electrodynamic force" };
export const ABNORMAL = "Abnormal resistance";
export const NORMAL = "Normal";

/** "2023-7-5-0-0-3-760" -> epoch milliseconds (Y-M-D-H-M-S-ms, not zero padded). */
export function parseDoorTime(s) {
  const p = String(s).trim().split("-").map((x) => parseInt(x, 10));
  return Date.UTC(p[0], p[1] - 1, p[2], p[3], p[4], p[5], p[6]);
}

export function loadStream(text) {
  const rows = lines(text);
  const header = rows[0].split(",").map((c) => c.trim());
  const iDt = header.indexOf("Datetime");
  if (iDt < 0) throw new Error("No 'Datetime' column - is this a door controller stream?");
  const cols = Object.fromEntries(header.map((h) => [h, []]));
  const datetime = [];
  for (let r = 1; r < rows.length; r++) {
    const cells = rows[r].split(",");
    datetime.push(cells[iDt].trim());
    header.forEach((h, j) => { if (j !== iDt) cols[h].push(parseFloat(cells[j])); });
  }
  cols.Datetime = datetime;
  cols.t = datetime.map(parseDoorTime);
  return { cols, n: datetime.length };
}

/** Segment id per row: a new cycle starts wherever the time gap exceeds GAP_SECONDS. */
export function segment(stream, gapSeconds = GAP_SECONDS) {
  const t = stream.cols.t;
  const seg = new Int32Array(stream.n);
  for (let i = 1; i < stream.n; i++) seg[i] = seg[i - 1] + ((t[i] - t[i - 1]) / 1000 > gapSeconds ? 1 : 0);
  return seg;
}

function segmentFeatures(cols, a, b) {
  const t = cols.t.slice(a, b);
  const f = { n: b - a, dur: (max(t) - min(t)) / 1000 };
  for (const [key, col] of Object.entries(SIGNALS)) {
    const x = cols[col].slice(a, b);
    f[`${key}_mean`] = mean(x);
    f[`${key}_max`] = max(x);
    f[`${key}_std`] = std(x);
    f[`${key}_med`] = median(x);
    f[`${key}_q90`] = percentile(x, 90);
    f[`${key}_auc`] = sum(x);
    arraySplit(x, 3).forEach((part, i) => { f[`${key}_m${i}`] = mean(part); });
  }
  const pos = cols["Door leaf position"].slice(a, b);
  f.p_range = max(pos) - min(pos);
  f.ratio_ce = f.c_mean / (f.e_mean + 1.0); // current per unit back-EMF
  f.closing = Math.trunc(max(cols["Door is closing"].slice(a, b)));
  return f;
}

/** Predictions per door cycle, plus the parsed stream for plotting. */
export function predict(text, model) {
  const stream = loadStream(text);
  const seg = segment(stream);
  const { cols } = stream;
  const bounds = [];
  let start = 0;
  for (let i = 1; i <= stream.n; i++) {
    if (i === stream.n || seg[i] !== seg[i - 1]) { bounds.push([start, i]); start = i; }
  }
  const feats = bounds.map(([a, b]) => segmentFeatures(cols, a, b));
  const proba = predictProba(model, feats).map((p) => p[1]);
  const result = bounds.map(([a, b], k) => ({
    start_time: cols.Datetime[a],
    end_time: cols.Datetime[b - 1],
    rows: [a, b],
    prediction: proba[k] >= 0.5 ? ABNORMAL : NORMAL,
    confidence: round(Math.max(proba[k], 1 - proba[k]), 4),
    mean_current_mA: round(feats[k].c_mean, 1),
    peak_current_mA: round(feats[k].c_max, 1),
  }));
  return { result, stream };
}
