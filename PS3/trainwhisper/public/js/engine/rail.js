// Rail corrugation: spectral features per rail side + random forest with tuned priors. Port of rail.py.
import { rfftPower } from "./fft.js";
import { predictProba } from "./forest.js";
import { f32, lines, max, mean, round, std } from "./util.js";

export const FS = 10000;
export const WHEEL_DIAMETER_M = 0.85;
export const TEETH = 90;
export const BANDS = [[50, 200], [200, 400], [400, 600], [600, 900], [900, 1300],
  [1300, 1800], [1800, 2500], [2500, 3500], [3500, 5000]];
const STAT_NAMES = ["rms", "peak", "kurt", ...BANDS.map((_, i) => `b${i}`)];

/** Parse a recording into float32 columns (pandas read_csv with dtype=float32). */
export function loadRecording(text) {
  const rows = lines(text);
  const header = rows[0].split(",");
  const n = rows.length - 1;
  const data = header.map(() => new Float32Array(n));
  for (let r = 0; r < n; r++) {
    const cells = rows[r + 1].split(",");
    for (let j = 0; j < header.length; j++) data[j][r] = parseFloat(cells[j]);
  }
  return { header, data, n };
}

function channelStats(x, freqs) {
  // numpy computes these in float32; each result is rounded back to float32 to match.
  const m = f32(mean(x));
  const c = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) c[i] = f32(x[i] - m);
  let s2 = 0;
  let s4 = 0;
  let peak = 0;
  for (let i = 0; i < c.length; i++) {
    const v2 = c[i] * c[i];
    s2 += v2;
    s4 += v2 * v2;
    if (Math.abs(c[i]) > peak) peak = Math.abs(c[i]);
  }
  const m2 = s2 / c.length;
  const row = [f32(Math.sqrt(m2)), f32(peak), f32(s4 / c.length / (m2 * m2 + 1e-12))];
  const power = rfftPower(c);
  for (const [lo, hi] of BANDS) {
    let e = 0;
    for (let k = 0; k < freqs.length; k++) if (freqs[k] >= lo && freqs[k] < hi) e += power[k];
    row.push(f32(Math.log10(f32(e) + 1e-12)));
  }
  return row;
}

/** Per-side aggregated features for one recording (1 s at 10 kHz, 64 axle boxes). */
export function extractFeatures(text) {
  const { header, data, n } = loadRecording(text);

  // Speed from the 90-tooth wheel encoder over a 1 s window.
  let toggles = 0;
  for (let i = 1; i < n; i++) toggles += Math.abs(f32(data[0][i] - data[0][i - 1]));
  const speed = (toggles / 2 / TEETH) * Math.PI * WHEEL_DIAMETER_M;

  const val = 1.0 / (n * (1 / FS)); // numpy.fft.rfftfreq
  const freqs = Float64Array.from({ length: Math.floor(n / 2) + 1 }, (_, k) => k * val);
  const buckets = { "1vib": [], "1shk": [], "2vib": [], "2shk": [] };
  for (let j = 1; j < header.length; j++) {
    const c = header[j];
    // Odd positions ride the Side I rail, even positions Side II.
    const pos = parseInt(c.split("position ")[1].split(" ")[0], 10);
    const side = pos % 2 === 1 ? 1 : 2;
    const kind = c.toLowerCase().startsWith("vibration") ? "vib" : "shk";
    buckets[`${side}${kind}`].push(channelStats(data[j], freqs));
  }

  const out = { _speed: speed };
  for (const side of [1, 2]) {
    for (const kind of ["vib", "shk"]) {
      const A = buckets[`${side}${kind}`];
      STAT_NAMES.forEach((nm, i) => {
        const col = A.map((r) => r[i]);
        out[`s${side}${kind}_${nm}_mean`] = mean(col);
        out[`s${side}${kind}_${nm}_max`] = max(col);
        out[`s${side}${kind}_${nm}_std`] = std(col);
      });
    }
  }
  // Corrugation is one-sided, so the asymmetry between rails carries the signal.
  for (const kind of ["vib", "shk"]) {
    for (const nm of STAT_NAMES) {
      out[`d${kind}_${nm}`] = out[`s1${kind}_${nm}_mean`] - out[`s2${kind}_${nm}_mean`];
      out[`dmax${kind}_${nm}`] = out[`s1${kind}_${nm}_max`] - out[`s2${kind}_${nm}_max`];
    }
  }
  return out;
}

/** features: extractFeatures() results, each with a `filename`. */
export function predictFeatures(features, model) {
  const clean = features.map((f) => {
    const r = {};
    for (const name of model.features) r[name] = Number.isFinite(f[name]) ? f[name] : -12.0;
    return r;
  });
  const proba = predictProba(model, clean);
  return features.map((f, i) => {
    const p = proba[i];
    let best = 0;
    for (let c = 1; c < p.length; c++) if (p[c] * model.priors[c] > p[best] * model.priors[best]) best = c;
    // No wheel-rail excitation when stationary, so no corrugation signature to detect.
    const stationary = f._speed < model.stationary_speed;
    return {
      file_id: f.filename,
      prediction: stationary ? "Normal" : model.classes[best],
      confidence: round(Math.max(...p), 4),
      speed_m_s: round(f._speed, 2),
    };
  });
}
