// Runs the models off the main thread. Receives { kind, files: [{ name, blob }] },
// posts { type: "progress" | "result" | "error" }.
//
// Each handler returns the payload the UI renders (see "UI payload contract" in README.md).
// When a model is swapped, keep these payload shapes and the UI keeps working.
import * as acv from "./engine/acv.js";
import * as door from "./engine/door.js";
import * as rail from "./engine/rail.js";
import * as shm from "./engine/shm.js";

const models = {};
async function model(name) {
  if (!models[name]) {
    const res = await fetch(new URL(`../models/${name}`, import.meta.url));
    if (!res.ok) throw new Error(`Could not load model ${name} (${res.status})`);
    models[name] = await res.json();
  }
  return models[name];
}

const decode = (buffer) => new TextDecoder().decode(buffer);
const progress = (done, total, label) => postMessage({ type: "progress", done, total, label });

/** Keep the plot light: min/max per bucket preserves peaks while dropping points. */
function decimate(xs, ys, buckets) {
  if (xs.length <= buckets * 2) return xs.map((x, i) => [x, ys[i]]);
  const size = xs.length / buckets;
  const out = [];
  for (let b = 0; b < buckets; b++) {
    const a = Math.floor(b * size);
    const e = Math.min(xs.length, Math.floor((b + 1) * size));
    let lo = a;
    let hi = a;
    for (let i = a; i < e; i++) {
      if (ys[i] < ys[lo]) lo = i;
      if (ys[i] > ys[hi]) hi = i;
    }
    for (const i of lo < hi ? [lo, hi] : [hi, lo]) out.push([xs[i], ys[i]]);
  }
  return out;
}

/** Rainflow histogram: equal-width amplitude bins, weighted by cycle count. */
function histogram(cycles, bins = 40) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const [a] of cycles) { lo = Math.min(lo, a); hi = Math.max(hi, a); }
  const width = (hi - lo) / bins || 1;
  const counts = new Array(bins).fill(0);
  for (const [a, n] of cycles) counts[Math.min(bins - 1, Math.floor((a - lo) / width))] += n;
  return counts.map((count, k) => ({ lo: lo + k * width, hi: lo + (k + 1) * width, count }));
}

const handlers = {
  async Door(files) {
    progress(0, 1, "segmenting cycles");
    const { result, stream } = door.predict(decode(await files[0].blob.arrayBuffer()), await model("door_rf.json"));
    const { cols } = stream;
    const current = cols[door.SIGNALS.c];
    const closing = cols["Door is closing"];
    const cycles = result.map((r) => {
      const [a, b] = r.rows;
      const idx = Array.from({ length: b - a }, (_, k) => a + k);
      let isClosing = false;
      if (closing) for (const i of idx) if (closing[i] >= 1) { isClosing = true; break; }
      return {
        // [sample index, current mA, epoch ms]
        points: decimate(idx, idx.map((i) => current[i]), 150).map(([i, y]) => [i, y, cols.t[i]]),
        abnormal: r.prediction === door.ABNORMAL,
        operation: closing ? (isClosing ? "Closing" : "Opening") : null,
        t0: cols.t[a],
        duration_s: (cols.t[b - 1] - cols.t[a]) / 1000,
      };
    });
    return { name: files[0].name, result, cycles };
  },

  async ACV(files) {
    progress(0, 2, "reading workbook");
    const XLSX = await import("../vendor/xlsx.mjs");
    const wb = XLSX.read(await files[0].blob.arrayBuffer(), {
      type: "array", dense: true, sheets: 0,
      cellText: false, cellHTML: false, cellFormula: false, cellStyles: false, cellNF: false,
    });
    const rows = XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]], { header: 1, raw: true, defval: null, blankrows: true });
    progress(1, 2, "ranking cars");
    const frame = acv.frameFromRows(rows);
    // Display-only context: recording window and train number, when the file carries them.
    const find = (re) => frame.header.find((h) => !/^Car \d/.test(h) && re.test(h));
    const timeCol = find(/time/i);
    const trainCol = find(/train/i);
    const times = timeCol ? frame.cols[timeCol].filter((v) => v !== null && v !== "") : [];
    return {
      name: files[0].name,
      ranked: acv.rankCars(frame),
      samples: frame.n,
      train: trainCol ? frame.cols[trainCol].find((v) => v !== null && v !== "") : null,
      window: times.length ? [times[0], times[times.length - 1]] : null,
    };
  },

  async "Rail corrugation"(files) {
    const m = await model("rail_rf.json");
    const feats = [];
    for (let i = 0; i < files.length; i++) {
      progress(i, files.length, files.length > 1
        ? `extracting features, file ${i + 1} of ${files.length}`
        : "extracting features");
      feats.push({ ...rail.extractFeatures(decode(await files[i].blob.arrayBuffer())), filename: files[i].name });
    }
    const result = rail.predictFeatures(feats, m);
    const bands = feats.map((f) => ({
      file_id: f.filename,
      speed: f._speed,
      stationary: f._speed < m.stationary_speed,
      side1: rail.BANDS.map((_, i) => f[`s1vib_b${i}_mean`]),
      side2: rail.BANDS.map((_, i) => f[`s2vib_b${i}_mean`]),
    }));
    return { result, bands, bandLabels: rail.BANDS.map(([lo, hi]) => `${lo}–${hi}`) };
  },

  async SHM(files) {
    const fit = await model("shm_fit.json");
    const result = [];
    const detail = [];
    for (let i = 0; i < files.length; i++) {
      progress(i, files.length, files.length > 1
        ? `rainflow counting, file ${i + 1} of ${files.length}`
        : "rainflow counting");
      const signal = shm.loadSignal(decode(await files[i].blob.arrayBuffer()));
      const { damage, cycles } = shm.predictSignal(signal, fit);
      result.push({ file_id: files[i].name, prediction: damage });
      let n = 0;
      let peak = 0;
      for (const [a, c] of cycles) { n += c; peak = Math.max(peak, a); }
      detail.push({ file_id: files[i].name, samples: signal.length, cycles: n, peak_amplitude: peak, histogram: histogram(cycles) });
    }
    return { result, detail, fit };
  },
};

onmessage = async ({ data }) => {
  try {
    const handler = handlers[data.kind];
    if (!handler) throw new Error(`Unknown subsystem: ${data.kind}`);
    postMessage({ type: "result", kind: data.kind, payload: await handler(data.files) });
  } catch (err) {
    postMessage({ type: "error", message: err && err.message ? err.message : String(err) });
  }
};
