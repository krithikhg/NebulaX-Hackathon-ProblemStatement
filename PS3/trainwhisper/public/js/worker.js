// Runs the models off the main thread. Receives { kind, files: [{ name, buffer }] },
// posts { type: "progress" | "result" | "error" }.
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

const handlers = {
  async Door(files) {
    progress(0, 1, "Segmenting door cycles…");
    const { result, stream } = door.predict(decode(files[0].buffer), await model("door_rf.json"));
    // Plot against sample order so the 20 s+ idle gaps between cycles don't flatten the trace.
    const current = stream.cols[door.SIGNALS.c];
    const cycles = result.map((r) => {
      const idx = Array.from({ length: r.rows[1] - r.rows[0] }, (_, k) => r.rows[0] + k);
      return {
        points: decimate(idx, idx.map((i) => current[i]), 150).map(([i, y]) => [i, y, stream.cols.t[i]]),
        abnormal: r.prediction === door.ABNORMAL,
      };
    });
    return { name: files[0].name, result, cycles };
  },

  async ACV(files) {
    progress(0, 1, "Reading workbook…");
    const XLSX = await import("../vendor/xlsx.mjs");
    const wb = XLSX.read(files[0].buffer, {
      type: "array", dense: true, sheets: 0,
      cellText: false, cellHTML: false, cellFormula: false, cellStyles: false, cellNF: false,
    });
    const rows = XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]], { header: 1, raw: true, defval: null, blankrows: true });
    progress(1, 2, "Ranking cars…");
    return { name: files[0].name, ranked: acv.rankCars(acv.frameFromRows(rows)) };
  },

  async "Rail corrugation"(files) {
    const m = await model("rail_rf.json");
    const feats = [];
    for (let i = 0; i < files.length; i++) {
      progress(i, files.length, `Extracting spectral features… ${i}/${files.length} files`);
      feats.push({ ...rail.extractFeatures(decode(files[i].buffer)), filename: files[i].name });
    }
    const result = rail.predictFeatures(feats, m);
    const bands = feats.map((f) => ({
      file_id: f.filename,
      speed: f._speed,
      side1: rail.BANDS.map((_, i) => f[`s1vib_b${i}_mean`]),
      side2: rail.BANDS.map((_, i) => f[`s2vib_b${i}_mean`]),
    }));
    return { result, bands, bandLabels: rail.BANDS.map(([lo, hi]) => `${lo}–${hi}`) };
  },

  async SHM(files) {
    const fit = await model("shm_fit.json");
    const result = [];
    let first = null;
    for (let i = 0; i < files.length; i++) {
      progress(i, files.length, `Counting rainflow cycles… ${i}/${files.length} files`);
      const { damage, cycles } = shm.predictSignal(shm.loadSignal(decode(files[i].buffer)), fit);
      result.push({ file_id: files[i].name, prediction: damage });
      if (!first) first = { name: files[i].name, cycles };
    }
    // Rainflow histogram of the first file: 60 equal-width amplitude bins, weighted by cycle count.
    let lo = Infinity;
    let hi = -Infinity;
    for (const [a] of first.cycles) { lo = Math.min(lo, a); hi = Math.max(hi, a); }
    const width = (hi - lo) / 60 || 1;
    const counts = new Array(60).fill(0);
    for (const [a, n] of first.cycles) counts[Math.min(59, Math.floor((a - lo) / width))] += n;
    const histogram = counts.map((count, k) => ({ lo: lo + k * width, hi: lo + (k + 1) * width, count }));
    return { result, histogram, histogramFile: first.name, fit };
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
