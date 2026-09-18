// ACV subsystem: rank cars by cabin-temperature deviation from the train median. Port of acv.py.
import { nanMean, nanMedian, toNumber } from "./util.js";

export const PRESSURE_WEIGHT = 0.5;
// "Car model" is an identifying column, NOT a car - the two-digit pattern keeps it out.
const CAR_COL_RE = /^Car (\d{2}) - (.+)$/;

/** pandas-style header de-duplication: repeated names become "name.1", "name.2", ... */
function dedupe(header) {
  const seen = new Map();
  return Array.from(header, (h) => {
    let name = h === null || h === undefined ? "" : String(h);
    const count = seen.get(name) || 0;
    seen.set(name, count + 1);
    if (count) name = `${name}.${count}`;
    return name;
  });
}

/** rows: array of arrays from the first sheet, header row first (pandas.read_excel defaults). */
export function frameFromRows(rows) {
  const header = dedupe(rows[0] || []);
  const body = rows.slice(1);
  const cols = {};
  header.forEach((h, j) => { cols[h] = body.map((r) => (r ? r[j] : null)); });
  return { header, cols, n: body.length };
}

function carColumns(header) {
  const out = new Map();
  for (const c of header) {
    const m = CAR_COL_RE.exec(c.trim());
    if (!m) continue;
    if (!out.has(m[1])) out.set(m[1], new Map());
    out.get(m[1]).set(m[2], c);
  }
  return out;
}

function findParam(params, ...keywords) {
  for (const [name, col] of params) {
    const low = name.toLowerCase();
    if (keywords.every((k) => low.includes(k.toLowerCase()))) return col;
  }
  return null;
}

const numericCol = (frame, col) => frame.cols[col].map(toNumber);

function cabinSeries(frame, params) {
  const col = findParam(params, "indoor", "average", "temperature")
    || findParam(params, "passenger", "cabin", "temperature")
    || findParam(params, "observation", "temperature");
  return col ? numericCol(frame, col) : null;
}

function lowPressureSeries(frame, params) {
  const cols = [...params].filter(([n]) => n.toLowerCase().includes("low pressure")).map(([, c]) => c);
  if (!cols.length) return null;
  const series = cols.map((c) => numericCol(frame, c));
  return series[0].map((_, i) => nanMean(series.map((s) => s[i])));
}

/** Mean deviation of each car from the across-car median, per timestamp. */
function deviation(seriesByCar) {
  const cars = [...seriesByCar.keys()];
  const data = cars.map((c) => seriesByCar.get(c));
  const dev = cars.map(() => []);
  for (let i = 0; i < data[0].length; i++) {
    const med = nanMedian(data.map((s) => s[i]));
    data.forEach((s, k) => dev[k].push(s[i] - med));
  }
  return new Map(cars.map((c, k) => [c, nanMean(dev[k])]));
}

/** pandas Series.abs().max(): NaN entries are skipped (a car with no data must not blank the rest). */
function nanMaxAbs(values) {
  let m = NaN;
  for (const v of values) if (!Number.isNaN(v) && !(Math.abs(v) <= m)) m = Math.abs(v);
  return m;
}

/** Per-car evidence table, most- to least-likely faulty. */
export function rankCars(frame) {
  const cars = carColumns(frame.header);
  if (!cars.size) throw new Error("No 'Car NN - <parameter>' columns found in this file.");
  const temps = new Map();
  const pressures = new Map();
  for (const [car, params] of cars) {
    const s = cabinSeries(frame, params);
    if (s) temps.set(car, s);
    const p = lowPressureSeries(frame, params);
    if (p) pressures.set(car, p);
  }
  if (!temps.size) throw new Error("No cabin-temperature channel found for any car.");

  const tempDev = deviation(temps);
  const score = new Map(tempDev);
  let pressDev = null;
  if (pressures.size === cars.size) {
    pressDev = deviation(pressures);
    // Low-side pressure FALLS when refrigerant is lost, hence the minus.
    const spread = nanMaxAbs(pressDev.values()) || 1.0;
    const tMax = nanMaxAbs(tempDev.values());
    for (const car of new Set([...tempDev.keys(), ...pressDev.keys()])) {
      const t = tempDev.has(car) ? tempDev.get(car) : NaN;
      score.set(car, t + PRESSURE_WEIGHT * (-pressDev.get(car) / spread) * tMax);
    }
  }
  const rows = [...score].map(([car, s]) => {
    const r = { car, score: s, cabin_temp_dev_C: tempDev.has(car) ? tempDev.get(car) : NaN };
    if (pressDev) r.low_pressure_dev = pressDev.get(car);
    return r;
  });
  // Descending by score, NaN last (pandas sort_values).
  const nan = (r) => Number.isNaN(r.score);
  return rows.sort((a, b) => (nan(a) || nan(b) ? nan(a) - nan(b) : b.score - a.score));
}
