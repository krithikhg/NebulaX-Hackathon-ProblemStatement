// Numeric helpers that mirror the numpy / pandas semantics the Python pipeline uses.

export const f32 = Math.fround;

/** Split CSV text into lines, dropping blank ones (pandas skips blank lines). */
export function lines(text) {
  const out = text.split(/\r?\n/);
  return out.filter((l) => l.trim() !== "");
}

export function sum(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s;
}

export function mean(a) {
  return a.length ? sum(a) / a.length : NaN;
}

export function max(a) {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
  return m;
}

export function min(a) {
  let m = Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] < m) m = a[i];
  return m;
}

/** Population standard deviation (numpy default, ddof=0). */
export function std(a) {
  const m = mean(a);
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - m) ** 2;
  return Math.sqrt(s / a.length);
}

/** numpy.percentile with the default linear interpolation. */
export function percentile(a, q) {
  const s = Float64Array.from(a).sort();
  const pos = ((s.length - 1) * q) / 100;
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, s.length - 1);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

export function median(a) {
  return percentile(a, 50);
}

/** numpy.array_split: the first (n % k) parts get one extra element. */
export function arraySplit(a, k) {
  const n = a.length;
  const base = Math.floor(n / k);
  const extra = n % k;
  const parts = [];
  let start = 0;
  for (let i = 0; i < k; i++) {
    const len = base + (i < extra ? 1 : 0);
    parts.push(a.slice(start, start + len));
    start += len;
  }
  return parts;
}

/** Mean / median over the non-NaN values (pandas skipna). */
export function nanMean(a) {
  let s = 0;
  let n = 0;
  for (const v of a) if (!Number.isNaN(v)) { s += v; n++; }
  return n ? s / n : NaN;
}

export function nanMedian(a) {
  const v = a.filter((x) => !Number.isNaN(x));
  return v.length ? median(v) : NaN;
}

/** pandas.to_numeric(errors="coerce") for a single cell. */
export function toNumber(v) {
  if (typeof v === "number") return v;
  if (typeof v === "string") {
    const t = v.trim();
    if (t === "") return NaN;
    const n = Number(t);
    return Number.isFinite(n) || /^[+-]?inf(inity)?$/i.test(t) ? n : NaN;
  }
  return NaN;
}

export function round(x, digits) {
  const p = 10 ** digits;
  return Math.round(x * p) / p;
}

export function basename(name) {
  return String(name).split(/[\/]/).pop();
}

/** Render rows (array of objects) as CSV with the given columns. */
export function toCsv(rows, columns) {
  const esc = (v) => {
    const s = v === null || v === undefined ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [columns.join(","), ...rows.map((r) => columns.map((c) => esc(r[c])).join(","))].join("\n") + "\n";
}

/** Python's float repr, so CSV output matches pandas.to_csv. */
export function pyFloat(x) {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? "" : x > 0 ? "inf" : "-inf";
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-4 || a >= 1e16)) {
    return x.toExponential().replace(/e([+-])(\d)$/, "e$10$2");
  }
  return Number.isInteger(x) ? x.toFixed(1) : String(x);
}
