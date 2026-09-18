// SHM subsystem: rainflow counting + fitted Miner's rule. Port of shm.py.
import { lines } from "./util.js";

/** Single-column stress file. No header row - the first line is already data. */
export function loadSignal(text) {
  const out = [];
  for (const l of lines(text)) {
    const v = Number(l.split(",")[0].trim());
    if (Number.isFinite(v)) out.push(v);
  }
  return out;
}

// Reversal points of the series, as rainflow 3.2 yields them (index, value).
function reversals(series) {
  const out = [];
  if (series.length < 2) return out;
  let x = series[1];
  let dLast = x - series[0];
  out.push(series[0]);
  let xNext;
  for (let i = 2; i < series.length; i++) {
    xNext = series[i];
    if (xNext === x) continue;
    const dNext = xNext - x;
    if (dLast * dNext < 0) out.push(x);
    x = xNext;
    dLast = dNext;
  }
  if (series.length > 2) out.push(xNext);
  return out;
}

/** Rainflow cycles (ASTM E1049-85 5.4.4) as [amplitude, count] pairs, amplitude = range / 2. */
export function cycles(series) {
  const out = [];
  const pts = [];
  let head = 0; // pts[head..] is the live stack; popleft advances head
  for (const p of reversals(series)) {
    pts.push(p);
    while (pts.length - head >= 3) {
      const x1 = pts[pts.length - 3];
      const x2 = pts[pts.length - 2];
      const x3 = pts[pts.length - 1];
      const X = Math.abs(x3 - x2);
      const Y = Math.abs(x2 - x1);
      if (X < Y) break;
      if (pts.length - head === 3) {
        out.push([Math.abs(pts[head] - pts[head + 1]) / 2, 0.5]);
        head++;
      } else {
        out.push([Y / 2, 1.0]);
        const last = pts.pop();
        pts.pop();
        pts.pop();
        pts.push(last);
      }
    }
  }
  for (; pts.length - head > 1; head++) out.push([Math.abs(pts[head] - pts[head + 1]) / 2, 0.5]);
  return out;
}

export function damageSum(cyc, m, cutoff) {
  let s = 0;
  for (const [amp, cnt] of cyc) if (amp >= cutoff) s += amp ** m * cnt;
  return s;
}

/** Miner's damage D = S / C for one stress segment, plus its cycles for plotting. */
export function predictSignal(signal, fit) {
  const cyc = cycles(signal);
  return { damage: damageSum(cyc, fit.m, fit.amplitude_cutoff) / fit.C, cycles: cyc };
}

/** Segments of identical service until Miner's D = 1 (fatigue failure). */
export function remainingLife(damagePerFile, segmentsSoFar = 1) {
  if (damagePerFile <= 0) return Infinity;
  return (1.0 - damagePerFile * segmentsSoFar) / damagePerFile;
}
