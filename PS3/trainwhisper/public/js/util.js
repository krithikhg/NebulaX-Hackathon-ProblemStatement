// Formatting helpers used by the views. The numeric work now happens on the
// server, so this is all that survives of the old js/engine/ ports.

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

/** Segments of identical service until Miner's D = 1 (fatigue failure). */
export function remainingLife(damagePerFile, segmentsSoFar = 1) {
  if (damagePerFile <= 0) return Infinity;
  return (1.0 - damagePerFile * segmentsSoFar) / damagePerFile;
}
