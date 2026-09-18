// Random-forest inference over trees exported by export_models.py.
import { f32 } from "./util.js";

/**
 * Average per-tree class probabilities, as sklearn's predict_proba does.
 * Inputs are cast to float32 first, matching sklearn's own input casting, so
 * split decisions agree with the Python model bit for bit.
 */
export function predictProba(forest, rows) {
  const nClasses = forest.classes.length;
  const idx = forest.features;
  return rows.map((row) => {
    const x = idx.map((name) => f32(row[name]));
    const acc = new Float64Array(nClasses);
    for (const t of forest.trees) {
      let node = 0;
      while (t.f[node] !== -1) {
        node = x[t.f[node]] <= t.t[node] ? t.l[node] : t.r[node];
      }
      const v = t.v[node];
      for (let c = 0; c < nClasses; c++) acc[c] += v[c];
    }
    for (let c = 0; c < nClasses; c++) acc[c] /= forest.trees.length;
    return Array.from(acc);
  });
}
