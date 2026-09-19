// Model validation results shown on the "How reliable is it?" tab.
// UPDATE THIS FILE when a model is replaced. It is the only place these numbers live.

/** Headline per subsystem: the selected model's out-of-sample score, in plain words. */
export const HEADLINES = {
  Door: {
    score: 1.0, metric: "IoU-weighted F1",
    plain: "All held-out cycles segmented with exact boundaries and classified correctly.",
    scale: "1.00 = every segment matched (IoU) with the correct label",
    validation: "Chronological hold-out on Train.csv",
  },
  ACV: {
    score: 0.958, metric: "Rank-decay score",
    plain: "Faulty car ranked first or near the top across the 6 labelled cases.",
    scale: "1.00 = faulty car always ranked 1st; partial credit for 2nd or 3rd",
    validation: "6 labelled cases, no parameters fitted to labels",
  },
  "Rail corrugation": {
    score: 0.743, metric: "Macro F1",
    plain: "Side I is the weakest class (F1 about 0.56).",
    scale: "Unweighted mean of per-class F1 over Normal, Side I, Side II",
    validation: "Nested CV, priors tuned inside each fold",
  },
  SHM: {
    score: 0.975, metric: "1 − MAPE",
    plain: "Mean absolute percentage error about 2.5%.",
    scale: "1.00 = exact; 0.90 = 10% mean error",
    validation: "Leave-one-out, 64 labelled segments",
  },
};

/** Every approach tried, including baselines and the ones rejected on evidence. */
export const BENCHMARK = [
  ["Door", "Gap segmentation + Random Forest", "Chronological holdout", "macro F1", 1.0, "selected"],
  ["Door", "Gap segmentation + fixed current threshold", "Chronological holdout", "macro F1", 0.86, "baseline"],
  ["ACV", "Cabin-temp deviation vs train median (+pressure)", "6 labelled cases", "rank decay", 0.958, "selected"],
  ["ACV", "Absolute cabin temperature", "6 labelled cases", "rank decay", 0.71, "baseline"],
  ["Rail corrugation", "Spectral features/side + tuned priors", "Nested CV", "macro F1", 0.743, "selected"],
  ["Rail corrugation", "Same features, untuned priors", "5-fold CV", "macro F1", 0.653, "ablation"],
  ["Rail corrugation", "Wavelength-normalised spectra", "5-fold CV", "macro F1", 0.556, "rejected"],
  ["Rail corrugation", "Per-axle-box model, aggregated", "Grouped CV", "macro F1", 0.633, "rejected"],
  ["Rail corrugation", "Always predict Normal", "-", "macro F1", 0.33, "baseline"],
  ["SHM", "Rainflow + fitted Miner (m, C)", "Leave-one-out", "1 − MAPE", 0.975, "selected"],
  ["SHM", "Mean of training labels", "Leave-one-out", "1 − MAPE", 0.0, "baseline"],
];
