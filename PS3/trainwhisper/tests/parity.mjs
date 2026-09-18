// Checks the browser engine (public/js/engine) against the Python pipeline's outputs.
//
//   cd tests && npm install && PS3_DATA_ROOT=/path/to/PS3/02_Datasets npm test
//
// Compares every test-set prediction with outputs/*.csv (written by predict.py)
// and checks the ACV ranking on every labelled training case.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import XLSX from "xlsx";

import * as acv from "../public/js/engine/acv.js";
import * as door from "../public/js/engine/door.js";
import * as rail from "../public/js/engine/rail.js";
import * as shm from "../public/js/engine/shm.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const APP = path.join(HERE, "..");
const DATA = process.env.PS3_DATA_ROOT || path.join(APP, "..", "02_Datasets");

const read = (...p) => fs.readFileSync(path.join(...p), "utf8");
const model = (name) => JSON.parse(read(APP, "public", "models", name));
const csv = (text) => {
  const [head, ...rows] = text.trim().split(/\r?\n/);
  const cols = head.split(",");
  return rows.map((r) => Object.fromEntries(r.split(",").map((v, i) => [cols[i], v])));
};
const byName = (a, b) => a.length - b.length || (a < b ? -1 : a > b ? 1 : 0);
const xlsxRows = (file) => {
  const wb = XLSX.read(fs.readFileSync(file), {
    type: "buffer", dense: true, sheets: 0,
    cellText: false, cellHTML: false, cellFormula: false, cellStyles: false, cellNF: false,
  });
  return XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]], { header: 1, raw: true, defval: null, blankrows: true });
};

let failures = 0;
function check(label, ok, detail = "") {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${detail ? `  (${detail})` : ""}`);
  if (!ok) failures++;
}

// Door
{
  const { result } = door.predict(read(DATA, "Door", "Test.csv"), model("door_rf.json"));
  const want = csv(read(APP, "outputs", "door_predictions.csv"));
  const same = result.length === want.length && result.every((r, i) =>
    r.start_time === want[i].start_time && r.end_time === want[i].end_time && r.prediction === want[i].prediction);
  check("door test predictions match", same, `${result.length} cycles`);

  const train = door.predict(read(DATA, "Door", "Train.csv"), model("door_rf.json")).result;
  const truth = csv(read(DATA, "Door", "Train_Segments_Answer.csv"));
  const exact = train.length === truth.length &&
    train.every((r, i) => r.start_time === truth[i].start_time && r.end_time === truth[i].end_time);
  check("door segmentation reproduces all training boundaries", exact, `${train.length}/${truth.length}`);
}

// ACV
{
  const ranked = acv.rankCars(acv.frameFromRows(xlsxRows(path.join(DATA, "ACV", "Test", "acv_test_case.xlsx"))));
  const got = ranked.map((r) => r.car).join("|");
  const want = csv(read(APP, "outputs", "acv_predictions.csv"))[0].ranked_cars;
  check("acv test ranking matches", got === want, got);

  // Top-3 per labelled training case, as printed by Python's acv.validate().
  const PY_TOP3 = {
    "acv_case_01.xlsx": "01|03|04", "acv_case_02.xlsx": "03|02|07", "acv_case_03.xlsx": "03|08|04",
    "acv_case_04.xlsx": "04|01|02", "acv_case_05.xlsx": "04|02|07", "acv_case_06.xlsx": "06|08|04",
  };
  for (const [file, want3] of Object.entries(PY_TOP3)) {
    const t = Date.now();
    const top3 = acv.rankCars(acv.frameFromRows(xlsxRows(path.join(DATA, "ACV", "Train", file)))).slice(0, 3).map((r) => r.car).join("|");
    check(`acv ${file} top-3 matches Python`, top3 === want3, `${top3}, ${((Date.now() - t) / 1000).toFixed(1)} s`);
  }
}

// SHM
{
  const fit = model("shm_fit.json");
  const want = csv(read(APP, "outputs", "shm_predictions.csv"));
  let worst = 0;
  for (const w of want) {
    const { damage } = shm.predictSignal(shm.loadSignal(read(DATA, "SHM", "Test", w.file_id)), fit);
    worst = Math.max(worst, Math.abs(damage - Number(w.prediction)) / Number(w.prediction));
  }
  check("shm test damage matches", worst < 1e-9, `max rel diff ${worst.toExponential(1)}`);
}

// Rail
{
  const m = model("rail_rf.json");
  const want = csv(read(APP, "outputs", "rail_predictions.csv"));
  const dir = path.join(DATA, "Rail_Corrugation", "Test");
  const files = fs.readdirSync(dir).sort(byName);
  const feats = files.map((f) => ({ ...rail.extractFeatures(read(dir, f)), filename: f }));
  const got = rail.predictFeatures(feats, m);
  const mism = got.filter((r, i) => r.file_id !== want[i].file_id || r.prediction !== want[i].prediction);
  check("rail test predictions match", got.length === want.length && !mism.length,
    `${got.length} files, ${mism.length} mismatches`);

  // Feature-level check against the cached Python training features.
  const cache = csv(read(APP, "models", "rail_train_features.csv"));
  const tdir = path.join(DATA, "Rail_Corrugation", "Train");
  let worst = 0;
  for (const row of cache.slice(0, 8)) {
    const f = rail.extractFeatures(read(tdir, row.filename));
    for (const k of m.features) worst = Math.max(worst, Math.abs(f[k] - Number(row[k])) / (Math.abs(Number(row[k])) + 1e-3));
  }
  check("rail features match Python cache (8 train files)", worst < 1e-4, `max rel diff ${worst.toExponential(1)}`);
}

console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
process.exit(failures ? 1 : 0);
