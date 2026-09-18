// TrainWhisper UI. Models run in a Web Worker (worker.js); this file handles input and rendering.
import { bars, segmentedLine } from "./charts.js";
import { remainingLife } from "./engine/shm.js";
import { pyFloat, toCsv } from "./engine/util.js";

const SAMPLE_BASE = "https://raw.githubusercontent.com/krithikhg/NebulaX-Hackathon-ProblemStatement/main/PS3/02_Datasets/";
const SAMPLES = {
  door: { kind: "Door", path: "Door/Test.csv" },
  acv: { kind: "ACV", path: "ACV/Test/acv_test_case.xlsx" },
  rail: { kind: "Rail corrugation", path: "Rail_Corrugation/Test/Test5.csv" },
  shm: { kind: "SHM", path: "SHM/Test/test01.csv" },
};

const BENCHMARK = [
  ["Door", "Gap segmentation + Random Forest", "Chronological holdout", "macro F1", 1.0, "selected"],
  ["Door", "Gap segmentation + fixed current threshold", "Chronological holdout", "macro F1", 0.86, "baseline"],
  ["ACV", "Cabin-temp deviation vs train median (+pressure)", "6 labelled cases", "rank decay", 0.958, "selected"],
  ["ACV", "Absolute cabin temperature", "6 labelled cases", "rank decay", 0.71, "baseline"],
  ["Rail", "Spectral features/side + tuned priors", "Nested CV", "macro F1", 0.743, "selected"],
  ["Rail", "Same features, untuned priors", "5-fold CV", "macro F1", 0.653, "ablation"],
  ["Rail", "Wavelength-normalised spectra", "5-fold CV", "macro F1", 0.556, "rejected"],
  ["Rail", "Per-axle-box model, aggregated", "Grouped CV", "macro F1", 0.633, "rejected"],
  ["Rail", "Always predict Normal", "—", "macro F1", 0.33, "baseline"],
  ["SHM", "Rainflow + fitted Miner (m, C)", "Leave-one-out", "1 − MAPE", 0.975, "selected"],
  ["SHM", "Mean of training labels", "Leave-one-out", "1 − MAPE", 0.0, "baseline"],
];

const $ = (sel) => document.querySelector(sel);
const results = $("#results");
const statusEl = $("#status");

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) node.append(c instanceof Node ? c : String(c));
  return node;
}

// ------------------------------------------------------------ tabs ----
const tabs = [...document.querySelectorAll('[role="tab"]')];
function selectTab(tab) {
  for (const t of tabs) {
    const on = t === tab;
    t.setAttribute("aria-selected", on);
    t.tabIndex = on ? 0 : -1;
    document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
  }
}
tabs.forEach((t, i) => {
  t.addEventListener("click", () => selectTab(t));
  t.addEventListener("keydown", (e) => {
    const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!d) return;
    const next = tabs[(i + d + tabs.length) % tabs.length];
    selectTab(next);
    next.focus();
  });
});

// ------------------------------------------------------- shared UI ----
const ICONS = {
  act: "M12 3 2 21h20L12 3Zm0 6v6m0 3v.01",
  plan: "M12 3 2 21h20L12 3Zm0 6v6m0 3v.01",
  watch: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm0 5v.01M12 11v6",
  healthy: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm-4 9 3 3 5-6",
};
const LEVELS = { "Act now": "act", Plan: "plan", Watch: "watch", Healthy: "healthy" };

function badge(level, message) {
  const key = LEVELS[level];
  const icon = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${ICONS[key]}"/></svg>`;
  const node = h("div", { class: `badge sev-${key}`, role: "note" });
  node.innerHTML = icon;
  node.append(h("div", {}, h("b", {}, level), message));
  return node;
}

function tile(label, value, sub) {
  return h("div", { class: "tile" }, h("div", { class: "label" }, label), h("div", { class: "value" }, value), sub ? h("div", { class: "sub" }, sub) : null);
}

/** columns: [{ key, label, num?, fmt?, pill? }] */
function table(rows, columns) {
  const head = h("tr", {}, columns.map((c) => h("th", { class: c.num ? "num" : "", scope: "col" }, c.label)));
  const body = rows.map((r) => h("tr", {}, columns.map((c) => {
    const v = c.fmt ? c.fmt(r[c.key], r) : r[c.key];
    const cell = c.pill ? h("span", { class: `pill ${c.pill(r[c.key], r)}` }, v) : v;
    return h("td", { class: c.num ? "num" : "" }, cell);
  })));
  return h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, head), h("tbody", {}, body)));
}

function download(rows, columns, filename, label = "Download predictions CSV") {
  return h("button", {
    class: "btn primary", type: "button",
    onclick: () => {
      const url = URL.createObjectURL(new Blob([toCsv(rows, columns)], { type: "text/csv" }));
      const a = h("a", { href: url, download: filename });
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    },
  }, label);
}

function explainer(textContent) {
  return h("details", {}, h("summary", {}, "How this works"), h("p", {}, textContent));
}

function chartCard(title, sub, legendItems, build) {
  const holder = h("div");
  const card = h("div", { class: "card" },
    h("h3", {}, title),
    sub ? h("p", { class: "sub" }, sub) : null,
    legendItems ? h("div", { class: "legend" }, legendItems.map(([label, color, shape]) => h("span", {}, h("i", { class: shape || "", style: `background:${color}` }), label))) : null,
    holder);
  queueMicrotask(() => build(holder));
  return card;
}

const fmt = (digits) => (v) => (Number.isFinite(v) ? v.toFixed(digits) : "—");
const predPill = (v) => (v === "Normal" ? "ok" : "bad");

// --------------------------------------------------------- renderers ----
function showDoor({ name, result, cycles }) {
  const bad = result.filter((r) => r.prediction !== "Normal");
  const meanOf = (rows) => rows.reduce((s, r) => s + r.mean_current_mA, 0) / rows.length;
  const normal = result.filter((r) => r.prediction === "Normal");
  const out = [];
  if (!bad.length) out.push(badge("Healthy", `${result.length} door cycles checked, all normal.`));
  else {
    out.push(badge(bad.length >= 5 ? "Act now" : "Plan",
      `${bad.length} of ${result.length} cycles show abnormal closing resistance. Inspect the slide rail for debris, ` +
      "check the rubber strip for jamming and the door leaf for deformation at the next engineering hours."));
  }
  out.push(h("div", { class: "tiles" },
    tile("Cycles detected", result.length),
    tile("Abnormal cycles", bad.length),
    tile("Mean current, abnormal cycles", bad.length ? `${meanOf(bad).toFixed(0)} mA` : "—",
      normal.length ? `vs ${meanOf(normal).toFixed(0)} mA for normal cycles` : null)));

  out.push(chartCard("Door motor current", `${name} — abnormal-resistance cycles shaded.`,
    [["Motor current", "var(--text-2)", "line"], ["Abnormal-resistance cycle", "var(--critical)", "span"]],
    (holder) => segmentedLine(holder, {
      cycles,
      yLabel: "Motor current (mA)",
      ariaLabel: `Door motor current across ${cycles.length} cycles; ${bad.length} abnormal cycles shaded`,
      describe: (p, ci) => [`${p[1].toFixed(0)} mA`, [
        `Cycle ${ci + 1} — ${result[ci].prediction}`,
        new Date(p[2]).toISOString().replace("T", " ").replace("Z", ""),
      ]],
    })));

  out.push(table(result, [
    { key: "start_time", label: "Start" },
    { key: "end_time", label: "End" },
    { key: "prediction", label: "Prediction", pill: predPill },
    { key: "confidence", label: "Confidence", num: true, fmt: fmt(3) },
    { key: "mean_current_mA", label: "Mean current (mA)", num: true, fmt: fmt(1) },
    { key: "peak_current_mA", label: "Peak current (mA)", num: true, fmt: fmt(1) },
  ]));
  out.push(h("div", { class: "actions" }, download(result, ["start_time", "end_time", "prediction"], "door_predictions.csv")));
  out.push(explainer(
    "Cycles are found by splitting the stream wherever the gap between consecutive samples exceeds 0.1 s — inside a " +
    "cycle the controller logs every 20 ms. On the labelled training stream this reproduced all 110 ground-truth " +
    "boundaries exactly. Each cycle is then classified from its motor current, voltage and back-EMF profile."));
  return out;
}

function showAcv({ name, ranked }) {
  const top = ranked[0];
  const out = [badge("Plan",
    `Car ${top.car} is the most likely refrigerant leak — its cabin runs ${top.cabin_temp_dev_C >= 0 ? "+" : ""}` +
    `${top.cabin_temp_dev_C.toFixed(2)} °C hotter than the train median under the same conditions. ` +
    "Pressure-test the circuit and check for oil traces at the joints.")];

  out.push(chartCard("Cabin temperature vs train median", "Cars in rank order, most likely leak first.",
    [["Most likely leak", "var(--serious)"], ["Other cars", "var(--neutral-bar)"]],
    (holder) => bars(holder, {
      labels: ranked.map((r) => `Car ${r.car}`),
      series: [{ name: "Deviation", values: ranked.map((r) => r.cabin_temp_dev_C), colors: ranked.map((_, i) => (i ? "var(--neutral-bar)" : "var(--serious)")) }],
      yLabel: "Deviation (°C)",
      ariaLabel: `Cabin temperature deviation per car; car ${top.car} highest`,
      tip: (_, i) => [`${ranked[i].cabin_temp_dev_C >= 0 ? "+" : ""}${ranked[i].cabin_temp_dev_C.toFixed(3)} °C`, [`Car ${ranked[i].car} — rank ${i + 1} of ${ranked.length}`]],
    })));

  const cols = [
    { key: "rank", label: "Rank", num: true },
    { key: "car", label: "Car" },
    { key: "score", label: "Score", num: true, fmt: fmt(4) },
    { key: "cabin_temp_dev_C", label: "Cabin temp dev (°C)", num: true, fmt: fmt(4) },
  ];
  if ("low_pressure_dev" in top) cols.push({ key: "low_pressure_dev", label: "Low-pressure dev", num: true, fmt: fmt(4) });
  out.push(table(ranked.map((r, i) => ({ ...r, rank: i + 1 })), cols));

  const row = { file_id: name.split(/[\\/]/).pop(), ranked_cars: ranked.map((r) => r.car).join("|") };
  out.push(h("div", { class: "actions" },
    download([row], ["file_id", "ranked_cars"], "acv_predictions.csv"),
    h("span", { class: "caption" }, "Submission row: ", h("code", {}, row.ranked_cars))));
  out.push(explainer(
    "The train is its own control group: at each timestamp every car's cabin temperature is compared with the median " +
    "across cars, so weather, time of day and setpoint changes cancel out. Cars are ranked by mean deviation. The loader " +
    "reads each file's own headers, so workbooks with 8 or 59 parameters per car both work."));
  return out;
}

function showRail({ result, bands, bandLabels }) {
  const faults = result.filter((r) => r.prediction !== "Normal");
  const out = [];
  if (!faults.length) out.push(badge("Healthy", `${result.length} recordings checked, no corrugation detected.`));
  else {
    const sides = [...new Set(faults.map((r) => r.prediction))].sort().join(", ");
    out.push(badge("Plan", `Corrugation detected in ${faults.length} of ${result.length} recordings (${sides}). ` +
      "Schedule rail grinding on the affected side and re-measure after the next grinding cycle."));
  }
  out.push(table(result, [
    { key: "file_id", label: "File" },
    { key: "prediction", label: "Prediction", pill: predPill },
    { key: "confidence", label: "Confidence", num: true, fmt: fmt(3) },
    { key: "speed_m_s", label: "Speed (m/s)", num: true, fmt: fmt(2) },
  ]));
  out.push(h("div", { class: "actions" }, download(result, ["file_id", "prediction"], "rail_predictions.csv")));

  const holder = h("div");
  const caption = h("p", { class: "caption" });
  const select = h("select", { "aria-label": "Recording to inspect" }, bands.map((b) => h("option", { value: b.file_id }, b.file_id)));
  const draw = () => {
    const b = bands.find((x) => x.file_id === select.value);
    holder.replaceChildren();
    const inner = h("div");
    holder.append(inner);
    bars(inner, {
      labels: bandLabels,
      series: [{ name: "Side I", values: b.side1, color: "var(--series-1)" }, { name: "Side II", values: b.side2, color: "var(--series-2)" }],
      yLabel: "Log energy",
      xLabel: "Frequency band (Hz)",
      height: 300,
      ariaLabel: `Vibration band energy per rail side for ${b.file_id}`,
      tip: (si, i) => [(si ? b.side2 : b.side1)[i].toFixed(3), [`${si ? "Side II" : "Side I"} · ${bandLabels[i]} Hz`, `Side I − Side II: ${(b.side1[i] - b.side2[i]).toFixed(3)}`]],
    });
    const r = result.find((x) => x.file_id === b.file_id);
    caption.textContent = `${b.file_id}: ${r.prediction}. Speed ${b.speed.toFixed(1)} m/s — corrugation shows as a one-sided energy excess, which is why the two sides are compared directly.`;
  };
  select.addEventListener("change", draw);
  const firstFault = faults[0];
  if (firstFault) select.value = firstFault.file_id;
  out.push(h("div", { class: "card" },
    h("h3", {}, "Inspect a recording"),
    h("p", { class: "sub" }, "Mean vibration log-energy per frequency band across each side's 32 axle boxes."),
    h("div", { class: "actions", style: "margin-bottom:8px" }, select),
    h("div", { class: "legend" }, h("span", {}, h("i", { style: "background:var(--series-1)" }), "Side I"), h("span", {}, h("i", { style: "background:var(--series-2)" }), "Side II")),
    holder, caption));
  queueMicrotask(draw);
  out.push(explainer(
    "Per channel: RMS, peak, kurtosis and log energy in 9 frequency bands, aggregated per rail side across its 32 axle " +
    "boxes, plus side-difference features. Raw speed is deliberately not a feature, and stationary recordings are " +
    "forced to Normal — with no wheel-rail excitation there is no corrugation signature to detect."));
  return out;
}

function showShm({ result, histogram, histogramFile, fit }) {
  const worst = result.reduce((a, b) => (b.prediction > a.prediction ? b : a));
  const left = remainingLife(worst.prediction);
  const leftText = left < 1 ? "less than one more segment" : `about ${left.toFixed(left < 10 ? 1 : 0)} more segments`;
  const out = [badge(worst.prediction < 0.5 ? "Watch" : "Plan",
    `Highest cumulative damage: ${worst.prediction.toFixed(3)} on ${worst.file_id} (Miner's rule fails at D = 1). ` +
    `At this rate the measurement point has ${leftText} of equivalent service before ` +
    "reaching the fatigue limit.")];

  out.push(table(result, [
    { key: "file_id", label: "File" },
    { key: "prediction", label: "Cumulative damage D", num: true, fmt: fmt(4) },
    { key: "life", label: "Segments to D = 1", num: true, fmt: (_, r) => { const v = remainingLife(r.prediction); return Number.isFinite(v) ? v.toFixed(0) : "—"; } },
  ]));
  out.push(h("div", { class: "actions" }, download(result.map((r) => ({ ...r, prediction: pyFloat(r.prediction) })), ["file_id", "prediction"], "shm_predictions.csv")));

  out.push(chartCard("Rainflow cycle histogram", `${histogramFile} — cycle count per stress-amplitude bin (log scale).`, null,
    (holder) => bars(holder, {
      labels: histogram.map((b) => b.lo.toFixed(1)),
      series: [{ name: "Cycles", values: histogram.map((b) => b.count), color: "var(--series-1)" }],
      yLabel: "Cycle count",
      xLabel: "Stress amplitude (bin start)",
      log: true,
      ariaLabel: `Rainflow cycle histogram for ${histogramFile}`,
      tip: (_, i) => [`${histogram[i].count} cycles`, [`Amplitude ${histogram[i].lo.toFixed(2)}–${histogram[i].hi.toFixed(2)}`]],
    })));

  if (result.length > 1) {
    out.push(chartCard("Cumulative damage per file", "Miner's rule predicts fatigue failure at D = 1.", null,
      (holder) => bars(holder, {
        labels: result.map((r) => r.file_id),
        series: [{ name: "Damage", values: result.map((r) => r.prediction), color: "var(--series-1)" }],
        yLabel: "Cumulative damage D",
          ariaLabel: "Cumulative fatigue damage per file",
        tip: (_, i) => [result[i].prediction.toFixed(4), [result[i].file_id]],
      })));
  }
  out.push(explainer(
    `Rainflow counting extracts stress cycles, then Miner's rule sums the damage using an S-N curve fitted to the ` +
    `labelled files: m = ${fit.m.toFixed(2)}, C = ${fit.C.toPrecision(3)}. Leave-one-out error is ${(fit.loo_mape * 100).toFixed(1)}%.`));
  return out;
}

const RENDER = { Door: showDoor, ACV: showAcv, "Rail corrugation": showRail, SHM: showShm };

// ---------------------------------------------------------- running ----
function detectSubsystem(name, head) {
  const lower = name.toLowerCase();
  if (lower.endsWith(".xlsx") || lower.endsWith(".xls")) return "ACV";
  const text = new TextDecoder().decode(head).slice(0, 400);
  if (text.includes("Datetime") && text.includes("Motor current")) return "Door";
  if (text.includes("Vibration of bearing")) return "Rail corrugation";
  const first = text.split(/\r?\n/)[0] || "";
  if (first && !first.includes(",")) return "SHM";
  return null;
}

function setStatus(message, { error = false, done, total } = {}) {
  statusEl.hidden = !message;
  statusEl.classList.toggle("error", error);
  statusEl.replaceChildren();
  if (!message) return;
  if (!error) statusEl.append(total ? h("progress", { max: total, value: done }) : h("progress"));
  statusEl.append(message);
}

let worker = null;
let busy = false;

function runWorker(kind, files) {
  if (!worker) worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  return new Promise((resolve, reject) => {
    worker.onmessage = ({ data }) => {
      if (data.type === "progress") setStatus(data.label, data);
      else if (data.type === "result") resolve(data.payload);
      else reject(new Error(data.message));
    };
    worker.onerror = (e) => reject(new Error(e.message || "The analysis worker failed to start."));
    worker.postMessage({ kind, files }, files.map((f) => f.buffer));
  });
}

async function analyse(files, forcedKind) {
  if (busy || !files.length) return;
  busy = true;
  document.querySelectorAll(".samples button").forEach((b) => { b.disabled = true; });
  results.classList.add("stale");
  try {
    const choice = document.querySelector('input[name="kind"]:checked').value;
    let kind = choice === "auto" ? forcedKind : choice;
    if (!kind) kind = detectSubsystem(files[0].name, new Uint8Array(files[0].buffer.slice(0, 2048)));
    if (!kind) throw new Error("Could not identify the subsystem — pick one above.");
    if ((kind === "Door" || kind === "ACV") && files.length > 1) {
      files = files.slice(0, 1);
    }
    setStatus(`Analysing ${files.length === 1 ? files[0].name : `${files.length} files`}…`);
    const payload = await runWorker(kind, files);
    const nodes = RENDER[kind](payload);
    results.replaceChildren(choice === "auto" ? h("p", { class: "detected" }, `Detected: ${kind}`) : "", ...nodes);
    setStatus("");
  } catch (err) {
    setStatus(err.message, { error: true });
    if (worker) { worker.terminate(); worker = null; }
  } finally {
    results.classList.remove("stale");
    busy = false;
    document.querySelectorAll(".samples button").forEach((b) => { b.disabled = false; });
  }
}

async function readFiles(fileList) {
  const files = await Promise.all([...fileList].map(async (f) => ({ name: f.name, buffer: await f.arrayBuffer() })));
  files.sort((a, b) => a.name.length - b.name.length || a.name.localeCompare(b.name));
  return files;
}

const input = $("#file-input");
input.addEventListener("change", async () => {
  if (input.files.length) await analyse(await readFiles(input.files));
  input.value = "";
});

const zone = $("#dropzone");
zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("dragging"); });
zone.addEventListener("dragleave", () => zone.classList.remove("dragging"));
zone.addEventListener("drop", async (e) => {
  e.preventDefault();
  zone.classList.remove("dragging");
  if (e.dataTransfer.files.length) await analyse(await readFiles(e.dataTransfer.files));
});

document.querySelectorAll("[data-sample]").forEach((btn) => btn.addEventListener("click", async () => {
  const { kind, path } = SAMPLES[btn.dataset.sample];
  document.querySelector('input[name="kind"][value="auto"]').checked = true;
  setStatus(`Downloading sample ${path.split("/").pop()}…`);
  try {
    const res = await fetch(SAMPLE_BASE + path);
    if (!res.ok) throw new Error(`Sample download failed (${res.status}).`);
    await analyse([{ name: path.split("/").pop(), buffer: await res.arrayBuffer() }], kind);
  } catch (err) {
    setStatus(err.message, { error: true });
  }
}));

// ------------------------------------------------------- benchmark ----
$("#bench-table").replaceWith(table(
  BENCHMARK.map(([Subsystem, Model, Validation, Metric, Score, Status]) => ({ Subsystem, Model, Validation, Metric, Score, Status })),
  [
    { key: "Subsystem", label: "Subsystem" },
    { key: "Model", label: "Model" },
    { key: "Validation", label: "Validation" },
    { key: "Metric", label: "Metric" },
    { key: "Score", label: "Score", num: true, fmt: fmt(3) },
    { key: "Status", label: "Status", pill: (v) => (v === "selected" ? "ok" : v === "rejected" ? "bad" : "") },
  ],
).querySelector("table"));
