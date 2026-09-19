// TrainWhisper UI shell. Models run in a Web Worker (worker.js); views.js turns each payload into
// page parts; this file handles routing, the side menu, uploads, the detail panel, zip and print.
import { BENCHMARK, HEADLINES } from "./benchmark.js";
import {
  $, KINDS, LEVELS, SYSTEMS, button, card, countUp, h, icon, iconButton, info, kindByKey, levelChip, now, saveBlob, segmented, stagger, stat, table, titled, toast,
} from "./ui.js";
import { hideTooltip, showTooltip } from "./charts.js";
import { VIEWS } from "./views.js";
import { makeZip } from "./zip.js";

const SAMPLE_BASE = "https://raw.githubusercontent.com/krithikhg/NebulaX-Hackathon-ProblemStatement/main/PS3/02_Datasets/";
const SAMPLES = {
  door: { kind: "Door", path: "Door/Test.csv" },
  acv: { kind: "ACV", path: "ACV/Test/acv_test_case.xlsx" },
  rail: { kind: "Rail corrugation", path: "Rail_Corrugation/Test/Test5.csv" },
  shm: { kind: "SHM", path: "SHM/Test/test02.csv" },
};
const SINGLE_FILE = new Set(["Door", "ACV"]);

const content = $("#content");
const statusEl = $("#status");

/** Latest result per subsystem: { payload, files, at, view } */
const session = new Map();
const tabOf = new Map(); // subsystem -> "summary" | "records" | "signals"
let route = "overview";
let notes = [];

// ------------------------------------------------------------- panel ----
const panel = $("#panel");
const scrim = $("#scrim");
let lastFocus = null;
function openPanel({ title, subtitle, chip, body }) {
  lastFocus = document.activeElement;
  $("#panel-title").replaceChildren(title);
  $("#panel-sub").replaceChildren(...[chip, subtitle ? h("span", {}, subtitle) : null].filter(Boolean));
  $("#panel-body").replaceChildren(...[body].flat(Infinity).filter(Boolean));
  document.querySelectorAll(".is-open").forEach((el) => el.classList.remove("is-open"));
  if (lastFocus && lastFocus !== document.body) lastFocus.classList.add("is-open");
  enhance($("#panel"));
  panel.hidden = false;
  scrim.hidden = false;
  requestAnimationFrame(() => panel.classList.add("open"));
  $("#panel-close").focus();
}
function closePanel() {
  if (panel.hidden) return;
  panel.classList.remove("open");
  hideTooltip();
  scrim.hidden = true;
  setTimeout(() => { panel.hidden = true; }, 180);
  document.querySelectorAll(".is-open").forEach((el) => el.classList.remove("is-open"));
  if (lastFocus && lastFocus.isConnected) lastFocus.focus();
}

/** Entrance order for list-like containers, and count-up for figures. */
const STAGGERED = ".page, .tab-body, .tiles, .stats, .stats-2, .prio, .strip, .train, .rec-grid, .gauges, .diverge, .checklist, .panel-body";
function enhance(root) {
  stagger(root, STAGGERED);
  if (root.matches && root.matches(".tab-body")) [...root.children].forEach((c, i) => c.style.setProperty("--i", i));
  countUp(root);
}

// Instant hover labels for dense marks (native title tooltips are slow to appear).
document.addEventListener("pointerover", (e) => {
  const t = e.target.closest && e.target.closest("[data-tip]");
  if (t) showTooltip(e, t.dataset.tip, t.dataset.tipSub ? [t.dataset.tipSub] : []);
});
document.addEventListener("pointerout", (e) => {
  const t = e.target.closest && e.target.closest("[data-tip]");
  if (t && !t.contains(e.relatedTarget)) hideTooltip();
});
$("#panel-close").addEventListener("click", closePanel);
scrim.addEventListener("click", closePanel);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closePanel(); closeMenu(); } });

// ------------------------------------------------------------ status ----
function setStatus(message, { error = false, done, total } = {}) {
  statusEl.hidden = !message;
  statusEl.classList.toggle("error", error);
  statusEl.replaceChildren();
  if (!message) return;
  statusEl.append(error ? icon("act", 16) : h("span", { class: "spinner", "aria-hidden": "true" }), h("span", {}, message));
  if (!error && total > 1) statusEl.append(h("progress", { max: total, value: done }));
  if (error) statusEl.append(h("button", { type: "button", class: "link", onclick: () => setStatus("") }, "Dismiss"));
}

// ------------------------------------------------------------ worker ----
let worker = null;
function runWorker(kind, files, label) {
  if (!worker) worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  return new Promise((resolve, reject) => {
    worker.onmessage = ({ data }) => {
      if (data.type === "progress") setStatus(`${label}: ${data.label}`, data);
      else if (data.type === "result") resolve(data.payload);
      else reject(new Error(data.message));
    };
    worker.onerror = (e) => reject(new Error(e.message || "The analysis could not start."));
    worker.postMessage({ kind, files });
  });
}

async function detect(file) {
  const lower = file.name.toLowerCase();
  if (lower.endsWith(".xlsx") || lower.endsWith(".xls")) return "ACV";
  const text = new TextDecoder().decode(await file.blob.slice(0, 2048).arrayBuffer()).slice(0, 600);
  if (text.includes("Datetime") && text.includes("Motor current")) return "Door";
  if (text.includes("Vibration of bearing")) return "Rail corrugation";
  const first = text.split(/\r?\n/)[0] || "";
  if (first && !first.includes(",") && Number.isFinite(Number(first.trim()))) return "SHM";
  return null;
}

const FRIENDLY_ERRORS = [
  [/Datetime/, "Not a door controller stream: no 'Datetime' column."],
  [/No 'Car NN/, "No per-car columns ('Car NN - ...') in this workbook."],
  [/cabin-temperature/, "No cabin temperature channel in this workbook."],
];
const friendly = (msg) => (FRIENDLY_ERRORS.find(([re]) => re.test(msg)) || [null, msg])[1];

let busy = false;
/** hint: subsystem to assume when a file can't be recognised (the page the user is on, or a sample). */
async function analyse(files, hint) {
  if (busy || !files.length) return;
  busy = true;
  document.body.classList.add("busy");
  notes = [];
  try {
    const groups = new Map();
    const unknown = [];
    for (const f of files) {
      const detected = await detect(f);
      const kind = detected || hint;
      if (!kind) { unknown.push(f.name); continue; }
      if (detected && hint && detected !== hint) notes.push(`${f.name} was recognised as ${SYSTEMS[detected].title} data.`);
      if (!groups.has(kind)) groups.set(kind, []);
      groups.get(kind).push(f);
    }
    if (unknown.length) notes.push(`Unrecognised: ${unknown.join(", ")}. Open the subsystem page and upload from there.`);
    const done = [];
    for (const kind of KINDS) {
      let group = groups.get(kind);
      if (!group) continue;
      group.sort((a, b) => a.name.length - b.name.length || a.name.localeCompare(b.name));
      if (SINGLE_FILE.has(kind) && group.length > 1) {
        notes.push(`${SYSTEMS[kind].title} takes one file. Used ${group[0].name}.`);
        group = group.slice(0, 1);
      }
      const label = `${SYSTEMS[kind].title}${group.length > 1 ? ` (${group.length} files)` : ""}`;
      setStatus(`${label}: starting`);
      const busyEls = [...document.querySelectorAll(`[data-kind="${SYSTEMS[kind].key}"]`)];
      busyEls.forEach((el) => el.classList.add("working"));
      try {
        const payload = await runWorker(kind, group, label);
        session.set(kind, { payload, files: group.map((f) => f.name), at: now() });
        tabOf.delete(kind);
        done.push(kind);
      } catch (err) {
        notes.push(`${SYSTEMS[kind].title}: ${friendly(err.message)}`);
        if (worker) { worker.terminate(); worker = null; }
      } finally {
        busyEls.forEach((el) => el.classList.remove("working"));
      }
    }
    setStatus("");
    // One result opens its page directly; a batch lands on the overview with a single summary.
    if (done.length === 1) {
      const v = viewOf(done[0]);
      toast([levelChip(v.level), h("b", {}, SYSTEMS[done[0]].title), h("span", {}, v.headline)]);
    } else if (done.length > 1) {
      const urgent = done.filter((k) => LEVELS[viewOf(k).level].rank >= 2).length;
      toast([icon("ok", 16), h("b", {}, `${done.length} subsystems analysed`), h("span", {}, urgent ? `${urgent} need action` : "no action needed")]);
    }
    if (done.length === 1) go(SYSTEMS[done[0]].key);
    else if (done.length > 1) go("overview");
    else render();
  } finally {
    busy = false;
    document.body.classList.remove("busy");
  }
}

/** Views are built once per result; panels they open are bound to the shared panel. */
function viewOf(kind) {
  const e = session.get(kind);
  if (!e.view) e.view = VIEWS[kind](e.payload, { openPanel });
  return e.view;
}

// ----------------------------------------------------------- routing ----
function go(r) {
  if (location.hash !== `#/${r}`) location.hash = `#/${r}`;
  else render();
}
window.addEventListener("hashchange", () => { closePanel(); closeMenu(); render(); });

function currentRoute() {
  const r = location.hash.replace(/^#\/?/, "") || "overview";
  return ["overview", "benchmark", "help"].includes(r) || kindByKey(r) ? r : "overview";
}

function render() {
  route = currentRoute();
  const kind = kindByKey(route);
  renderMenu();
  renderCrumbs(kind);
  hideTooltip();
  const page = route === "overview" ? overviewPage() : route === "benchmark" ? benchmarkPage() : route === "help" ? helpPage() : systemPage(kind);
  content.replaceChildren(...[notes.length ? noticeBox() : null, page].filter(Boolean));
  enhance(content);
  window.scrollTo(0, 0);
}

function noticeBox() {
  return h("div", { class: "notice", role: "note" }, icon("info", 16),
    h("div", {}, notes.map((m) => h("p", {}, m))),
    h("button", { type: "button", class: "link", onclick: () => { notes = []; render(); } }, "Dismiss"));
}

// ------------------------------------------------------------- menu ----
function renderMenu() {
  const item = (r, label, iconName, extra) => h("a", { href: `#/${r}`, class: "menu-item", "aria-current": route === r ? "page" : null },
    icon(iconName, 18), h("span", {}, label), extra || null);
  const dot = (kind) => (session.has(kind)
    ? h("i", { class: `dot sev-${viewOf(kind).level}`, title: LEVELS[viewOf(kind).level].label }) : null);
  $("#menu").replaceChildren(
    item("overview", "Overview", "grid"),
    h("div", { class: "menu-label" }, "Subsystems"),
    ...KINDS.map((k) => item(SYSTEMS[k].key, SYSTEMS[k].title, k, dot(k))),
    h("div", { class: "menu-label" }, "Model"),
    item("benchmark", "Benchmark", "chart"),
    item("help", "Help", "help"));

  const n = session.size;
  $("#export-card").replaceChildren(
    h("div", { class: "side-card-head" }, h("b", {}, "Submission"), info("Zips one *_predictions.csv per analysed subsystem, at the top level of predictions.zip.")),
    h("div", { class: "meter" }, h("i", { style: `width:${(n / 4) * 100}%` })),
    h("div", { class: "side-card-sub" }, `${n} of 4 subsystems analysed`),
    button("predictions.zip", { kind: "primary block", iconName: "download", onclick: downloadZip, attrs: { disabled: n ? null : true, id: "zip-report" } }),
    button("Print report", { kind: "block", iconName: "print", onclick: () => printReport(KINDS.filter((k) => session.has(k))), attrs: { disabled: n ? null : true, id: "print-report" } }));
}

function renderCrumbs(kind) {
  const label = kind ? SYSTEMS[kind].title : { overview: "Overview", benchmark: "Benchmark", help: "Help" }[route];
  $("#crumbs").replaceChildren(h("a", { href: "#/overview" }, "Dashboard"), h("span", { class: "sep", "aria-hidden": "true" }, "›"), h("span", { "aria-current": "page" }, label));
}

const side = $("#side");
const toggle = $("#menu-toggle");
function closeMenu() { side.classList.remove("open"); toggle.setAttribute("aria-expanded", "false"); }
toggle.addEventListener("click", () => {
  const open = !side.classList.contains("open");
  side.classList.toggle("open", open);
  toggle.setAttribute("aria-expanded", String(open));
});

// ------------------------------------------------------------ pages ----
function sampleButton(key, label = "Sample") {
  return h("button", { type: "button", class: "btn small", "data-sample": key, onclick: (e) => { e.stopPropagation(); loadSample(key); } }, icon("sample", 14), label);
}
function uploadButton(kind, label = "Upload") {
  return h("button", { type: "button", class: "btn small", onclick: (e) => { e.stopPropagation(); pickFiles(kind); } }, icon("upload", 14), label);
}

function overviewPage() {
  const tiles = KINDS.map((k) => {
    const s = SYSTEMS[k];
    const e = session.get(k);
    const v = e ? viewOf(k) : null;
    const tile = h("article", { class: `tile${e ? " done" : ""}`, "data-kind": s.key },
      h("div", { class: "tile-head" },
        h("span", { class: "tile-icon" }, icon(k, 18)),
        h("div", { class: "tile-name" }, h("b", {}, s.title), h("small", {}, s.full)),
        info(`${s.what} Input: ${s.needs}.`)),
      e ? [
        h("div", { class: "tile-row" }, h("span", { class: "tile-value" }, v.tile.value), levelChip(v.level)),
        h("div", { class: "tile-caption" }, v.tile.caption),
        h("div", { class: "tile-mini" }, v.mini),
        h("a", { class: "tile-go", href: `#/${s.key}`, "aria-label": `Open ${s.title}` }, icon("arrow", 16)),
      ] : [
        h("div", { class: "tile-empty" }, icon("file", 20), h("span", {}, s.needs)),
        h("div", { class: "tile-actions" }, uploadButton(k), sampleButton(s.key)),
      ]);
    if (e) {
      tile.tabIndex = 0;
      tile.addEventListener("click", (ev) => { if (!ev.target.closest("button, a")) go(s.key); });
      tile.addEventListener("keydown", (ev) => { if (ev.key === "Enter" && ev.target === tile) go(s.key); });
    }
    addDropTarget(tile, k);
    return tile;
  });

  const kinds = KINDS.filter((k) => session.has(k)).sort((a, b) => LEVELS[viewOf(b).level].rank - LEVELS[viewOf(a).level].rank);
  const priority = kinds.length ? card(titled("h3", "Priority", "Analysed subsystems, most urgent first."),
    h("div", { class: "prio" }, kinds.map((k) => {
      const v = viewOf(k);
      return h("a", { class: "prio-row", href: `#/${SYSTEMS[k].key}` },
        levelChip(v.level),
        h("b", {}, SYSTEMS[k].title),
        h("span", { class: "prio-head" }, v.headline),
        h("span", { class: "prio-next" }, v.actions[0]),
        icon("arrow", 16));
    }))) : null;

  const drop = h("button", { type: "button", class: "dropzone", onclick: () => pickFiles(null) },
    icon("upload", 18), h("span", {}, h("b", {}, "Drop files anywhere"), " or browse. Mixed subsystems are sorted automatically."));

  return h("div", { class: "page", "data-state": session.size ? "done" : null },
    h("div", { class: "page-head" }, h("h1", {}, "Overview")),
    h("div", { class: "tiles" }, tiles),
    priority,
    drop);
}

function systemPage(kind) {
  const s = SYSTEMS[kind];
  const e = session.get(kind);
  const head = h("div", { class: "page-head" },
    h("div", {}, h("h1", {}, s.title, h("small", {}, s.full)),
      e ? h("div", { class: "page-meta" }, levelChip(viewOf(kind).level), h("span", {}, `${e.files.length === 1 ? e.files[0] : `${e.files.length} files`} · ${e.at}`)) : null),
    e ? h("div", { class: "page-actions" },
      iconButton("download", `Download ${s.csv}`, () => saveBlob(new Blob([viewOf(kind).csv.text], { type: "text/csv" }), viewOf(kind).csv.filename)),
      iconButton("print", "Print job sheet", () => printReport([kind])),
      iconButton("upload", "Replace data", () => pickFiles(kind))) : null);

  if (!e) {
    const empty = h("button", { type: "button", class: "empty", onclick: () => pickFiles(kind) },
      h("span", { class: "tile-icon big" }, icon(kind, 26)),
      h("b", {}, "No data yet"),
      h("span", {}, `${s.needs}. Drop it here or browse.`));
    addDropTarget(empty, kind);
    return h("div", { class: "page" }, head, empty, h("div", { class: "center" }, sampleButton(s.key, "Load sample data")));
  }

  const v = viewOf(kind);
  const tab = tabOf.get(kind) || "summary";
  const body = h("div", { class: "tab-body" });
  const draw = (t) => {
    tabOf.set(kind, t);
    hideTooltip();
    if (t === "summary") {
      body.replaceChildren(
        h("div", { class: "stats" }, v.stats),
        h("div", { class: "split" }, v.main,
          card(titled("h3", "Actions", `When: ${LEVELS[v.level].when}.`),
            h("ul", { class: "checklist" }, v.actions.map((a) => h("li", {}, h("label", {}, h("input", { type: "checkbox" }), h("span", {}, a))))))));
    } else if (t === "records") body.replaceChildren(v.records);
    else body.replaceChildren(...v.signals, h("p", { class: "method" }, icon("info", 14), v.method));
    enhance(body);
  };
  draw(tab);
  return h("div", { class: "page", "data-state": "done", "data-kind": s.key },
    head,
    h("div", { class: "headline-row" },
      h("p", { class: "headline" }, v.headline),
      segmented([["summary", "Summary"], ["records", "Records"], ["signals", "Signals"]], tab, draw, "View")),
    body);
}

function benchmarkPage() {
  return h("div", { class: "page" },
    h("div", { class: "page-head" }, h("h1", {}, "Benchmark", info("Out-of-sample validation scores of the selected models, using each subsystem's official metric."))),
    h("div", { class: "stats" }, KINDS.map((k) => {
      const b = HEADLINES[k];
      return stat({ label: SYSTEMS[k].title, value: b.score.toFixed(3), caption: b.metric, tip: `${b.plain} Scale: ${b.scale}. Validation: ${b.validation}.` });
    })),
    card(titled("h3", "All approaches", "Door uses a chronological split (one continuous stream). Rail selects its top-40 features inside each fold. SHM uses leave-one-out."),
      table(BENCHMARK.map(([Subsystem, Model, Validation, Metric, Score, Status]) => ({ Subsystem: SYSTEMS[Subsystem].title, Model, Validation, Metric, Score, Status })), [
        { key: "Subsystem", label: "Subsystem" },
        { key: "Model", label: "Approach" },
        { key: "Validation", label: "Validation" },
        { key: "Metric", label: "Metric" },
        { key: "Score", label: "Score", num: true, fmt: (v) => v.toFixed(3) },
        { key: "Status", label: "Status", fmt: (v) => levelChip(v === "selected" ? "ok" : v === "rejected" ? "act" : "watch", v[0].toUpperCase() + v.slice(1)) },
      ])));
}

function helpPage() {
  return h("div", { class: "page" },
    h("div", { class: "page-head" }, h("h1", {}, "Help")),
    h("div", { class: "split even" },
      card("Workflow", h("ol", { class: "steps" },
        h("li", {}, "Upload on the Overview (any mix of files) or on a subsystem page."),
        h("li", {}, "Select any item in the Summary visual to open its details."),
        h("li", {}, "Download the CSV, or predictions.zip from the side menu."))),
      card("Status", h("ul", { class: "levels" }, Object.keys(LEVELS).map((l) => h("li", {}, levelChip(l), h("span", {}, LEVELS[l].when)))))),
    card("Inputs", table(KINDS.map((k) => ({ s: SYSTEMS[k].title, full: SYSTEMS[k].full, needs: SYSTEMS[k].needs, csv: SYSTEMS[k].csv })), [
      { key: "s", label: "Subsystem" }, { key: "full", label: "Full name" }, { key: "needs", label: "Input" }, { key: "csv", label: "Output", fmt: (v) => h("code", {}, v) },
    ])));
}

// ------------------------------------------------------------- input ----
const input = $("#file-input");
let pickHint = null;
function pickFiles(kind) { pickHint = kind; input.click(); }
input.addEventListener("change", async () => {
  if (input.files.length) await analyse([...input.files].map((f) => ({ name: f.name, blob: f })), pickHint);
  input.value = "";
});
$("#upload-btn").addEventListener("click", () => pickFiles(kindByKey(route) || null));

// Per-element drop targets carry a subsystem hint; anywhere else on the page uses the current page.
let dropHint = null;
function addDropTarget(el, kind) {
  el.addEventListener("dragover", (e) => { e.preventDefault(); dropHint = kind; el.classList.add("drop-on"); });
  el.addEventListener("dragleave", () => { dropHint = null; el.classList.remove("drop-on"); });
  el.addEventListener("drop", () => el.classList.remove("drop-on"));
}
let depth = 0;
const overlay = $("#drop-overlay");
window.addEventListener("dragenter", (e) => { if (e.dataTransfer?.types?.includes("Files")) { depth++; overlay.classList.add("show"); } });
window.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) overlay.classList.remove("show"); });
window.addEventListener("dragover", (e) => { e.preventDefault(); });
window.addEventListener("drop", async (e) => {
  e.preventDefault();
  depth = 0;
  overlay.classList.remove("show");
  const hint = dropHint || kindByKey(route) || null;
  dropHint = null;
  if (e.dataTransfer.files.length) await analyse([...e.dataTransfer.files].map((f) => ({ name: f.name, blob: f })), hint);
});

async function loadSample(key) {
  if (busy) return;
  const { kind, path } = SAMPLES[key];
  const name = path.split("/").pop();
  setStatus(`Downloading sample ${name}`);
  try {
    const res = await fetch(SAMPLE_BASE + path);
    if (!res.ok) throw new Error(`Sample download failed (${res.status}).`);
    await analyse([{ name, blob: await res.blob() }], kind);
  } catch (err) {
    setStatus(err.message, { error: true });
  }
}

// ----------------------------------------------------- zip and print ----
function downloadZip() {
  const files = KINDS.filter((k) => session.has(k)).map((k) => ({ name: viewOf(k).csv.filename, text: viewOf(k).csv.text }));
  if (files.length) saveBlob(makeZip(files), "predictions.zip");
}

function printReport(kinds) {
  if (!kinds.length) return;
  const worst = kinds.reduce((a, k) => Math.max(a, LEVELS[viewOf(k).level].rank), 0);
  $("#print-root").replaceChildren(
    h("header", { class: "p-head" },
      h("div", {}, h("h1", {}, kinds.length === 1 ? `${SYSTEMS[kinds[0]].title} job sheet` : "Condition report"),
        h("p", {}, `TrainWhisper · ${now()}`)),
      h("div", {}, "Overall: ", levelChip(Object.keys(LEVELS).find((l) => LEVELS[l].rank === worst)))),
    ...kinds.map((k) => {
      const v = viewOf(k);
      const { files, at } = session.get(k);
      return h("section", { class: `p-sec sev-${v.level}` },
        h("div", { class: "p-sec-head" }, h("h2", {}, `${SYSTEMS[k].title} (${SYSTEMS[k].full})`), levelChip(v.level)),
        h("p", { class: "p-meta" }, `${files.length === 1 ? files[0] : `${files.length} files`} · analysed ${at} · ${LEVELS[v.level].when}`),
        h("p", { class: "p-headline" }, v.headline),
        h("ul", { class: "p-check" }, v.actions.map((t) => h("li", {}, t))),
        v.print.rows.length ? [h("h3", {}, v.print.note), h("table", {}, h("thead", {}, h("tr", {}, v.print.columns.map((c) => h("th", {}, c)))),
          h("tbody", {}, v.print.rows.map((r) => h("tr", {}, r.map((c) => h("td", {}, c))))))] : h("p", {}, v.print.note),
        h("div", { class: "p-sign" }, h("span", {}, "Actioned by: ____________________"), h("span", {}, "Date: __________"), h("span", {}, "Work order: __________")));
    }));
  document.body.classList.add("printing");
  const done = () => { document.body.classList.remove("printing"); window.removeEventListener("afterprint", done); };
  window.addEventListener("afterprint", done);
  window.print();
}

// ------------------------------------------------------------- start ----
$("#local-note").replaceChildren(icon("cloud", 14), h("span", {}, "Cloud processing"), info("Files are analysed on the TrainWhisper server and are not stored after the response."));
render();
