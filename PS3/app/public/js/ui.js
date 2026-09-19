// Shared UI building blocks: element builder, status levels, icons, info tips, tables, formatting.
import { hideTooltip, showTooltip } from "./charts.js";

export const $ = (sel, root = document) => root.querySelector(sel);

export function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) if (c !== null && c !== undefined && c !== false) node.append(c instanceof Node ? c : String(c));
  return node;
}

// ------------------------------------------------------------ icons ----
const PATHS = {
  Door: "M4 3h16v18H4zM12 3v18M9 12H7m10 0h-2",
  ACV: "M12 2v20M3.3 7l17.4 10M3.3 17 20.7 7M9 4l3 2 3-2M9 20l3-2 3 2",
  "Rail corrugation": "M8 3 5 21M16 3l3 18M6.5 8h11M6 13h12M5.3 18h13.4",
  SHM: "M2 12h4l3-8 4 16 3-8h6",
  act: "M12 3 2 21h20L12 3Zm0 6v6m0 3v.01",
  plan: "M8 2v4m8-4v4M3 9h18M5 4h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Z",
  watch: "M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Zm10 3a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z",
  ok: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm-4 9 3 3 5-6",
  upload: "M12 16V4m0 0-5 5m5-5 5 5M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2",
  download: "M12 4v12m0 0-5-5m5 5 5-5M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2",
  print: "M6 9V3h12v6M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2M6 14h12v7H6z",
  info: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm0 8v5m0-8v.01",
  grid: "M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z",
  chart: "M4 20V10m6 10V4m6 16v-7m6 7H2",
  help: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm-2.5 6.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6V14m0 3v.01",
  close: "M6 6l12 12M18 6 6 18",
  arrow: "M5 12h14m-6-6 6 6-6 6",
  menu: "M4 6h16M4 12h16M4 18h16",
  lock: "M5 11h14v10H5zM8 11V7a4 4 0 0 1 8 0v4",
  cloud: "M7 18a4 4 0 0 1 0-8 5.5 5.5 0 0 1 10.6 1.5A3.5 3.5 0 0 1 17 18H7Z",
  file: "M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-6-6Zm0 0v6h6",
  home: "M3 11 12 3l9 8M5 9.5V20a1 1 0 0 0 1 1h4v-6h4v6h4a1 1 0 0 0 1-1V9.5",
};

export function icon(name, size = 20) {
  const span = h("span", { class: `icon i-${String(name).split(" ")[0].toLowerCase()}`, "aria-hidden": "true" });
  span.innerHTML = `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="${PATHS[name]}"/></svg>`;
  return span;
}

// ------------------------------------------------------ status levels ----
// One vocabulary for every subsystem: colour + icon + word, never colour alone.
// Statuses report only what the model found in the data; they do not prescribe urgency or actions.
export const LEVELS = {
  fault: { label: "Fault detected", icon: "act", desc: "The model flagged a fault in this data.", rank: 2 },
  na: { label: "Not assessed", icon: "info", desc: "The data could not be assessed, e.g. a rail recording taken while the train was stationary.", rank: 1 },
  ok: { label: "No fault", icon: "ok", desc: "The model flagged no fault in this data.", rank: 0 },
};

export function levelChip(level, text = LEVELS[level].label) {
  return h("span", { class: `chip sev-${level}` }, icon(LEVELS[level] ? LEVELS[level].icon : level, 13), text);
}

// ---------------------------------------------------------- systems ----
export const SYSTEMS = {
  Door: {
    key: "door", title: "Door", full: "Saloon door controller",
    what: "Segments the door controller stream into open/close cycles and flags abnormal resistance.",
    needs: "1 continuous stream (.csv)", csv: "door_predictions.csv",
  },
  ACV: {
    key: "acv", title: "ACV", full: "Air-Conditioning & Ventilation",
    what: "Ranks the 8 cars by likelihood of a refrigerant leak, using the other cars as the baseline.",
    needs: "1 train workbook (.xlsx)", csv: "acv_predictions.csv",
  },
  "Rail corrugation": {
    key: "rail", title: "Rail Corrugation", full: "Axle-box vibration",
    what: "Classifies each recording as Normal, Side I or Side II corrugation.",
    needs: "1 or more recordings (.csv)", csv: "rail_predictions.csv",
  },
  SHM: {
    key: "shm", title: "SHM", full: "Structural Health Monitoring",
    what: "Estimates cumulative fatigue damage D per stress segment (Miner's rule, failure at D = 1).",
    needs: "1 or more stress segments (.csv)", csv: "shm_predictions.csv",
  },
};
export const KINDS = Object.keys(SYSTEMS);
export const kindByKey = (key) => KINDS.find((k) => SYSTEMS[k].key === key);

// --------------------------------------------------------- info tips ----
/** Small (i) that reveals text on hover, focus or tap. */
export function info(text, label = "More information") {
  const b = h("button", { type: "button", class: "info", "aria-label": `${label}: ${text}` }, icon("info", 15));
  const on = (e) => {
    const r = b.getBoundingClientRect();
    showTooltip(e && e.clientX ? e : { clientX: r.right, clientY: r.bottom }, text, []);
  };
  b.addEventListener("pointerenter", on);
  b.addEventListener("focus", () => on());
  b.addEventListener("pointerleave", hideTooltip);
  b.addEventListener("blur", hideTooltip);
  b.addEventListener("click", (e) => { e.stopPropagation(); on(); });
  return b;
}

/** Title with an optional (i) beside it. */
export function titled(tag, text, tip) {
  return h(tag, { class: "titled" }, text, tip ? info(tip) : null);
}

// -------------------------------------------------------- formatting ----
export const fmt = (digits) => (v) => (Number.isFinite(v) ? v.toFixed(digits) : "-");
export const int = (v) => Math.round(v).toLocaleString("en-GB");
export const signed = (v, digits = 2) => {
  const s = Math.abs(v).toFixed(digits);
  return Number(s) === 0 ? s : `${v >= 0 ? "+" : "−"}${s}`;
};

/** Model probability -> High / Medium / Low. */
export function certainty(p) {
  if (!Number.isFinite(p)) return { word: "-", cls: "" };
  if (p >= 0.9) return { word: "High", cls: "c-high" };
  if (p >= 0.7) return { word: "Medium", cls: "c-med" };
  return { word: "Low", cls: "c-low" };
}
export function certaintyCell(p) {
  const c = certainty(p);
  return h("span", { class: `certainty ${c.cls}`, title: `p = ${Number.isFinite(p) ? p.toFixed(3) : "-"}` },
    h("i", { "aria-hidden": "true" }, h("b"), h("b"), h("b")), c.word);
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const pad = (n, w = 2) => String(n).padStart(w, "0");
/** Epoch ms (recorded clock time, stored as UTC) -> "14:05:09". */
export const clock = (ms) => { const d = new Date(ms); return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`; };
export const day = (ms) => { const d = new Date(ms); return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`; };
/** Excel serial date -> epoch ms. */
export const excelDate = (v) => (typeof v === "number" && v > 20000 && v < 80000 ? Math.round((v - 25569) * 86400000) : null);
export const now = () => { const d = new Date(); return `${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()}, ${pad(d.getHours())}:${pad(d.getMinutes())}`; };

// ------------------------------------------------------------ layout ----
/** Outlined card. title may be a string or a node. */
export function card(title, ...body) {
  return h("section", { class: "card" },
    title ? (typeof title === "string" ? h("h3", { class: "card-title" }, title) : title) : null, ...body);
}

/** Stat card: label, big value, optional chip, caption and icon badge. */
export function stat({ label, value, chip, caption, tip, iconName }) {
  return h("div", { class: `stat${iconName ? " has-ico" : ""}` },
    h("div", { class: "stat-label" }, label, tip ? info(tip) : null),
    h("div", { class: "stat-row" }, h("span", { class: "stat-value" }, value), chip || null),
    caption ? h("div", { class: "stat-caption" }, caption) : null,
    iconName ? h("span", { class: "stat-ico" }, icon(iconName, 20)) : null);
}

export function legend(items) {
  return h("div", { class: "legend" }, items.map(([label, cls]) => h("span", {}, h("i", { class: cls }), label)));
}

/** Segmented control with a sliding thumb. options: [[value, label]] */
export function segmented(options, value, onchange, label) {
  const thumb = h("span", { class: "seg-thumb", "aria-hidden": "true" });
  const group = h("div", { class: "segmented", role: "group", "aria-label": label }, thumb);
  const place = (btn) => {
    if (!btn || !btn.offsetWidth) return;
    thumb.style.width = `${btn.offsetWidth}px`;
    thumb.style.transform = `translateX(${btn.offsetLeft}px)`;
  };
  const buttons = options.map(([v, text]) => h("button", {
    type: "button", "aria-pressed": String(v === value),
    onclick: (e) => {
      buttons.forEach((b) => b.setAttribute("aria-pressed", String(b === e.currentTarget)));
      place(e.currentTarget);
      onchange(v);
    },
  }, text));
  group.append(...buttons);
  const current = () => buttons.find((b) => b.getAttribute("aria-pressed") === "true");
  new ResizeObserver(() => {
    place(current());
    requestAnimationFrame(() => group.classList.add("ready"));
  }).observe(group);
  return group;
}

// ------------------------------------------------------------ motion ----
export const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Give children of list-like containers an index, used by CSS for staggered entrances. */
export function stagger(root, selectors) {
  for (const el of root.querySelectorAll(selectors)) {
    [...el.children].forEach((c, i) => c.style.setProperty("--i", i));
  }
}

/** Count numbers up from zero, keeping their formatting ("683 mA", "+22%", "1,999", "0.824"). */
export function countUp(root, selector = ".stat-value, .tile-value") {
  if (reducedMotion()) return;
  const re = /\d[\d,]*(?:\.\d+)?/g;
  for (const el of root.querySelectorAll(selector)) {
    if (el.children.length) continue;
    // Keep the final text on the node: views are re-inserted when switching tabs.
    const text = el.dataset.final || (el.dataset.final = el.textContent);
    const run = (el._countRun = (el._countRun || 0) + 1);
    // Numbers that follow a word ("Car 01", "Test5") are identifiers, not quantities.
    const live = [...text.matchAll(re)]
      .map((m) => ({ m: m[0], at: m.index }))
      .filter(({ at }) => !/[A-Za-z]\s*$/.test(text.slice(0, at)));
    if (!live.length) continue;
    const parsed = live.map(({ m, at }) => ({ at, len: m.length, v: Number(m.replace(/,/g, "")), dec: (m.split(".")[1] || "").length, comma: m.includes(",") }));
    const fmtNum = (x, p) => (p.comma ? Math.round(x).toLocaleString("en-GB") : x.toFixed(p.dec));
    const t0 = performance.now();
    const dur = 700;
    const frame = (t) => {
      if (el._countRun !== run) return;
      const k = Math.min(1, Math.max(0, (t - t0) / dur));
      const e = 1 - (1 - k) ** 3;
      let out = "";
      let pos = 0;
      for (const p of parsed) { out += text.slice(pos, p.at) + fmtNum(p.v * e, p); pos = p.at + p.len; }
      el.textContent = k < 1 ? out + text.slice(pos) : text;
      if (k < 1) requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }
}

/** Short-lived notification, bottom right. */
export function toast(content, { href, label = "Open" } = {}) {
  let host = document.getElementById("toasts");
  if (!host) { host = h("div", { id: "toasts", class: "toasts", "aria-live": "polite" }); document.body.append(host); }
  const node = h("div", { class: "toast" }, h("div", { class: "toast-body" }, content), href ? h("a", { href, class: "toast-link" }, label, icon("arrow", 14)) : null);
  host.append(node);
  const bye = () => { node.classList.add("out"); setTimeout(() => node.remove(), 250); };
  const timer = setTimeout(bye, 4500);
  node.addEventListener("click", () => { clearTimeout(timer); bye(); });
}

// ------------------------------------------------------------ tables ----
/**
 * columns: [{ key, label, num?, fmt?(v, row) -> Node|string, title? }]
 * opts: { onOpen(row, i), rowClass(row), maxHeight, caption }
 */
export function table(rows, columns, opts = {}) {
  const head = h("tr", {}, columns.map((c) => h("th", { class: c.num ? "num" : "", scope: "col", title: c.title }, c.label)),
    opts.onOpen ? h("th", { class: "open-col" }, h("span", { class: "sr-only" }, "Open")) : null);
  const trs = rows.map((r, i) => {
    const tr = h("tr", { class: opts.rowClass ? opts.rowClass(r) : "" },
      columns.map((c) => h("td", { class: c.num ? "num" : "" }, c.fmt ? c.fmt(r[c.key], r) : r[c.key])),
      opts.onOpen ? h("td", { class: "open-col" }, icon("arrow", 16)) : null);
    if (opts.onOpen) {
      tr.tabIndex = 0;
      tr.classList.add("selectable");
      const open = () => opts.onOpen(r, i);
      tr.addEventListener("click", open);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
    }
    return tr;
  });
  return h("div", { class: "table-wrap", style: opts.maxHeight ? `max-height:${opts.maxHeight}px` : null },
    h("table", {}, opts.caption ? h("caption", { class: "sr-only" }, opts.caption) : null, h("thead", {}, head), h("tbody", {}, trs)));
}

// --------------------------------------------------------- downloads ----
export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = h("a", { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function button(label, { kind = "", iconName, onclick, title, attrs = {} } = {}) {
  return h("button", { type: "button", class: `btn ${kind}`, onclick, title, ...attrs }, iconName ? icon(iconName, 16) : null, label);
}

export function iconButton(name, label, onclick) {
  const b = h("button", { type: "button", class: "icon-btn", "aria-label": label, onclick }, icon(name, 18));
  b.addEventListener("pointerenter", (e) => showTooltip(e, label, []));
  b.addEventListener("pointerleave", hideTooltip);
  return b;
}
