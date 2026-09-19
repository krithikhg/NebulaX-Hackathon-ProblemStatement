// ============================================================ SHM stream ====
// Cumulative, per-component fatigue monitoring for SHM.
//
// Each stress segment is filed under a free-text **component** label, so fatigue
// is tracked separately for every monitored component. This module owns its own
// store, server calls and DOM so it drops into the SHM page cleanly:
//   mountComponentPicker()  -> always-visible field to name the component
//   mountShmStream()         -> the cumulative dashboard (per component + all)
//   getSelectedStream()      -> the component new uploads are filed under
//   refreshStreams()         -> reload the persisted per-component state
//
// State lives server-side (Cloud Storage when configured), so totals survive
// restarts and are shared across instances.

import { lines } from "./charts.js";
import { button, card, h, int, levelChip, stat, table, titled } from "./ui.js";

export const DEFAULT_STREAM = "Component 01";
// Report-only status vocabulary, matching the app's LEVELS: failure at D = 1.
const levelOf = (d) => (d >= 1 ? "fault" : "ok");

const store = { streams: [], selected: null, listeners: new Set() };
const notify = () => store.listeners.forEach((fn) => fn());

export const getSelectedStream = () => store.selected || DEFAULT_STREAM;

export function setSelectedStream(label) {
  const v = String(label || "").trim();
  if (v) { store.selected = v; notify(); }
}

export function subscribe(fn) { store.listeners.add(fn); return () => store.listeners.delete(fn); }

export async function refreshStreams() {
  try {
    const res = await fetch("/api/shm/state");
    const data = await res.json();
    store.streams = (data.streams || []).filter((s) => s && s.stream);
  } catch { /* offline: keep current */ }
  notify();
}

async function resetStream(label) {
  if (label && !window.confirm(`Reset cumulative damage for "${label}"?`)) return;
  const body = new FormData();
  if (label) body.append("stream", label);
  try { await fetch("/api/shm/state/reset", { method: "POST", body }); } catch { /* ignore */ }
  await refreshStreams();
}

const perSegment = (s) => (s.n_segments ? s.D / s.n_segments : 0);
const segmentsToFailure = (s) => {
  const per = perSegment(s);
  return per > 0 ? (1 - s.D) / per : Infinity;
};
const lifeText = (v) => (v === Infinity ? "-" : v <= 0 ? "0" : v < 10 ? v.toFixed(1) : int(v));

const chartHolder = (build) => {
  const holder = h("div");
  queueMicrotask(() => build(holder));
  return holder;
};

function projection(state, per) {
  const last = state.history[state.history.length - 1];
  const need = (1 - state.D) / per;
  const span = Math.max(1, Math.min(need, Math.max(10, state.n_segments * 3)));
  return [[last.n, state.D], [last.n + span, Math.min(1, state.D + span * per)]];
}

// --------------------------------------------------------- component field --
// One cached element so re-renders never steal focus from the input.
let _picker = null;

export function mountComponentPicker() {
  if (_picker) return _picker;
  const input = h("input", {
    class: "shm-stream-input", list: "shm-component-options",
    placeholder: "e.g. bogie frame / carbody mount 03", value: getSelectedStream(),
    oninput: (e) => setSelectedStream(e.target.value),
  });
  const options = h("datalist", { id: "shm-component-options" });
  _picker = h("label", { class: "shm-component-picker" }, h("span", {}, "Component"), input, options);

  const sync = () => {
    if (document.activeElement !== input) input.value = getSelectedStream();
    options.replaceChildren(...store.streams.map((s) => h("option", { value: s.stream })));
  };
  subscribe(sync);
  refreshStreams();
  sync();
  return _picker;
}

// ------------------------------------------------------------------- view --
function build() {
  const selected = store.selected || (store.streams[0] && store.streams[0].stream) || DEFAULT_STREAM;
  const current = store.streams.find((s) => s.stream === selected)
    || { stream: selected, D: 0, n_segments: 0, n_samples: 0, history: [] };
  const per = perSegment(current);

  const head = h("div", { class: "shm-stream-head" },
    h("div", { class: "shm-stream-title" }, h("b", {}, current.stream)),
    button("Reset this component", { onclick: () => resetStream(current.stream) }));

  const stats = h("div", { class: "stats" },
    stat({ label: "Cumulative damage D", value: current.D.toFixed(4), chip: levelChip(levelOf(current.D)),
      tip: "Running sum of per-segment Miner damage for this component; failure is expected at D = 1." }),
    stat({ label: "Segments", value: int(current.n_segments) }),
    stat({ label: "Stress samples", value: int(current.n_samples) }),
    stat({ label: "Segments to D = 1", value: lifeText(segmentsToFailure(current)),
      tip: "At the current per-segment damage rate." }));

  const timeline = current.history && current.history.length >= 2
    ? card(titled("h3", "Cumulative damage over time", "Each point is one ingested segment of this component."),
        chartHolder((holder) => lines(holder, {
          series: [
            { name: "D", color: "var(--series-1)", points: current.history.map((e) => [e.n, e.D]) },
            ...(per > 0 ? [{ name: "projection", color: "var(--muted)", dash: "6 4", points: projection(current, per) }] : []),
          ],
          xLabel: "Segment", yLabel: "Cumulative D", height: 240,
          fmtX: (v) => String(Math.round(v)), fmtY: (v) => v.toFixed(2),
          ariaLabel: `Cumulative damage for ${current.stream}`,
        })))
    : null;

  const components = store.streams.length
    ? card(titled("h3", "Tracked components", "Every component with persisted cumulative state."),
        table(store.streams.map((s) => ({ ...s, level: levelOf(s.D), life: lifeText(segmentsToFailure(s)) })), [
          { key: "stream", label: "Component" },
          { key: "D", label: "D", num: true, fmt: (v) => v.toFixed(4) },
          { key: "n_segments", label: "Segments", num: true, fmt: int },
          { key: "life", label: "Segments to D = 1", num: true },
          { key: "level", label: "Status", fmt: (v) => levelChip(v) },
        ]))
    : null;

  return [card(titled("h3", "Component fatigue (live)",
    "Every ingested segment is added to its component's running Miner damage."), head, stats),
    timeline, components].filter(Boolean);
}

let _dashboard = null;

export function mountShmStream() {
  if (_dashboard) return _dashboard;
  const root = h("div", { class: "shm-stream" });
  const render = () => root.replaceChildren(...build());
  subscribe(render);
  refreshStreams();
  render();
  _dashboard = root;
  return _dashboard;
}
