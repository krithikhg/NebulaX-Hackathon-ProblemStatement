// ============================================================ SHM stream ====
// Cumulative, per-stream fatigue monitoring for SHM.
//
// This module is deliberately self-contained (its own store, server calls and
// DOM) so it can be dropped into the SHM view without touching the rest of the
// UI. Public surface:
//   mountShmStream()        -> element that renders itself from the store
//   getSelectedStream()     -> the stream label new uploads are added to
//   setSelectedStream(label)
//   refreshStreams()        -> reload the persisted per-stream state
//
// State lives server-side (persisted to Cloud Storage when configured), so the
// running totals survive restarts and are shared across instances.

import { lines } from "./charts.js";
import { button, card, h, int, levelChip, stat, table, titled } from "./ui.js";

export const DEFAULT_STREAM = "Train 01 / bogie frame";
const BANDS = [[0.8, "act"], [0.5, "plan"], [0.25, "watch"], [0, "ok"]];
const levelOf = (d) => (BANDS.find(([t]) => d >= t) || BANDS[3])[1];
const STATUS = { ok: "Normal", watch: "Monitor", plan: "Plan", act: "Act now" };

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

function build() {
  const selected = store.selected || (store.streams[0] && store.streams[0].stream) || DEFAULT_STREAM;
  const current = store.streams.find((s) => s.stream === selected)
    || { stream: selected, D: 0, n_segments: 0, n_samples: 0, history: [] };
  const per = perSegment(current);

  const input = h("input", {
    class: "shm-stream-input", list: "shm-stream-options",
    value: selected, placeholder: "e.g. Train 03 / bogie frame",
    onchange: (e) => setSelectedStream(e.target.value),
  });
  const options = h("datalist", { id: "shm-stream-options" },
    store.streams.map((s) => h("option", { value: s.stream })));

  const head = h("div", { class: "shm-stream-head" },
    h("label", {}, h("span", {}, "Monitored stream"), input, options),
    button("Reset this stream", { onclick: () => resetStream(current.stream) }));

  const stats = h("div", { class: "stats" },
    stat({ label: "Cumulative damage D", value: current.D.toFixed(4), chip: levelChip(levelOf(current.D)),
      tip: "Running sum of per-segment Miner damage; failure is expected at D = 1." }),
    stat({ label: "Segments", value: int(current.n_segments) }),
    stat({ label: "Stress samples", value: int(current.n_samples) }),
    stat({ label: "Segments to D = 1", value: lifeText(segmentsToFailure(current)),
      tip: "At the current per-segment damage rate." }));

  const timeline = current.history && current.history.length >= 2
    ? card(titled("h3", "Cumulative damage over time", "Each point is one ingested segment."),
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

  const fleet = store.streams.length
    ? card(titled("h3", "Fleet", "All monitored streams with persisted cumulative state."),
        table(store.streams.map((s) => ({ ...s, level: levelOf(s.D), life: lifeText(segmentsToFailure(s)) })), [
          { key: "stream", label: "Stream" },
          { key: "D", label: "D", num: true, fmt: (v) => v.toFixed(4) },
          { key: "n_segments", label: "Segments", num: true, fmt: int },
          { key: "life", label: "Segments to D = 1", num: true },
          { key: "level", label: "Status", fmt: (v) => levelChip(v, STATUS[v]) },
        ]))
    : null;

  return [card(titled("h3", "Cumulative fatigue (live)",
    "Persisted per stream: every ingested segment is added to the running Miner damage."), head, stats),
    timeline, fleet].filter(Boolean);
}

export function mountShmStream() {
  const root = h("div", { class: "shm-stream" });
  const render = () => root.replaceChildren(...build());
  subscribe(render);
  refreshStreams();
  render();
  return root;
}
