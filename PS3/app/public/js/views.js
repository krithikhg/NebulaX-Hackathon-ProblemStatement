// One view per subsystem. Each takes the worker payload plus { openPanel } and returns:
//   { level, headline, tile: { value, caption }, mini: Node, stats: [], main: Node,
//     records: Node, signals: Node[], method, csv: { filename, text }, print: { columns, rows, note } }
// app.js lays these out identically for every subsystem.
import { bars, lines, segmentedLine } from "./charts.js";
import { mountShmStream } from "./shm_stream.js";
import { pyFloat, remainingLife, toCsv } from "./util.js";
import {
  LEVELS, card, certainty, certaintyCell, clock, day, excelDate, fmt, h, int, legend, levelChip, segmented, signed, stat, table, titled,
} from "./ui.js";

const NORMAL = "Normal";
const mean = (xs) => xs.reduce((s, x) => s + x, 0) / xs.length;
const pct = (x) => `${Math.round(x * 100)}%`;
const medianBy = (rows, key) => {
  const s = [...rows].sort((a, b) => key(a) - key(b));
  return s[Math.floor((s.length - 1) / 2)];
};

/** Chart that draws once it is attached (charts size themselves to their container). */
function chartHolder(build) {
  const holder = h("div");
  queueMicrotask(() => build(holder));
  return holder;
}

function kv(pairs) {
  return h("dl", { class: "kv" }, pairs.filter(Boolean).map(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]));
}

function filterable(rows, isProblem, columns, opts) {
  const holder = h("div");
  let mode = rows.some(isProblem) ? "flagged" : "all";
  const draw = () => holder.replaceChildren(table(mode === "all" ? rows : rows.filter(isProblem), columns, opts));
  draw();
  const flagged = rows.filter(isProblem).length;
  return h("section", { class: "card" },
    h("div", { class: "card-head" }, h("h3", { class: "card-title" }, `${rows.length} records`),
      flagged ? segmented([["flagged", `Flagged (${flagged})`], ["all", "All"]], mode, (v) => { mode = v; draw(); }, "Filter records") : null),
    holder);
}

// =============================================================== Door ====
export function doorView({ name, result, cycles }, { openPanel }) {
  const n = result.length;
  const rows = result.map((r, i) => ({ ...r, i, ...cycles[i], bad: r.prediction !== NORMAL }));
  const badRows = rows.filter((r) => r.bad);
  const okRows = rows.filter((r) => !r.bad);
  const bad = badRows.length;
  const level = bad ? "fault" : "ok";
  const meanBad = bad ? mean(badRows.map((r) => r.mean_current_mA)) : null;
  const meanOk = okRows.length ? mean(okRows.map((r) => r.mean_current_mA)) : null;
  const effort = meanBad && meanOk ? meanBad / meanOk - 1 : null;
  const opens = rows.filter((r) => r.operation === "Opening").length;
  const closes = rows.filter((r) => r.operation === "Closing").length;

  const open = (i) => {
    const r = rows[i];
    const pool = okRows.filter((o) => o.i !== i && (!r.operation || o.operation === r.operation));
    const ref = pool.length ? medianBy(pool, (o) => o.mean_current_mA) : null;
    const rel = ref ? r.mean_current_mA / ref.mean_current_mA - 1 : null;
    const toSeries = (c) => c.points.map(([, y, t]) => [(t - c.t0) / 1000, y]);
    const series = [{ name: `Cycle ${i + 1}`, color: r.bad ? "var(--critical)" : "var(--series-1)", points: toSeries(r) }];
    if (ref) series.push({ name: `Median normal (cycle ${ref.i + 1})`, color: "var(--neutral-bar)", dash: "5 4", points: toSeries(ref) });
    openPanel({
      title: `Cycle ${i + 1}`,
      chip: levelChip(r.bad ? "fault" : "ok", r.prediction),
      subtitle: `${r.operation || "Cycle"} · ${day(r.t0)} ${clock(r.t0)}`,
      body: [
        h("div", { class: "stats-2" },
          stat({ label: "Mean current", value: `${int(r.mean_current_mA)} mA`, caption: rel !== null ? `${signed(rel * 100, 0)}% vs normal` : null, tip: rel !== null ? "Compared with the median normal cycle in the same direction." : null }),
          stat({ label: "Peak current", value: `${int(r.peak_current_mA)} mA` }),
          stat({ label: "Duration", value: `${r.duration_s.toFixed(1)} s` })),
        titled("h4", "Motor current", "Higher current for the same movement means the motor is working against extra resistance."),
        legend([[`Cycle ${i + 1}`, r.bad ? "sw-line-bad" : "sw-line-sel"], ["Median normal cycle, same direction", "sw-line-ref"]]),
        chartHolder((holder) => lines(holder, {
          series, xLabel: "Time from cycle start (s)", yLabel: "Current (mA)", height: 220,
          fmtX: (v) => `${v.toFixed(2)} s`, fmtY: (v) => `${v.toFixed(0)} mA`,
          ariaLabel: `Motor current for cycle ${i + 1} against the median normal cycle`,
        })),
      ],
    });
  };

  const strip = h("div", { class: "strip", role: "group", "aria-label": "Door cycles in time order" },
    rows.map((r) => h("button", {
      type: "button", class: `strip-bar ${r.bad ? "bad" : "ok"}`,
      "aria-label": `Cycle ${r.i + 1}, ${r.operation || ""} ${clock(r.t0)}, ${r.prediction}`,
      "data-tip": `Cycle ${r.i + 1}: ${r.prediction}`, "data-tip-sub": `${r.operation || ""} ${clock(r.t0)} · ${int(r.mean_current_mA)} mA`.trim(),
      onclick: () => open(r.i),
    })));
  const main = card(titled("h3", "Cycle timeline", "Each bar is one open or close cycle, in time order. Select a bar to see its current profile."),
    strip,
    h("div", { class: "strip-axis" }, h("span", {}, `${day(rows[0].t0)} ${clock(rows[0].t0)}`), h("span", {}, clock(rows[n - 1].t0))),
    legend([["Abnormal resistance", "sw-bad"], ["Normal", "sw-ok"]]));

  const cols = [
    { key: "i", label: "Cycle", num: true, fmt: (v) => v + 1 },
    { key: "t0", label: "Start", fmt: (v) => clock(v) },
    { key: "operation", label: "Direction", fmt: (v) => v || "-" },
    { key: "duration_s", label: "Duration (s)", num: true, fmt: (v) => v.toFixed(1) },
    { key: "prediction", label: "Status", fmt: (v) => levelChip(v === NORMAL ? "ok" : "fault", v) },
    { key: "mean_current_mA", label: "Mean current (mA)", num: true, fmt: (v) => int(v) },
  ];
  const records = filterable(rows, (r) => r.bad, cols, { maxHeight: 520, onOpen: (r) => open(r.i), rowClass: (r) => (r.bad ? "row-bad" : ""), caption: "Door cycles" });

  const signals = [
    card(titled("h3", "Motor current, full stream", "Idle time between cycles removed. Shaded cycles were classified as abnormal resistance."),
      legend([["Motor current", "sw-line-sel"], ["Abnormal resistance", "sw-span"]]),
      chartHolder((holder) => segmentedLine(holder, {
        cycles,
        yLabel: "Motor current (mA)",
        ariaLabel: `Door motor current across ${n} cycles; ${bad} abnormal cycles shaded`,
        describe: (p, ci) => [`${p[1].toFixed(0)} mA`, [`Cycle ${ci + 1}: ${result[ci].prediction}`, clock(p[2])]],
      }))),
  ];

  return {
    level,
    headline: bad ? `${bad} of ${n} cycles with abnormal resistance` : `All ${n} cycles normal`,
    tile: { value: `${bad} / ${n}`, caption: "cycles abnormal" },
    mini: h("div", { class: "mini-strip" }, rows.map((r) => h("i", { class: r.bad ? "bad" : "ok" }))),
    stats: [
      stat({ label: "Cycles detected", value: int(n), caption: opens || closes ? `${opens} open · ${closes} close` : null, tip: "Cycles are split wherever the door motion state flips or the leaf position jumps discontinuously." }),
      stat({ label: "Abnormal resistance", value: int(bad), chip: bad ? levelChip(level, pct(bad / n)) : null }),
      stat({ label: "Mean current, abnormal", value: meanBad ? `${int(meanBad)} mA` : "-", caption: meanOk ? `normal: ${int(meanOk)} mA` : null }),
      stat({ label: "Current increase", value: effort !== null ? `${signed(effort * 100, 0)}%` : "-", caption: "abnormal vs normal" }),
    ],
    main,
    records, signals,
    method: "Motion-state segmentation, then a per-operation median current template from normal cycles; a cycle is scored by its largest sustained excursion against the normal IQR band.",
    csv: { filename: "door_predictions.csv", text: toCsv(result, ["start_time", "end_time", "prediction"]) },
    print: {
      note: bad ? "Abnormal cycles" : `All ${n} cycles normal.`,
      columns: ["Cycle", "Start", "Direction", "Mean current"],
      rows: badRows.map((r) => [r.i + 1, `${day(r.t0)} ${clock(r.t0)}`, r.operation || "-", `${int(r.mean_current_mA)} mA`]),
    },
  };
}

// ================================================================ ACV ====
function divergeChart(ranked, highlight) {
  const maxAbs = Math.max(...ranked.map((r) => Math.abs(r.cabin_temp_dev_C)).filter(Number.isFinite), 1e-9);
  return h("div", { class: "diverge", role: "list" },
    ranked.map((r, i) => {
      const v = r.cabin_temp_dev_C;
      const w = Number.isFinite(v) ? (Math.abs(v) / maxAbs) * 50 : 0;
      return h("div", { class: `div-row${r.car === highlight ? " hl" : ""}`, role: "listitem" },
        h("span", { class: "div-label" }, `Car ${r.car}`),
        h("span", { class: "div-track" }, h("i", { class: `div-bar ${v >= 0 ? "pos" : "neg"} ${i === 0 ? "top" : ""}`, style: `width:${w}%;${v >= 0 ? "left:50%" : "right:50%"}` })),
        h("span", { class: "div-val" }, Number.isFinite(v) ? `${signed(v)} °C` : "-"));
    }),
    h("div", { class: "div-axis" }, h("span", {}, "cooler"), h("span", {}, "warmer")));
}

export function acvView({ name, ranked, samples, train, window: win }, { openPanel }) {
  const top = ranked[0];
  const second = ranked[1];
  const last = ranked[ranked.length - 1];
  const spread = top.score - last.score;
  const lead = spread > 0 && second ? (top.score - second.score) / spread : 1;
  const rankOf = new Map(ranked.map((r, i) => [r.car, i + 1]));
  const order = ranked.map((r) => r.car);
  const t0 = win ? excelDate(win[0]) : null;
  const t1 = win ? excelDate(win[1]) : null;
  const tag = (rank) => (rank === 1 ? "leak" : rank <= 3 ? "next" : "ok");

  const open = (car) => {
    const r = ranked.find((x) => x.car === car);
    const rank = rankOf.get(car);
    openPanel({
      title: `Car ${car}`,
      chip: rank === 1 ? levelChip("fault", "Suspected leak") : h("span", { class: "chip sev-neutral" }, `Rank ${rank}`),
      subtitle: `Rank ${rank} of ${ranked.length}`,
      body: [
        h("div", { class: "stats-2" },
          stat({ label: "Cabin temp vs median", value: `${signed(r.cabin_temp_dev_C, 3)} °C`, tip: "Mean difference between this car's cabin temperature and the median of all cars at the same timestamp." }),
          Math.abs(r.score - r.cabin_temp_dev_C) > 1e-9 ? stat({ label: "Leak score", value: r.score.toFixed(3) }) : null,
          "low_pressure_dev" in r ? stat({ label: "Low-side pressure vs median", value: fmt(3)(r.low_pressure_dev) }) : null),
        titled("h4", "All cars", "Positive means warmer than the train median, i.e. less cooling."),
        divergeChart(ranked, car),
      ],
    });
  };

  const cars = [...ranked].sort((a, b) => a.car.localeCompare(b.car, undefined, { numeric: true }));
  const trainNode = h("div", { class: "train", role: "group", "aria-label": "Cars in train order" },
    cars.map((c, k) => {
      const rank = rankOf.get(c.car);
      return h("button", {
        type: "button", onclick: () => open(c.car),
        class: `car car-${tag(rank)}${k === 0 ? " cab-l" : ""}${k === cars.length - 1 ? " cab-r" : ""}`,
        "aria-label": `Car ${c.car}, rank ${rank}`,
        "data-tip": `Car ${c.car}: rank ${rank} of ${cars.length}`,
        "data-tip-sub": Number.isFinite(c.cabin_temp_dev_C) ? `${signed(c.cabin_temp_dev_C, 3)} °C vs median` : "",
      },
      h("span", { class: "car-body" }, h("span", { class: "windows" }, h("i"), h("i"), h("i")), h("span", { class: "car-rank" }, rank)),
      h("b", {}, c.car),
      h("span", { class: "car-dev" }, Number.isFinite(c.cabin_temp_dev_C) ? `${signed(c.cabin_temp_dev_C)}°` : "-"));
    }));
  const main = card(titled("h3", "Train consist", "Cars in physical order. The number on each car is its leak rank. Select a car for details."),
    trainNode,
    legend([["Rank 1", "sw-leak"], ["Rank 2 to 3", "sw-next"], ["Rank 4+", "sw-ok-car"]]));

  const cols = [
    { key: "rank", label: "Rank", num: true },
    { key: "car", label: "Car" },
    { key: "cabin_temp_dev_C", label: "Cabin temp vs median (°C)", num: true, fmt: (v) => (Number.isFinite(v) ? signed(v, 3) : "-") },
  ];
  if (ranked.some((r) => Math.abs(r.score - r.cabin_temp_dev_C) > 1e-9)) cols.push({ key: "score", label: "Leak score", num: true, fmt: fmt(3) });
  if ("low_pressure_dev" in top) cols.push({ key: "low_pressure_dev", label: "Low-side pressure vs median", num: true, fmt: fmt(3) });
  const records = card(`${ranked.length} cars`, table(ranked.map((r, i) => ({ ...r, rank: i + 1 })), cols, { onOpen: (r) => open(r.car), caption: "Cars ranked by leak likelihood" }));

  const row = { file_id: name.split(/[\\/]/).pop(), ranked_cars: order.join("|") };
  const signals = [card(titled("h3", "Cabin temperature vs train median", "Mean over all timestamps. Ambient and set-point changes cancel out because every car sees them."),
    divergeChart(ranked, null),
    h("p", { class: "caption" }, "Submission row: ", h("code", {}, row.ranked_cars)))];

  return {
    level: "fault",
    headline: `Car ${top.car}: suspected refrigerant leak`,
    tile: { value: `Car ${top.car}`, caption: "suspected leak" },
    mini: h("div", { class: "mini-train" }, cars.map((c) => h("i", { class: `m-${tag(rankOf.get(c.car))}` }))),
    stats: [
      stat({ label: "Suspected car", value: `Car ${top.car}`, caption: `${signed(top.cabin_temp_dev_C)} °C vs median` }),
      stat({ label: "Next most likely", value: order.slice(1, 3).map((c) => `Car ${c}`).join(", "), caption: "ranks 2 and 3" }),
      stat({ label: "Separation", value: pct(lead), caption: "rank 1 lead over rank 2", tip: "Lead of rank 1 over rank 2, as a share of the score spread between the first and last car. Higher means the top car stands out more." }),
      stat({ label: "Samples", value: samples ? int(samples) : "-", caption: [train ? `Train ${train}` : null, t0 && t1 ? `${day(t0)} to ${day(t1)}` : null].filter(Boolean).join(" · ") || null }),
    ],
    main,
    records, signals,
    method: "Each car's cabin temperature is compared with the median of all cars at every timestamp; cars are ranked by mean deviation. Low-side pressure is added when every car logs it.",
    csv: { filename: "acv_predictions.csv", text: toCsv([row], ["file_id", "ranked_cars"]) },
    print: {
      note: "Cars by leak likelihood",
      columns: ["Rank", "Car", "Cabin temp vs median"],
      rows: ranked.map((r, i) => [i + 1, `Car ${r.car}`, Number.isFinite(r.cabin_temp_dev_C) ? `${signed(r.cabin_temp_dev_C)} °C` : "-"]),
    },
  };
}

// =============================================================== Rail ====
function trackDiagram(side, stationary) {
  // Top-down view: sleepers across two rails. Side I on top (odd axle-box positions), Side II below.
  const W = 640;
  const H = 130;
  const sleepers = Array.from({ length: 16 }, (_, k) => `<rect class="sleeper" x="${70 + k * 36}" y="16" width="14" height="98" rx="2"/>`).join("");
  const wave = (y) => `M70 ${y}` + Array.from({ length: 36 }, (_, k) => ` q 8 ${k % 2 ? 7 : -7} 16 0`).join("");
  const rail = (y, label, on) => `
    <rect class="rail ${on ? "rail-bad" : ""}" x="60" y="${y - 5}" width="${W - 70}" height="10" rx="2"/>
    ${on ? `<path class="wave" d="${wave(y)}"/>` : ""}
    <text class="rail-label" x="52" y="${y + 4}" text-anchor="end">${label}</text>`;
  const svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${stationary ? "Not assessed, train stationary" : side ? `Corrugation on ${side}` : "Both rails normal"}">
    ${sleepers}${rail(40, "Side I", side === "Side I")}${rail(90, "Side II", side === "Side II")}
  </svg>`;
  return h("div", { class: `track${stationary ? " track-na" : ""}`, html: svg });
}

function bandChart(b, bandLabels) {
  return chartHolder((holder) => bars(holder, {
    labels: bandLabels,
    series: [{ name: "Side I", values: b.side1, color: "var(--series-1)" }, { name: "Side II", values: b.side2, color: "var(--series-2)" }],
    yLabel: "Log energy", xLabel: "Frequency band (Hz)", height: 240,
    ariaLabel: `Vibration band energy per rail side for ${b.file_id}`,
    tip: (si, k) => [(si ? b.side2 : b.side1)[k].toFixed(3), [`${si ? "Side II" : "Side I"} · ${bandLabels[k]} Hz`, `Side I − Side II: ${(b.side1[k] - b.side2[k]).toFixed(3)}`]],
  }));
}

export function railView({ result, bands, bandLabels }, { openPanel }) {
  const n = result.length;
  const rows = result.map((r, i) => {
    const stationary = !!(bands[i] && bands[i].stationary);
    return { ...r, i, stationary, kmh: r.speed_m_s * 3.6, status: stationary ? "Not assessed" : r.prediction };
  });
  const faults = rows.filter((r) => !r.stationary && r.prediction !== NORMAL);
  const still = rows.filter((r) => r.stationary);
  const s1 = faults.filter((r) => r.prediction === "Side I").length;
  const s2 = faults.filter((r) => r.prediction === "Side II").length;
  const level = faults.length ? "fault" : still.length ? "na" : "ok";
  const statusChip = (r) => (r.stationary ? levelChip("na") : r.prediction === NORMAL ? levelChip("ok", NORMAL) : levelChip("fault", r.prediction));

  const open = (i) => {
    const r = rows[i];
    openPanel({
      title: r.file_id,
      chip: statusChip(r),
      subtitle: r.stationary ? "Train stationary" : `${r.kmh.toFixed(0)} km/h`,
      body: [
        trackDiagram(r.stationary || r.prediction === NORMAL ? null : r.prediction, r.stationary),
        h("div", { class: "stats-2" },
          stat({ label: "Confidence", value: r.stationary ? "-" : certainty(r.confidence).word, caption: r.stationary ? null : `p = ${r.confidence.toFixed(3)}` }),
          stat({ label: "Speed", value: `${r.kmh.toFixed(0)} km/h`, caption: `${r.speed_m_s.toFixed(2)} m/s` })),
        titled("h4", "Vibration energy per band", "Mean over each side's 32 axle boxes. Corrugation shows as a one-sided excess."),
        legend([["Side I", "sw-s1"], ["Side II", "sw-s2"]]),
        bandChart(bands[i], bandLabels),
      ],
    });
  };

  let main;
  if (n === 1) {
    const r = rows[0];
    main = card(titled("h3", "Rail side", "Side I: axle-box positions 1, 3, 5, 7. Side II: positions 2, 4, 6, 8."),
      trackDiagram(r.stationary || r.prediction === NORMAL ? null : r.prediction, r.stationary),
      h("div", { class: "card-foot" }, h("button", { type: "button", class: "btn", onclick: () => open(0) }, "Recording details")));
  } else {
    main = card(titled("h3", `${n} recordings`, "Select a recording for its track view and vibration spectrum."),
      h("div", { class: "rec-grid" }, rows.map((r) => h("button", {
        type: "button", class: `rec rec-${r.stationary ? "na" : r.prediction === NORMAL ? "ok" : "bad"}`,
        onclick: () => open(r.i), "data-tip": `${r.file_id}: ${r.status}`, "data-tip-sub": r.stationary ? "speed < 0.5 m/s" : `${r.kmh.toFixed(0)} km/h · ${certainty(r.confidence).word} confidence`, "aria-label": `${r.file_id}, ${r.status}`,
      }, h("span", {}, r.file_id.replace(/\.csv$/i, "")), r.prediction !== NORMAL && !r.stationary ? h("b", {}, r.prediction === "Side I" ? "I" : "II") : null))),
      legend([["Corrugation (side shown)", "sw-act"], ["Normal", "sw-ok"], ["Not assessed", "sw-na"]]));
  }

  const cols = [
    { key: "file_id", label: "Recording" },
    { key: "status", label: "Status", fmt: (_, r) => statusChip(r) },
    { key: "confidence", label: "Confidence", fmt: (v, r) => (r.stationary ? "-" : certaintyCell(v)) },
    { key: "kmh", label: "Speed (km/h)", num: true, fmt: (v) => v.toFixed(0) },
  ];
  const records = filterable(rows, (r) => r.status !== NORMAL, cols, { maxHeight: 520, onOpen: (r) => open(r.i), rowClass: (r) => (r.status !== NORMAL ? "row-bad" : ""), caption: "Rail recordings" });

  // Signals: spectrum for any recording, chosen from a list.
  const select = h("select", { "aria-label": "Recording" }, rows.map((r) => h("option", { value: r.i }, `${r.file_id} (${r.status})`)));
  const spec = h("div");
  const drawSpec = () => spec.replaceChildren(bandChart(bands[+select.value], bandLabels));
  select.addEventListener("change", drawSpec);
  select.value = String(faults.length ? faults[0].i : 0);
  drawSpec();
  const signals = [card(h("div", { class: "card-head" }, titled("h3", "Vibration energy per band", "Mean log energy across each side's 32 axle boxes."), select),
    legend([["Side I", "sw-s1"], ["Side II", "sw-s2"]]), spec)];

  const headline = n === 1
    ? (faults.length ? `Corrugation on ${faults[0].prediction}` : still.length ? "Not assessed: train stationary" : "No corrugation detected")
    : (faults.length ? `Corrugation in ${faults.length} of ${n} recordings` : `No corrugation in ${n - still.length} assessed recordings`);

  return {
    level, headline,
    tile: n === 1 ? { value: faults.length ? faults[0].prediction : still.length ? "N/A" : NORMAL, caption: "1 recording" } : { value: `${faults.length} / ${n}`, caption: "with corrugation" },
    mini: h("div", { class: "mini-rails" }, h("i", { class: s1 ? "bad" : "" }), h("i", { class: s2 ? "bad" : "" })),
    stats: n === 1 ? [
      stat({ label: "Classification", value: rows[0].stationary ? "Not assessed" : rows[0].prediction }),
      stat({ label: "Confidence", value: rows[0].stationary ? "-" : certainty(rows[0].confidence).word, caption: rows[0].stationary ? null : `p = ${rows[0].confidence.toFixed(3)}`, tip: "Random forest probability for the chosen class. High ≥ 0.9, Medium ≥ 0.7, Low below that." }),
      stat({ label: "Speed", value: `${rows[0].kmh.toFixed(0)} km/h`, caption: "from 90-tooth speed sensor" }),
    ] : [
      stat({ label: "Recordings", value: int(n) }),
      stat({ label: "Side I", value: int(s1), chip: s1 ? levelChip("fault", "corrugation") : null }),
      stat({ label: "Side II", value: int(s2), chip: s2 ? levelChip("fault", "corrugation") : null }),
      stat({ label: "Not assessed", value: int(still.length), caption: "speed < 0.5 m/s", tip: "Stationary recordings have no wheel-rail excitation, so they are reported as Normal in the CSV." }),
    ],
    main, records, signals,
    method: "Per-channel RMS, peak, kurtosis and 9-band log energy, aggregated per rail side plus Side I − Side II differences; random forest with class priors tuned for macro F1.",
    csv: { filename: "rail_predictions.csv", text: toCsv(result, ["file_id", "prediction"]) },
    print: {
      note: faults.length || still.length ? "Flagged recordings" : `All ${n} recordings normal.`,
      columns: ["Recording", "Status", "Speed"],
      rows: rows.filter((r) => r.status !== NORMAL).map((r) => [r.file_id, r.status, `${r.kmh.toFixed(0)} km/h`]),
    },
  };
}

// ================================================================ SHM ====
// Miner's rule: fatigue failure is expected at D = 1. No other thresholds are assumed.
const shmLevel = (d) => (d >= 1 ? "fault" : "ok");
const lifeLeft = (d) => {
  const v = remainingLife(d);
  if (!Number.isFinite(v)) return "-";
  return v < 1 ? "< 1" : v < 10 ? v.toFixed(1) : int(v);
};

export function shmView({ result, detail, fit }, { openPanel }) {
  const n = result.length;
  const rows = result.map((r, i) => ({ ...r, i, level: shmLevel(r.prediction), ...(detail ? detail[i] : {}) }));
  const sorted = [...rows].sort((a, b) => b.prediction - a.prediction);
  const worst = sorted[0];
  const level = worst.level;
  const failed = rows.filter((r) => r.level === "fault");
  const median = [...rows].sort((a, b) => a.prediction - b.prediction)[Math.floor((n - 1) / 2)].prediction;

  const gauge = (r, big) => h("div", { class: `g-track${big ? " big" : ""}` },
    h("i", { class: `g-fill${r.level === "fault" ? " sev-fault" : ""}`, style: `width:${Math.min(100, r.prediction * 100)}%` }));

  const open = (i) => {
    const r = rows[i];
    openPanel({
      title: r.file_id,
      chip: levelChip(r.level),
      subtitle: `D = ${r.prediction.toFixed(4)}`,
      body: [
        gauge(r, true),
        h("div", { class: "g-ticks", "aria-hidden": "true" }, ["0", "0.25", "0.5", "0.75", "1"].map((t) => h("span", {}, t))),
        h("div", { class: "stats-2" },
          stat({ label: "Cumulative damage D", value: r.prediction.toFixed(3) }),
          stat({ label: "Segments to D = 1", value: lifeLeft(r.prediction), tip: "Further segments of the same length and loading before Miner's sum reaches 1." }),
          r.cycles ? stat({ label: "Rainflow cycles", value: int(r.cycles) }) : null,
          r.peak_amplitude ? stat({ label: "Max amplitude", value: r.peak_amplitude.toFixed(1) }) : null),
        r.histogram ? [titled("h4", "Rainflow histogram", "Cycle count per stress-amplitude bin, log scale. Damage grows with amplitude to the power m."),
          chartHolder((holder) => bars(holder, {
            labels: r.histogram.map((b) => b.lo.toFixed(1)),
            series: [{ name: "Cycles", values: r.histogram.map((b) => b.count), color: "var(--series-1)" }],
            yLabel: "Cycles", xLabel: "Stress amplitude", log: true, height: 220,
            ariaLabel: `Rainflow cycle histogram for ${r.file_id}`,
            tip: (_, k) => [`${r.histogram[k].count} cycles`, [`Amplitude ${r.histogram[k].lo.toFixed(2)} to ${r.histogram[k].hi.toFixed(2)}`]],
          }))] : null,
      ],
    });
  };

  const main = h("div", { class: "stack" },
    card(titled("h3", "Cumulative damage D", "Miner's cumulative damage per segment. Fatigue failure is expected at D = 1; bars are drawn on a 0 to 1 scale."),
      h("div", { class: `gauges${n > 10 ? " scroll" : ""}` }, sorted.map((r) => h("button", {
        type: "button", class: "gauge-row", onclick: () => open(r.i), "aria-label": `${r.file_id}, D ${r.prediction.toFixed(3)}, ${r.level}`,
      }, h("span", { class: "g-name" }, r.file_id), gauge(r), h("span", { class: "g-val" }, r.prediction.toFixed(3))))),
      legend([["D < 1", "sw-s1"], ["D ≥ 1, failure expected (Miner's rule)", "sw-act"]])),
    // Cumulative, per-stream monitoring (persisted server-side).
    mountShmStream());

  const cols = [
    { key: "file_id", label: "File" },
    { key: "prediction", label: "D", num: true, fmt: fmt(4) },
    { key: "level", label: "Status", fmt: (v) => levelChip(v) },
    { key: "prediction", label: "Segments to D = 1", num: true, fmt: (v) => lifeLeft(v) },
    { key: "cycles", label: "Rainflow cycles", num: true, fmt: (v) => (v ? int(v) : "-") },
  ];
  const records = filterable(rows, (r) => r.level === "fault", cols, { maxHeight: 520, onOpen: (r) => open(r.i), rowClass: (r) => (r.level === "fault" ? "row-bad" : ""), caption: "Stress segments" });

  const signals = [card(titled("h3", "S-N fit", "D = Σ nᵢ·σᵢᵐ / C, fitted to the 64 labelled segments. m = 5 is a typical exponent for welded steel."),
    kv([["m", fit.m.toFixed(2)], ["C", fit.C.toPrecision(3)], ["Amplitude cut-off", String(fit.amplitude_cutoff)], ["Leave-one-out MAPE", `${(fit.loo_mape * 100).toFixed(1)}%`]]))];

  return {
    level,
    headline: n === 1 ? `Cumulative damage D = ${worst.prediction.toFixed(2)}` : failed.length ? `${failed.length} of ${n} segments at D ≥ 1` : `All ${n} segments below D = 1`,
    tile: { value: worst.prediction.toFixed(2), caption: n === 1 ? "cumulative damage D" : `max D, ${n} segments` },
    mini: h("div", { class: "mini-bars" }, sorted.slice(0, 16).map((r) => h("i", { class: r.level === "fault" ? "sev-fault" : "", style: `height:${Math.max(8, Math.min(100, r.prediction * 100))}%` }))),
    stats: n === 1 ? [
      stat({ label: "Cumulative damage D", value: worst.prediction.toFixed(3), chip: levelChip(level), tip: "Miner's rule: fatigue failure is expected at D = 1." }),
      stat({ label: "Segments to D = 1", value: lifeLeft(worst.prediction), tip: "Further segments of the same length and loading before D reaches 1." }),
      stat({ label: "Rainflow cycles", value: worst.cycles ? int(worst.cycles) : "-" }),
    ] : [
      stat({ label: "Segments", value: int(n) }),
      stat({ label: "Max D", value: worst.prediction.toFixed(3), chip: levelChip(level), caption: worst.file_id }),
      stat({ label: "D ≥ 1", value: int(failed.length), caption: "failure expected (Miner's rule)" }),
      stat({ label: "Median D", value: median.toFixed(3) }),
    ],
    main, records, signals,
    method: `Rainflow counting (ASTM E1049), then Miner's rule with a fitted S-N curve (m = ${fit.m.toFixed(2)}). Leave-one-out MAPE ${(fit.loo_mape * 100).toFixed(1)}%.`,
    csv: { filename: "shm_predictions.csv", text: toCsv(result.map((r) => ({ ...r, prediction: pyFloat(r.prediction) })), ["file_id", "prediction"]) },
    print: {
      note: "Segments by cumulative damage",
      columns: ["File", "D", "Status"],
      rows: sorted.slice(0, 20).map((r) => [r.file_id, r.prediction.toFixed(3), LEVELS[r.level].label]),
    },
  };
}

export const VIEWS = { Door: doorView, ACV: acvView, "Rail corrugation": railView, SHM: shmView };
