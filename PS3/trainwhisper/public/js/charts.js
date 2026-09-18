// Small SVG chart kit: a segmented line chart and a bar chart, both with hover tooltips.
const NS = "http://www.w3.org/2000/svg";
const tooltip = () => document.getElementById("tooltip");

function el(name, attrs = {}, parent) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function text(parent, x, y, str, attrs = {}) {
  const t = el("text", { x, y, ...attrs }, parent);
  t.textContent = str;
  return t;
}

/** Tooltip body: value first (strong), then secondary lines. Always textContent. */
export function showTooltip(evt, value, lines = []) {
  const tip = tooltip();
  tip.replaceChildren();
  const strong = document.createElement("strong");
  strong.textContent = value;
  tip.appendChild(strong);
  for (const l of lines) {
    const d = document.createElement("div");
    d.className = "muted";
    d.textContent = l;
    tip.appendChild(d);
  }
  tip.hidden = false;
  const pad = 14;
  const { innerWidth: w, innerHeight: h } = window;
  const r = tip.getBoundingClientRect();
  let x = evt.clientX + pad;
  let y = evt.clientY + pad;
  if (x + r.width > w - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > h - 8) y = evt.clientY - r.height - pad;
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${Math.max(8, y)}px`;
}

export function hideTooltip() {
  tooltip().hidden = true;
}

function niceTicks(lo, hi, count = 5) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const step0 = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  // ticks always enclose the data: floor the low end, ceil the high end
  const first = Math.floor(lo / step + 1e-9);
  const last = Math.ceil(hi / step - 1e-9);
  const ticks = [];
  for (let k = first; k <= last; k++) ticks.push(+(k * step).toPrecision(12));
  return ticks;
}

const fmtTick = (v) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(0) : String(+v.toPrecision(4)));

/** Re-render on container resize; returns the container. */
function responsive(container, height, draw) {
  container.classList.add("chart");
  let last = 0;
  const render = () => {
    const width = Math.round(container.clientWidth);
    if (!width || width === last) return;
    last = width;
    container.replaceChildren();
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img" }, container);
    draw(svg, width, height);
  };
  new ResizeObserver(render).observe(container);
  render();
  return container;
}

/**
 * Door-style trace: one path per cycle along sample order, shaded where flagged.
 * cycles: [{ points: [[x, y, meta]], abnormal }]
 */
export function segmentedLine(container, { cycles, yLabel, ariaLabel, describe }) {
  const all = cycles.flatMap((c) => c.points);
  const xMin = all[0][0];
  const xMax = all[all.length - 1][0];
  let yMax = 0;
  for (const p of all) yMax = Math.max(yMax, p[1]);
  const yTicks = niceTicks(0, yMax);
  const yTop = yTicks[yTicks.length - 1];

  return responsive(container, 280, (svg, W, H) => {
    svg.setAttribute("aria-label", ariaLabel);
    const m = { l: 56, r: 12, t: 8, b: 36 };
    const sx = (x) => m.l + ((x - xMin) / (xMax - xMin || 1)) * (W - m.l - m.r);
    const sy = (y) => H - m.b - (y / yTop) * (H - m.t - m.b);

    for (const c of cycles) {
      if (!c.abnormal) continue;
      const a = sx(c.points[0][0]);
      const b = sx(c.points[c.points.length - 1][0]);
      el("rect", { x: a - 1, y: m.t, width: Math.max(3, b - a + 2), height: H - m.t - m.b, fill: "var(--critical)", opacity: 0.18 }, svg);
    }
    for (const t of yTicks) {
      el("line", { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t), class: t === 0 ? "baseline" : "grid" }, svg);
      text(svg, m.l - 6, sy(t) + 4, fmtTick(t), { "text-anchor": "end" });
    }
    text(svg, 14, (m.t + H - m.b) / 2, yLabel, { class: "axis-title", "text-anchor": "middle", transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})` });

    // x ticks: cycle numbers, thinned to fit
    const every = Math.max(1, Math.ceil(cycles.length / Math.max(2, Math.floor((W - m.l - m.r) / 44))));
    cycles.forEach((c, i) => {
      if (i % every) return;
      const x = sx(c.points[0][0]);
      text(svg, x, H - m.b + 16, String(i + 1), { "text-anchor": "middle" });
    });
    text(svg, (m.l + W - m.r) / 2, H - 4, "Door cycle (idle gaps between cycles removed)", { class: "axis-title", "text-anchor": "middle" });

    for (const c of cycles) {
      const d = c.points.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
      el("path", { d, fill: "none", stroke: "var(--text-2)", "stroke-width": 1.5, "stroke-linejoin": "round" }, svg);
    }

    // crosshair + nearest-sample tooltip
    const cross = el("line", { y1: m.t, y2: H - m.b, stroke: "var(--muted)", "stroke-width": 1, visibility: "hidden" }, svg);
    const dot = el("circle", { r: 4, fill: "var(--surface)", stroke: "var(--text)", "stroke-width": 2, visibility: "hidden" }, svg);
    const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, class: "hit" }, svg);
    const flat = cycles.flatMap((c, ci) => c.points.map((p) => [p, ci]));
    hit.addEventListener("pointermove", (evt) => {
      const box = svg.getBoundingClientRect();
      const x = xMin + ((evt.clientX - box.left) * (W / box.width) - m.l) / (W - m.l - m.r) * (xMax - xMin);
      let lo = 0;
      let hi = flat.length - 1;
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (flat[mid][0][0] < x) lo = mid; else hi = mid; }
      const [p, ci] = Math.abs(flat[lo][0][0] - x) < Math.abs(flat[hi][0][0] - x) ? flat[lo] : flat[hi];
      cross.setAttribute("x1", sx(p[0])); cross.setAttribute("x2", sx(p[0]));
      dot.setAttribute("cx", sx(p[0])); dot.setAttribute("cy", sy(p[1]));
      cross.setAttribute("visibility", "visible"); dot.setAttribute("visibility", "visible");
      const [value, lines] = describe(p, ci);
      showTooltip(evt, value, lines);
    });
    hit.addEventListener("pointerleave", () => {
      cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden");
      hideTooltip();
    });
  });
}

// Bar path with 4px rounded data-end, square at the baseline.
function barPath(x, w, y0, y1) {
  const r = Math.min(4, w / 2, Math.abs(y1 - y0));
  if (y1 <= y0) {
    return `M${x},${y0}V${y1 + r}Q${x},${y1} ${x + r},${y1}H${x + w - r}Q${x + w},${y1} ${x + w},${y1 + r}V${y0}Z`;
  }
  return `M${x},${y0}V${y1 - r}Q${x},${y1} ${x + r},${y1}H${x + w - r}Q${x + w},${y1} ${x + w},${y1 - r}V${y0}Z`;
}

/**
 * labels: category names; series: [{ name, values, color | colors[] }].
 * Options: yLabel, xLabel, log, tip(seriesIndex, i) -> [value, lines]. Labels rotate only when crowded.
 */
export function bars(container, { labels, series, yLabel, xLabel, log = false, height = 260, ariaLabel, tip }) {
  const vals = series.flatMap((s) => s.values).filter(Number.isFinite);
  const tf = log ? (v) => Math.log10(v) : (v) => v;
  let lo;
  let hi;
  let ticks;
  if (log) {
    const pos = vals.filter((v) => v > 0);
    lo = Math.floor(Math.log10(Math.min(...pos)));
    hi = Math.ceil(Math.log10(Math.max(...pos)));
    if (hi === lo) hi += 1;
    ticks = Array.from({ length: hi - lo + 1 }, (_, i) => lo + i);
  } else {
    ticks = niceTicks(Math.min(0, ...vals), Math.max(0, ...vals));
    lo = ticks[0];
    hi = ticks[ticks.length - 1];
  }

  return responsive(container, height, (svg, W, H) => {
    svg.setAttribute("aria-label", ariaLabel);
    const longest = Math.max(...labels.map((l) => String(l).length));
    // rotate category labels only when they would collide horizontally
    const rotate = ((W - 64) / labels.length) < longest * 6.5 + 8;
    const m = { l: 56, r: 8, t: 8, b: (rotate ? Math.min(90, 12 + longest * 5.5) : 26) + (xLabel ? 18 : 0) };
    const sy = (v) => H - m.b - ((tf(v) - lo) / (hi - lo)) * (H - m.t - m.b);
    const base = log ? H - m.b : sy(0);
    const band = (W - m.l - m.r) / labels.length;
    const gap = 2;
    const inner = Math.min(band * 0.8, 64);
    const bw = (inner - gap * (series.length - 1)) / series.length;

    for (const t of ticks) {
      const y = log ? H - m.b - ((t - lo) / (hi - lo)) * (H - m.t - m.b) : sy(t);
      el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "grid" }, svg);
      text(svg, m.l - 6, y + 4, log ? fmtTick(10 ** t) : fmtTick(t), { "text-anchor": "end" });
    }
    text(svg, 14, (m.t + H - m.b) / 2, yLabel, { class: "axis-title", "text-anchor": "middle", transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})` });
    if (xLabel) text(svg, (m.l + W - m.r) / 2, H - 4, xLabel, { class: "axis-title", "text-anchor": "middle" });

    const every = rotate ? Math.max(1, Math.ceil(labels.length / Math.floor((W - m.l - m.r) / 14))) : Math.max(1, Math.ceil(labels.length * 40 / (W - m.l - m.r)));
    labels.forEach((lab, i) => {
      if (i % every) return;
      const x = m.l + band * (i + 0.5);
      const y = H - m.b + 14;
      text(svg, x, y, lab, rotate ? { "text-anchor": "end", transform: `rotate(-45 ${x} ${y - 4})` } : { "text-anchor": "middle" });
    });

    const marks = [];
    series.forEach((s, si) => {
      s.values.forEach((v, i) => {
        if (!Number.isFinite(v) || (log && v <= 0)) return;
        const x = m.l + band * i + (band - inner) / 2 + si * (bw + gap);
        const fill = s.colors ? s.colors[i] : s.color;
        const p = el("path", { d: barPath(x, Math.max(1, bw), base, sy(v)), fill, class: "mark" }, svg);
        marks.push(p);
        // hit target: full band height for this bar, wider than the mark
        const h = el("rect", { x: x - gap, y: m.t, width: bw + gap * 2, height: H - m.t - m.b, class: "hit", tabindex: 0 }, svg);
        const on = (evt) => {
          marks.forEach((mk) => mk.classList.toggle("dim", mk !== p));
          const [value, lines] = tip(si, i);
          const r = h.getBoundingClientRect();
          showTooltip(evt.clientX ? evt : { clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 }, value, lines);
        };
        const off = () => { marks.forEach((mk) => mk.classList.remove("dim")); hideTooltip(); };
        h.addEventListener("pointermove", on);
        h.addEventListener("focus", on);
        h.addEventListener("pointerleave", off);
        h.addEventListener("blur", off);
      });
    });
    if (!log) el("line", { x1: m.l, x2: W - m.r, y1: base, y2: base, class: "baseline" }, svg);
  });
}
