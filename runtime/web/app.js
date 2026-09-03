/* Dashboard. One WebSocket in, canvas out.
 *
 * The rule this file follows: NOTHING here is allowed to cost the backend
 * anything. The socket is a pure consumer, it never asks for a frame and never
 * acknowledges one, so a stalled tab drops frames instead of stalling
 * perception. Every canvas is redrawn from the last payload, so a dropped
 * frame costs a redraw and not a gap.
 *
 * Numbers that are unavailable render as unavailable, never as zero. A
 * simulator that is not running and a simulator that is idle look identical on
 * a bar chart, and only one of them means anything. */
const $ = id => document.getElementById(id);
const css = v => getComputedStyle(document.documentElement)
  .getPropertyValue(v).trim();
const fmt = (v, d = 1) => (v == null || !isFinite(v)) ? "—"
  : v.toLocaleString(undefined, {minimumFractionDigits: d,
                                 maximumFractionDigits: d});
const b64 = (s, T) => { const b = atob(s), a = new Uint8Array(b.length);
  for (let i = 0; i < b.length; i++) a[i] = b.charCodeAt(i);
  return new T(a.buffer); };

let LAST = null, MAP = null;
const STAGE = [["ground", "--s1"], ["detect", "--s2"], ["label", "--s5"],
               ["accumulate", "--s3"], ["drivability", "--s4"]];
const DRIVE = [["drivable", "--ok"], ["marginal", "--warn"],
               ["non_drivable", "--no"]];

/* ------------------------------------------------------------- transport */
function connect(){
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = e => { LAST = JSON.parse(e.data); render(LAST); };
  // Reconnect rather than die. The backend gets restarted constantly during
  // development and a dashboard that needs a manual refresh after every
  // restart is a dashboard nobody leaves open.
  ws.onclose = () => { setState("offline", "err"); setTimeout(connect, 1200); };
  ws.onerror = () => ws.close();
}
function setState(text, cls){
  const p = $("p-state");
  p.className = "pill " + (cls || "");
  p.querySelector("span").textContent = text;
}

/* ------------------------------------------------------------- rendering */
function render(s){
  const t = s.telemetry || {}, st = s.stats || {}, f = s.frame;
  if (f) MAP = f;

  setState(s.state === "running" ? "running"
         : s.state === "error" ? "error"
         : s.state === "loading" ? "loading model" : "idle",
    {running: "run", error: "err", loading: "load"}[s.state] || "");
  $("p-dev").textContent = t.gpu_name || "cpu";
  $("p-frame").textContent = MAP ? `frame ${MAP.frame}` : "frame —";
  if (s.err) $("p-frame").textContent = s.err.slice(0, 60);

  kpis(s, t, st);
  budget(st, MAP);
  latency(s.hist || [], st);
  resources(t);
  detections(MAP, s.classes || []);
  terrain(MAP);
  drawMap(MAP);
}

function kpis(s, t, st){
  const used = st.median ? st.median / s.budget_ms : null;
  const grade = v => v == null ? "" : v < 0.6 ? "good" : v < 0.9 ? "warn" : "bad";
  const cards = [
    {k: "FPS", v: st.fps != null ? fmt(st.fps, 1) : "—", cls: "good",
     s: st.n ? `over ${st.n} sweeps` : "waiting"},
    {k: "FRAME", v: st.median != null ? fmt(st.median, 1) : "—", u: "ms",
     cls: grade(used), s: st.p99 != null ? `p99 ${fmt(st.p99, 1)} ms` : ""},
    {k: "BUDGET", v: used != null ? fmt(100 * used, 0) : "—", u: "%",
     cls: grade(used), s: "of 100 ms at 10 Hz"},
    {k: "BACKEND GPU", v: t.gpu_backend_mb != null ? fmt(t.gpu_backend_mb, 0) : "—",
     u: "MB", cls: "", s: t.gpu_backend_reserved_mb != null
       ? `${fmt(t.gpu_backend_reserved_mb, 0)} MB reserved` : ""},
    {k: "BACKEND CPU", v: t.cpu_backend != null ? fmt(t.cpu_backend, 2) : "—",
     u: "cores", cls: "", s: t.cores_total ? `of ${t.cores_total}` : ""},
    {k: "OBJECTS", v: MAP ? Object.values(MAP.counts || {})
        .reduce((a, b) => a + b, 0) : "—", u: "",
     cls: "", s: MAP ? `${MAP.clusters} clusters examined` : ""},
  ];
  $("kpis").innerHTML = cards.map(c => `<div class="kpi ${c.cls}">
    <div class="k">${c.k}</div>
    <div class="v">${c.v}${c.u ? `<span class="u">${c.u}</span>` : ""}</div>
    <div class="s">${c.s || ""}</div></div>`).join("");
}

/* The frame budget: one stacked bar against the sensor's period, so the empty
   space to the right IS the headroom rather than something to work out. */
function budget(st, f){
  const c = $("c-budget"), d = fit(c);
  const W = c.width / dpr(), H = c.height / dpr();
  d.clearRect(0, 0, W, H);
  if (!f || !f.ms){ return; }
  const period = 100, L = 8, R = 108, TR = W - L - R;
  const x = v => L + v / period * TR;
  const y = 20, h = 34;

  d.fillStyle = "#0a0f17";
  d.strokeStyle = css("--line");
  round(d, L, y, TR, h, 7); d.fill(); d.stroke();

  let acc = 0;
  const parts = [];
  for (const [k, v] of STAGE){
    const ms = f.ms[k]; if (ms == null) continue;
    const x0 = x(acc), w = Math.max(x(acc + ms) - x0 - 1.5, 1);
    d.fillStyle = css(v);
    round(d, x0, y, w, h, 3); d.fill();
    parts.push({n: k, c: css(v), ms});
    acc += ms;
  }
  // the deadline
  d.strokeStyle = css("--no"); d.setLineDash([4, 4]); d.lineWidth = 1.5;
  d.beginPath(); d.moveTo(x(period), y - 9); d.lineTo(x(period), y + h + 9);
  d.stroke(); d.setLineDash([]);
  d.fillStyle = css("--ink3"); d.font = "10px " + css("--mono");
  d.fillText("100 ms", x(period) - 22, y - 13);

  d.fillStyle = css("--ink"); d.font = "600 13px " + css("--mono");
  d.fillText(`${fmt(acc, 1)} ms`, x(Math.min(acc, period)) + 10, y + 16);
  d.fillStyle = css("--ink3"); d.font = "10.5px " + css("--mono");
  d.fillText(`${fmt(100 * acc / period, 0)}% used`,
             x(Math.min(acc, period)) + 10, y + 30);

  $("lg-budget").innerHTML = parts.map(p =>
    `<span><i style="background:${p.c}"></i>${p.n} ${fmt(p.ms, 1)} ms</span>`)
    .join("") + (f.read_ms != null
      ? `<span style="color:#3f4a5c">read ${fmt(f.read_ms, 1)} ms, excluded:
         a sensor hands points over in memory</span>` : "");
  $("budget-sub").textContent = st.median != null
    ? `median ${fmt(st.median, 1)} ms · p95 ${fmt(st.p95, 1)} · p99 ${fmt(st.p99, 1)}`
    : "—";
}

function latency(h, st){
  const c = $("c-lat"), d = fit(c);
  const W = c.width / dpr(), H = c.height / dpr();
  d.clearRect(0, 0, W, H);
  if (!h.length) return;
  const B = 22, top = 10, PH = H - B - top;
  const max = Math.max(100, ...h) * 1.05;
  const yv = v => top + PH - (v / max) * PH;

  d.strokeStyle = css("--line"); d.lineWidth = 1;
  for (const g of [25, 50, 75, 100]){
    d.beginPath(); d.moveTo(0, yv(g)); d.lineTo(W, yv(g)); d.stroke();
  }
  const bw = Math.max(W / h.length, 1);
  h.forEach((v, i) => {
    // colour by whether that individual frame made the deadline: the point of
    // a per-frame view is to show the ones that did not
    d.fillStyle = v > 100 ? css("--no") : v > 60 ? css("--warn") : css("--s1");
    const y = yv(v);
    d.fillRect(i * bw, y, Math.max(bw - 1, 1), top + PH - y);
  });
  d.strokeStyle = css("--no"); d.setLineDash([5, 4]); d.lineWidth = 1.5;
  d.beginPath(); d.moveTo(0, yv(100)); d.lineTo(W, yv(100)); d.stroke();
  d.setLineDash([]);
  d.fillStyle = css("--ink3"); d.font = "10px " + css("--mono");
  d.fillText("100 ms deadline", 4, yv(100) - 5);
  const over = h.filter(v => v > 100).length;
  $("lat-sub").textContent =
    `${h.length} sweeps · ${over} over budget (${fmt(100 * over / h.length, 1)}%)`;
}

/* Resource attribution. Ours is measured; the remainder is labelled as the
   remainder, because per-process GPU memory is not exposed on this driver and
   claiming otherwise would be inventing a number. */
function resources(t){
  const rows = [];
  const bar = segs => `<div class="track">` + segs.map(s =>
    `<i style="width:${Math.max(0, Math.min(100, s.w))}%;background:${s.c}"></i>`)
    .join("") + `</div>`;

  if (t.cores_total){
    const b = t.cpu_backend, sim = t.cpu_sim, n = t.cores_total;
    rows.push(`<div class="res-row">
      <div class="res-top"><b>CPU</b>
        <span>backend ${b != null ? fmt(b, 2) + " cores" : "—"}</span>
        <span class="r">${sim != null ? "simulator " + fmt(sim, 2)
          : `<span class="absent">simulator not running</span>`}
          &nbsp;/&nbsp;${n} cores</span></div>
      ${bar([{w: 100 * (b || 0) / n, c: css("--s1")},
             {w: 100 * (sim || 0) / n, c: css("--s2")}])}</div>`);
  }
  if (t.gpu_total_mb){
    const ours = t.gpu_backend_reserved_mb || 0, other = t.gpu_other_mb || 0;
    const tot = t.gpu_total_mb;
    rows.push(`<div class="res-row">
      <div class="res-top"><b>GPU memory</b>
        <span>backend ${fmt(ours, 0)} MB</span>
        <span class="r">other ${fmt(other, 0)} MB / ${fmt(tot / 1024, 1)} GB</span>
      </div>
      ${bar([{w: 100 * ours / tot, c: css("--s1")},
             {w: 100 * other / tot, c: css("--s2")}])}</div>`);
  }
  if (t.gpu_util != null){
    rows.push(`<div class="res-row">
      <div class="res-top"><b>GPU utilisation</b>
        <span>whole card</span>
        <span class="r">${fmt(t.gpu_util, 0)}% · ${fmt(t.gpu_clock_mhz, 0)} MHz
          · ${fmt(t.gpu_temp_c, 0)}&deg;C${
          isFinite(t.gpu_power_w) ? " · " + fmt(t.gpu_power_w, 1) + " W" : ""}</span>
      </div>
      ${bar([{w: t.gpu_util, c: css("--accent2")}])}</div>`);
  }
  $("res").innerHTML = rows.join("") || `<div class="empty">waiting</div>`;
  $("res-note").innerHTML =
    `CPU is measured per process for both, directly. GPU memory is exact for ` +
    `the backend and inferred for everything else: this driver does not expose ` +
    `per-process GPU memory, so "other" is the total minus ours and includes ` +
    `the desktop as well as the simulator. Utilisation is whole-card for the ` +
    `same reason and is not split, because a split would be invented.`;
}

function detections(f, classes){
  if (!f){ $("det").innerHTML = `<div class="empty">waiting</div>`; return; }
  const c = f.counts || {}, keys = Object.keys(c).sort((a, b) => c[b] - c[a]);
  const max = Math.max(1, ...keys.map(k => c[k]));
  const hue = ["--s1", "--s2", "--s3", "--s4", "--s5"];
  $("det").innerHTML = keys.length ? `<div class="rowlist">` + keys.map((k, i) =>
    `<div class="row"><i class="sw" style="background:${css(hue[i % 5])}"></i>
      <span class="n">${k}</span>
      <span class="bar2"><i style="width:${100 * c[k] / max}%;
        background:${css(hue[i % 5])}"></i></span>
      <span class="v">${c[k]}</span></div>`).join("") + `</div>`
    : `<div class="empty">no objects in this sweep</div>`;
  $("det-sub").textContent =
    `${f.clusters} clusters · ${(f.npts || 0).toLocaleString()} points`;
}

function terrain(f){
  if (!f){ $("terr").innerHTML = `<div class="empty">waiting</div>`; return; }
  const d = f.drive || {}, n = f.drive_n || {};
  $("terr").innerHTML = `<div class="rowlist">` +
    DRIVE.concat([["unknown", "--ink3"]]).map(([k, v]) =>
      `<div class="row"><i class="sw" style="background:${css(v)}"></i>
        <span class="n">${k.replace("_", " ")}</span>
        <span class="bar2"><i style="width:${100 * (d[k] || 0)}%;
          background:${css(v)}"></i></span>
        <span class="v">${fmt(100 * (d[k] || 0), 1)}%</span></div>`).join("") +
    `</div>`;
  const seen = 1 - (d.unknown || 0);
  $("terr-sub").textContent =
    `${(n.drivable || 0).toLocaleString()} drivable of ` +
    `${fmt(100 * seen, 0)}% seen`;
}

/* The adaptive 2.5D map.
 *
 * Three things this has to show that a flat classification raster cannot, and
 * all three are the actual claim of the system:
 *
 *   ELEVATION. It is the "2.5" in 2.5D. The cell height was being computed
 *   every frame and thrown away, so the panel drew coloured squares on a
 *   plane and there was nothing in the picture a 2D occupancy grid could not
 *   also produce.
 *
 *   RELIEF. Height as a colour is nearly unreadable across a road with 30 cm
 *   of total relief. Height as a SHADE, lit from a fixed angle, is readable at
 *   a centimetre: the eye is far better at surface orientation than at
 *   absolute value. So the fill carries the class and the shading carries the
 *   shape, and neither has to fight the other for the same channel.
 *
 *   FOVEATION. Cell size grows with range, which is the whole design, and it
 *   is invisible at this zoom because a 5 cm cell is a fifth of a pixel. So
 *   the tier boundaries are drawn and labelled instead of implied.
 *
 * The oblique view tilts the ground plane and lifts each cell by its own
 * height. It is a projection, not a 3D scene: no camera, no depth buffer, one
 * pass over the cells in draw order. That is deliberate, because this has to
 * redraw inside a 100 ms frame budget on a machine that is also running
 * perception and a simulator. */
let MAP_MODE = "drive", MAP_VIEW = "obl";

function drawMap(f){
  const c = $("c-map"), d = fit(c);
  const W = c.width / dpr(), H = c.height / dpr();
  d.clearRect(0, 0, W, H);
  if (!f || !f.n_cells){ return; }

  // int16 offsets from the vehicle's own cell, so the payload is a constant
  // size however long the run has been going
  const gx = f.origin[0], gy = f.origin[1];
  const rx = b64(f.ix, Int16Array), ry = b64(f.iy, Int16Array),
        cl = b64(f.cls, Uint8Array), res = f.res;
  const zq = f.z ? b64(f.z, Int16Array) : null;
  const z0 = f.z0 || 0;
  const n = rx.length;

  // extent in CELL units, vehicle at the origin of the offsets
  let x0 = -20 / res, x1 = 20 / res, y0 = -20 / res, y1 = 20 / res;
  for (let i = 0; i < n; i++){
    if (rx[i] < x0) x0 = rx[i]; if (rx[i] > x1) x1 = rx[i];
    if (ry[i] < y0) y0 = ry[i]; if (ry[i] > y1) y1 = ry[i];
  }
  const spanX = x1 - x0 + 1, spanY = y1 - y0 + 1;
  const tilt = MAP_VIEW === "obl" ? 0.52 : 1.0;   // vertical squash
  // EQUAL ASPECT in the ground plane before the tilt, so a square cell stays
  // square and the tilt is the only thing changing proportions
  const s = Math.min(W / spanX, H / (spanY * tilt)) * 0.94;
  const cw = Math.max(s, 0.8);
  const ox = (W - spanX * s) / 2, oy = (H - spanY * s * tilt) / 2;
  const X = i => ox + (i - x0) * s;
  const Y = (j, zm) => oy + (y1 - j) * s * tilt - (MAP_VIEW === "obl"
    ? (zm || 0) * 0.001 * s / res * 0.55 : 0);

  // height range, from the data rather than assumed: a road with 30 cm of
  // relief and a junction with 4 m must both be readable
  let zlo = 1e9, zhi = -1e9;
  if (zq) for (let i = 0; i < n; i++){
    if (zq[i] < zlo) zlo = zq[i]; if (zq[i] > zhi) zhi = zq[i];
  }
  const zsp = Math.max(zhi - zlo, 1);

  const DR = [css("--ok"), css("--warn"), css("--no")];
  const rgb = h => {            // one hue, light to dark, for elevation
    const t = Math.max(0, Math.min(1, h));
    return `rgb(${Math.round(28 + 90 * t)},${Math.round(70 + 130 * t)},` +
           `${Math.round(110 + 120 * t)})`;
  };

  // Painter's order: far cells first, so a nearer cell lifted by its height
  // covers the one behind it instead of poking through. Without this the
  // oblique view is a mess of overlapping squares and reads as noise.
  const idx = new Int32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  if (MAP_VIEW === "obl") idx.sort((a, b) => ry[b] - ry[a]);

  for (let k = 0; k < n; k++){
    const i = idx[k];
    const zm = zq ? zq[i] : 0;
    const t = zq ? (zm - zlo) / zsp : 0.5;
    let fill;
    if (MAP_MODE === "elev") fill = rgb(t);
    else {
      // class as hue, height as brightness. Shading a classified surface is
      // what makes it read as a surface at all; flat fills read as a chart.
      fill = DR[cl[i]] || "#334";
      d.globalAlpha = 0.55 + 0.45 * t;
    }
    d.fillStyle = fill;
    const px = X(rx[i]), py = Y(ry[i], zm);
    d.fillRect(px, py, cw + 0.6, cw + 0.6);
    // a lit top edge on raised cells: one line per cell, and it is what turns
    // a field of squares into something with a surface normal
    if (MAP_VIEW === "obl" && zq && t > 0.12){
      d.globalAlpha = Math.min(0.5, t * 0.6);
      d.fillStyle = "#dff3ff";
      d.fillRect(px, py, cw + 0.6, Math.max(cw * 0.22, 0.7));
    }
    d.globalAlpha = 1;
  }

  // TIER RINGS. Where the cell size changes, drawn because at this zoom the
  // change itself is sub-pixel and would otherwise be invisible.
  const vx = X(0), vy = Y(0, 0);
  d.save();
  d.setLineDash([3, 5]);
  d.lineWidth = 1;
  for (const t of (f.tiers || [])){
    const rr = t.half_extent / res * s;
    if (rr < 14 || rr > Math.max(W, H)) continue;
    d.strokeStyle = "#3ba7ff44";
    d.beginPath();
    d.ellipse(vx, vy, rr, rr * tilt, 0, 0, 7);
    d.stroke();
    d.setLineDash([]);
    d.fillStyle = "#5f7a9a";
    d.font = "9.5px " + css("--mono");
    d.fillText(`${(t.res * 100).toFixed(0)} cm`, vx + rr * 0.7,
               vy - rr * tilt * 0.7);
    d.setLineDash([3, 5]);
  }
  d.restore();

  // detections, lifted onto the surface like everything else
  for (const b of f.boxes || []){
    const [bx, by, , dx, dy, , yaw] = b.b;
    const ci = bx / res - gx, cj = by / res - gy;
    d.save();
    d.translate(X(ci), Y(cj, zhi > -1e8 ? zhi * 0.35 : 0));
    d.scale(1, tilt);
    d.rotate(-yaw);
    d.strokeStyle = css("--accent");
    d.lineWidth = 1.6 / Math.max(tilt, 0.3);
    d.strokeRect(-dx / 2 * s / res, -dy / 2 * s / res,
                 dx * s / res, dy * s / res);
    d.restore();
  }

  // the vehicle
  d.fillStyle = css("--accent2");
  d.beginPath(); d.ellipse(vx, vy, 4.5, 4.5 * tilt, 0, 0, 7); d.fill();
  d.strokeStyle = css("--accent2"); d.globalAlpha = .3;
  d.beginPath(); d.ellipse(vx, vy, 20 / res * s, 20 / res * s * tilt, 0, 0, 7);
  d.stroke(); d.globalAlpha = 1;

  const relief = zq ? (zsp / 1000).toFixed(2) : "—";
  $("map-sub").textContent =
    `${f.n_cells.toLocaleString()} cells · ${res} m · ${relief} m relief · ` +
    `${f.boxes.length} boxes`;
  $("lg-map").innerHTML = (MAP_MODE === "elev"
    ? `<span><i style="background:${rgb(0)}"></i>low</span>` +
      `<span><i style="background:${rgb(1)}"></i>high</span>`
    : DRIVE.map(([k, v]) =>
        `<span><i style="background:${css(v)}"></i>${k.replace("_", " ")}</span>`)
       .join("") +
      `<span style="color:#4a5568">brightness is height</span>`)
    + `<span><i style="background:${css("--accent")}"></i>detection</span>`
    + `<span><i style="background:${css("--accent2")}"></i>vehicle, 20 m</span>`
    + `<span style="color:#3f4a5c">dashed rings mark where cell size changes</span>`;
}

/* ---------------------------------------------------------------- canvas */
function dpr(){ return Math.min(window.devicePixelRatio || 1, 2); }
function fit(c){
  const r = dpr(), w = c.clientWidth, h = c.clientHeight || c.height;
  if (c.width !== w * r || c.height !== h * r){
    c.width = w * r; c.height = h * r;
  }
  const d = c.getContext("2d");
  d.setTransform(r, 0, 0, r, 0, 0);
  return d;
}
function round(d, x, y, w, h, r){
  d.beginPath();
  d.moveTo(x + r, y); d.arcTo(x + w, y, x + w, y + h, r);
  d.arcTo(x + w, y + h, x, y + h, r); d.arcTo(x, y + h, x, y, r);
  d.arcTo(x, y, x + w, y, r); d.closePath();
}

/* ---------------------------------------------------------------- control */
$("go").onclick = async () => {
  await fetch("/api/start", {method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({root: $("root").value, loop: true})});
};
$("halt").onclick = () => fetch("/api/stop", {method: "POST"});

// Mode switches redraw from the LAST payload rather than asking the backend
// for anything. Changing how a frame is drawn is not a reason to make
// perception produce another one.
function seg(id, attr, set){
  $(id).querySelectorAll("button").forEach(b => b.onclick = () => {
    $(id).querySelectorAll("button").forEach(x => x.classList.remove("on"));
    b.classList.add("on");
    set(b.dataset[attr]);
    if (MAP) drawMap(MAP);
  });
}
seg("map-modes", "m", v => MAP_MODE = v);
seg("map-views", "v", v => MAP_VIEW = v);
addEventListener("resize", () => LAST && render(LAST));
connect();
