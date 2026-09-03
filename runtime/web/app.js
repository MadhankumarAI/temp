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

/* The 2.5D drivability lattice, vehicle-centred. Cells arrive as world indices
   so the view follows the vehicle without the backend having to re-centre
   anything. */
function drawMap(f){
  const c = $("c-map"), d = fit(c);
  const W = c.width / dpr(), H = c.height / dpr();
  d.clearRect(0, 0, W, H);
  if (!f || !f.n_cells){ return; }
  // int16 offsets from the vehicle's own cell, so the payload is a constant
  // size however long the run has been going. Absolute world indices are
  // rebuilt here, which keeps the rest of this function unchanged.
  const gx = f.origin[0], gy = f.origin[1];
  const rx = b64(f.ix, Int16Array), ry = b64(f.iy, Int16Array),
        cl = b64(f.cls, Uint8Array), res = f.res;
  const ix = new Int32Array(rx.length), iy = new Int32Array(ry.length);
  for (let i = 0; i < rx.length; i++){ ix[i] = rx[i] + gx; iy[i] = ry[i] + gy; }
  const px = f.pose[0] / res, py = f.pose[1] / res;

  // fit the cloud, but keep the vehicle in view even when the map runs away
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < ix.length; i++){
    if (ix[i] < x0) x0 = ix[i]; if (ix[i] > x1) x1 = ix[i];
    if (iy[i] < y0) y0 = iy[i]; if (iy[i] > y1) y1 = iy[i];
  }
  x0 = Math.min(x0, px - 20); x1 = Math.max(x1, px + 20);
  y0 = Math.min(y0, py - 20); y1 = Math.max(y1, py + 20);
  // EQUAL ASPECT: a top-down view on unequal axes reports wrong shapes
  const s = Math.min(W / (x1 - x0 + 1), H / (y1 - y0 + 1));
  const ox = (W - (x1 - x0 + 1) * s) / 2, oy = (H - (y1 - y0 + 1) * s) / 2;
  const X = i => ox + (i - x0) * s, Y = j => H - oy - (j - y0) * s;

  const col = [css("--ok"), css("--warn"), css("--no")];
  const cell = Math.max(s, 1);
  for (let i = 0; i < ix.length; i++){
    d.fillStyle = col[cl[i]] || "#333";
    d.fillRect(X(ix[i]), Y(iy[i]) - cell, cell, cell);
  }

  for (const b of f.boxes || []){
    const [bx, by, , dx, dy, , yaw] = b.b;
    d.save();
    d.translate(X(bx / res), Y(by / res));
    d.rotate(-yaw);
    d.strokeStyle = css("--accent"); d.lineWidth = 1.5;
    d.strokeRect(-dx / 2 * s / res, -dy / 2 * s / res,
                 dx * s / res, dy * s / res);
    d.restore();
  }

  // the vehicle
  d.fillStyle = css("--accent2");
  d.beginPath(); d.arc(X(px), Y(py), 4, 0, 7); d.fill();
  d.strokeStyle = css("--accent2"); d.globalAlpha = .35;
  d.beginPath(); d.arc(X(px), Y(py), 20 / res * s, 0, 7); d.stroke();
  d.globalAlpha = 1;

  $("map-sub").textContent =
    `${f.n_cells.toLocaleString()} cells · ${res} m · ${f.boxes.length} boxes`;
  $("lg-map").innerHTML = DRIVE.map(([k, v]) =>
    `<span><i style="background:${css(v)}"></i>${k.replace("_", " ")}</span>`)
    .join("") + `<span><i style="background:${css("--accent")}"></i>detection</span>`
    + `<span><i style="background:${css("--accent2")}"></i>vehicle, 20 m ring</span>`;
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
addEventListener("resize", () => LAST && render(LAST));
connect();
