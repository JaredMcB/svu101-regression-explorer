/* regression.js — SVU 101 Regression Explorer
 *
 * Everything numerical happens here, in the browser, so that clicking a line
 * into the plot redraws immediately. The server is only ever asked to fetch a
 * Google Sheet.
 *
 * The line is set two ways: click two points on the training plot (the line
 * goes through both), or type the slope and intercept.
 *
 * Layout: six panes in two rows. Training row is the data, the fitted line,
 * and the loss surface. Test row is the held-out data, the same line meeting
 * it, and a prediction box.
 */
'use strict';

// ---------------------------------------------------------------- palette --
const CSS   = getComputedStyle(document.documentElement);
const pick  = (name, fallback) => CSS.getPropertyValue(name).trim() || fallback;
const COLOR = {
  train: pick('--train', '#9e1b32'),
  test:  pick('--test',  '#1a2c99'),
  line:  pick('--line',  '#16193a'),
  resid: pick('--resid', '#8b93b8'),
  navy:  pick('--navy',  '#1a2c99'),
  peri:  pick('--peri',  '#5682f3'),
  grid:  pick('--border', '#d5dcf0'),
};
const FONT = { family: '"DM Sans", system-ui, sans-serif', size: 11, color: '#16193a' };

// ------------------------------------------------------------------ state --
const state = {
  xName: 'x', yName: 'y',
  rows: [],            // every [x, y] pair, as loaded
  train: [], test: [],
  m: 0, b: 0,
  fit: null,           // {m, b} least-squares solution, computed on load
  lossGrid: null,      // cached surface so redraws do not recompute it
  anchors: [],         // the 0, 1 or 2 points clicked to define the line
};

const $ = (id) => document.getElementById(id);

/* ============================================================== numbers == */

/** Format to `digits` significant figures, then drop trailing decimal zeros.
 *  Significant figures rather than decimal places because slopes here are
 *  routinely small -- 0.0826 has to survive as 0.08258, not collapse to 0.08.
 *  Trailing zeros are only stripped after a decimal point, or 1200 would be
 *  printed as 12. */
function fmt(v, digits = 4) {
  if (!Number.isFinite(v)) return '—';
  if (v === 0) return '0';
  const mag = Math.abs(v);
  if (mag >= 1e7 || mag < 1e-4) return v.toExponential(2);
  const dec = Math.min(20, Math.max(0, digits - 1 - Math.floor(Math.log10(mag))));
  const s = v.toFixed(dec);
  return s.includes('.') ? s.replace(/0+$/, '').replace(/\.$/, '') : s;
}

const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

function sd(a) {
  if (a.length < 2) return 0;
  const mu = mean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - mu) ** 2, 0) / (a.length - 1));
}

/** Ordinary least squares through the given pairs. */
function leastSquares(pairs) {
  const n = pairs.length;
  if (n < 2) return { m: 0, b: 0 };
  const xs = pairs.map((p) => p[0]);
  const ys = pairs.map((p) => p[1]);
  const xb = mean(xs);
  const yb = mean(ys);
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i += 1) {
    sxy += (xs[i] - xb) * (ys[i] - yb);
    sxx += (xs[i] - xb) ** 2;
  }
  if (sxx === 0) return { m: 0, b: yb };          // all x identical
  const m = sxy / sxx;
  return { m, b: yb - m * xb };
}

/** Mean squared error of the line y = b + m x on these pairs. */
function mse(pairs, m, b) {
  if (!pairs.length) return NaN;
  let s = 0;
  for (const [x, y] of pairs) s += (y - (b + m * x)) ** 2;
  return s / pairs.length;
}

/** The line through two points, or null if they are stacked vertically --
 *  a vertical line has no slope, so y = b + m x cannot describe it. */
function lineThrough([x1, y1], [x2, y2]) {
  if (x1 === x2) return null;
  const m = (y2 - y1) / (x2 - x1);
  return { m, b: y1 - m * x1 };
}

/* -------------------------------------------------------------- shuffle -- */

/** mulberry32 — a small seeded PRNG, so a given seed always splits the same
 *  way. That matters in class: everyone sees identical training and test sets. */
function rng(seed) {
  let a = (seed >>> 0) || 1;
  return function next() {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function splitData(rows, trainPct, seed) {
  const idx = rows.map((_, i) => i);
  const rand = rng(seed);
  for (let i = idx.length - 1; i > 0; i -= 1) {      // Fisher–Yates
    const j = Math.floor(rand() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  // Floor rather than round, so the held-out set gets the remainder: with 16
  // rows an 80/20 split is 12 and 4, which is what the Week 1 activity uses.
  let nTrain = Math.floor((trainPct / 100) * rows.length);
  nTrain = Math.min(Math.max(nTrain, 2), rows.length - 1);
  return {
    train: idx.slice(0, nTrain).map((i) => rows[i]),
    test: idx.slice(nTrain).map((i) => rows[i]),
  };
}

/* ================================================================= CSV === */

/** Minimal CSV reader: handles quoted fields, commas inside quotes, CRLF. */
function parseCSV(text) {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  const s = text.replace(/^﻿/, '');
  for (let i = 0; i < s.length; i += 1) {
    const c = s[i];
    if (quoted) {
      if (c === '"') {
        if (s[i + 1] === '"') { field += '"'; i += 1; } else { quoted = false; }
      } else { field += c; }
    } else if (c === '"') { quoted = true;
    } else if (c === ',') { row.push(field); field = '';
    } else if (c === '\n' || c === '\r') {
      if (c === '\r' && s[i + 1] === '\n') i += 1;
      row.push(field); field = '';
      if (row.some((v) => v.trim() !== '')) rows.push(row);
      row = [];
    } else { field += c; }
  }
  row.push(field);
  if (row.some((v) => v.trim() !== '')) rows.push(row);
  return rows;
}

const isNum = (v) => v !== null && v !== undefined && v.trim() !== ''
  && Number.isFinite(Number(v.trim()));

/** Rows of CSV -> {xName, yName, pairs, skipped}. Mirrors app.py's rules so
 *  that pasting a file and reading the sheet behave identically. */
function readPairs(rows) {
  if (!rows.length) throw new Error('That data is empty.');
  const header = rows[0];
  if (header.length < 2) {
    throw new Error('Two columns are needed: the predictor in column A and '
      + 'the response in column B, each named in row 1.');
  }
  if (isNum(header[0]) && isNum(header[1])) {
    throw new Error('Row 1 should be the variable names, not data. Put the '
      + "predictor's name in A1 and the response's name in B1.");
  }
  const pairs = [];
  let skipped = 0;
  for (const r of rows.slice(1)) {
    if (r.length < 2 || !isNum(r[0]) || !isNum(r[1])) { skipped += 1; continue; }
    pairs.push([Number(r[0].trim()), Number(r[1].trim())]);
  }
  if (pairs.length < 4) {
    throw new Error(`Only ${pairs.length} usable rows were found. An 80/20 `
      + 'split needs at least 4, and really wants 10 or more.');
  }
  return {
    xName: (header[0] || '').trim() || 'x',
    yName: (header[1] || '').trim() || 'y',
    pairs,
    skipped,
  };
}

/* ================================================================ KaTeX == */

/** Render the two standing formulas once. KaTeX loads with `defer`, so it may
 *  not exist yet on first call; retry on window load rather than racing it. */
function renderFormulas() {
  if (typeof katex === 'undefined') return false;
  katex.render(
    String.raw`\text{MSE}=\frac{1}{n}\sum_{i=1}^{n}\bigl(y_i-\hat{y}_i\bigr)^{2}`,
    $('mse-formula'), { throwOnError: false, displayMode: false },
  );
  katex.render(
    String.raw`\hat{y}_i = b + m\,x_i`,
    $('pred-formula-tex'), { throwOnError: false, displayMode: false },
  );
  return true;
}
if (!renderFormulas()) window.addEventListener('load', renderFormulas);

/* ============================================================== loading == */

function setStatus(msg, kind = '') {
  const el = $('load-status');
  el.textContent = msg;
  el.className = `status ${kind}`;
}

function loadData({ xName, yName, pairs, skipped }) {
  state.xName = xName;
  state.yName = yName;
  state.rows = pairs;
  state.fit = leastSquares(pairs);

  // Start at the flat line through the mean: the honest "no relationship"
  // baseline, and a sane place to begin guessing from.
  state.m = 0;
  state.b = Number(mean(pairs.map((p) => p[1])).toFixed(4));

  $('loader').hidden = true;
  $('databar').hidden = false;
  $('workspace').hidden = false;
  document.body.classList.add('has-data');
  $('pred-x-label').textContent = xName;
  $('db-vars').textContent = `${xName}  →  ${yName}`;

  applySplit();
  clearAnchors();
  syncInputs();
  redrawAll();

  const note = skipped ? ` (${skipped} row${skipped === 1 ? '' : 's'} skipped)` : '';
  setStatus(`Loaded ${pairs.length} rows: ${xName} → ${yName}${note}.`, 'ok');
  observePlots();
}

function applySplit() {
  const pct = Number($('train-pct').value) || 80;
  const seed = Number($('seed').value) || 0;
  const { train, test } = splitData(state.rows, pct, seed);
  state.train = train;
  state.test = test;
  state.fit = leastSquares(train);      // the target is fit on training only
  state.lossGrid = null;                // surface depends on the training set
  $('db-counts').innerHTML =
    `${state.rows.length} rows`
    + `<span class="pill train">${train.length} training</span>`
    + `<span class="pill test">${test.length} test</span>`;
}

/* ------------------------------------------------------------- controls -- */

/** The (m, b) window plotted by the loss surface.
 *
 *  Scaling both axes by the same multiple of the data's spread makes the
 *  surface look like a trough, because MSE is far more sensitive to the slope
 *  than to the intercept: a step in m is levered by x, a step in b is not.
 *
 *  So instead of a fixed number of standard deviations, pick a target rise in
 *  loss and ask how far you have to move along each axis to climb it. For the
 *  quadratic MSE surface that is
 *
 *      MSE(m* + dm, b*) - MSE(m*, b*) = dm^2 * mean(x^2)
 *      MSE(m*, b* + db) - MSE(m*, b*) = db^2
 *
 *  so half-widths of sqrt(rise / mean(x^2)) and sqrt(rise) climb equally far
 *  in both directions. With rise set to the variance of y -- a natural "how
 *  wrong could you be" scale -- the result reads as a bowl rather than a
 *  canyon, and sits near the bottom of the paraboloid instead of up its walls.
 */
function lossWindow() {
  const xs = state.train.map((p) => p[0]);
  const ys = state.train.map((p) => p[1]);
  const varY = Math.max(sd(ys) ** 2, 1e-9);            // the target rise
  const meanX2 = Math.max(mean(xs.map((x) => x * x)), 1e-9);
  const mHalf = Math.sqrt(varY / meanX2) || 1;
  const bHalf = Math.sqrt(varY) || 1;
  let mLo = state.fit.m - mHalf;
  let mHi = state.fit.m + mHalf;
  let bLo = state.fit.b - bHalf;
  let bHi = state.fit.b + bHalf;
  // Keep the current line -- typed or clicked -- inside the visible window.
  mLo = Math.min(mLo, state.m); mHi = Math.max(mHi, state.m);
  bLo = Math.min(bLo, state.b); bHi = Math.max(bHi, state.b);
  return { mLo, mHi, bLo, bHi };
}

/** Push state back into the controls.
 *
 *  Never rewrite the box the user is currently typing in. Reformatting on
 *  every keystroke ate the decimal point -- typing "0.08" became "0", "0",
 *  "0", "8" -- which looked exactly like the field refusing to accept floats.
 */
function syncInputs() {
  const active = document.activeElement;
  if (active !== $('m')) $('m').value = fmt(state.m, 6);
  if (active !== $('b')) $('b').value = fmt(state.b, 6);
}

/* ================================================================ plots == */

const layoutBase = (xTitle, yTitle) => ({
  font: FONT,
  margin: { l: 52, r: 14, t: 10, b: 46 },
  showlegend: false,
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  hovermode: 'closest',
  xaxis: { title: { text: xTitle, font: { size: 12 } }, gridcolor: COLOR.grid,
           zeroline: false, linecolor: COLOR.grid },
  yaxis: { title: { text: yTitle, font: { size: 12 } }, gridcolor: COLOR.grid,
           zeroline: false, linecolor: COLOR.grid },
});

const CONFIG = { displayModeBar: false, responsive: true };

/** Axis window for a scatter pane: the data, padded.
 *
 *  Fixed rather than autoscaled, so a steep line cannot rescale the plot out
 *  from under the mouse between the first click and the second. */
function axisWindow(pairs) {
  const xs = pairs.map((p) => p[0]);
  const ys = pairs.map((p) => p[1]);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const xPad = (xMax - xMin || 1) * 0.08;
  const yPad = (yMax - yMin || 1) * 0.12;
  return { x: [xMin - xPad, xMax + xPad], y: [yMin - yPad, yMax + yPad] };
}

/** One scatter pane: points, the current line, optional residual whiskers,
 *  and -- on the training pane -- the points clicked to define the line. */
function drawScatter(divId, pairs, color, showResid, anchors = []) {
  const div = $(divId);
  if (!pairs.length) { Plotly.purge(div); return; }

  const xs = pairs.map((p) => p[0]);
  const ys = pairs.map((p) => p[1]);
  // Run the line edge to edge; the fixed axes clip it to the plot.
  const win = axisWindow(pairs);
  const [lo, hi] = win.x;

  const traces = [];

  if (showResid) {
    // One trace holding every whisker, separated by nulls: far fewer objects
    // for Plotly to manage than one trace per point.
    const rx = [];
    const ry = [];
    for (const [x, y] of pairs) {
      rx.push(x, x, null);
      ry.push(y, state.b + state.m * x, null);
    }
    traces.push({
      x: rx, y: ry, mode: 'lines', type: 'scatter', hoverinfo: 'skip',
      line: { color: COLOR.resid, width: 1.4, dash: 'dot' },
    });
  }

  traces.push({
    x: [lo, hi],
    y: [state.b + state.m * lo, state.b + state.m * hi],
    mode: 'lines', type: 'scatter', hoverinfo: 'skip',
    line: { color: COLOR.line, width: 2 },
  });

  traces.push({
    x: xs, y: ys, mode: 'markers', type: 'scatter',
    marker: { color, size: 8, line: { color: '#fff', width: 1 } },
    hovertemplate: `${state.xName}: %{x}<br>${state.yName}: %{y}<extra></extra>`,
  });

  if (anchors.length) {
    // Rings, numbered, so the room can see which two points made the line.
    traces.push({
      x: anchors.map((a) => a[0]), y: anchors.map((a) => a[1]),
      mode: 'markers+text', type: 'scatter', hoverinfo: 'skip',
      cliponaxis: false,                // keep the number visible at the plot's edge
      text: anchors.map((_, i) => String(i + 1)), textposition: 'top center',
      textfont: { color: COLOR.line, size: 12 },
      marker: { size: 15, color: 'rgba(0,0,0,0)', line: { color: COLOR.line, width: 2.5 } },
    });
  }

  const layout = layoutBase(state.xName, state.yName);
  layout.xaxis.range = win.x;
  layout.yaxis.range = win.y;
  layout.xaxis.fixedrange = true;     // no zoom or pan: a click is always a click
  layout.yaxis.fixedrange = true;
  layout.dragmode = false;
  Plotly.react(div, traces, layout, CONFIG);
}

const drawTrain = () => drawScatter('train-plot', state.train, COLOR.train,
  $('show-resid').checked, state.anchors);
const drawTest = () => drawScatter('test-plot', state.test, COLOR.test,
  $('show-resid-test').checked);

/** The MSE surface over (m, b).
 *
 *  Cached, because recomputing 45x45 cells on every redraw is wasteful
 *  -- but keyed on the actual range, because a typed or clicked slope outside
 *  the current window widens it, and a stale grid would leave the marker
 *  floating off the edge of the surface it is supposed to be sitting on. */
function lossSurface() {
  const { mLo, mHi, bLo, bHi } = lossWindow();
  const key = `${mLo},${mHi},${bLo},${bHi},${state.train.length}`;
  if (state.lossGrid && state.lossGrid.key === key) return state.lossGrid;
  const N = 45;
  const ms = Array.from({ length: N }, (_, i) => mLo + ((mHi - mLo) * i) / (N - 1));
  const bs = Array.from({ length: N }, (_, i) => bLo + ((bHi - bLo) * i) / (N - 1));
  // Plotly's z is indexed [row=y][col=x]; here y is b and x is m.
  const z = bs.map((b) => ms.map((m) => mse(state.train, m, b)));
  state.lossGrid = { ms, bs, z, key };
  return state.lossGrid;
}

function drawLoss() {
  const div = $('loss-plot');
  if (!state.train.length) { Plotly.purge(div); return; }
  const { ms, bs, z } = lossSurface();
  const zHere = mse(state.train, state.m, state.b);
  let zMin = Infinity;
  let zMax = -Infinity;
  for (const row of z) for (const v of row) { if (v < zMin) zMin = v; if (v > zMax) zMax = v; }
  const floor = zMin - (zMax - zMin) * 0.06;   // a little air under the bowl

  const traces = [{
    type: 'surface', x: ms, y: bs, z,
    // mist -> periwinkle -> navy: light at the bottom of the bowl, dark up
    // the walls, so depth reads at a glance.
    colorscale: [[0, '#eff1f8'], [0.25, '#b9c8f6'], [0.5, '#5682f3'],
                 [0.78, '#1a2c99'], [1, '#0d1550']],
    opacity: 0.94, showscale: false,
    contours: {
      z: { show: true, usecolormap: true, width: 1,
           project: { z: true }, start: zMin, end: zMax,
           size: Math.max((zMax - zMin) / 14, 1e-9) },
    },
    hovertemplate: 'm: %{x:.4g}<br>b: %{y:.4g}<br>MSE: %{z:.4g}<extra></extra>',
  }, {
    // Guideline: straight down from your line's loss to the floor.
    type: 'scatter3d', mode: 'lines',
    x: [state.m, state.m], y: [state.b, state.b], z: [floor, zHere],
    line: { color: COLOR.train, width: 5 }, hoverinfo: 'skip',
  }, {
    // and out to each wall, so m and b can be read off the floor.
    type: 'scatter3d', mode: 'lines',
    x: [ms[0], state.m, state.m],
    y: [state.b, state.b, bs[0]],
    z: [floor, floor, floor],
    line: { color: COLOR.train, width: 2, dash: 'dot' }, hoverinfo: 'skip',
  }, {
    type: 'scatter3d', mode: 'markers',
    x: [state.m], y: [state.b], z: [zHere],
    marker: { color: COLOR.train, size: 6, line: { color: '#fff', width: 1 } },
    hovertemplate: `your line<br>m: ${fmt(state.m)}<br>b: ${fmt(state.b)}`
                 + `<br>MSE: ${fmt(zHere)}<extra></extra>`,
  }, {
    // the bottom of the bowl
    type: 'scatter3d', mode: 'markers',
    x: [state.fit.m], y: [state.fit.b],
    z: [mse(state.train, state.fit.m, state.fit.b)],
    marker: { color: '#fff', size: 4, symbol: 'circle',
              line: { color: COLOR.navy, width: 2 } },
    hovertemplate: 'least squares<br>m: ' + fmt(state.fit.m)
                 + '<br>b: ' + fmt(state.fit.b) + '<extra></extra>',
  }];

  const axis = (title) => ({
    title: { text: title, font: { size: 10 } },
    gridcolor: COLOR.grid, zerolinecolor: COLOR.grid,
    backgroundcolor: 'rgba(239,241,248,0.55)', showbackground: true,
    tickfont: { size: 9 },
  });

  Plotly.react(div, traces, {
    font: FONT,
    margin: { l: 0, r: 0, t: 0, b: 0 },
    showlegend: false,
    paper_bgcolor: 'rgba(0,0,0,0)',
    scene: {
      xaxis: axis('slope  m'),
      yaxis: axis('intercept  b'),
      zaxis: { ...axis('MSE'), range: [floor, zMax] },
      camera: { eye: { x: 1.55, y: -1.6, z: 0.82 } },
      aspectmode: 'cube',
    },
  }, { displayModeBar: false, responsive: true });
}

/* --------------------------------------------------------------- tables -- */

function drawTable(tableId, pairs) {
  const t = $(tableId);
  const head = `<thead><tr><th>#</th><th>${state.xName}</th>`
             + `<th>${state.yName}</th><th>predicted</th><th>residual</th></tr></thead>`;
  const body = pairs.map(([x, y], i) => {
    const p = state.b + state.m * x;
    const r = y - p;
    const cls = r >= 0 ? 'pos' : 'neg';
    return `<tr><td>${i + 1}</td><td>${fmt(x)}</td><td>${fmt(y)}</td>`
         + `<td>${fmt(p)}</td><td class="resid ${cls}">${r >= 0 ? '+' : ''}${fmt(r)}</td></tr>`;
  }).join('');
  t.innerHTML = `${head}<tbody>${body}</tbody>`;
}

/* ------------------------------------------------------------- readouts -- */

function drawReadouts() {
  const trainMSE = mse(state.train, state.m, state.b);
  const testMSE = mse(state.test, state.m, state.b);
  const bestMSE = mse(state.train, state.fit.m, state.fit.b);

  $('train-mse').innerHTML =
    `<span class="lbl">Training MSE</span><br>`
    + `<span class="big">${fmt(trainMSE)}</span>`
    + `<span class="lbl"> &nbsp;·&nbsp; RMSE ${fmt(Math.sqrt(trainMSE))} ${state.yName}</span>`;

  $('test-mse').innerHTML =
    `<span class="lbl">Test MSE</span><br>`
    + `<span class="big test">${fmt(testMSE)}</span>`
    + `<span class="lbl"> &nbsp;·&nbsp; RMSE ${fmt(Math.sqrt(testMSE))} ${state.yName}</span>`;

  const gap = trainMSE - bestMSE;
  $('loss-readout').innerHTML =
    `<span class="lbl">Your line</span> m = <strong>${fmt(state.m)}</strong>, `
    + `b = <strong>${fmt(state.b)}</strong> &nbsp;→&nbsp; MSE <strong>${fmt(trainMSE)}</strong><br>`
    + `<span class="lbl">The bottom of the bowl is at MSE ${fmt(bestMSE)} — `
    + `you are ${gap <= 1e-9 ? 'there' : `${fmt(gap)} above it`}.</span>`;

  drawPrediction();
}

function drawPrediction() {
  const raw = $('pred-x').value.trim();
  const out = $('pred-out');
  const warn = $('pred-warn');
  const line = `<span class="units">using ${state.yName} = ${fmt(state.b)} `
             + `${state.m < 0 ? '−' : '+'} ${fmt(Math.abs(state.m))} × ${state.xName}</span>`;

  if (raw === '' || !Number.isFinite(Number(raw))) {
    out.innerHTML = `<span class="pred-empty">Enter a value to get an estimate.</span><br>${line}`;
    warn.hidden = true;
    return;
  }
  const x = Number(raw);
  const yhat = state.b + state.m * x;
  out.innerHTML = `<span class="value">${fmt(yhat)}</span>`
    + `<span class="units">estimated ${state.yName} when ${state.xName} = ${fmt(x)}</span>`
    + `<br>${line}`;

  // The model will answer for any number at all. Say when that answer is
  // being made outside anything it ever saw -- this is the Week 1 point.
  const xs = state.rows.map((p) => p[0]);
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  if (x < lo || x > hi) {
    warn.textContent = `Careful: the data only runs from ${fmt(lo)} to ${fmt(hi)}. `
      + 'The model answers anyway — it has no way to tell you it is guessing.';
    warn.hidden = false;
  } else {
    warn.hidden = true;
  }
}

function redrawAll() {
  drawTrain();
  drawTest();
  drawLoss();
  drawTable('train-table', state.train);
  drawTable('test-table', state.test);
  drawReadouts();
}

/* ---------------------------------------------------------- resizing ---- */

/** The panes are flex children in a viewport-height grid, so a plot's box can
 *  change size without the window changing size -- which Plotly's own
 *  `responsive` option does not notice. Watch the containers instead. */
let plotObserver = null;
function observePlots() {
  if (plotObserver || typeof ResizeObserver === 'undefined') return;
  let pending = null;
  plotObserver = new ResizeObserver((entries) => {
    cancelAnimationFrame(pending);
    pending = requestAnimationFrame(() => {
      for (const e of entries) {
        if (e.target.offsetParent !== null) Plotly.Plots.resize(e.target);
      }
    });
  });
  ['train-plot', 'test-plot', 'loss-plot'].forEach((id) => plotObserver.observe($(id)));
}

/* ================================================================ events = */

// ---- tabs
document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('is-active'));
    document.querySelectorAll('.tabpane').forEach((p) => p.classList.remove('is-active'));
    tab.classList.add('is-active');
    document.querySelector(`[data-pane="${tab.dataset.tab}"]`).classList.add('is-active');
  });
});

// ---- load: google sheet (through the server, which handles CORS)
$('load-sheet').addEventListener('click', async () => {
  const url = $('sheet-url').value.trim();
  if (!url) { setStatus('Paste the link to your sheet first.', 'err'); return; }
  setStatus('Reading the sheet…');
  try {
    const res = await fetch(`/api/sheet?url=${encodeURIComponent(url)}`);
    const data = await res.json();
    if (!res.ok) { setStatus(data.error || 'Could not read that sheet.', 'err'); return; }
    loadData({ xName: data.x_name, yName: data.y_name,
               pairs: data.rows, skipped: data.skipped });
  } catch (err) {
    setStatus(`Could not reach the server: ${err.message}`, 'err');
  }
});
$('sheet-url').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') $('load-sheet').click();
});

// ---- load: pasted csv
$('load-paste').addEventListener('click', () => {
  try {
    loadData(readPairs(parseCSV($('paste-csv').value)));
  } catch (err) { setStatus(err.message, 'err'); }
});

// ---- load: file
$('csv-file').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      loadData(readPairs(parseCSV(reader.result)));
    } catch (err) { setStatus(err.message, 'err'); }
  };
  reader.onerror = () => setStatus('Could not read that file.', 'err');
  reader.readAsText(file);
});

// ---- split controls
function resplit() {
  if (!state.rows.length) return;
  applySplit();
  clearAnchors();
  redrawAll();
}
$('train-pct').addEventListener('change', resplit);
$('reshuffle').addEventListener('click', () => {
  $('seed').value = Number($('seed').value || 0) + 1;
  resplit();
});
$('seed').addEventListener('change', resplit);

// ---- the line
/** Set the line. Anything but a click (typing, Reveal best, Reset) clears the
 *  clicked points, since the new line no longer runs through them. */
function setLine(m, b, { fromClicks = false } = {}) {
  state.m = m;
  state.b = b;
  if (!fromClicks) clearAnchors();
  syncInputs();
  redrawAll();
}

const HINT_START = 'Click <strong>two points</strong> on the plot to draw your line '
  + '— or type the slope and intercept.';
const setHint = (html) => { $('click-hint').innerHTML = html; };
const pt = ([x, y]) => `(${fmt(x)}, ${fmt(y)})`;

function clearAnchors() {
  state.anchors = [];
  setHint(HINT_START);
}

/** Where a click landed, in data units -- or null if it missed the plot area.
 *  Within 12 px of a training point it snaps to that point exactly, so a
 *  student can choose two real data points to run the line through. */
function clickToData(div, ev) {
  const fl = div._fullLayout;
  if (!fl || !fl.xaxis || !fl.yaxis) return null;
  const xa = fl.xaxis;
  const ya = fl.yaxis;
  const box = div.getBoundingClientRect();
  const px = ev.clientX - box.left - xa._offset;
  const py = ev.clientY - box.top - ya._offset;
  if (px < 0 || px > xa._length || py < 0 || py > ya._length) return null;
  let best = null;
  let bestD = 12;
  for (const [x, y] of state.train) {
    const d = Math.hypot(xa.l2p(x) - px, ya.l2p(y) - py);
    if (d <= bestD) { best = [x, y]; bestD = d; }
  }
  return best || [xa.p2l(px), ya.p2l(py)];
}

function onTrainClick(p) {
  if (state.anchors.length === 2) state.anchors = [];   // a third click starts over
  state.anchors.push(p);
  if (state.anchors.length === 1) {
    setHint(`Point 1 at <strong>${pt(p)}</strong>. Now click a second point.`);
    drawTrain();
    return;
  }
  const [p1, p2] = state.anchors;
  const line = lineThrough(p1, p2);
  if (!line) {
    state.anchors.pop();
    setHint(p1[1] === p2[1]
      ? 'That is the same point twice. Click a different second point.'
      : 'Those two points are straight above each other — the line through them '
        + 'would be vertical. Click a second point to the left or right.');
    drawTrain();
    return;
  }
  setLine(line.m, line.b, { fromClicks: true });
  setHint(`Line through <strong>${pt(state.anchors[0])}</strong> and `
    + `<strong>${pt(state.anchors[1])}</strong>. Click again to start a new line.`);
}

$('train-plot').addEventListener('click', (ev) => {
  if (!state.train.length) return;
  const p = clickToData($('train-plot'), ev);
  if (p) onTrainClick(p);
});

/** Read a typed number.
 *
 *  These are text inputs, not number inputs: Safari's number spinner rejects
 *  intermediate states and made small decimals awkward to enter. Partial input
 *  ("-", "0.", ".") is left alone rather than treated as an error, so the field
 *  never fights the person typing into it.
 */
function readTyped(el, apply) {
  const raw = el.value.trim();
  if (raw === '' || /^[-+]?\.?$/.test(raw) || /[.]$/.test(raw)) {
    el.classList.remove('bad');
    return;
  }
  const v = Number(raw);
  if (Number.isFinite(v)) { el.classList.remove('bad'); apply(v); }
  else { el.classList.add('bad'); }
}
$('m').addEventListener('input', () => readTyped($('m'), (v) => setLine(v, state.b)));
$('b').addEventListener('input', () => readTyped($('b'), (v) => setLine(state.m, v)));
// Tidy the field up once they are done with it, not while they are in it.
$('m').addEventListener('blur', syncInputs);
$('b').addEventListener('blur', syncInputs);

$('fit').addEventListener('click', () => {
  if (!state.train.length) return;
  setLine(Number(state.fit.m.toFixed(6)), Number(state.fit.b.toFixed(6)));
});
$('reset-line').addEventListener('click', () => {
  if (!state.train.length) return;
  setLine(0, Number(mean(state.train.map((p) => p[1])).toFixed(4)));
});

// ---- residual toggles and prediction
$('show-resid').addEventListener('change', drawTrain);
$('show-resid-test').addEventListener('change', drawTest);
$('pred-x').addEventListener('input', drawPrediction);

// ---- sample data and going back to the loader
$('load-demo').addEventListener('click', async () => {
  setStatus('Loading the Week 1 data…');
  try {
    const res = await fetch('/sample-data.csv');
    loadData(readPairs(parseCSV(await res.text())));
  } catch (err) { setStatus(`Could not load the sample: ${err.message}`, 'err'); }
});
$('change-data').addEventListener('click', () => {
  $('loader').hidden = false;
  $('databar').hidden = true;
  $('workspace').hidden = true;
  document.body.classList.remove('has-data');
  setStatus('');
});
