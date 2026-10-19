const fs = require('fs'), vm = require('vm'), path = require('path');
const src = fs.readFileSync(path.join(__dirname,'static/regression.js'), 'utf8')
  + '\n;globalThis.__api = { leastSquares, mse, splitData, readPairs, parseCSV, fmt, sd, mean, rng, state, isNum, lossWindow, syncInputs, lineThrough, axisWindow };';

const el = () => new Proxy({ value:'', textContent:'', innerHTML:'', checked:false,
  hidden:false, files:[], dataset:{}, style:{}, offsetParent:null,
  classList:{ add(){}, remove(){}, contains(){return false} },
  addEventListener(){}, scrollIntoView(){}, querySelector(){return el()} },
  { get:(t,k)=> k in t ? t[k] : (()=>{}) , set:(t,k,v)=>{t[k]=v; return true} });

const ctx = {
  document: { getElementById: el, querySelectorAll: () => [], querySelector: el,
              documentElement: {} },
  getComputedStyle: () => ({ getPropertyValue: () => '' }),
  Plotly: { react(){}, purge(){} },
  fetch: async () => ({ ok:false, json: async()=>({}), text: async()=>'' }),
  FileReader: function(){}, console,
  window: { addEventListener(){} },
  ResizeObserver: function(){ this.observe = ()=>{}; },
  requestAnimationFrame: (fn)=>fn(), cancelAnimationFrame: ()=>{},
};
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(src, ctx);
const A = ctx.__api;

let pass = 0, fail = 0;
const near = (a, b, tol=1e-6) => Math.abs(a-b) <= tol;
function check(name, cond, got, want) {
  if (cond) { pass++; console.log(`  ok   ${name}`); }
  else { fail++; console.log(`  FAIL ${name}\n       got  ${got}\n       want ${want}`); }
}

console.log('\n--- least squares against the Week 1 hand-computed values ---');
const TRAIN = [[21,2.4],[26,2.2],[33,3.6],[38,3.1],[45,4.6],[49,4.0],
               [54,5.4],[60,4.9],[67,6.4],[72,5.8],[79,7.2],[86,7.6]];
const TEST  = [[30,3.0],[52,4.4],[64,6.0],[91,8.1]];
const f = A.leastSquares(TRAIN);
check('slope  m = 0.082577', near(f.m, 0.082577, 1e-6), f.m, 0.082577);
check('intercept b = 0.431365', near(f.b, 0.431365, 1e-6), f.b, 0.431365);

// SSE 2.1314 over 12 points -> MSE
const trainMSE = A.mse(TRAIN, f.m, f.b);
check('training MSE = 2.1314/12', near(trainMSE, 2.1314/12, 1e-5), trainMSE, 2.1314/12);
const testMSE = A.mse(TEST, f.m, f.b);
check('test MSE = 0.2184/4', near(testMSE, 0.2184/4, 1e-4), testMSE, 0.2184/4);

console.log('\n--- MSE is minimised at the least-squares point ---');
let worse = 0;
for (const dm of [-0.01,-0.003,0.003,0.01]) for (const db of [-0.3,-0.1,0.1,0.3]) {
  if (A.mse(TRAIN, f.m+dm, f.b+db) > trainMSE) worse++;
}
check('all 16 nearby lines are worse', worse === 16, worse, 16);

console.log('\n--- degenerate inputs do not throw ---');
check('single point', Number.isFinite(A.leastSquares([[1,2]]).b), 'finite', 'finite');
check('identical x (vertical)', A.leastSquares([[5,1],[5,9]]).m === 0, 'm=0', 'm=0');
check('empty MSE is NaN not crash', Number.isNaN(A.mse([], 1, 1)), 'NaN', 'NaN');

console.log('\n--- 80/20 split ---');
const rows = Array.from({length: 16}, (_, i) => [i, i*2]);
const s = A.splitData(rows, 80, 1);
check('12 train / 4 test from 16', s.train.length===12 && s.test.length===4,
      `${s.train.length}/${s.test.length}`, '12/4');
check('no point in both sets',
      !s.train.some(t => s.test.some(u => u[0]===t[0])), 'disjoint', 'disjoint');
check('every point used once', s.train.length + s.test.length === 16,
      s.train.length + s.test.length, 16);
const s2 = A.splitData(rows, 80, 1);
check('same seed -> same split',
      JSON.stringify(s.train)===JSON.stringify(s2.train), 'same', 'same');
const s3 = A.splitData(rows, 80, 2);
check('different seed -> different split',
      JSON.stringify(s.train)!==JSON.stringify(s3.train), 'different', 'different');
const tiny = A.splitData([[1,1],[2,2],[3,3],[4,4]], 100, 1);
check('always holds at least one out', tiny.test.length >= 1, tiny.test.length, '>=1');
const small = A.splitData([[1,1],[2,2],[3,3],[4,4]], 10, 1);
check('always keeps at least two to train', small.train.length >= 2, small.train.length, '>=2');

console.log('\n--- CSV parsing ---');
const csv = fs.readFileSync(path.join(__dirname,'sample-data.csv'),'utf8');
const p = A.readPairs(A.parseCSV(csv));
check('16 rows from sample-data.csv', p.pairs.length===16, p.pairs.length, 16);
check('x name from A1', p.xName==='Battery used (%)', p.xName, 'Battery used (%)');
check('y name from B1', p.yName==='Screen time (hours)', p.yName, 'Screen time (hours)');

const quoted = A.readPairs(A.parseCSV('"Weight, kg","Height, cm"\n1,2\n3,4\n5,6\n7,8\n'));
check('commas inside quotes', quoted.xName==='Weight, kg', quoted.xName, 'Weight, kg');
const crlf = A.readPairs(A.parseCSV('a,b\r\n1,2\r\n3,4\r\n5,6\r\n7,8\r\n'));
check('CRLF line endings', crlf.pairs.length===4, crlf.pairs.length, 4);
const messy = A.readPairs(A.parseCSV('Battery,Screen\n21,2.4\n\n26,oops\n33,3.6\n38,3.1\n45,4.6\n49,4.0\n'));
check('skips non-numeric rows', messy.pairs.length===5 && messy.skipped===1,
      `${messy.pairs.length}/${messy.skipped}`, '5/1');

for (const [bad, why] of [['', 'empty'], ['21,2.4\n26,2.2\n33,3.6\n38,3.1\n', 'numeric header'],
                          ['x\n1\n2\n3\n4\n', 'one column'], ['x,y\n1,2\n3,4\n', 'too few rows']]) {
  let threw = false;
  try { A.readPairs(A.parseCSV(bad)); } catch { threw = true; }
  check(`rejects ${why}`, threw, threw, true);
}

console.log('\n--- number formatting ---');
for (const [v, want] of [[0,'0'], [0.082577,'0.08258'], [2.5,'2.5'], [1234.5678,'1235'],
                         [1200,'1200'], [100,'100'], [-0.0826,'-0.0826'], [0.5,'0.5'],
                         [7,'7'], [0.000123,'0.000123']]) {
  check(`fmt(${v})`, A.fmt(v)===want, A.fmt(v), want);
}
check('fmt(NaN)', A.fmt(NaN)==='—', A.fmt(NaN), '—');


console.log('\n--- loss surface is bowl-shaped, not a canyon ---');
// Equal rise along each axis at the edge of the plotted window is what makes
// the paraboloid read as a bowl rather than a trough.
A.state.train = TRAIN;
A.state.fit = A.leastSquares(TRAIN);
A.state.m = A.state.fit.m; A.state.b = A.state.fit.b;
const R = A.lossWindow();
const z0 = A.mse(TRAIN, A.state.fit.m, A.state.fit.b);
const riseM = A.mse(TRAIN, R.mHi, A.state.fit.b) - z0;
const riseB = A.mse(TRAIN, A.state.fit.m, R.bHi) - z0;
check('rise along m ≈ rise along b', Math.abs(riseM/riseB - 1) < 0.05,
      `ratio ${(riseM/riseB).toFixed(3)}`, 'ratio ≈ 1');
check('window sits near the bottom, not up the walls', riseM < 20 * z0,
      `rise ${riseM.toFixed(2)} vs MSE_min ${z0.toFixed(2)}`, 'modest');
check('least-squares point is inside the window',
      A.state.fit.m > R.mLo && A.state.fit.m < R.mHi
      && A.state.fit.b > R.bLo && A.state.fit.b < R.bHi, 'inside', 'inside');
// typing outside must widen the window, not leave the marker off the surface
A.state.m = 5;
const R2 = A.lossWindow();
check('window widens to contain a typed value', R2.mHi >= 5, R2.mHi, '>= 5');
A.state.m = A.state.fit.m;

console.log('\n--- clicking two points makes the line through them ---');
// Two Week 1 training points: (21, 2.4) and (86, 7.6).
const L = A.lineThrough([21, 2.4], [86, 7.6]);
check('slope = rise/run = 5.2/65 = 0.08', near(L.m, 0.08), L.m, 0.08);
check('intercept = 2.4 - 0.08*21 = 0.72', near(L.b, 0.72), L.b, 0.72);
check('line passes through point 1', near(L.b + L.m * 21, 2.4), L.b + L.m * 21, 2.4);
check('line passes through point 2', near(L.b + L.m * 86, 7.6), L.b + L.m * 86, 7.6);
const R3 = A.lineThrough([86, 7.6], [21, 2.4]);
check('order of the two clicks does not matter', near(R3.m, L.m) && near(R3.b, L.b),
      `${R3.m}, ${R3.b}`, `${L.m}, ${L.b}`);
check('vertical pair has no line', A.lineThrough([40, 1], [40, 5]) === null,
      A.lineThrough([40, 1], [40, 5]), null);
const neg = A.lineThrough([0, 10], [5, 0]);
check('negative slope', near(neg.m, -2) && near(neg.b, 10), `${neg.m}, ${neg.b}`, '-2, 10');
const W = A.axisWindow(TRAIN);
check('axis window contains every training point',
      TRAIN.every(([x, y]) => x > W.x[0] && x < W.x[1] && y > W.y[0] && y < W.y[1]),
      JSON.stringify(W), 'all inside');

console.log('\n--- typing a decimal is not eaten by the redraw ---');
// The old bug: syncInputs() reformatted the focused field on every keystroke,
// so "0.08" became "0","0","0","8". syncInputs must skip the active element.
let mField = { value: '' };
ctx.document.activeElement = mField;
const realGet = ctx.document.getElementById;
ctx.document.getElementById = (id) => (id === 'm' ? mField : realGet(id));
A.state.m = 0.0826; A.state.b = 0.43;
mField.value = '0.';                 // mid-typing state
A.syncInputs();
check('focused field is left alone', mField.value === '0.', mField.value, '0.');
ctx.document.activeElement = null;
A.syncInputs();
check('unfocused field is formatted', mField.value === '0.0826', mField.value, '0.0826');
ctx.document.getElementById = realGet;

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
