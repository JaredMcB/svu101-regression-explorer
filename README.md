# Regression Explorer

An in-class applet for **SVU 101: AI Literacy, Week 1**. Students put paired
data in a Google Sheet; the applet splits it 80/20, and they hunt for the line
by hand while watching the loss surface underneath them.

Six panes, two rows:

| | Training row | Test row |
|---|---|---|
| **1** | the training data, with residuals | the held-out data, with residuals |
| **2** | scatter + your line, `m` and `b` fields, training MSE | the *same* line meeting new points, test MSE |
| **3** | the 3-D loss surface, with a marker on your line | a prediction maker |

---

## Running it locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py          # http://127.0.0.1:5001
```

Port 5001 rather than 5000, because on macOS the **AirPlay Receiver** listens
on 5000 and the clash produces a misleading "Address already in use". Set
`PORT` to use something else. Render supplies its own port, so this only
affects local runs.

## Deploying to Render

Same shape as ClassBot. Push to a repo, then in Render: **New → Blueprint**,
point it at the repo, and `render.yaml` does the rest.

No database, no API keys, no environment variables. The free plan is fine —
a cold start costs the first person of the day a few seconds. Bump to
`starter` in `render.yaml` before a class if that bothers you.

---

## The Google Sheet

Two columns. **Row 1 holds the variable names** — whatever you type there is
what the axes get labelled.

| | A | B |
|---|---|---|
| **1** | `Battery used (%)` | `Screen time (hours)` |
| **2** | 21 | 2.4 |
| **3** | 26 | 2.2 |

Column A is the predictor, column B is the response. Data starts in row 2.

**The sheet must be shared.** In Google Sheets: **Share → General access →
Anyone with the link → Viewer**. Without that, Google hands back a sign-in page
instead of data — the applet detects this and says so in plain language rather
than failing silently.

Rows that are not two numbers are skipped and counted, so a blank line or a
note in the margin will not break anything. If row 1 is numeric the applet
stops and tells you, rather than quietly eating a data point as a header.

Students can also **paste CSV** or **upload a file** — both are read entirely
in the browser and never reach the server.

---

## How it is put together

**All the statistics and plotting happen in the browser** (`static/regression.js`)
so that dragging the slope redraws instantly. The Flask app does exactly two
things: serves the page, and proxies Google Sheets.

The proxy exists because a browser cannot fetch a sheet directly — Google
serves no CORS headers — and because it keeps URL handling in one auditable
place. Note that **the server never fetches the URL the user pasted.** It pulls
the document id and tab id out of it and builds a `docs.google.com` URL itself,
so a crafted link cannot point the server at an internal address.

```
app.py                 Flask: /, /api/sheet, /healthz
static/regression.js   all the maths and every plot
static/style.css       the SVU 101 deck palette and type
templates/index.html   the six panes
sample-data.csv        the Week 1 screen-time data
```

Plotly and KaTeX come from their CDNs; nothing else is vendored.

### Look and feel

Styled to the **SVU 101 slide deck**, not to ClassBot: navy `#1A2C99` as the
dominant colour, the deck's 45° crimson-to-blue gradient across the top bar,
periwinkle `#5682F3` as the accent, mist `#EFF1F8` ground, and the same
Space Grotesk / DM Sans pairing. Training data is crimson, test data navy.

### One screen

Once data is loaded the loader collapses to a one-line strip and both rows fit
in the viewport without scrolling — the point of the applet is seeing the
training and test halves *at the same time*. Below 700px of window height it
falls back to ordinary page scrolling, because forcing two rows into a short
window squashes the plots into uselessness.

### Why the loss surface looks like a bowl

MSE is far more sensitive to the slope than to the intercept — a step in `m`
is levered by `x`, a step in `b` is not. Scaling both axes by the same number
of standard deviations therefore draws a canyon, not a bowl.

Instead the window is sized by *curvature*: pick a target rise in loss and ask
how far you must travel along each axis to climb it. For this quadratic
surface that is `sqrt(rise / mean(x²))` in `m` and `sqrt(rise)` in `b`. With
the rise set to `var(y)`, both directions climb equally far and the view sits
near the bottom of the paraboloid rather than up its walls. There is a test
asserting the two rises stay within 5% of each other.

The white ringed marker is the least-squares minimum; the crimson marker with
the dropped guidelines is wherever the student's line currently sits.

---

## Teaching notes

**The split is seeded.** Everyone on the same seed sees the same training and
test sets, which matters when you are walking the room through one picture.
*Reshuffle* bumps the seed by one. 80% of 16 rows floors to 12 training and 4
test — deliberately, so the Week 1 lesson guide's numbers match what is on
screen.

**Start flat.** The line opens at `m = 0`, `b = ȳ` — the honest "no
relationship" baseline. Its loss is the variance of `y`, which is a nice thing
to notice out loud before anyone starts guessing.

**Hold off on "Reveal the best line."** It is the Week 1 Step 4 moment. Let the
room guess, collect slopes on the poll, *then* press it. The readout under the
surface always shows how far above the bottom of the bowl the current line
sits, so students can tell they are close without being told the answer.

**The prediction pane is where Week 1's real lesson lands.** Type a predictor
value outside the range of the data and the applet still answers — with a
warning that it is extrapolating. That warning is a thing *we* added. The model
has no idea. This is the concrete version of "a model does not know what it
does not know," and it sets up the Week 2 `pierogi` demonstration.

**Test MSE will usually beat training MSE on small class data sets.** That
surprises people. It is a sample-size artefact, not evidence the model is
better on new data — worth saying so if a student notices, because noticing is
the good outcome.

**The formulas are rendered with KaTeX and left unfilled** — the general
`MSE = (1/n) Σ (yᵢ − ŷᵢ)²` and `ŷᵢ = b + m xᵢ`, so students connect the symbols
to the numbers moving on screen rather than reading an answer.

**Sample data** loads the Week 1 screen-time set with one button, for when you
want the picture up before anyone has built a sheet.

---

## Tests

```bash
node test_regression.js
```

42 checks. The numerical core is verified against the values worked out by
hand for the Week 1 lesson guide (`m = 0.082577`, `b = 0.431365`, training SSE
`2.1314`), plus split invariants, CSV edge cases, number formatting, the
bowl-shape property of the loss window, and the rule that `syncInputs()` must
never rewrite the field being typed into.

That last one is a real bug that shipped: reformatting on every keystroke ate
the decimal point, so typing `0.08` produced `0`, `0`, `0`, `8` — which looked
exactly like the field rejecting floats. The `m` and `b` boxes are now plain
text inputs that tolerate partial entry (`-`, `0.`, `.`) and only tidy
themselves up on blur.

If a change does not show up in the browser, Safari is caching: **Cmd-Option-R**
for a hard reload.
