"""
app.py -- Flask web layer for the Regression Explorer.

An in-class applet for SVU 101: AI Literacy, Week 1. Students put paired data
in a Google Sheet, the applet splits it 80/20, and they hunt for the line by
hand while watching the loss surface underneath them.

All of the statistics and plotting happen in the browser (static/regression.js)
so that dragging the slope is instant. The server does exactly two things:

  1. serves the page, and
  2. proxies Google Sheets, because the browser cannot fetch a sheet directly
     (no CORS headers) and because doing it here keeps the URL parsing in one
     auditable place.

Routes:
  GET  /                  the applet
  GET  /sample-data.csv   the Week 1 screen-time data, for the demo button
  GET  /api/sheet         ?url=<google sheet url> -> {x_name, y_name, rows}
  GET  /healthz           liveness probe for Render
"""

import csv
import io
import os
import re
import urllib.error
import urllib.request

from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__)

# A Google Sheets document id, as it appears in /spreadsheets/d/<id>/edit
_SHEET_ID = re.compile(r"/spreadsheets/d/([a-zA-Z0-9_-]{20,})")
# The tab within the document, as ?gid=<n> or #gid=<n>
_SHEET_GID = re.compile(r"[#&?]gid=([0-9]+)")

FETCH_TIMEOUT = 15          # seconds
MAX_BYTES = 2 * 1024 * 1024  # 2 MB is far more data than a class will paste


class SheetError(Exception):
    """Something went wrong that the user can act on."""


def sheet_csv_url(user_url: str) -> str:
    """Build Google's CSV export URL from a pasted Sheets link.

    Note that we never fetch the URL the user gave us. We pull the document id
    and tab id out of it and construct a docs.google.com URL ourselves, so a
    pasted link cannot point this server at an internal address.
    """
    doc = _SHEET_ID.search(user_url or "")
    if not doc:
        raise SheetError(
            "That does not look like a Google Sheets link. It should look "
            "like https://docs.google.com/spreadsheets/d/…/edit"
        )
    gid = _SHEET_GID.search(user_url)
    gid = gid.group(1) if gid else "0"
    return (f"https://docs.google.com/spreadsheets/d/{doc.group(1)}"
            f"/export?format=csv&gid={gid}")


def fetch_csv(url: str) -> str:
    """Fetch the export URL and hand back CSV text."""
    req = urllib.request.Request(url, headers={"User-Agent": "SVU101-Regression-Explorer"})
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
            final = resp.geturl()
            # A sheet that is not shared redirects to a Google sign-in page,
            # which arrives as a 200 full of HTML rather than as an error.
            if "accounts.google.com" in final or "ServiceLogin" in final:
                raise SheetError(
                    "That sheet is private. In Google Sheets choose "
                    "Share → General access → Anyone with the link "
                    "→ Viewer, then paste the link again."
                )
            ctype = resp.headers.get("Content-Type", "")
            body = resp.read(MAX_BYTES)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise SheetError(
                "Google refused access to that sheet. Set Share → "
                "Anyone with the link → Viewer and try again."
            ) from exc
        if exc.code == 404:
            raise SheetError("No sheet found at that link.") from exc
        raise SheetError(f"Google returned an error ({exc.code}).") from exc
    except urllib.error.URLError as exc:
        raise SheetError(f"Could not reach Google Sheets: {exc.reason}") from exc

    if "text/html" in ctype:
        raise SheetError(
            "Google sent back a web page instead of data, which almost always "
            "means the sheet is not shared. Set Share → Anyone with the "
            "link → Viewer."
        )
    return body.decode("utf-8-sig", errors="replace")


def parse_pairs(text: str):
    """Read CSV text into headers plus numeric (x, y) rows.

    The first row names the variables: cell 1 is the independent variable,
    cell 2 is the dependent variable. Whatever the instructor types there is
    what the axes get labelled. Rows that are not two numbers are skipped and
    counted, so a stray blank line or a note in the margin is not fatal.
    """
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if any((c or "").strip() for c in r)]
    if not rows:
        raise SheetError("That sheet is empty.")

    header = rows[0]
    if len(header) < 2:
        raise SheetError(
            "The sheet needs two columns: the predictor in column A and the "
            "response in column B, each with a name in row 1."
        )
    x_name = (header[0] or "").strip() or "x"
    y_name = (header[1] or "").strip() or "y"

    # A header row of two numbers means the names are missing, not that the
    # data starts at row 1 -- say so rather than silently eating a data point.
    def is_number(cell):
        try:
            float((cell or "").strip())
            return True
        except ValueError:
            return False

    if is_number(header[0]) and is_number(header[1]):
        raise SheetError(
            "Row 1 should be the variable names, not data. Put the predictor's "
            "name in A1 and the response's name in B1."
        )

    pairs, skipped = [], 0
    for row in rows[1:]:
        if len(row) < 2:
            skipped += 1
            continue
        try:
            pairs.append([float(row[0].strip()), float(row[1].strip())])
        except ValueError:
            skipped += 1

    if len(pairs) < 4:
        raise SheetError(
            f"Only {len(pairs)} usable rows were found. An 80/20 split needs "
            "at least 4, and really wants 10 or more."
        )
    return {"x_name": x_name, "y_name": y_name, "rows": pairs, "skipped": skipped}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/sample-data.csv")
def sample_data():
    """The Week 1 screen-time data, so the applet has something to show
    without anyone having to build a sheet first."""
    return send_from_directory(app.root_path, "sample-data.csv",
                               mimetype="text/csv")


@app.get("/api/sheet")
def api_sheet():
    try:
        data = parse_pairs(fetch_csv(sheet_csv_url(request.args.get("url", ""))))
    except SheetError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:                      # noqa: BLE001 - last-resort guard
        app.logger.exception("sheet fetch failed")
        return jsonify({"error": "Something went wrong reading that sheet."}), 500
    return jsonify(data)


@app.get("/healthz")
def healthz():
    return "ok", 200


if __name__ == "__main__":
    # 5001, not 5000: on macOS the AirPlay Receiver holds port 5000 by default,
    # which produces a confusing "Address already in use" on a fresh machine.
    app.run(debug=True, port=int(os.getenv("PORT", "5001")))
