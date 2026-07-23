"""Download the APS Failure dataset from the UCI repository.

Usage: python scripts/fetch_data.py
Writes data/raw/aps_failure_training_set.csv and aps_failure_test_set.csv.

The UCI zip contains the two csv files with a 20-line license/header
preamble; this script strips the preamble so the files start at the
column header row, matching what src/fleet_copilot/data.py expects.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

URL = "https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def strip_preamble(text: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("class,"):
            return "\n".join(lines[i:]) + "\n"
    raise ValueError("no header row found")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    print("downloading", URL)
    blob = urlopen(URL).read()
    zf = zipfile.ZipFile(io.BytesIO(blob))
    for name in zf.namelist():
        base = Path(name).name
        if base.endswith(".csv"):
            text = zf.read(name).decode("utf-8", errors="replace")
            out = RAW / base
            out.write_text(strip_preamble(text))
            print("wrote", out, f"({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
