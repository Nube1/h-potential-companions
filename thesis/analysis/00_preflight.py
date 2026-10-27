"""Check the analysis extract against the data dictionary before any modelling.

Python replacement for 00_preflight.R. Reads the extract named by the
HPOTENTIAL_DATA environment variable (default data/nhats_analysis.csv), checks
that every required column in data_dictionary.csv is present, and records the
package versions used.

    HPOTENTIAL_DATA=path/to/extract.csv python analysis/00_preflight.py
"""

from __future__ import annotations

import csv
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    data_path = Path(os.environ.get("HPOTENTIAL_DATA", "data/nhats_analysis.csv"))
    dictionary_path = HERE / "data_dictionary.csv"
    if not data_path.exists():
        sys.exit(
            f"NHATS analysis extract not found at '{data_path}'. "
            "Set HPOTENTIAL_DATA to the approved de-identified extract."
        )
    with dictionary_path.open(newline="") as fh:
        required = [r["analysis_name"] for r in csv.DictReader(fh) if r["required"] == "yes"]
    with data_path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = sum(1 for _ in reader)
    missing = [c for c in required if c not in header]
    if missing:
        sys.exit("Required columns missing: " + ", ".join(missing))
    out = HERE.parent / "outputs"
    out.mkdir(exist_ok=True)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (out / "session-info.txt").write_text(
        f"python {sys.version}\nplatform {platform.platform()}\n\n{freeze}", encoding="utf-8"
    )
    print(f"Preflight passed: {rows} rows; {len(header)} columns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
