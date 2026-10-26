"""Check the delivered results and PDFs before release.

Run after ``reproduce.py`` and ``build_papers.py``. Exits non-zero on failure.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]

# Headline numbers quoted in paper 2 (drift scenario, social domain).
EXPECTED_DRIFT = {"coverage95": 0.72, "bias": 0.5006630333845059}


def main() -> int:
    failures: list[str] = []
    manifest = json.loads((ROOT / "results/manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["csv_sha256"].items():
        actual = hashlib.sha256((ROOT / "results" / name).read_bytes()).hexdigest()
        if actual != digest:
            failures.append(f"{name}: hash differs from manifest")

    with (ROOT / "results/measurement.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    drift = next(
        r for r in rows if r["scenario"] == "social_intercept_shift" and r["domain"] == "2"
    )
    for key, value in EXPECTED_DRIFT.items():
        if not math.isclose(float(drift[key]), value, rel_tol=1e-9):
            failures.append(f"drift {key}={drift[key]}, expected {value}")

    for md in sorted((ROOT / "papers").glob("0*.md")):
        pdf, tex = md.with_suffix(".pdf"), md.with_suffix(".tex")
        if not pdf.exists() or not tex.exists():
            failures.append(f"{md.name}: missing .pdf or .tex")
            continue
        reader = PdfReader(pdf)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        for word in ("Abstract", "References"):
            if word not in text:
                failures.append(f"{pdf.name}: no '{word}' section")
        if len(reader.pages) < 4:
            failures.append(f"{pdf.name}: only {len(reader.pages)} pages")
        print(f"{pdf.name}: {len(reader.pages)} pages")

    if failures:
        print("FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    print("CSV hashes, key numeric claims and PDF completeness verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
