"""Head-to-head comparison of the candidate scores, end to end, in Python.

Replaces the R scaffold 03_empirical_comparison.R. It runs the real-world
pipeline (multiple imputation, the joint factor model, every candidate score
and comparator, competing-risk evaluation, bootstrap internal validation) on
the extract named by HPOTENTIAL_DATA, then fills the thesis results chapters.

    HPOTENTIAL_DATA=path/to/extract.csv python analysis/03_empirical_comparison.py

Participant-level outputs are written with RESTRICTED in their names and must
stay on the machine that holds the data.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

THESIS = Path(__file__).resolve().parents[1]
PIPELINE = THESIS.parent / "realworld-pipeline"


def main() -> int:
    data = Path(os.environ.get("HPOTENTIAL_DATA", "data/nhats_analysis.csv")).resolve()
    run_dir = PIPELINE / "outputs" / "thesis"
    subprocess.run([sys.executable, str(THESIS / "analysis" / "00_preflight.py")], check=True)
    cli = [sys.executable, "-m", "hpotential.cli"]
    subprocess.run(
        [*cli, "run", "--data", str(data), "--output", str(run_dir), "--figures"],
        cwd=PIPELINE,
        check=True,
    )
    subprocess.run(
        [*cli, "thesis-export", "--run", str(run_dir), "--thesis", str(THESIS)],
        cwd=PIPELINE,
        check=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
