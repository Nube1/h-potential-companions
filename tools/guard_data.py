#!/usr/bin/env python3
"""Refuse to commit participant data or anything that looks like it.

Used by pre-commit on the files being committed and by CI on every tracked
file. Exits 1 and names each offending file.

Blocked:
* statistical data formats (.dta, .sav, .sas7bdat, .xpt, .rds, .feather,
  .parquet) outside allowed sample folders;
* any file with RESTRICTED, participant_scores or predictions_ in its name;
* anything under a data/raw/ or data/processed/ folder;
* CSV files whose header carries the NHATS person ID (spid);
* CSV files over 2 MB, and any file over 50 MB not stored in Git LFS.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DATA_SUFFIXES = {".dta", ".sav", ".sas7bdat", ".xpt", ".rds", ".feather", ".parquet", ".pq"}
NAME_MARKERS = ("restricted", "participant_scores", "predictions_")
BLOCKED_DIRS = ("data/raw/", "data/processed/")
ALLOWED_DIRS = ("data/sample/", "data/templates/", "tests/fixtures/")
ID_COLUMNS = {"spid"}
MAX_CSV = 2 * 1024 * 1024
MAX_FILE = 50 * 1024 * 1024


def _lfs_files() -> set[str]:
    try:
        out = subprocess.run(["git", "lfs", "ls-files", "-n"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return set()
    return set(out.stdout.split())


def problems(path_text: str, lfs: set[str]) -> list[str]:
    path = Path(path_text)
    posix = path.as_posix()
    low = posix.lower()
    found = []
    allowed = any(d in low for d in ALLOWED_DIRS)
    if path.suffix.lower() in DATA_SUFFIXES and not allowed:
        found.append(f"data file format {path.suffix}")
    if any(m in path.name.lower() for m in NAME_MARKERS):
        found.append("participant-level output by name")
    if any(d in low for d in BLOCKED_DIRS):
        found.append("inside a raw or processed data folder")
    if not path.exists():
        return found
    size = path.stat().st_size
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", errors="replace") as f:
            header = {c.strip().strip('"').lower() for c in f.readline().split(",")}
        if header & ID_COLUMNS and not allowed:
            found.append("CSV carries the NHATS person ID")
        if size > MAX_CSV and not low.startswith("archive/"):
            found.append(f"CSV of {size / 1e6:.1f} MB")
    if size > MAX_FILE and posix not in lfs:
        found.append(f"{size / 1e6:.0f} MB file not in Git LFS")
    return found


def main(argv: list[str]) -> int:
    files = (
        argv
        or subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.splitlines()
    )
    lfs = _lfs_files()
    bad = {f: p for f in files if (p := problems(f, lfs))}
    for f, p in bad.items():
        print(f"BLOCKED {f}: {'; '.join(p)}")
    if bad:
        print(
            "\nParticipant data and RESTRICTED outputs stay on the analysis machine. "
            "If a file is safe, move it to a data/sample/ folder or rename it."
        )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
