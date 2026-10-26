from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REQUIRED = [
    "pyproject.toml",
    "configs/study.yaml",
    "notebooks/H_Potential_Real_World_Pipeline.ipynb",
    ".github/workflows/ci.yml",
    "docs/DATA_CONTRACT.md",
    "docs/PROTOCOL_AND_GAPS.md",
]


def main():
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("Missing release files: " + ", ".join(missing))
    hashes = {}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts:
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps({"status": "pass", "files": len(hashes), "sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()

