from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class StudyConfig:
    raw: dict
    path: Path

    @property
    def seed(self):
        return int(self.raw["study"]["seed"])

    @property
    def horizons(self):
        return [float(x) for x in self.raw["outcomes"]["horizons_days"]]

    @property
    def indicators(self):
        return self.raw["measurement"]["domains"]

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()


def load_config(path: str | Path) -> StudyConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("study", "data", "measurement", "outcomes", "validation"):
        if key not in raw:
            raise ValueError(f"Configuration is missing section: {key}")
    if not 0 < float(raw["measurement"]["alpha"]) < 1:
        raise ValueError("measurement.alpha must be strictly between 0 and 1")
    return StudyConfig(raw, path)
