"""Shared fixtures: the published results and a seeded generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

#: Directory holding the JSON written by the published analysis scripts.
FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def published() -> dict[str, dict[str, Any]]:
    """Every published result file, keyed by stem."""
    return {p.stem: json.loads(p.read_text()) for p in sorted(FIXTURE_DIR.glob("*.json"))}


@pytest.fixture
def rng() -> np.random.Generator:
    """A generator with a fixed seed, so any test using it is reproducible."""
    return np.random.default_rng(20260815)
