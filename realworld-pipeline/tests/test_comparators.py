"""Charlson weights and the EQ-5D crosswalk score as declared in the layout."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hpotential.nhats import Layout
from hpotential.nhats.ingest import eq5d_levels, score_item

LAYOUT = Layout.load(Path(__file__).resolve().parents[1] / "configs/nhats_layout.yaml")


def frame(values: dict, n=1):
    return pd.DataFrame({LAYOUT.name(k): [v] * n for k, v in values.items()})


def test_charlson_weights():
    spec = LAYOUT.raw["comparators"]["charlson_index"]
    none = {s: 2 for s in spec["weights"]}
    assert score_item(frame(none), LAYOUT, spec)[0] == 0
    cancer_and_stroke = {**none, "hc{r}disescn10": 1, "hc{r}disescn8": 1}
    assert score_item(frame(cancer_and_stroke), LAYOUT, spec)[0] == 3
    every = {s: 1 for s in spec["weights"]}
    assert score_item(frame(every), LAYOUT, spec)[0] == 8


def _eq5d_frame(level: int):
    spec = LAYOUT.raw["comparators"]["qaly_proxy"]
    vals = {}
    for dim, d in spec["dimensions"].items():
        for s in d["sources"]:
            if d.get("sum"):
                vals[s] = {1: 1, 3: 4}[level]  # sum 4 (level 1) or 16 (level 3)
            else:
                vals[s] = d["present"][0] if level == 3 else 99
    return frame(vals), spec


def test_eq5d_best_and_worst_states():
    best, spec = _eq5d_frame(1)
    assert (eq5d_levels(best, LAYOUT, spec).iloc[0] == 1).all()
    assert score_item(best, LAYOUT, spec)[0] == pytest.approx(1.0)
    worst, _ = _eq5d_frame(3)
    assert (eq5d_levels(worst, LAYOUT, spec).iloc[0] == 3).all()
    # The UK TTO tariff's worst state, 33333, is valued at -0.594.
    assert score_item(worst, LAYOUT, spec)[0] == pytest.approx(-0.594, abs=1e-9)


def test_optional_comparators_do_not_block_check():
    names = LAYOUT.optional_variables()
    assert set(names) == {"adl_iadl_count", "qaly_proxy"}
    assert not set(np.concatenate(list(names.values()))) & set(LAYOUT.baseline_variables()) - {
        LAYOUT.name(s) for s in ("pc{r}walk6blks", "pc{r}up20stair")
    } - {LAYOUT.name(f"hc{{r}}depresan{i}") for i in range(1, 5)}
