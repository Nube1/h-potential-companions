import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hpotential.cli import main
from hpotential.config import load_config
from hpotential.data import required_columns
from hpotential.nhats import Layout, build, check, write_synthetic_nhats
from hpotential.nhats.ingest import _outcome_times, score_item

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def layout():
    return Layout.load(ROOT / "configs/nhats_layout.yaml")


@pytest.fixture(scope="module")
def nhats_dir(tmp_path_factory, layout):
    d = tmp_path_factory.mktemp("nhats")
    write_synthetic_nhats(d, layout, n=600, seed=7)
    return d


def test_layout_names_follow_rounds(layout):
    assert layout.name("hc{r}health") == "hc1health"
    assert layout.name("r{k}status", 4) == "r4status"
    assert "w1anfinwgt0" in layout.baseline_variables()


def test_check_passes_on_complete_files(nhats_dir, layout):
    report = check(nhats_dir, layout)
    assert report.ok, report.to_dict()
    assert report.rows["sp"] == 600


def test_check_names_missing_variables(tmp_path, layout, nhats_dir):
    sp = pd.read_stata(nhats_dir / "NHATS_Round_1_SP_File.dta").drop(columns=["hc1health"])
    for f in nhats_dir.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    sp.to_stata(tmp_path / "NHATS_Round_1_SP_File.dta", write_index=False, version=118)
    report = check(tmp_path, layout)
    assert not report.ok
    assert "hc1health" in report.missing_variables["sp"]
    with pytest.raises(ValueError, match="layout check failed"):
        build(tmp_path, layout)


def test_build_meets_the_data_contract(nhats_dir, layout):
    frame, flow = build(nhats_dir, layout)
    cfg = load_config(ROOT / "configs/study.yaml")
    assert required_columns(cfg) <= set(frame.columns)
    assert frame["participant_id"].is_unique
    assert (frame["survey_weight"] > 0).all()
    assert set(frame["mortality_event"].dropna().unique()) <= {0.0, 1.0}
    assert (frame["mortality_time"] > 0).all()
    assert flow["analytic_rows"] == len(frame) <= flow["sp_file_rows"]
    # No NHATS special code survives as a value.
    numeric = frame.select_dtypes("number")
    assert not numeric.isin(layout.special_missing).any().any()


def test_special_codes_become_missing(layout):
    frame = pd.DataFrame({"hc1health": [1, 5, -8, -1]})
    out = score_item(frame, layout, {"kind": "reverse", "source": "hc{r}health", "range": [1, 5]})
    assert out.tolist()[:2] == [5.0, 1.0]
    assert out.iloc[2:].isna().all()


def test_counts_need_half_the_items(layout):
    frame = pd.DataFrame({"a1": [1, 1, -8], "b1": [2, -7, -8], "c1": [1, -7, -9]})
    spec = {"kind": "count_present", "sources": ["a{r}", "b{r}", "c{r}"], "present": [1]}
    out = score_item(frame, layout, spec)
    assert out.iloc[0] == 2.0
    assert np.isnan(out.iloc[1])  # one of three observed
    assert np.isnan(out.iloc[2])


def test_outcome_times_use_interval_midpoints(layout):
    known = pd.DataFrame({2: [True, True, True], 3: [True, True, False]})
    event = pd.DataFrame({2: [False, True, False], 3: [True, False, False]})
    t, e = _outcome_times(known, event, layout)
    d = layout.days_per_round
    assert t.tolist() == [1.5 * d, 0.5 * d, 1.0 * d]
    assert e.tolist() == [1.0, 1.0, 0.0]


def test_demo_chain_draws_every_figure(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "out"
    code = main(
        [
            "demo",
            "--workdir",
            str(tmp_path / "work"),
            "--output",
            str(out),
            "--n",
            "700",
            "--reps",
            "20",
        ]
    )
    assert code == 0
    index = json.loads((out / "figures" / "figure_index.json").read_text())
    names = {e["name"] for e in index}
    assert {
        "bio_04_km_mortality",
        "med_01_profile_by_morbidity",
        "pol_03_income_gap_sources",
    } <= names
    for e in index:
        assert (out / "figures" / f"{e['name']}.pdf").exists()
        assert e["caption"]
    gap = next(e for e in index if e["name"] == "pol_03_income_gap_sources")["numbers"]
    assert gap["total"] == pytest.approx(gap["direct_check"], abs=1e-4)
