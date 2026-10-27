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
    assert layout.name("hc{r}health") == "hc5health"
    assert layout.name("r{k}status", 4) == "r4status"
    assert "w5anfinwgt0" in layout.baseline_variables()


def test_check_passes_on_complete_files(nhats_dir, layout):
    report = check(nhats_dir, layout)
    assert report.ok, report.to_dict()
    assert report.rows["sp"] == 600


def test_check_names_missing_variables(tmp_path, layout, nhats_dir):
    sp = pd.read_stata(nhats_dir / "NHATS_Round_5_SP_File.dta").drop(columns=["sn5dnumsn"])
    for f in nhats_dir.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    sp.to_stata(tmp_path / "NHATS_Round_5_SP_File.dta", write_index=False, version=118)
    report = check(tmp_path, layout)
    assert not report.ok
    assert "sn5dnumsn" in report.missing_variables["sp"]
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
    frame = pd.DataFrame({"hc5health": [1, 5, -8, -1]})
    out = score_item(frame, layout, {"kind": "reverse", "source": "hc{r}health", "range": [1, 5]})
    assert out.tolist()[:2] == [5.0, 1.0]
    assert out.iloc[2:].isna().all()


def test_counts_need_half_the_items(layout):
    frame = pd.DataFrame({"a5": [1, 1, -8], "b5": [2, -7, -8], "c5": [1, -7, -9]})
    spec = {"kind": "count_present", "sources": ["a{r}", "b{r}", "c{r}"], "present": [1]}
    out = score_item(frame, layout, spec)
    assert out.iloc[0] == 2.0
    assert np.isnan(out.iloc[1])  # one of three observed
    assert np.isnan(out.iloc[2])


def test_outcome_times_use_interval_midpoints(layout):
    known = pd.DataFrame({6: [True, True, True], 7: [True, True, False]})
    event = pd.DataFrame({6: [False, True, False], 7: [True, False, False]})
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


def test_baseline_variables_found_in_tracker_and_bad_codes_flagged(tmp_path):
    """NHATS keeps weights and design variables in the tracker file; the build
    must find them there. An answer outside the declared codes is reported."""
    from hpotential.nhats import Layout, build, check, write_synthetic_nhats
    from hpotential.nhats.ingest import _latest_tracker, find_file, read_table

    layout = Layout.load(ROOT / "configs/nhats_layout.yaml")
    write_synthetic_nhats(tmp_path, layout, 400, fmt=".csv")
    pid = layout.name(layout.raw["ids"]["person"])
    sp_path = find_file(tmp_path, layout.name(layout.raw["files"]["sp"]))
    _, tr_path = _latest_tracker(tmp_path, layout)
    sp, tr = read_table(sp_path), read_table(tr_path)
    design = [layout.name(v) for v in layout.raw["design"].values()]
    tr = tr.merge(sp[[pid, *design]], on=pid)
    sp = sp.drop(columns=design)
    sp.loc[0, layout.name("sc{r}deatsfdf")] = 9
    sp.to_csv(sp_path, index=False)
    tr.to_csv(tr_path, index=False)
    report = check(tmp_path, layout)
    assert report.ok, report.to_dict()
    assert report.unexpected_codes[layout.name("sc{r}deatsfdf")] == [9.0]
    frame, _ = build(tmp_path, layout)
    assert frame["survey_weight"].notna().mean() > 0.9
