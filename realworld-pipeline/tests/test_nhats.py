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
    sp.loc[0, layout.name("sc{r}deatsfdf")] = 12
    sp.to_csv(sp_path, index=False)
    tr.to_csv(tr_path, index=False)
    report = check(tmp_path, layout)
    assert report.ok, report.to_dict()
    assert report.unexpected_codes[layout.name("sc{r}deatsfdf")] == [12.0]
    frame, _ = build(tmp_path, layout)
    assert frame["survey_weight"].notna().mean() > 0.9


# ---- Coding checked against the NHATS documents (thesis appendix on data checks)


def test_activity_codes_follow_the_user_guide(layout):
    """Derived 'self and difficulty' codes: 2 and 4 mean no limitation."""
    spec = layout.raw["indicators"]["bio_function"]
    cols = [layout.name(s) for s in spec["sources"]]
    rows = {
        "no limitation": [2] * 12,
        "dk if alone, no difficulty": [4] * 12,
        "did not do alone": [1] * 12,
        "difficulty": [3] * 12,
        "dk if alone, difficulty": [5] * 12,
        "half unknown": [2] * 6 + [6, 7, 8, 6, 7, 8],
        "mostly unknown": [2] * 5 + [6] * 7,
    }
    frame = pd.DataFrame(rows.values(), columns=cols, index=rows.keys())
    out = score_item(frame, layout, spec)  # activities done without a limitation
    assert out["no limitation"] == 12 and out["dk if alone, no difficulty"] == 12
    assert out["did not do alone"] == 0 and out["difficulty"] == 0
    assert out["dk if alone, difficulty"] == 0
    assert out["half unknown"] == 12  # six answered, all without limitation, prorated
    assert np.isnan(out["mostly unknown"])  # under half answered


def test_skipped_home_items_read_as_no_barrier(layout):
    spec = layout.raw["indicators"]["sys_environment"]
    cols = [layout.name(s) for s in spec["sources"]]  # stairs, levels, same floor, ramp
    frame = pd.DataFrame(
        [[2, -1, -1, 1], [2, 1, -1, 1], [1, 3, 2, 2], [2, -8, -8, -8]], columns=cols
    )
    out = score_item(frame, layout, spec)
    assert out.tolist()[:3] == [0.0, 0.0, 4.0]
    assert np.isnan(out.iloc[3])  # one of four answered


def test_build_uses_round_one_items_income_file_and_sample_rule(tmp_path, layout):
    write_synthetic_nhats(tmp_path, layout, n=800, seed=3)
    frame, flow = build(tmp_path, layout)
    # Decedents and facility-only cases are not in the analytic sample.
    assert flow["with_sp_interview"] < flow["sp_file_rows"]
    # Education and gender for the continuing sample come from Round 1.
    filled = flow["filled_from_earlier_round"]
    assert filled["el5higstschl"] > 0.4 * flow["with_sp_interview"]
    assert filled["r5dgender"] > 0
    assert frame["sys_education"].isna().mean() < 0.1
    assert frame["sex"].isna().mean() < 0.01
    # Income comes from the 20 imputations and is equivalised for couples.
    assert frame["income"].notna().mean() > 0.95
    report = check(tmp_path, layout)
    assert report.files.get("income")
    assert "62" in report.tracker_status_values["r10status"]


def test_income_prefers_twenty_imputations_and_equivalises(layout):
    spec = layout.raw["indicators"]["sys_economic"]
    frame = pd.DataFrame({layout.name(s): [10_000.0, 10_000.0] for s in spec["imputations"]})
    for s in spec["fallback_imputations"]:
        frame[layout.name(s)] = [99.0, 99.0]
    frame[layout.name(spec["couple"]["source"])] = [1, 5]  # married, widowed
    out = score_item(frame, layout, spec)
    assert out.iloc[1] == pytest.approx(-np.log1p(10_000.0))
    assert out.iloc[0] == pytest.approx(-np.log1p(10_000.0 / np.sqrt(2)))
    # Without the income file, the five SP-file imputations are used.
    frame = frame.drop(columns=[layout.name(s) for s in spec["imputations"]])
    assert score_item(frame, layout, spec).iloc[1] == pytest.approx(-np.log1p(99.0))


def test_deaths_without_proxy_count_and_untraced_rounds_do_not_extend_follow_up(tmp_path, layout):
    """86 (deceased, no proxy) is a death; 79 (unable to locate) is not a sign of life."""
    from hpotential.nhats.ingest import _latest_tracker, read_table

    write_synthetic_nhats(tmp_path, layout, n=400, seed=5, fmt=".csv")
    first, _ = build(tmp_path, layout)
    a, b = (int(x) for x in first["participant_id"].iloc[:2])
    _, tpath = _latest_tracker(tmp_path, layout)
    tr = read_table(tpath)
    pid = layout.name(layout.raw["ids"]["person"])
    for k in layout.follow_ups:
        col = layout.name("r{k}status", k)
        tr.loc[tr[pid] == a, col] = 86 if k >= 7 else 60  # dies between Rounds 6 and 7, no proxy
        tr.loc[tr[pid] == b, col] = 60 if k == 6 else 79  # seen in Round 6, then untraced
    tr.to_csv(tpath, index=False)
    frame, _ = build(tmp_path, layout)
    rows = frame.set_index("participant_id")
    d = layout.days_per_round
    assert rows.loc[str(a), "mortality_event"] == 1
    assert rows.loc[str(a), "mortality_time"] == pytest.approx(1.5 * d)
    assert rows.loc[str(b), "mortality_event"] == 0
    assert rows.loc[str(b), "mortality_time"] == pytest.approx(1.0 * d)  # censored at Round 6


def test_income_file_found_under_another_name(tmp_path, layout):
    from hpotential.nhats.ingest import _baseline_files

    write_synthetic_nhats(tmp_path, layout, n=200, seed=9, fmt=".csv")
    stem = layout.name(layout.raw["files"]["income"])
    src = next(p for p in tmp_path.iterdir() if p.stem.lower() == stem)
    src.rename(tmp_path / "NHATS_Round_5_Income_Imputations_v2.csv")
    labels = dict(_baseline_files(tmp_path, layout))
    assert labels["income"].name == "NHATS_Round_5_Income_Imputations_v2.csv"
    _, flow = build(tmp_path, layout)
    assert flow["income_source"].startswith("20 interval-regression")


def test_proxy_flags_reach_the_imputation_model(tmp_path, layout):
    from hpotential.measurement import _aux_frame

    write_synthetic_nhats(tmp_path, layout, n=600, seed=11, fmt=".csv")
    frame, _ = build(tmp_path, layout)
    assert frame["proxy_respondent"].mean() > 0
    # Everyone flagged as not asked is a proxy case, and their cognition is missing.
    flagged = frame["cognition_not_asked"] == 1
    assert flagged.any() and (frame.loc[flagged, "proxy_respondent"] == 1).all()
    assert frame.loc[flagged, "bio_cognition"].isna().all()
    cfg = load_config(ROOT / "configs/study.yaml")
    aux = _aux_frame(frame, cfg)
    assert {"proxy_respondent", "cognition_not_asked", "dementia_reported"} <= set(aux.columns)


def test_not_asked_floor_is_a_sensitivity_rule_only():
    import copy

    from hpotential.config import StudyConfig
    from hpotential.measurement import _indicator_frame

    cfg = load_config(ROOT / "configs/study.yaml")
    cols = [c for d in cfg.indicators.values() for c in d["indicators"]]
    df = pd.DataFrame({c: [1.0, 2.0] for c in cols})
    df["bio_cognition"] = [np.nan, np.nan]
    df["cognition_not_asked"] = [1, 0]
    assert _indicator_frame(df, cfg)["bio_cognition"].isna().all()  # primary: left to imputation
    raw = copy.deepcopy(cfg.raw)
    raw["measurement"]["not_asked_floor"] = {
        "flag": "cognition_not_asked",
        "indicators": ["bio_cognition"],
        "value": 0,
    }
    out = _indicator_frame(df, StudyConfig(raw, cfg.path))["bio_cognition"]
    assert out.iloc[0] == 0 and np.isnan(out.iloc[1])
