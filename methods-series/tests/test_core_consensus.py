"""Tests for the Delphi consensus rule, the indices and the concordance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpotential.core.consensus import (
    DelphiConsensusEngine,
    kendalls_w,
    scale_cvi,
    simulate_panel,
    stability,
)


def _frame(rows: list[list[int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[f"item{j + 1}" for j in range(len(rows[0]))])


# ---------------------------------------------------------------- the rule
def test_default_iqr_threshold_is_two():
    assert DelphiConsensusEngine().rule.iqr_threshold == 2.0
    assert DelphiConsensusEngine().rule.median_min == 7.0
    assert DelphiConsensusEngine().rule.cvi_min == 0.78


def test_retention_needs_all_three_criteria():
    engine = DelphiConsensusEngine()
    # 12 panellists. Item 1 passes everything with a tight spread. Item 2 is
    # endorsed as strongly but the endorsement is spread from 7 to 9. Item 3
    # sits below the relevance cut for most of the panel.
    ratings = _frame(
        [
            [8, 9, 5],
            [8, 9, 5],
            [8, 7, 6],
            [8, 9, 6],
            [8, 7, 7],
            [9, 9, 7],
            [8, 7, 5],
            [8, 9, 6],
            [8, 7, 5],
            [8, 9, 6],
            [8, 7, 6],
            [8, 9, 5],
        ]
    )
    stats = engine.item_statistics(ratings)
    assert stats.loc["item1", "decision"] == "RETAIN"
    assert stats.loc["item1", "binding"] == ""
    assert stats.loc["item2", "iqr"] == pytest.approx(2.0)
    assert stats.loc["item2", "decision"] == "RETAIN"
    assert stats.loc["item3", "decision"] == "DROP"
    assert "CVI" in stats.loc["item3", "binding"]
    assert "median" in stats.loc["item3", "binding"]


def test_the_same_item_is_dropped_on_dispersion_at_a_threshold_of_one_point_five():
    """Item 2 above passes at 2.0 and fails at 1.5 on spread alone."""
    ratings = _frame([[9], [9], [7], [9], [7], [9], [7], [9], [7], [9], [7], [9]])
    stats = DelphiConsensusEngine(iqr_threshold=1.5).item_statistics(ratings)
    assert stats.loc["item1", "i_cvi"] == pytest.approx(1.0)
    assert stats.loc["item1", "decision"] == "DROP"
    assert stats.loc["item1", "binding"] == "IQR"
    assert DelphiConsensusEngine().item_statistics(ratings).loc["item1", "decision"] == "RETAIN"


def test_at_a_threshold_of_two_the_dispersion_criterion_never_binds():
    """A consequence of the default worth stating: on a 1 to 9 scale with a
    relevance cut of 7, I-CVI >= 0.78 forces the lower quartile to 7 or more
    and the upper quartile to 9 or less, so the IQR cannot exceed 2. Checked
    here over a large grid of panels; no counterexample exists."""
    rng = np.random.default_rng(0)
    engine = DelphiConsensusEngine()
    passing = 0
    for mu in (7.0, 7.5, 8.0, 8.5):
        for sigma in (0.8, 1.2, 2.0, 3.0):
            for n_panel in (6, 9, 12, 20, 30, 60):
                draws = np.clip(np.rint(rng.normal(mu, sigma, size=(400, n_panel))), 1, 9)
                frame = pd.DataFrame(draws.T, columns=[f"s{i}" for i in range(400)])
                stats = engine.item_statistics(frame)
                keep = stats[stats["ok_cvi"]]
                passing += len(keep)
                assert keep["ok_iqr"].all()
    assert passing > 5000


def test_binding_criterion_is_reported_for_every_dropped_item():
    engine = DelphiConsensusEngine()
    ratings = simulate_panel(30, [8.4, 6.0, 4.5, 8.0], sigma=1.2, seed=3)
    result = engine.analyse(ratings)
    dropped = result.items[result.items["decision"] == "DROP"]
    assert (dropped["binding"].str.len() > 0).all()
    counts = result.binding_counts()
    assert counts.sum() >= len(dropped)


def test_threshold_of_one_point_five_drops_a_well_endorsed_item_more_often():
    """The reason the default is 2.0, reproduced on a small simulation.

    ``hpotential.analyses.paper3_consensus`` reports a false-drop rate of 0.694 at an IQR
    threshold of 1.5 and 0.023 at 2.0, for an item with true endorsement 0.90
    on a panel of 30 with sigma = 1.2. This is the same experiment with fewer
    replications, so it checks the direction and the rough size, not the digit.
    """
    rng = np.random.default_rng(20260815)
    n_sim, n_panel = 600, 30
    draws = np.clip(np.rint(rng.normal(8.04, 1.2, size=(n_sim, n_panel))), 1, 9)
    strict = DelphiConsensusEngine(iqr_threshold=1.5)
    relaxed = DelphiConsensusEngine(iqr_threshold=2.0)

    def false_drop(engine: DelphiConsensusEngine) -> float:
        frame = pd.DataFrame(draws.T, columns=[f"s{i}" for i in range(n_sim)])
        stats = engine.item_statistics(frame)
        return float((stats["decision"] != "RETAIN").mean())

    strict_rate = false_drop(strict)
    relaxed_rate = false_drop(relaxed)
    assert strict_rate > 0.55
    assert relaxed_rate < 0.10
    assert strict_rate - relaxed_rate > 0.4


def test_ratings_outside_the_scale_are_rejected():
    engine = DelphiConsensusEngine()
    with pytest.raises(ValueError, match="outside the scale"):
        engine.item_statistics(_frame([[10, 8], [9, 7], [8, 7]]))


# ------------------------------------------------------------------- indices
def test_scale_cvi_average_and_universal_agreement():
    # 4 panellists, 2 items. Item 1: all four relevant. Item 2: three of four.
    ratings = _frame([[8, 8], [9, 9], [7, 6], [8, 7]])
    out = scale_cvi(ratings)
    assert out["s_cvi_ave"] == pytest.approx((1.0 + 0.75) / 2)
    assert out["s_cvi_ua"] == pytest.approx(0.5)
    assert out["n_raters"] == 4


def test_group_cvi_and_priority_flag():
    """An item the patients endorse but the pooled rule drops must be flagged."""
    engine = DelphiConsensusEngine()
    patients = [[9, 5]] * 6
    clinicians = [[4, 8]] * 18 + [[5, 8]] * 6
    ratings = _frame(patients + clinicians)
    groups = ["patient"] * 6 + ["clinician"] * 24
    result = engine.analyse(ratings, groups=groups, priority_group="patient")

    assert result.group_cvi.loc["item1", "patient"] == pytest.approx(1.0)
    assert result.group_cvi.loc["item1", "clinician"] == pytest.approx(0.0)
    assert result.items.loc["item1", "decision"] == "DROP"

    flags = result.priority_flags
    assert list(flags["item"]) == ["item1"]
    assert flags.loc[0, "n_group"] == 6
    assert flags.loc[0, "group_i_cvi"] == pytest.approx(1.0)


def test_priority_flag_is_empty_when_the_pooled_rule_agrees():
    engine = DelphiConsensusEngine()
    ratings = _frame([[8, 8]] * 20)
    groups = ["patient"] * 10 + ["clinician"] * 10
    result = engine.analyse(ratings, groups=groups, priority_group="patient")
    assert result.priority_flags.empty
    assert result.retained == ["item1", "item2"]


# --------------------------------------------------------------- Kendall's W
def test_kendalls_w_known_answer_with_ties():
    """Hand-computed example, worked through in the comments.

    Three raters, four items::

        r1: 9 8 8 5    midranks 4.0 2.5 2.5 1.0    T = 2^3 - 2 = 6
        r2: 9 7 7 6    midranks 4.0 2.5 2.5 1.0    T = 6
        r3: 8 8 6 6    midranks 3.5 3.5 1.5 1.5    T = 6 + 6 = 12

    Rank sums 11.5, 8.5, 6.5, 3.5 with mean 7.5, so
    S = 16 + 1 + 1 + 16 = 34. With M = 3 and J = 4,
    M^2 (J^3 - J) = 540 and M sum T = 3 * 24 = 72, giving
    W = 12 * 34 / (540 - 72) = 408 / 468 = 34 / 39.
    """
    ratings = _frame([[9, 8, 8, 5], [9, 7, 7, 6], [8, 8, 6, 6]])
    out = kendalls_w(ratings)
    assert out["tie_correction"] == pytest.approx(24.0)
    assert out["W"] == pytest.approx(34.0 / 39.0)
    assert out["W_uncorrected"] == pytest.approx(408.0 / 540.0)
    assert out["chi2"] == pytest.approx(3 * 3 * 34.0 / 39.0)
    assert out["df"] == 3


def test_tie_correction_gives_exactly_one_for_perfect_agreement_with_ties():
    """Without the correction, identical rankings with ties score 0.9, not 1."""
    ratings = _frame([[9, 9, 7, 5]] * 3)
    out = kendalls_w(ratings)
    assert out["W"] == pytest.approx(1.0)
    assert out["W_uncorrected"] == pytest.approx(0.9)


def test_kendalls_w_is_one_for_perfect_agreement_without_ties():
    ratings = _frame([[9, 7, 5, 3]] * 4)
    assert kendalls_w(ratings)["W"] == pytest.approx(1.0)


def test_kendalls_w_is_nan_when_every_rater_ties_everything():
    out = kendalls_w(_frame([[7, 7, 7]] * 3))
    assert np.isnan(out["W"])


def test_kendalls_w_needs_two_items():
    with pytest.raises(ValueError, match="two items"):
        kendalls_w(_frame([[7], [8], [9]]))


# ---------------------------------------------------------------- stability
def test_stability_compares_two_rounds_and_reports_attrition():
    round1 = simulate_panel(30, [8.3, 6.0], sigma=1.0, seed=1)
    round2 = simulate_panel(24, [8.3, 6.0], sigma=0.7, seed=2)
    out = stability(round1, round2)
    assert list(out.index) == ["item01", "item02"]
    assert set(out.columns) >= {"i_cvi_r1", "i_cvi_r2", "delta_i_cvi", "stable"}
    assert (out["delta_i_cvi"] >= 0).all()
    assert out.attrs["attrition"] == pytest.approx(1 - 24 / 30)


def test_stability_needs_shared_items():
    a = simulate_panel(10, [8.0], seed=1)
    b = simulate_panel(10, [8.0], seed=2).rename(columns={"item01": "other"})
    with pytest.raises(ValueError, match="share no items"):
        stability(a, b)
