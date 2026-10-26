"""Unit tests for the costed access rule and the early-assessment model."""

from __future__ import annotations

import numpy as np
import pytest

from hpotential.core.access import AccessModel, attainable_capability, costed_statistic
from hpotential.core.hta import HTAModel, evpi, evppi, smooth_conditional_mean


# ------------------------------------------------------------------- access
def test_costed_statistic_reduces_to_the_uncosted_ratio():
    one = np.ones(5)
    v = costed_statistic(np.full(5, 0.8), np.full(5, 0.4), one, one, one, one)
    assert np.allclose(v, 2.0)


def test_closed_form_matches_a_direct_evaluation():
    model = AccessModel.low_resource()
    assert model.closed_form_check(np.random.default_rng(20260815), 20_000) < 1e-12


def test_attainable_capability_rises_with_access_and_falls_with_friction():
    kwargs = {
        "life": np.array([18.0]),
        "quality": np.array([0.7]),
        "supply": np.array([0.6]),
        "efficacy": np.array([0.8]),
        "demand": np.array([1.2]),
        "cost_barrier": np.array([0.3]),
        "psi": np.array([0.2]),
    }
    low = attainable_capability(access=np.array([0.3]), **kwargs)
    high = attainable_capability(access=np.array([0.9]), **kwargs)
    blocked = attainable_capability(
        access=np.array([0.9]), **{**kwargs, "cost_barrier": np.array([3.0])}
    )
    assert high > low
    assert blocked < high


def test_psa_is_reproducible_and_the_rules_can_disagree():
    model = AccessModel.high_resource()
    a = model.psa(np.random.default_rng(1), 20_000)
    b = model.psa(np.random.default_rng(1), 20_000)
    assert a == b
    assert 0.3 < a["P_disagreement"] < 0.6


def test_low_resource_setting_leaves_no_room_for_disagreement():
    """Not an approximation: over these ranges T/A and V both exceed one always."""
    model = AccessModel.low_resource()
    record = model.psa(np.random.default_rng(2), 50_000)
    assert record["P_disagreement"] == 0.0
    assert record["P_uncosted_rule_favours_access"] == 1.0
    assert record["P_costed_rule_favours_access"] == 1.0


# ---------------------------------------------------------------------- HTA
def test_smooth_conditional_mean_tracks_a_linear_relation():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 10, 20_000)
    y = 3.0 * x + rng.normal(0, 1.0, 20_000)
    fitted = smooth_conditional_mean(x, y, 60)
    assert np.corrcoef(fitted, 3.0 * x)[0, 1] > 0.99


def test_smooth_conditional_mean_checks_its_shapes():
    with pytest.raises(ValueError, match="same shape"):
        smooth_conditional_mean(np.zeros(5), np.zeros(4))


def test_evpi_is_zero_when_the_decision_never_changes():
    assert evpi(np.full(1000, 500.0)) == pytest.approx(0.0)
    assert evpi(np.full(1000, -500.0)) == pytest.approx(0.0)
    assert evpi(np.array([-100.0, 100.0])) > 0.0


def test_evppi_is_near_zero_for_an_irrelevant_input():
    rng = np.random.default_rng(6)
    inb = rng.normal(0, 1000, 60_000)
    irrelevant = rng.uniform(0, 1, 60_000)
    assert abs(evppi(inb, irrelevant)) < 0.05 * evpi(inb)


def test_evppi_is_bounded_above_by_evpi():
    rng = np.random.default_rng(7)
    driver = rng.normal(0, 1, 60_000)
    inb = 1000 * driver + rng.normal(0, 200, 60_000)
    assert evppi(inb, driver) <= evpi(inb) + 1e-6


def test_hta_model_is_reproducible():
    model = HTAModel()
    a = model.summarise(model.run_scenario(np.random.default_rng(3), "central", 20_000))
    b = model.summarise(model.run_scenario(np.random.default_rng(3), "central", 20_000))
    assert a["mean_INB"] == b["mean_INB"]


def test_hta_scenarios_are_ordered_as_their_names_say():
    model = HTAModel()
    rng = np.random.default_rng(20260815)
    probabilities = {
        name: model.summarise(model.run_scenario(rng, name, 40_000))["P_cost_effective_QALY"]
        for name in ("optimistic", "central", "limited")
    }
    assert probabilities["optimistic"] > probabilities["central"] > probabilities["limited"]


def test_hta_rejects_an_unknown_scenario():
    with pytest.raises(KeyError, match="unknown scenario"):
        HTAModel().sample(np.random.default_rng(0), "nonsense", 10)


def test_bmp_and_qaly_are_not_interchangeable():
    """The paper's central caution, as a number rather than an assertion."""
    model = HTAModel()
    summary = model.summarise(model.run_scenario(np.random.default_rng(5), "central", 60_000))
    ratio = summary["mean_incremental_bmp"] / summary["mean_incremental_qaly"]
    assert ratio > 1.5
    assert summary["P_decision_disagreement"] > 0.02


def test_population_multiplier_discounts_over_the_horizon():
    model = HTAModel()
    undiscounted = HTAModel(discount=0.0)
    assert model.population_multiplier() < undiscounted.population_multiplier()
    assert undiscounted.population_multiplier() == pytest.approx(model.population)
