"""Unit tests for the Cox partial likelihood and the prognostic pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from hpotential.core.measurement import MeasurementModel
from hpotential.core.prognostic import (
    attenuated_beta,
    cox_fit,
    events_inflation,
    pipeline_replicate,
    required_events,
    simulate_survival,
)


def _survival(n=20_000, beta=-0.12, fraction=0.22, seed=20260815):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    time, event = simulate_survival(beta * z, fraction, rng)
    return (z - z.mean()) / z.std(), time, event


def test_cox_fit_recovers_the_true_coefficient():
    z, time, event = _survival()
    fit = cox_fit(z, time, event)
    assert fit.converged
    assert fit.beta == pytest.approx(-0.12, abs=0.02)
    assert fit.se == pytest.approx(0.015, abs=0.005)
    assert fit.hazard_ratio == pytest.approx(np.exp(fit.beta))


def test_cox_fit_beats_a_difference_of_group_means():
    """The estimator matters, not just the sign.

    A difference of group means, ``mean(z | event) - mean(z | no event)``, is
    not the partial-likelihood estimate. On this simulation at a true
    coefficient of -0.12 it returns -0.1400 where the partial likelihood returns
    -0.1244, so it is further from the truth and biased away from the null.
    """
    z, time, event = _survival()
    fit = cox_fit(z, time, event)
    difference = float(z[event == 1].mean() - z[event == 0].mean())
    assert fit.beta == pytest.approx(-0.1244, abs=1e-3)
    assert difference == pytest.approx(-0.1400, abs=1e-3)
    assert abs(fit.beta - (-0.12)) < abs(difference - (-0.12))


def test_cox_fit_is_invariant_to_the_order_of_the_rows():
    z, time, event = _survival(n=4000)
    rng = np.random.default_rng(1)
    order = rng.permutation(z.size)
    a = cox_fit(z, time, event)
    b = cox_fit(z[order], time[order], event[order])
    assert a.beta == pytest.approx(b.beta, rel=1e-9)
    assert a.se == pytest.approx(b.se, rel=1e-9)


def test_cox_fit_returns_nan_when_nothing_happens():
    fit = cox_fit(np.zeros(10), np.arange(10.0), np.zeros(10, dtype=int))
    assert np.isnan(fit.beta)
    assert not fit.significant


def test_cox_fit_checks_its_input_shapes():
    with pytest.raises(ValueError, match="same shape"):
        cox_fit(np.zeros(5), np.zeros(4), np.zeros(5))


def test_simulate_survival_hits_the_requested_event_fraction():
    rng = np.random.default_rng(2)
    time, event = simulate_survival(np.zeros(50_000), 0.38, rng)
    assert event.mean() == pytest.approx(0.38, abs=0.005)
    assert np.all(time > 0)


def test_simulate_survival_rejects_an_impossible_fraction():
    with pytest.raises(ValueError, match="event_fraction"):
        simulate_survival(np.zeros(10), 0.0, np.random.default_rng(0))


def test_attenuation_prediction_is_the_product():
    assert attenuated_beta(-0.12, 0.8361) == pytest.approx(-0.100332, abs=1e-6)


def test_events_inflation_matches_the_corollary():
    assert events_inflation(0.877819) == pytest.approx(1.2977, abs=1e-4)
    assert events_inflation(0.8361) == pytest.approx(1.4305, abs=1e-4)
    assert events_inflation(1.0) == pytest.approx(1.0)


def test_events_inflation_rejects_an_impossible_determinacy():
    with pytest.raises(ValueError, match="rho_d"):
        events_inflation(0.0)


def test_required_events_scales_as_the_inflation_factor():
    oracle = required_events(-0.12, 0.90, 0.05, 1.0)
    scored = required_events(-0.12, 0.90, 0.05, 0.8361)
    assert oracle == pytest.approx(729.68, abs=0.5)
    assert scored / oracle == pytest.approx(events_inflation(0.8361), rel=1e-9)


def test_pipeline_replicate_returns_both_fits():
    model = MeasurementModel.balanced(4, 0.75)
    record = pipeline_replicate(3000, "mortality", 0.22, model, -0.12, np.random.default_rng(4))
    assert record is not None
    assert record.events > 0
    assert 0.7 < record.rho_hat < 0.95
    assert record.beta_observed > record.beta_oracle  # attenuated toward zero


def test_pipeline_replicate_is_reproducible():
    model = MeasurementModel.balanced(4, 0.75)
    a = pipeline_replicate(1500, "m", 0.22, model, -0.12, np.random.default_rng(9))
    b = pipeline_replicate(1500, "m", 0.22, model, -0.12, np.random.default_rng(9))
    assert a.beta_observed == b.beta_observed
    assert a.beta_oracle == b.beta_oracle
