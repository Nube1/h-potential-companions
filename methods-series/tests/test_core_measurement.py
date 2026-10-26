"""Unit tests for the measurement model, identification and ML fitting."""

from __future__ import annotations

import numpy as np
import pytest

from hpotential.core import measurement as m


def test_balanced_model_matches_the_closed_form_helper():
    model = m.MeasurementModel.balanced(5, 0.7)
    assert model.determinacy() == pytest.approx(m.second_order_determinacy(5, 0.7))
    assert model.n_indicators == 15
    assert model.n_domains == 3


def test_model_arrays_are_read_only():
    model = m.MeasurementModel.balanced(4, 0.75)
    with pytest.raises(ValueError):
        model.gamma[0] = 0.1


def test_model_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="one entry per domain"):
        m.MeasurementModel(np.zeros((6, 3)), np.array([0.8, 0.7]), np.array([0.3, 0.4]), np.ones(6))


def test_sigma_is_symmetric_positive_definite():
    sigma = m.MeasurementModel.balanced(4, 0.75).sigma
    assert np.allclose(sigma, sigma.T)
    assert np.all(np.linalg.eigvalsh(sigma) > 0)


def test_marker_free_indices_fix_one_loading_per_domain():
    marker, free = m.marker_free_indices(12, 4)
    assert marker == [0, 4, 8]
    assert len(free) == 9
    assert set(marker).isdisjoint(free)


def test_vech_keeps_the_lower_triangle():
    a = np.arange(9.0).reshape(3, 3)
    assert m.vech(a).size == 6


def test_planned_design_is_locally_identified():
    report = m.identification_report(0.75, "planned")
    assert report["n_free"] == 27
    assert report["n_moments"] == 78
    assert report["rank"] == 27
    assert report["identified"] is True
    assert report["condition_number"] < 50


def test_two_domain_control_is_not_identified():
    """The check is only informative if it can detect a model that fails."""
    report = m.identification_report(
        0.75, "two domains", np.array([0.80, 0.70]), np.array([0.36, 0.51]), 4
    )
    assert report["n_free"] == 18
    assert report["rank"] == 17
    assert report["identified"] is False
    assert report["condition_number"] > 1e8


def test_two_indicators_per_domain_is_identified():
    report = m.identification_report(0.75, "two per domain", per_domain=2)
    assert report["rank"] == report["n_free"] == 15
    assert report["identified"] is True


def test_simulation_is_reproducible_from_the_generator():
    model = m.MeasurementModel.balanced(4, 0.75)
    a = model.simulate(200, np.random.default_rng(7))
    b = model.simulate(200, np.random.default_rng(7))
    c = model.simulate(200, np.random.default_rng(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_simulated_covariance_approaches_the_model_covariance():
    model = m.MeasurementModel.balanced(4, 0.75)
    y = model.simulate(200_000, np.random.default_rng(3))
    assert np.max(np.abs(np.cov(y, rowvar=False) - model.sigma)) < 0.02


def test_regression_scores_have_variance_equal_to_squared_determinacy():
    model = m.MeasurementModel.balanced(4, 0.75)
    zeta, _, y = m.simulate_indicators(model, 100_000, np.random.default_rng(11))
    scores = m.regression_scores(y, model.sigma, model.cov_zeta_y)
    rho = model.determinacy()
    assert scores.var() == pytest.approx(rho**2, abs=0.01)
    assert np.corrcoef(scores, zeta)[0, 1] == pytest.approx(rho, abs=0.01)


def test_cfa_recovers_the_population_parameters():
    model = m.MeasurementModel.balanced(4, 0.75)
    y = model.simulate(20_000, np.random.default_rng(5))
    fit = m.fit_cfa(np.cov(y, rowvar=False))
    assert fit.converged and fit.admissible
    assert fit.determinacy() == pytest.approx(model.determinacy(), abs=0.01)
    assert fit.discrepancy < 0.05


def test_cfa_rejects_a_singular_covariance():
    with pytest.raises(ValueError, match="positive definite"):
        m.fit_cfa(np.zeros((12, 12)))


def test_expected_information_is_positive_definite_and_scales_with_n():
    model = m.MeasurementModel.balanced(4, 0.75)
    y = model.simulate(4000, np.random.default_rng(13))
    fit = m.fit_cfa(np.cov(y, rowvar=False))
    _, free = m.marker_free_indices(12, 4)
    info = m.expected_information(fit.packed, 4000, 12, 3, 4, free)
    assert np.all(np.linalg.eigvalsh(info) > 0)
    doubled = m.expected_information(fit.packed, 8000, 12, 3, 4, free)
    assert np.allclose(doubled, 2 * info)


def test_sample_size_replicate_reports_convergence_and_coverage():
    model = m.MeasurementModel.marker_scaled(4, 0.90)
    record = m.sample_size_replicate(600, model, np.random.default_rng(17))
    assert record is not None
    assert record["converged"] == 1
    assert record["admissible"] == 1
    assert len(record["cover"]) == 3
    assert set(record["cover"]) <= {0, 1}
    assert 0.0 < record["rho"] < 1.0


def test_sample_size_replicate_is_reproducible():
    model = m.MeasurementModel.marker_scaled(4, 0.90)
    a = m.sample_size_replicate(400, model, np.random.default_rng(21))
    b = m.sample_size_replicate(400, model, np.random.default_rng(21))
    assert a["gamma"] == b["gamma"]


def test_coverage_is_counted_rather_than_assumed():
    """Nominal 95 per cent coverage is realised, not written in.

    Fifty replicates give a coarse estimate, so the bound is wide. The point is
    that the number moves with the data: a hardcoded 0.95 would pass any bound
    and tell the reader nothing.
    """
    model = m.MeasurementModel.marker_scaled(4, 0.90)
    covers = []
    for rep in range(50):
        record = m.sample_size_replicate(800, model, np.random.default_rng([99, rep]))
        if record is not None and "cover" in record:
            covers.extend(record["cover"])
    realised = float(np.mean(covers))
    assert 0.80 < realised < 1.0
    assert len(set(covers)) == 2  # both outcomes occur; it is not a constant


def test_marker_scaled_model_fixes_the_first_loading_of_each_domain():
    model = m.MeasurementModel.marker_scaled(4, 0.90)
    assert model.lambda_matrix[0, 0] == 1.0
    assert model.lambda_matrix[4, 1] == 1.0
    assert model.lambda_matrix[1, 0] == pytest.approx(0.90)


def test_marker_scaled_fit_recovers_gamma_but_a_balanced_one_does_not():
    """The scale mismatch is real, and worth a test rather than a comment."""
    matched = m.MeasurementModel.marker_scaled(4, 0.90)
    y = matched.simulate(40_000, np.random.default_rng(31))
    fit = m.fit_cfa(np.cov(y, rowvar=False))
    assert np.allclose(fit.gamma, m.GAMMA_PLANNED, atol=0.03)

    mismatched = m.MeasurementModel.balanced(4, 0.75)
    y = mismatched.simulate(40_000, np.random.default_rng(31))
    fit = m.fit_cfa(np.cov(y, rowvar=False))
    assert np.allclose(fit.gamma, 0.75 * m.GAMMA_PLANNED, atol=0.03)


def test_determinacy_curve_carries_the_ceiling_in_its_attributes():
    curve = m.determinacy_curve([4, 8, 40], [0.55, 0.85])
    rho_inf, c = m.determinacy_ceiling()
    assert curve.attrs["ceiling"] == pytest.approx(rho_inf)
    assert curve.attrs["c"] == pytest.approx(c)
    assert curve.to_numpy().max() < rho_inf
