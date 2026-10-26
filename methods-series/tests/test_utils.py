"""Unit tests for the plotting style, the numerical helpers and the statistics."""

from __future__ import annotations

import numpy as np
import pytest

from hpotential.utils import numerical, plotting, stats


# ------------------------------------------------------------------ plotting
def test_palette_is_ordered_and_wraps():
    assert plotting.series_color(0) == plotting.SERIES[0]
    assert plotting.series_color(len(plotting.SERIES)) == plotting.SERIES[0]
    assert len(set(plotting.SERIES)) == len(plotting.SERIES)


def test_apply_style_sets_the_series_defaults():
    import matplotlib.pyplot as plt

    plotting.apply_style()
    assert plt.rcParams["figure.dpi"] == 200
    assert plt.rcParams["axes.grid"] is True
    assert plt.rcParams["legend.frameon"] is False


def test_save_figure_creates_missing_directories(tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    written = plotting.save_figure(fig, tmp_path / "deep" / "nested" / "figure.png")
    assert written.exists()
    assert written.stat().st_size > 0


# ----------------------------------------------------------------- numerical
def test_convergence_order_recovers_a_known_slope():
    h = np.array([0.4, 0.2, 0.1, 0.05])
    assert numerical.convergence_order(h, 3.0 * h**2) == pytest.approx(2.0)
    assert numerical.convergence_order(h, 0.5 * h**4) == pytest.approx(4.0)


def test_convergence_order_needs_two_points():
    with pytest.raises(ValueError, match="at least two"):
        numerical.convergence_order(np.array([0.1]), np.array([1.0]))


def test_pairwise_orders_agree_with_the_fitted_slope():
    h = np.array([0.4, 0.2, 0.1, 0.05])
    pairs = numerical.pairwise_orders(h, 3.0 * h**2)
    assert np.allclose(pairs, 2.0)


def test_richardson_removes_the_leading_error_term():
    exact = 1.0
    coarse = exact + 0.04
    fine = exact + 0.01
    assert numerical.richardson(coarse, fine, 2.0, 2) == pytest.approx(exact, abs=1e-12)


def test_richardson_rejects_a_ratio_of_one():
    with pytest.raises(ValueError, match="ratio must exceed one"):
        numerical.richardson(1.0, 1.0, 1.0, 2)


def test_safe_cholesky_passes_a_clean_matrix_through():
    a = np.array([[4.0, 1.0], [1.0, 3.0]])
    chol = numerical.safe_cholesky(a)
    assert np.allclose(chol @ chol.T, a)


def test_safe_cholesky_rescues_a_barely_indefinite_matrix():
    a = np.eye(3)
    a[2, 2] = -1e-14
    chol = numerical.safe_cholesky(a)
    assert np.allclose(chol @ chol.T, a, atol=1e-9)


def test_safe_cholesky_refuses_a_genuinely_indefinite_matrix():
    a = np.array([[1.0, 0.0], [0.0, -5.0]])
    with pytest.raises(np.linalg.LinAlgError, match="not close to positive definite"):
        numerical.safe_cholesky(a)


# --------------------------------------------------------------------- stats
def test_credible_interval_is_equal_tailed():
    lower, upper = stats.credible_interval(np.arange(101.0), 0.90)
    assert lower == pytest.approx(5.0)
    assert upper == pytest.approx(95.0)


def test_credible_interval_rejects_an_impossible_level():
    with pytest.raises(ValueError, match="strictly between"):
        stats.credible_interval(np.arange(10.0), 1.0)


def test_binned_conditional_mean_recovers_a_linear_relation():
    rng = np.random.default_rng(11)
    x = rng.uniform(0, 5, 10_000)
    y = 2.0 * x + rng.normal(0, 0.5, 10_000)
    fitted = stats.binned_conditional_mean(x, y, 50)
    assert np.corrcoef(fitted, 2.0 * x)[0, 1] > 0.99


def test_bootstrap_ci_is_reproducible_and_brackets_the_mean():
    values = np.random.default_rng(12).normal(5.0, 1.0, 500)
    a = stats.bootstrap_ci(values, np.random.default_rng(1), n_boot=400)
    b = stats.bootstrap_ci(values, np.random.default_rng(1), n_boot=400)
    assert a == b
    assert a[0] < values.mean() < a[1]


def test_relative_bias_is_zero_for_an_unbiased_estimator():
    truth = np.array([0.8, 0.7, 0.62])
    estimates = np.tile(truth, (100, 1))
    assert np.allclose(stats.relative_bias(estimates, truth), 0.0)


def test_relative_bias_refuses_a_zero_true_value():
    with pytest.raises(ValueError, match="undefined at a true value of zero"):
        stats.relative_bias(np.ones((5, 2)), np.array([1.0, 0.0]))


def test_summarise_reports_the_expected_keys():
    out = stats.summarise(np.random.default_rng(13).normal(0, 1, 1000))
    assert set(out) == {"mean", "sd", "median", "lower", "upper", "n"}
    assert out["n"] == 1000
    assert out["lower"] < out["median"] < out["upper"]
