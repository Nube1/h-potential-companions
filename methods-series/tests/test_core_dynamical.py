"""Unit tests for the solvers, the Fokker-Planck scheme and the Kalman filter."""

from __future__ import annotations

import numpy as np
import pytest

from hpotential.core.dynamical import (
    FokkerPlanck,
    LinearSDE,
    convergence_order,
    euler_maruyama,
    kalman_loglik,
    profile_sigma,
    rk4,
    simulate_state,
    stable_time_step,
    strong_error_study,
    weak_error_study,
)


@pytest.fixture
def sde() -> LinearSDE:
    return LinearSDE(1.2, 0.6, 0.45)


def test_sde_rejects_a_non_reverting_process():
    with pytest.raises(ValueError, match="b must be positive"):
        LinearSDE(1.2, 0.0, 0.45)


def test_exact_solution_starts_and_settles_where_it_should(sde):
    assert sde.exact_solution(0.0, 3.0) == pytest.approx(3.0)
    assert sde.exact_solution(200.0, 3.0) == pytest.approx(sde.mean_reversion_level)


def test_stationary_density_integrates_to_one(sde):
    x = np.linspace(-6, 10, 20_001)
    assert np.trapezoid(sde.stationary_pdf(x), x) == pytest.approx(1.0, abs=1e-9)


def test_stationary_density_is_refused_for_multiplicative_noise():
    with pytest.raises(ValueError, match="additive noise"):
        LinearSDE(multiplicative=True).stationary_pdf(np.zeros(3))


def test_rk4_attains_fourth_order(sde):
    steps = np.array([0.4, 0.2, 0.1, 0.05, 0.025, 0.0125])
    exact = sde.exact_solution(6.0, 3.0)
    errors = np.array([abs(rk4(lambda t, x: sde.drift(x), 3.0, 6.0, h) - exact) for h in steps])
    assert 3.9 < convergence_order(steps, errors) < 4.2
    assert np.all(np.diff(errors) < 0)


def test_rk4_rejects_a_step_larger_than_the_interval(sde):
    with pytest.raises(ValueError, match="smaller than t_end"):
        rk4(lambda t, x: sde.drift(x), 3.0, 1.0, 2.0)


def test_euler_maruyama_needs_a_source_of_noise(sde):
    with pytest.raises(ValueError, match="rng or base_increments"):
        euler_maruyama(sde, 3.0, 6.0, 0.1, 10)


def test_euler_maruyama_rejects_an_incommensurate_reference_grid(sde):
    fine = np.zeros((100, 5))
    with pytest.raises(ValueError, match="divide evenly"):
        euler_maruyama(sde, 3.0, 6.0, 6.0 / 7, 5, base_increments=fine)


def test_euler_maruyama_is_reproducible(sde):
    a = euler_maruyama(sde, 3.0, 6.0, 0.05, 100, rng=np.random.default_rng(1))
    b = euler_maruyama(sde, 3.0, 6.0, 0.05, 100, rng=np.random.default_rng(1))
    assert np.array_equal(a, b)


def test_strong_order_is_one_for_additive_and_one_half_for_multiplicative(sde):
    counts = [75, 150, 300, 600, 1200]
    additive = strong_error_study(sde, 3.0, 6.0, counts, 2000, np.random.default_rng(20260815))
    multiplicative = strong_error_study(
        LinearSDE(1.2, 0.6, 0.45, multiplicative=True),
        3.0,
        6.0,
        counts,
        2000,
        np.random.default_rng(20260816),
    )
    assert 0.90 < additive["strong_order"] < 1.15
    assert 0.40 < multiplicative["strong_order"] < 0.65
    assert additive["strong_order"] > multiplicative["strong_order"]


def test_weak_order_is_one_and_carries_no_monte_carlo_noise(sde):
    a = weak_error_study(sde, 3.0, 6.0, [75, 150, 300, 600, 1200])
    b = weak_error_study(sde, 3.0, 6.0, [75, 150, 300, 600, 1200])
    assert a["weak_errors"] == b["weak_errors"]
    assert 0.95 < a["weak_order"] < 1.05


def test_stable_time_step_honours_both_limits(sde):
    dx = 0.1
    step = stable_time_step(sde, dx, -2.0, 8.0)
    diffusive = 0.4 * dx * dx / (2 * 0.5 * sde.sigma**2)
    advective = 0.4 * dx / max(abs(sde.drift(-2.0)), abs(sde.drift(8.0)))
    assert step == pytest.approx(min(diffusive, advective))


def test_fokker_planck_conserves_mass_without_renormalising(sde):
    """The mass check is only informative if the scheme is left alone."""
    solver = FokkerPlanck(sde)
    _, _, dx = solver.grid(200)
    n_steps = int(np.ceil(5.0 / stable_time_step(sde, dx, -2.0, 8.0)))
    solution = solver.solve(200, n_steps, 5.0)
    assert solution.mass_drift < 1e-14
    assert solution.min_density >= 0.0
    # Every intermediate mass is conserved too, not just the final one.
    assert np.max(np.abs(solution.mass - solution.mass[0])) < 1e-14


def test_fokker_planck_converges_to_the_stationary_density(sde):
    solver = FokkerPlanck(sde)
    widths, errors = [], []
    for nx in (100, 200, 400):
        _, _, dx = solver.grid(nx)
        n_steps = int(np.ceil(25.0 / stable_time_step(sde, dx, -2.0, 8.0)))
        solution = solver.solve(nx, n_steps, 25.0)
        widths.append(dx)
        errors.append(solution.l1_error(sde.stationary_pdf(solution.centres)))
    assert 0.85 < convergence_order(np.array(widths), np.array(errors)) < 1.10
    assert errors[-1] < errors[0]


def test_fokker_planck_refuses_multiplicative_noise():
    with pytest.raises(ValueError, match="state-independent"):
        FokkerPlanck(LinearSDE(multiplicative=True))


def test_kalman_loglik_peaks_near_the_true_diffusion(sde):
    times = np.linspace(0, 6, 25)
    y = simulate_state(sde, times, np.random.default_rng(3), 0.35, 3)
    grid = np.linspace(0.2, 0.9, 15)
    values = [kalman_loglik(y, times, sde.a, sde.b, s, 0.35) for s in grid]
    assert abs(grid[int(np.argmax(values))] - sde.sigma) < 0.15


def test_kalman_loglik_checks_its_shapes(sde):
    with pytest.raises(ValueError, match="one row per observation"):
        kalman_loglik(np.zeros((4, 2)), np.linspace(0, 1, 5), sde.a, sde.b, 0.4, 0.3)


def test_more_indicators_narrow_the_profile_interval(sde):
    """The design, not the optimiser, decides whether the variances separate."""
    times = np.linspace(0, 6, 5)
    grid = np.linspace(0.12, 1.2, 30)
    widths = []
    for n_indicators in (1, 3):
        rng = np.random.default_rng([20260815, n_indicators])
        total = np.zeros_like(grid)
        for _ in range(20):
            y = simulate_state(sde, times, rng, 0.35, n_indicators)
            total += profile_sigma(y, times, sde, grid)
        total -= total.max()
        inside = grid[total >= -1.92]
        widths.append(float(inside.max() - inside.min()))
    assert widths[1] < widths[0]


def test_convergence_order_rejects_a_non_positive_error():
    with pytest.raises(ValueError, match="positive"):
        convergence_order(np.array([0.1, 0.05]), np.array([1.0, 0.0]))
