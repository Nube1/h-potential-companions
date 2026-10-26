"""Tests for second-order factor score determinacy and its ceiling."""

from __future__ import annotations

import numpy as np
import pytest

from hpotential.core import measurement as det

# Values from hpotential.analyses.paper1_validation, stored in tests/fixtures/p1_determinacy.json.
C_PLANNED = 3.3586
CEILING = 0.877819
NUMERIC_AT_400 = 0.877594


def test_ceiling_closed_form():
    rho, c = det.determinacy_ceiling()
    assert c == pytest.approx(C_PLANNED, abs=5e-5)
    assert rho == pytest.approx(CEILING, abs=5e-7)


def test_closed_form_matches_the_numerical_limit_to_three_decimals():
    """The stated ceiling result: adding indicators cannot pass sqrt(c/(1+c))."""
    rho_inf, _ = det.determinacy_ceiling()
    numeric = det.second_order_determinacy(400, 0.85)
    assert numeric == pytest.approx(NUMERIC_AT_400, abs=5e-6)
    assert round(numeric, 3) == round(rho_inf, 3) == 0.878
    assert numeric == pytest.approx(rho_inf, abs=5e-4)
    assert numeric < rho_inf  # the limit is approached from below


def test_determinacy_rises_with_indicators_but_stays_under_the_ceiling():
    rho_inf, _ = det.determinacy_ceiling()
    values = [det.second_order_determinacy(m, 0.75) for m in (3, 4, 8, 20, 40, 200)]
    assert all(b > a for a, b in zip(values, values[1:], strict=False))
    assert max(values) < rho_inf


def test_determinacy_rises_with_the_loading_but_stays_under_the_ceiling():
    rho_inf, _ = det.determinacy_ceiling()
    values = [det.second_order_determinacy(8, lam) for lam in (0.55, 0.65, 0.75, 0.85)]
    assert all(b > a for a, b in zip(values, values[1:], strict=False))
    assert max(values) < rho_inf


def test_values_match_the_published_analysis_table():
    # tests/fixtures/p1_determinacy.json, rounded to four decimals there
    assert round(det.second_order_determinacy(4, 0.55), 4) == 0.7700
    assert round(det.second_order_determinacy(8, 0.55), 4) == 0.8182
    assert round(det.second_order_determinacy(40, 0.55), 4) == 0.8647
    assert round(det.second_order_determinacy(4, 0.85), 4) == 0.8563
    assert round(det.second_order_determinacy(40, 0.85), 4) == 0.8756


def test_the_planned_design_never_reaches_the_individual_use_benchmark():
    """No design in the grid reaches 0.90, because the ceiling is below it."""
    curve = det.determinacy_curve(range(3, 41), [0.55, 0.65, 0.75, 0.85])
    assert curve.to_numpy().max() < 0.90
    assert curve.attrs["ceiling"] == pytest.approx(CEILING, abs=5e-7)
    assert det.required_second_order_signal(0.90) == pytest.approx(4.2632, abs=5e-5)
    assert det.required_second_order_signal(0.90) > curve.attrs["c"]


def test_a_stronger_second_order_model_raises_the_ceiling():
    weak, c_weak = det.determinacy_ceiling(np.array([0.5, 0.5, 0.5]), np.array([0.6, 0.6, 0.6]))
    strong, c_strong = det.determinacy_ceiling(np.array([0.9, 0.9, 0.9]), np.array([0.2, 0.2, 0.2]))
    assert c_strong > c_weak
    assert strong > weak
    assert strong == pytest.approx(np.sqrt(c_strong / (1 + c_strong)))


def test_general_form_agrees_with_the_balanced_helper():
    m, lam = 5, 0.7
    gamma, psi = det.GAMMA_PLANNED, det.PSI_PLANNED
    lam_matrix = np.zeros((3 * m, 3))
    for k in range(3):
        lam_matrix[k * m : (k + 1) * m, k] = lam
    theta = np.repeat(1.0 - lam**2 * (gamma**2 + psi), m)
    assert det.rho_determinacy(lam_matrix, gamma, psi, theta) == pytest.approx(
        det.second_order_determinacy(m, lam)
    )


def test_determinacy_is_a_correlation():
    value = det.second_order_determinacy(6, 0.7)
    assert 0.0 <= value <= 1.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"psi": np.array([0.0, 0.5, 0.5])}, "positive"),
        ({"gamma": np.array([0.8, 0.7])}, "one entry per domain"),
    ],
)
def test_rho_determinacy_guards_its_inputs(kwargs, message):
    lam = np.zeros((6, 3))
    for k in range(3):
        lam[k * 2 : (k + 1) * 2, k] = 0.7
    args = {
        "lambda_matrix": lam,
        "gamma": det.GAMMA_PLANNED,
        "psi": det.PSI_PLANNED,
        "theta": np.full(6, 0.5),
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        det.rho_determinacy(**args)


def test_required_signal_rejects_impossible_targets():
    with pytest.raises(ValueError, match="between 0 and 1"):
        det.required_second_order_signal(1.0)
