"""Tests for the graded response model: convention, algebra, recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpotential.core import irt

A_TRUE = [1.4, 1.0, 1.8, 1.2, 1.6, 0.9]
B_TRUE = [
    [-1.2, -0.2, 0.9],
    [-1.6, -0.3, 1.2],
    [-0.9, 0.1, 1.1],
    [-1.4, 0.0, 1.0],
    [-1.1, 0.2, 1.3],
    [-1.8, -0.4, 0.8],
]


# ------------------------------------------------------------- the algebra
def test_category_probabilities_sum_to_one():
    theta = np.linspace(-4, 4, 101)
    for a, b in zip(A_TRUE, B_TRUE, strict=False):
        probs = irt.category_probabilities(theta, a, b)
        assert probs.shape == (theta.size, len(b) + 1)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-12)
        assert (probs >= 0).all()


def test_boundary_conventions_hold():
    """P*_1 = 1 and P*_{K+1} = 0, so the extreme categories are differences."""
    theta = np.array([-2.0, 0.0, 2.0])
    a, b = 1.5, np.array([-1.0, 0.0, 1.0])
    probs = irt.category_probabilities(theta, a, b)
    expected_lowest = 1.0 - 1.0 / (1.0 + np.exp(-a * (theta - b[0])))
    expected_highest = 1.0 / (1.0 + np.exp(-a * (theta - b[-1])))
    assert np.allclose(probs[:, 0], expected_lowest)
    assert np.allclose(probs[:, -1], expected_highest)


def test_k_minus_one_thresholds_for_k_categories():
    probs = irt.category_probabilities(0.0, 1.0, [-1.0, 0.0, 1.0])
    assert probs.shape[1] == 4


def test_unordered_thresholds_are_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        irt.category_probabilities(0.0, 1.0, [0.5, -0.5])
    with pytest.raises(ValueError, match="positive"):
        irt.category_probabilities(0.0, -1.0, [-0.5, 0.5])


def test_probability_of_the_top_category_increases_with_theta():
    theta = np.linspace(-3, 3, 50)
    probs = irt.category_probabilities(theta, 1.3, [-1.0, 0.0, 1.0])
    assert np.all(np.diff(probs[:, -1]) > 0)
    assert np.all(np.diff(probs[:, 0]) < 0)


def test_expected_score_is_monotone_and_bounded():
    theta = np.linspace(-4, 4, 60)
    scores = irt.expected_score(theta, 1.5, [-1.0, 0.0, 1.0])
    assert np.all(np.diff(scores) > 0)
    assert scores.min() > 0.0
    assert scores.max() < 3.0


# ---------------------------------------------------------------- information
def test_item_information_matches_a_numerical_derivative():
    theta = np.array([-1.0, 0.0, 0.75])
    a, b = 1.6, [-1.0, 0.2, 1.1]
    step = 1e-5
    probs = irt.category_probabilities(theta, a, b)
    up = irt.category_probabilities(theta + step, a, b)
    down = irt.category_probabilities(theta - step, a, b)
    derivative = (up - down) / (2 * step)
    numeric = (derivative**2 / probs).sum(axis=1)
    assert np.allclose(irt.item_information(theta, a, b), numeric, rtol=1e-5)


def test_information_peaks_near_the_middle_threshold():
    theta = np.linspace(-4, 4, 401)
    info = irt.item_information(theta, 2.0, [-1.0, 0.0, 1.0])
    assert abs(theta[info.argmax()]) < 0.2


def test_more_discriminating_items_carry_more_information():
    low = irt.item_information(0.0, 0.7, [-1.0, 0.0, 1.0])
    high = irt.item_information(0.0, 2.0, [-1.0, 0.0, 1.0])
    assert high > low


def test_test_information_adds_and_the_standard_error_inverts_it():
    theta = np.array([-1.0, 0.0, 1.0])
    total = irt.test_information(theta, A_TRUE, B_TRUE)
    piecewise = sum(irt.item_information(theta, a, b) for a, b in zip(A_TRUE, B_TRUE, strict=False))
    assert np.allclose(total, piecewise)
    assert np.allclose(irt.standard_error(total), 1.0 / np.sqrt(total))
    assert np.isinf(irt.standard_error(np.array([0.0]))[0])


# ------------------------------------------------------------------ recovery
@pytest.fixture(scope="module")
def fitted():
    responses, theta = irt.simulate_grm(3000, A_TRUE, B_TRUE, seed=20260815)
    model = irt.GradedResponseModel().fit(responses)
    return model, responses, theta


def test_parameters_are_recovered_from_generated_data(fitted):
    model, _, _ = fitted
    assert model.converged
    a_hat = model.a
    assert np.corrcoef(a_hat, A_TRUE)[0, 1] > 0.9
    assert np.abs(a_hat - np.array(A_TRUE)).mean() < 0.2
    b_hat = np.concatenate(model.b)
    b_true = np.concatenate([np.asarray(b) for b in B_TRUE])
    assert np.abs(b_hat - b_true).mean() < 0.12
    assert np.abs(b_hat - b_true).max() < 0.30


def test_estimated_thresholds_stay_ordered(fitted):
    model, _, _ = fitted
    for item in model.items:
        assert np.all(np.diff(item.b) > 0)
        assert item.a > 0
        assert item.n_categories == len(item.b) + 1
    assert model.boundary_items == []


def test_parameter_frame_names_thresholds_from_two(fitted):
    model, _, _ = fitted
    frame = model.parameter_frame()
    assert list(frame.columns) == ["a", "n_categories", "b2", "b3", "b4"]
    assert frame.shape[0] == len(A_TRUE)


def test_eap_scores_track_the_generating_trait(fitted):
    model, responses, theta = fitted
    scores = model.score(responses)
    assert np.corrcoef(scores["theta_eap"], theta)[0, 1] > 0.85
    assert (scores["se_eap"] > 0).all()
    # EAP scores shrink towards the prior mean
    assert scores["theta_eap"].std() < theta.std()


def test_the_model_likelihood_beats_a_less_discriminating_fit(fitted):
    model, responses, _ = fitted
    worse = irt.GradedResponseModel(max_em_iter=1).fit(responses)
    assert model.loglik > worse.loglik


def test_fit_accepts_a_frame_and_keeps_item_names():
    responses, _ = irt.simulate_grm(400, A_TRUE[:3], B_TRUE[:3], seed=5)
    frame = pd.DataFrame(responses, columns=["alpha", "beta", "gamma"])
    model = irt.GradedResponseModel(max_em_iter=60).fit(frame)
    assert [it.name for it in model.items] == ["alpha", "beta", "gamma"]


def test_missing_responses_are_skipped():
    responses, _ = irt.simulate_grm(600, A_TRUE[:4], B_TRUE[:4], seed=6)
    holed = responses.astype(float)
    holed[:50, 0] = np.nan
    model = irt.GradedResponseModel(max_em_iter=80).fit(holed)
    assert np.isfinite(model.loglik)
    assert model.a[0] > 0


def test_empty_categories_are_refused():
    responses, _ = irt.simulate_grm(300, A_TRUE[:2], B_TRUE[:2], seed=7)
    responses[responses[:, 0] == 1, 0] = 0  # category 1 now unused
    with pytest.raises(ValueError, match="0..K-1"):
        irt.GradedResponseModel().fit(responses)


def test_scoring_before_fitting_raises():
    with pytest.raises(RuntimeError, match="fit the model"):
        irt.GradedResponseModel().score(np.zeros((5, 3)))
