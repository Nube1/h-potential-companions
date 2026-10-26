"""Tests for classical test theory and exploratory factor analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpotential.core import ctt


def _one_factor_data(n: int = 600, p: int = 8, loading: float = 0.7, seed: int = 0):
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n)
    resid = rng.normal(size=(n, p)) * np.sqrt(1 - loading**2)
    values = loading * theta[:, None] + resid
    return pd.DataFrame(values, columns=[f"i{j + 1}" for j in range(p)])


def _three_factor_data(n: int = 800, per: int = 3, loading: float = 0.75, seed: int = 3):
    rng = np.random.default_rng(seed)
    phi = np.array([[1.0, 0.4, 0.3], [0.4, 1.0, 0.35], [0.3, 0.35, 1.0]])
    factors = rng.normal(size=(n, 3)) @ np.linalg.cholesky(phi).T
    lam = np.zeros((3 * per, 3))
    for k in range(3):
        lam[k * per : (k + 1) * per, k] = loading
    values = factors @ lam.T + rng.normal(size=(n, 3 * per)) * np.sqrt(1 - loading**2)
    names = [f"d{k + 1}_{j + 1}" for k in range(3) for j in range(per)]
    return pd.DataFrame(values, columns=names), phi


# -------------------------------------------------------------------- alpha
def test_cronbach_alpha_runs_as_a_plain_function():
    """The draft defined this as a method without ``self``, so it could not run."""
    data = _one_factor_data()
    alpha = ctt.cronbach_alpha(data)
    assert 0.75 < alpha < 0.95


def test_alpha_matches_the_formula_written_out():
    data = _one_factor_data(n=200, p=5, seed=11)
    values = data.to_numpy()
    k = values.shape[1]
    expected = k / (k - 1) * (1 - values.var(axis=0, ddof=1).sum() / values.sum(axis=1).var(ddof=1))
    assert ctt.cronbach_alpha(data) == pytest.approx(expected)


def test_alpha_grows_with_the_number_of_items_at_fixed_loading():
    """Why alpha is not evidence of unidimensionality: it tracks test length."""
    short = ctt.cronbach_alpha(_one_factor_data(p=4, seed=5))
    long = ctt.cronbach_alpha(_one_factor_data(p=16, seed=5))
    assert long > short + 0.1


def test_alpha_is_high_for_two_correlated_factors():
    """A two-factor set still returns a comfortable alpha."""
    data, _ = _three_factor_data(per=5)
    assert ctt.cronbach_alpha(data) > 0.8


def test_alpha_needs_variance_in_the_total():
    constant = pd.DataFrame(np.ones((10, 3)))
    with pytest.raises(ValueError, match="zero variance"):
        ctt.cronbach_alpha(constant)


# -------------------------------------------------------------------- omega
def test_omega_recovers_a_congeneric_reliability():
    """With equal loadings, omega and alpha agree; the model value is known."""
    loading, p = 0.7, 8
    data = _one_factor_data(n=4000, p=p, loading=loading, seed=2)
    expected = (p * loading) ** 2 / ((p * loading) ** 2 + p * (1 - loading**2))
    assert ctt.mcdonalds_omega(data) == pytest.approx(expected, abs=0.02)
    assert ctt.mcdonalds_omega(data) == pytest.approx(ctt.cronbach_alpha(data), abs=0.02)


def test_omega_from_a_correlation_matrix_matches_omega_from_data():
    data = _one_factor_data(n=1000, seed=4)
    corr = np.corrcoef(data.to_numpy(), rowvar=False)
    assert ctt.mcdonalds_omega(corr=corr) == pytest.approx(ctt.mcdonalds_omega(data), abs=1e-6)


# ------------------------------------------------------------- factorability
def test_kmo_and_bartlett_on_structured_and_unstructured_data():
    structured = _one_factor_data(n=500, p=8, seed=6)
    assert ctt.kmo(structured)["kmo_total"] > 0.8
    bart = ctt.bartlett_sphericity(structured)
    assert bart["df"] == 28
    assert bart["p_value"] < 1e-6

    rng = np.random.default_rng(7)
    noise = pd.DataFrame(rng.normal(size=(500, 8)))
    assert ctt.kmo(noise)["kmo_total"] < 0.6
    assert ctt.bartlett_sphericity(noise)["p_value"] > 0.01


def test_item_total_statistics_uses_the_rest_score():
    data = _one_factor_data(n=400, p=6, seed=8)
    stats = ctt.item_total_statistics(data)
    assert list(stats.index) == list(data.columns)
    assert (stats["corrected_item_total_r"] > 0.4).all()
    # the uncorrected correlation with the total is larger for every item
    total = data.sum(axis=1)
    for name in data.columns:
        uncorrected = np.corrcoef(data[name], total)[0, 1]
        assert uncorrected > stats.loc[name, "corrected_item_total_r"]


# ---------------------------------------------------------------------- EFA
def test_efa_promax_recovers_simple_structure_and_factor_correlations():
    data, phi = _three_factor_data()
    result = ctt.efa(data, n_factors=3, rotation="promax")
    loadings = result.loadings.to_numpy()
    # each item loads highest on the factor of the domain that generated it
    assignment = np.abs(loadings).argmax(axis=1)
    for block in range(3):
        block_assignment = assignment[block * 3 : (block + 1) * 3]
        assert len(set(block_assignment)) == 1
    assert len(set(assignment)) == 3
    # salient loadings are large and cross-loadings are small
    salient = np.abs(loadings)[np.arange(9), assignment]
    assert salient.min() > 0.6
    off = np.abs(loadings).sum(axis=1) - salient
    assert off.max() < 0.25
    # the factor correlations come back in the right region
    est = result.factor_correlations.to_numpy()
    order = [assignment[0], assignment[3], assignment[6]]
    reordered = est[np.ix_(order, order)]
    assert np.abs(reordered - phi).max() < 0.15


def test_efa_reproduces_the_correlation_matrix():
    data, _ = _three_factor_data()
    result = ctt.efa(data, n_factors=3, rotation="promax")
    pattern = result.loadings.to_numpy()
    phi = result.factor_correlations.to_numpy()
    implied = pattern @ phi @ pattern.T
    observed = np.corrcoef(data.to_numpy(), rowvar=False)
    off = ~np.eye(observed.shape[0], dtype=bool)
    assert np.abs(implied - observed)[off].max() < 0.05


def test_varimax_leaves_factors_uncorrelated():
    data, _ = _three_factor_data()
    result = ctt.efa(data, n_factors=3, rotation="varimax")
    assert np.allclose(result.factor_correlations.to_numpy(), np.eye(3))


def test_efa_rejects_impossible_factor_counts():
    data = _one_factor_data(p=4)
    with pytest.raises(ValueError, match="n_factors"):
        ctt.efa(data, n_factors=4)


def test_factor_analyzer_backend_agrees_when_it_is_installed():
    factor_analyzer = pytest.importorskip("factor_analyzer")
    del factor_analyzer
    data, _ = _three_factor_data()
    try:
        other = ctt.efa(data, n_factors=3, rotation="promax", backend="factor_analyzer")
    except RuntimeError as exc:  # incompatible optional dependency versions
        pytest.skip(str(exc))
    ours = ctt.efa(data, n_factors=3, rotation="promax")
    assert np.abs(ours.loadings.to_numpy()).max(axis=1).min() == pytest.approx(
        np.abs(other.loadings.to_numpy()).max(axis=1).min(), abs=0.1
    )


# ------------------------------------------------------------------- wrapper
def test_classical_item_analysis_summary():
    data = _one_factor_data(n=300, p=6, seed=9)
    analysis = ctt.ClassicalItemAnalysis(data)
    summary = analysis.summary()
    assert summary["n_obs"] == 300
    assert summary["n_items"] == 6
    assert 0.0 < summary["alpha"] < 1.0
    assert 0.0 < summary["omega"] < 1.0
    assert summary["bartlett_p"] < 0.01
    assert analysis.efa(n_factors=1).loadings.shape == (6, 1)
