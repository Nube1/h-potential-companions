"""The CFA engine recovers a known model and behaves sensibly under invariance."""

import numpy as np
import pytest

from hpotential.cfa import fit_cfa, invariance, rubin

PATTERN = {"bio": [0, 1, 2], "soc": [3, 4, 5], "sys": [6, 7, 8]}
LAM = np.array([0.8, 0.7, 0.6, 0.75, 0.65, 0.55, 0.7, 0.6, 0.5])
PHI = np.array([[1.0, 0.5, 0.4], [0.5, 1.0, 0.3], [0.4, 0.3, 1.0]])


def simulate(n, seed, shift=0.0, intercept_drift=0.0, with_eta=False):
    rng = np.random.default_rng(seed)
    eta = rng.multivariate_normal(np.full(3, shift), PHI, n)
    L = np.zeros((9, 3))
    L[np.arange(9), np.repeat([0, 1, 2], 3)] = LAM
    theta = 1 - LAM**2
    x = eta @ L.T + rng.normal(0, np.sqrt(theta), (n, 9))
    x[:, 3] += intercept_drift
    return (x, eta) if with_eta else x


def test_recovers_loadings_and_correlations():
    fit = fit_cfa(simulate(4000, 1), PATTERN)
    std = np.array([e for e, _ in fit.standardised_loadings()])
    assert np.allclose(std, LAM, atol=0.05)
    R = fit.factor_correlations()
    assert R[0, 1] == pytest.approx(0.5, abs=0.06)
    assert fit.df == 24
    assert fit.cfi > 0.99 and fit.rmsea < 0.03
    se = np.array([s for _, s in fit.standardised_loadings()])
    assert np.all((se > 0.003) & (se < 0.05))


def test_second_order_and_determinacy():
    fit = fit_cfa(simulate(4000, 2), PATTERN)
    gam = np.array([e for e, _ in fit.second_order()])
    # phi_12 = g1 g2 etc.: g1 = sqrt(.5*.4/.3)
    assert gam[0] == pytest.approx(np.sqrt(0.5 * 0.4 / 0.3), abs=0.08)
    det = fit.determinacy()
    x, eta = simulate(20000, 3, with_eta=True)
    scores = fit.scores(x)
    for i, name in enumerate(["bio", "soc", "sys"]):
        assert np.corrcoef(scores[:, i], eta[:, i])[0, 1] == pytest.approx(det[name], abs=0.02)
    g = fit.general_scores(x)
    assert 0 < det["general"] <= 1 and np.corrcoef(g, eta.sum(axis=1))[0, 1] > 0.8


def test_reliability_and_htmt():
    fit = fit_cfa(simulate(4000, 4), PATTERN)
    rel = fit.reliability()
    lam = LAM[:3]
    omega = lam.sum() ** 2 / (lam.sum() ** 2 + (1 - lam**2).sum())
    assert rel["bio"]["omega"] == pytest.approx(omega, abs=0.03)
    assert rel["bio"]["ave"] == pytest.approx(np.mean(lam**2), abs=0.04)
    h = fit.htmt()
    assert 0.3 < h[0, 1] < 0.7


def test_invariance_detects_intercept_drift():
    same = invariance([simulate(2000, 5), simulate(2000, 6, shift=0.3)], PATTERN)
    assert abs(same["metric"]["delta_cfi"]) < 0.01
    assert abs(same["scalar"]["delta_cfi"]) < 0.01
    drift = invariance([simulate(2000, 7), simulate(2000, 8, intercept_drift=0.6)], PATTERN)
    assert drift["scalar"]["delta_cfi"] < -0.01


def test_rubin_pooling():
    out = rubin([1.0, 1.2, 0.8], [0.04, 0.04, 0.04])
    assert out["estimate"] == pytest.approx(1.0)
    assert out["se"] == pytest.approx(np.sqrt(0.04 + (4 / 3) * 0.04), rel=1e-9)


def test_rmsea_interval_brackets_estimate():
    fit = fit_cfa(simulate(1500, 9), PATTERN)
    lo, hi = fit.rmsea_ci()
    assert lo <= fit.rmsea <= hi
    assert 0 <= lo < hi < 0.1
    rel = fit.reliability()["bio"]
    assert 0 < rel["alpha"] <= rel["omega"] + 0.02
