"""WLSMV: the pieces against known answers, and recovery from coarsened data."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from hpotential.wlsmv import bvn_cdf, fit_wlsmv, polychoric, polyserial

LAM = np.array([0.8, 0.7, 0.6, 0.75, 0.65, 0.55, 0.7, 0.6, 0.5])
PHI = np.array([[1.0, 0.5, 0.4], [0.5, 1.0, 0.3], [0.4, 0.3, 1.0]])
PATTERN = {"a": [0, 1, 2], "b": [3, 4, 5], "c": [6, 7, 8]}


def test_bivariate_normal_cdf_matches_scipy():
    for rho in (-0.7, 0.0, 0.3, 0.9):
        mvn = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
        for h, k in ((-1.0, 0.5), (0.2, 0.2), (1.5, -0.3)):
            assert bvn_cdf(h, k, rho) == pytest.approx(mvn.cdf([h, k]), abs=1e-6)
    assert bvn_cdf(np.inf, 0.4, 0.5) == pytest.approx(0.6554, abs=1e-4)


def test_polychoric_and_polyserial_recover_rho():
    rng = np.random.default_rng(1)
    z = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 20000)
    x = np.digitize(z[:, 0], [-1.0, 0.0, 0.8])
    y = np.digitize(z[:, 1], [-0.5, 0.6])
    assert polychoric(x, y) == pytest.approx(0.5, abs=0.03)
    assert polyserial(x.astype(float), z[:, 1]) == pytest.approx(0.5, abs=0.03)
    # Pearson on the coarsened codes understates rho; the polychoric does not.
    assert np.corrcoef(x, y)[0, 1] < 0.46


def test_wlsmv_recovers_loadings_from_ordinal_indicators():
    rng = np.random.default_rng(2)
    n = 3000
    eta = rng.multivariate_normal(np.zeros(3), PHI, n)
    L = np.zeros((9, 3))
    L[np.arange(9), np.repeat([0, 1, 2], 3)] = LAM
    ystar = eta @ L.T + rng.normal(0, np.sqrt(1 - LAM**2), (n, 9))
    X = ystar.copy()
    for j in (1, 2, 4, 5, 7, 8):  # six ordinal, three continuous
        X[:, j] = np.digitize(ystar[:, j], [-0.8, 0.2, 1.0])
    fit = fit_wlsmv(X, PATTERN, reps=60, seed=3)
    assert fit.ordinal.sum() == 6
    assert np.allclose(np.abs(fit.loadings), LAM, atol=0.06)
    assert fit.factor_corr[0, 1] == pytest.approx(0.5, abs=0.07)
    assert fit.df == 24
    assert fit.cfi > 0.98 and fit.rmsea < 0.03
    assert np.all((fit.loadings_se > 0.005) & (fit.loadings_se < 0.06))
