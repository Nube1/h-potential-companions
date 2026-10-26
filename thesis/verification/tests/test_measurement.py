"""Chapter 5: reliability, convergent and discriminant validity, shrinkage."""
import math
import numpy as np
import pytest
from hpv import measurement as m


BIOMEDICAL = [0.82, 0.74, 0.68, 0.55]


def test_definition_composite_reliability_and_ave():
    """Definitions 5.1 and 5.2. Omega uses the loadings as estimated; AVE is
    the mean squared loading, and the two answer different questions."""
    assert m.composite_reliability(BIOMEDICAL) == pytest.approx(0.794, abs=1e-3)
    assert m.average_variance_extracted(BIOMEDICAL) == pytest.approx(0.496,
                                                                     abs=1e-3)
    with pytest.raises(ValueError):
        m.composite_reliability([0.8])


def test_alpha_equals_omega_only_under_equal_loadings():
    """Definition 5.1. Alpha assumes every indicator carries the construct
    equally well. Where they do, the two agree; where they do not, alpha is
    the lower figure."""
    equal = [0.7, 0.7, 0.7, 0.7]
    assert m.cronbach_alpha_from_loadings(equal) == pytest.approx(
        m.composite_reliability(equal), abs=1e-12)
    assert (m.cronbach_alpha_from_loadings(BIOMEDICAL)
            < m.composite_reliability(BIOMEDICAL))


def test_worked_example_reliability_and_ave_from_loadings():
    """Worked Example 5.1: omega 0.79 with AVE 0.50 is a factor that is
    reliable and broad, which is a report about the indicator set and not a
    failure of the domain."""
    omega = m.composite_reliability(BIOMEDICAL)
    ave = m.average_variance_extracted(BIOMEDICAL)
    alpha = m.cronbach_alpha_from_loadings(BIOMEDICAL)
    assert omega == pytest.approx(0.794, abs=1e-3)
    assert alpha == pytest.approx(0.789, abs=1e-3)
    assert ave == pytest.approx(0.496, abs=1e-3)
    assert omega > 0.70 and ave < 0.50
    assert math.sqrt(ave) == pytest.approx(0.704, abs=1e-3)


def test_definition_fornell_larcker():
    """Definition 5.3: a domain must share more variance with its own
    indicators than with any other domain."""
    ave = m.average_variance_extracted(BIOMEDICAL)
    assert m.fornell_larcker(ave, [0.62, 0.55]) is True
    assert m.fornell_larcker(ave, [0.62, 0.78]) is False


WITHIN_SOCIAL = [0.52, 0.47, 0.44]
WITHIN_SYSTEMIC = [0.49, 0.41, 0.38, 0.45, 0.36, 0.42]
CROSS_BLOCK = [0.34, 0.29, 0.31, 0.36,
               0.30, 0.26, 0.28, 0.33,
               0.27, 0.24, 0.25, 0.30]


def test_definition_and_worked_example_htmt():
    """Definition 5.4 and Worked Example 5.2: the HTMT ratio for Social
    Modifiers against Systemic Burdens, and the distance to the 0.85 line."""
    ratio = m.htmt(WITHIN_SOCIAL, WITHIN_SYSTEMIC, CROSS_BLOCK)
    assert ratio == pytest.approx(0.659, abs=1e-3)
    assert ratio < 0.85
    inflated = [c * 1.2903 for c in CROSS_BLOCK]
    assert m.htmt(WITHIN_SOCIAL, WITHIN_SYSTEMIC, inflated) == pytest.approx(
        0.850, abs=2e-3)


def test_htmt_rises_with_cross_construct_correlation():
    base = m.htmt(WITHIN_SOCIAL, WITHIN_SYSTEMIC, CROSS_BLOCK)
    higher = m.htmt(WITHIN_SOCIAL, WITHIN_SYSTEMIC,
                    [c + 0.05 for c in CROSS_BLOCK])
    assert higher > base


def test_definition_factor_determinacy():
    """Definition 5.5: determinacy is the correlation between the factor and
    its score, and its square is the reliability of the score."""
    rho = m.shrinkage_factor(BIOMEDICAL)
    det = m.factor_determinacy(BIOMEDICAL)
    assert rho == pytest.approx(0.820, abs=1e-3)
    assert det == pytest.approx(0.906, abs=1e-3)
    assert det * det == pytest.approx(rho, abs=1e-12)


def test_proposition_shrinkage_compresses_towards_the_mean():
    """Proposition 5.2: the posterior mean is the true score multiplied by
    rho = I/(1+I), so every score is pulled towards zero, and a profile
    carrying less information is pulled further."""
    rho = m.shrinkage_factor(BIOMEDICAL)
    assert 0.0 < rho < 1.0
    # noiseless indicators for a true score of -2: the score returns rho*-2
    noiseless = [lam * -2.0 for lam in BIOMEDICAL]
    assert m.posterior_mean(BIOMEDICAL, noiseless) == pytest.approx(
        rho * -2.0, abs=1e-12)
    # two weaker indicators carry less information, so compress harder
    sparse = [0.68, 0.55]
    assert m.shrinkage_factor(sparse) < rho
    assert m.posterior_mean(sparse, [lam * -2.0 for lam in sparse]) > \
        m.posterior_mean(BIOMEDICAL, noiseless)


def test_worked_example_shrinkage_on_a_simulated_latent_variable():
    """Worked Example 5.3: a true score of -2.00 comes back at about -1.64
    with four indicators and about -1.15 with two, against a predicted -1.13."""
    lam = np.array(BIOMEDICAL)
    theta = 1.0 - lam ** 2
    info = float(np.sum(lam ** 2 / theta))
    rng = np.random.default_rng(20260902)
    n = 200_000
    z = rng.standard_normal(n)
    y = np.outer(z, lam) + rng.standard_normal((n, 4)) * np.sqrt(theta)
    post = (y @ (lam / theta)) / (1.0 + info)
    extreme = post[(z > -2.05) & (z < -1.95)]
    assert extreme.mean() == pytest.approx(-1.64, abs=0.02)
    assert m.shrinkage_factor(BIOMEDICAL) * -2.0 == pytest.approx(-1.64,
                                                                  abs=0.01)
    assert np.corrcoef(z, post)[0, 1] == pytest.approx(
        m.factor_determinacy(BIOMEDICAL), abs=0.01)
    lam2 = lam[2:]
    theta2 = 1.0 - lam2 ** 2
    info2 = float(np.sum(lam2 ** 2 / theta2))
    post2 = (y[:, 2:] @ (lam2 / theta2)) / (1.0 + info2)
    assert post2[(z > -2.05) & (z < -1.95)].mean() == pytest.approx(-1.15,
                                                                    abs=0.02)
    assert m.shrinkage_factor([0.68, 0.55]) * -2.0 == pytest.approx(-1.13,
                                                                    abs=0.01)


def test_proposition_second_order_layer_with_three_factors_is_just_identified():
    """Proposition 5.1: with three first-order factors the higher-order layer
    has as many free parameters as it has covariances to reproduce, so it
    reproduces the factor correlation matrix exactly."""
    gamma = np.array([0.82, 0.78, 0.31])
    phi = np.outer(gamma, gamma)
    np.fill_diagonal(phi, 1.0)
    # recovering the loadings from the three correlations alone
    g1 = math.sqrt(phi[0, 1] * phi[0, 2] / phi[1, 2])
    g2 = phi[0, 1] / g1
    g3 = phi[0, 2] / g1
    assert (g1, g2, g3) == pytest.approx(tuple(gamma), abs=1e-12)
    # three free loadings plus three disturbances against six moments: df = 0
    moments = 3 * (3 + 1) // 2
    free = 3 + 3
    assert moments - free == 0
    implied = np.outer([g1, g2, g3], [g1, g2, g3])
    psi = 1.0 - np.array([g1, g2, g3]) ** 2
    np.fill_diagonal(implied, np.diag(implied) + psi)
    assert np.allclose(implied, phi)
