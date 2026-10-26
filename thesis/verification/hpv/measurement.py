"""Psychometrics of the measurement model: Chapter 5.

Composite reliability, average variance extracted, the two discriminant
validity criteria, and the shrinkage of a factor score. Standardised
indicators are assumed throughout, so an indicator variance is 1 and a
residual variance is 1 - lambda**2.
"""
from __future__ import annotations
import math
from typing import Sequence


def _check_loadings(loadings: Sequence[float]) -> None:
    if len(loadings) < 2:
        raise ValueError("a factor needs at least two indicators")
    for lam in loadings:
        if not -1.0 < lam < 1.0:
            raise ValueError(f"standardised loading {lam} outside (-1,1)")


# --- Definition 5.1: composite reliability ---------------------------------
def composite_reliability(loadings: Sequence[float]) -> float:
    """Omega: (sum lambda)^2 / [(sum lambda)^2 + sum(1 - lambda^2)]."""
    _check_loadings(loadings)
    s = sum(loadings)
    theta = sum(1.0 - lam * lam for lam in loadings)
    return s * s / (s * s + theta)


def cronbach_alpha_from_loadings(loadings: Sequence[float]) -> float:
    """Alpha implied by the same one-factor model, for comparison with omega.

    Equal to omega when the loadings are equal (tau-equivalence) and below it
    otherwise.
    """
    _check_loadings(loadings)
    k = len(loadings)
    s = sum(loadings)
    total = s * s + sum(1.0 - lam * lam for lam in loadings)
    return (k / (k - 1.0)) * (1.0 - k / total)


# --- Definition 5.2: average variance extracted ----------------------------
def average_variance_extracted(loadings: Sequence[float]) -> float:
    """AVE: the mean squared standardised loading."""
    _check_loadings(loadings)
    return sum(lam * lam for lam in loadings) / len(loadings)


# --- Definition 5.3: the Fornell-Larcker criterion -------------------------
def fornell_larcker(ave: float, factor_correlations: Sequence[float]) -> bool:
    """True when sqrt(AVE) exceeds every correlation with another factor."""
    if not 0.0 < ave <= 1.0:
        raise ValueError("AVE must lie in (0,1]")
    return math.sqrt(ave) > max(abs(r) for r in factor_correlations)


# --- Definition 5.4: the heterotrait-monotrait ratio -----------------------
def htmt(within_i: Sequence[float], within_j: Sequence[float],
         cross: Sequence[float]) -> float:
    """Mean cross-construct correlation over the geometric mean of the two
    mean within-construct correlations. Inputs are the off-diagonal
    correlations of block i, of block j, and the full i-by-j cross block."""
    for block in (within_i, within_j, cross):
        if len(block) == 0:
            raise ValueError("empty correlation block")
    mi = sum(within_i) / len(within_i)
    mj = sum(within_j) / len(within_j)
    if mi <= 0 or mj <= 0:
        raise ValueError("mean within-construct correlation must be positive")
    mc = sum(cross) / len(cross)
    return mc / math.sqrt(mi * mj)


# --- Proposition 5.2: shrinkage of the factor score ------------------------
def score_information(loadings: Sequence[float]) -> float:
    """Fisher information the indicators carry about the factor,
    I = sum lambda_j^2 / theta_j, with the factor variance fixed at one."""
    _check_loadings(loadings)
    return sum(lam * lam / (1.0 - lam * lam) for lam in loadings)


def shrinkage_factor(loadings: Sequence[float]) -> float:
    """rho = I/(1+I): the factor by which the posterior mean compresses the
    true score towards the population mean, and the reliability of the
    score."""
    info = score_information(loadings)
    return info / (1.0 + info)


def factor_determinacy(loadings: Sequence[float]) -> float:
    """Correlation between the factor and its score, sqrt(rho)."""
    return math.sqrt(shrinkage_factor(loadings))


def posterior_mean(loadings: Sequence[float],
                   responses: Sequence[float]) -> float:
    """E[zeta | y] for a standardised one-factor model with normal errors."""
    _check_loadings(loadings)
    if len(loadings) != len(responses):
        raise ValueError("one response per loading")
    info = score_information(loadings)
    weighted = sum(lam * y / (1.0 - lam * lam)
                   for lam, y in zip(loadings, responses))
    return weighted / (1.0 + info)
