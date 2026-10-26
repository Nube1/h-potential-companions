"""Capability Realisation Function and the results that follow from it.

Covers Appendix A (axioms, CES family, relational rival, thresholds) and
Appendix E (characterisation, decomposition, policy gradient, equity).
"""
from __future__ import annotations
import math
from typing import Sequence


# --- Definition A.2: the domain profile ------------------------------------
def check_profile(p: float, s: float, r: float) -> None:
    for name, v in (("p", p), ("s", s), ("r", r)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name}={v} outside [0,1]")


# --- Definition A.6 / eq. (E.2): CES contextual mean ------------------------
def contextual_mean(s: float, r: float, rho: float = 0.0,
                    alpha: float = 0.5) -> float:
    """Weighted power mean of the two enablers.

    rho = 1  arithmetic (the additive comparator, outside the admissible family)
    rho = 0  geometric  (the primary CRF)
    rho -> -inf  minimum (the bottleneck limit)
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly inside (0,1)")
    if rho > 1.0:
        raise ValueError("rho must be at most 1")
    if rho == 0.0:
        return s ** alpha * r ** (1 - alpha)
    if s == 0.0 or r == 0.0:
        # For rho <= 0 a zero enabler drives the mean to zero (essential
        # enablement). For 0 < rho <= 1 the power mean stays finite and must
        # be evaluated, not short-circuited: 0**rho = 0 is well defined.
        if rho < 0.0:
            return 0.0
        return (alpha * s ** rho + (1 - alpha) * r ** rho) ** (1.0 / rho)
    if rho < -700:                       # numerical stand-in for the limit
        return min(s, r)
    return (alpha * s ** rho + (1 - alpha) * r ** rho) ** (1.0 / rho)


def H(p: float, s: float, r: float, rho: float = 0.0,
      alpha: float = 0.5) -> float:
    """Realised capability, eq. (E.1) with its geometric limit (E.2)."""
    check_profile(p, s, r)
    return p * contextual_mean(s, r, rho, alpha)


def H_additive(p: float, s: float, r: float,
               w=(1 / 3, 1 / 3, 1 / 3)) -> float:
    """Definition A.7: the additive comparator the axioms exclude."""
    check_profile(p, s, r)
    return w[0] * p + w[1] * s + w[2] * r


# --- Definition A.11: the relational rival ----------------------------------
def H_relational(p: float, s: float, r: float,
                 beta: Sequence[float] = (1.0, 0.5, 0.5, 0.25, 0.25)) -> float:
    """Normalised relational aggregator; denies the capacity ceiling."""
    check_profile(p, s, r)
    bP, bS, bR, bPS, bPR = beta
    K = sum(beta)
    x = bP * p + bS * s + bR * r + bPS * p * s + bPR * p * r
    return (1 - math.exp(-x)) / (1 - math.exp(-K))


# --- Definition A.13: deprivation counting ----------------------------------
def deprivation_count(profile, z=(0.5, 0.5, 0.5), w=(1 / 3, 1 / 3, 1 / 3)):
    return sum(wj for xj, zj, wj in zip(profile, z, w) if xj < zj)


# --- Proposition E.4: marginal effects, geometric case ----------------------
def gradients(p: float, s: float, r: float, alpha: float = 0.5):
    """dH/dp, dH/ds, dH/dr for the primary CRF."""
    h = H(p, s, r, 0.0, alpha)
    return (h / p if p else float("nan"),
            alpha * h / s if s else float("nan"),
            (1 - alpha) * h / r if r else float("nan"))


def elasticities(p: float, s: float, r: float, alpha: float = 0.5,
                 h_step: float = 1e-6):
    """Proposition E.4: constant elasticities (1, alpha, 1-alpha).

    Computed by numerical differentiation of H rather than from the analytic
    gradients. Using the gradients would make any test of this function
    circular: elasticity = (alpha*H/s)*(s/H) = alpha holds by algebra for any
    H whatsoever. Differentiating H itself is what actually checks the claim.
    """
    def d_log(idx: int) -> float:
        args = [p, s, r]
        lo, hi = list(args), list(args)
        lo[idx] *= (1.0 - h_step)
        hi[idx] *= (1.0 + h_step)
        return ((math.log(H(*hi, rho=0.0, alpha=alpha))
                 - math.log(H(*lo, rho=0.0, alpha=alpha)))
                / (math.log(hi[idx]) - math.log(lo[idx])))
    return (d_log(0), d_log(1), d_log(2))


def cross_partials(p: float, s: float, r: float, alpha: float = 0.5):
    """Proposition E.5: complementarity. Both strictly positive."""
    return (alpha * s ** (alpha - 1) * r ** (1 - alpha),
            (1 - alpha) * s ** alpha * r ** (-alpha))


# --- Theorem E.6: exact capability-gap decomposition ------------------------
def gap_decomposition(a, b, alpha: float = 0.5):
    """Split log(H_a/H_b) into biomedical, social and structural parts.

    Returns (biomedical, social, structural, total). The three parts sum to
    the total exactly, with no residual interaction term.
    """
    pa, sa, ra = a
    pb, sb, rb = b
    bio = math.log(pa / pb)
    soc = alpha * math.log(sa / sb)
    stc = (1 - alpha) * math.log(ra / rb)
    total = math.log(H(pa, sa, ra, 0.0, alpha) / H(pb, sb, rb, 0.0, alpha))
    return bio, soc, stc, total


# --- Definition E.9: equity-sensitive population potential ------------------
def equity_ede(values: Sequence[float], eps: float = 0.0) -> float:
    """Atkinson equally distributed equivalent of a set of H values."""
    v = list(values)
    if any(x <= 0 for x in v):
        raise ValueError("EDE requires strictly positive values")
    n = len(v)
    if abs(eps - 1.0) < 1e-12:
        return math.exp(sum(math.log(x) for x in v) / n)
    return (sum(x ** (1 - eps) for x in v) / n) ** (1 / (1 - eps))


def social_weight(h_i: float, ede: float, eps: float) -> float:
    """Proposition E.10: marginal social weight (EDE/H_i)^eps."""
    return (ede / h_i) ** eps
