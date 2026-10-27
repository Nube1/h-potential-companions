"""Candidate realisation functions, ported from thesis/analysis/01_capability_realisation.R.

Each function takes biomedical potential P, social enablement S and systemic
burden B (so structural freedom is R = 1 - B) and returns one score per person.
The forms are the competing hypotheses of Chapter 2: additive, additive with a
P x S interaction, the geometric realisation function, its CES generalisation,
and a saturating relational form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REL_BETA = (1.0, 0.5, 0.5, 0.25, 0.25)  # relational form: P, S, R, P*S, P*R


def _check(P, S, B):
    P, S, B = (np.asarray(x, float) for x in (P, S, B))
    if not (P.shape == S.shape == B.shape):
        raise ValueError("P, S and B must have the same length")
    if not np.all(np.isfinite(np.concatenate([P.ravel(), S.ravel(), B.ravel()]))):
        raise ValueError("P, S and B must be finite")
    if np.any(P < 0) or np.any((S < 0) | (S > 1)) or np.any((B < 0) | (B > 1)):
        raise ValueError("require P >= 0 and S, B in [0, 1]")
    return P, S, B


def crf(P, S, B, alpha: float = 0.5, rho: float = 0.0) -> np.ndarray:
    """Capability realisation function H = P [alpha S^rho + (1-alpha) R^rho]^(1/rho).

    rho = 0 is the geometric form P S^alpha R^(1-alpha). For rho < 0 the score
    is zero whenever S or R is zero (essential enablement).
    """
    if not 0 < alpha < 1 or rho > 0:
        raise ValueError("require 0 < alpha < 1 and rho <= 0")
    P, S, B = _check(P, S, B)
    R = 1 - B
    if rho == 0:
        return P * S**alpha * R ** (1 - alpha)
    out = np.zeros_like(P)
    ok = (S > 0) & (R > 0)
    out[ok] = P[ok] * (alpha * S[ok] ** rho + (1 - alpha) * R[ok] ** rho) ** (1 / rho)
    return out


def crf_gap(a, b, alpha: float = 0.5) -> dict:
    """Exact log decomposition of H(a) / H(b) for profiles given as (P, S, B)."""
    (pa, sa, ba), (pb, sb, bb) = a, b
    return {
        "biomedical": float(np.log(pa / pb)),
        "social": float(alpha * np.log(sa / sb)),
        "structural": float((1 - alpha) * np.log((1 - ba) / (1 - bb))),
    }


def hp_candidates(P, S, B, alpha=0.5, rho=0.0, beta=(0, 1, 1, 1, 0)) -> pd.DataFrame:
    """All five candidate forms, as in the R function of the same name."""
    if len(beta) != 5:
        raise ValueError("beta must have five entries")
    P, S, B = _check(P, S, B)
    R = 1 - B
    additive = beta[0] + beta[1] * P + beta[2] * S - beta[3] * B
    interaction = additive + beta[4] * P * S
    r = REL_BETA
    exponent = r[0] * P + r[1] * S + r[2] * R + r[3] * P * S + r[4] * P * R
    relational = (1 - np.exp(-exponent)) / (1 - np.exp(-sum(r)))
    return pd.DataFrame(
        {
            "additive": additive,
            "interaction": interaction,
            "multiplicative": P * S**alpha * R ** (1 - alpha),
            "ces": crf(P, S, B, alpha=alpha, rho=rho),
            "relational": relational,
        }
    )


__all__ = ["crf", "crf_gap", "hp_candidates"]
