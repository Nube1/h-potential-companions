"""Net benefit, budget impact and distributional evaluation: Appendix A.11."""
from __future__ import annotations
import math
from typing import Sequence


def net_health_benefit(dQ: float, dC: float, lam: float) -> float:
    """NHB = dQ - dC/lambda, in units of health."""
    if lam <= 0:
        raise ValueError("lambda must be positive")
    return dQ - dC / lam


def inmb(dQ: float, dC: float, lam: float) -> float:
    """Incremental net monetary benefit."""
    return lam * dQ - dC


def budget_impact(N: Sequence[float], dc: Sequence[float]) -> float:
    return sum(n * c for n, c in zip(N, dc))


def displaced_health(dC: float, lam: float) -> float:
    """The health lost elsewhere when this budget is spent here."""
    return dC / lam


def discounted(values: Sequence[float], r: float) -> float:
    return sum(v / (1 + r) ** t for t, v in enumerate(values))


def icer(dC: float, dE: float) -> float:
    if dE == 0:
        raise ZeroDivisionError("ICER undefined when the effect difference is zero")
    return dC / dE


def daly(L0: float, w: float, E_L: float):
    """Appendix D Algorithm 1: YLD, YLL and their sum."""
    yld = w * E_L
    yll = L0 - E_L
    return yld, yll, yld + yll


def biomedical_potential(L0: float, Q: float, dalys: float) -> float:
    """BP = (L_adj/L0)(1 - DALY/L0) with L_adj = L0*Q."""
    return Q * (1.0 - dalys / L0)
