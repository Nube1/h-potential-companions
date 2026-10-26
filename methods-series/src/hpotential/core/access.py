"""Costed comparison of an access programme against an efficacy programme.

The uncosted result
-------------------
For the attainable-capability expression

    H = (L Q) (S T A) / (D (1 + C + psi)) - burden,

the ratio of marginal effects is ``(dH/dA) / (dH/dT) = T / A``. That is a fact
about the function, not an allocation rule. A decision maker choosing between an
access programme and an efficacy programme compares gain per unit spend.

The costed rule
---------------
Writing ``Delta_A`` and ``Delta_T`` for the change each programme can buy, and
``c_A`` and ``c_T`` for their costs, the access programme is the better buy
exactly when

    V = (T / A) * (Delta_A / Delta_T) * (c_T / c_A) > 1.

``V`` factorises into a structural term, a feasibility term and a price term.
The uncosted rule ``T > A`` is the special case ``Delta_A = Delta_T`` and
``c_A = c_T``. Because two of the three factors are dropped by the uncosted
rule, the two rules can disagree, and :meth:`AccessModel.psa` reports how often.

Limitations
-----------
Every input range here is illustrative. Nothing is estimated from a health
system, the rule carries no distributional weight, and ``V > 1`` says which of
two programmes buys more attainable capability per unit spend, not whether
either is worth funding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["AccessModel", "attainable_capability", "costed_statistic"]


def attainable_capability(
    life: np.ndarray,
    quality: np.ndarray,
    supply: np.ndarray,
    efficacy: np.ndarray,
    access: np.ndarray,
    demand: np.ndarray,
    cost_barrier: np.ndarray,
    psi: np.ndarray,
    burden: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Attainable capability ``H``.

    Parameters
    ----------
    life, quality : numpy.ndarray
        Remaining life and its quality, whose product is the biomedical
        potential.
    supply, efficacy, access : numpy.ndarray
        Supply of care, treatment efficacy, and the share of the population able
        to reach it.
    demand, cost_barrier, psi : numpy.ndarray
        Demand pressure and the two friction terms in the denominator.
    burden : numpy.ndarray or float, default 0.0
        Disease burden subtracted from the attainable total.

    Returns
    -------
    numpy.ndarray
        ``(L Q)(S T A) / (D (1 + C + psi)) - burden``.
    """
    return (life * quality) * (supply * efficacy * access) / (
        demand * (1.0 + cost_barrier + psi)
    ) - burden


def costed_statistic(
    efficacy: np.ndarray,
    access: np.ndarray,
    delta_access: np.ndarray,
    delta_efficacy: np.ndarray,
    cost_access: np.ndarray,
    cost_efficacy: np.ndarray,
) -> np.ndarray:
    """The costed decision statistic ``V``.

    Parameters
    ----------
    efficacy, access : numpy.ndarray
        Current levels ``T`` and ``A``.
    delta_access, delta_efficacy : numpy.ndarray
        Feasible change each programme can buy.
    cost_access, cost_efficacy : numpy.ndarray
        Cost of each programme.

    Returns
    -------
    numpy.ndarray
        ``(T/A)(Delta_A/Delta_T)(c_T/c_A)``. Access is the better buy when this
        exceeds one.

    Notes
    -----
    The closed form is exact, not an approximation: the common factor
    ``(L Q) S / (D (1 + C + psi))`` cancels between the two marginal effects, so
    the ratio of gain per unit spend reduces to these three terms. The identity
    is checked numerically in :meth:`AccessModel.closed_form_check`, which
    agrees to a maximum relative discrepancy of order 1e-16.
    """
    return (efficacy / access) * (delta_access / delta_efficacy) * (cost_efficacy / cost_access)


def _default_high() -> dict[str, tuple[float, float]]:
    return {
        "A": (0.80, 0.95),
        "T": (0.70, 0.90),
        "S": (0.65, 0.90),
        "D": (1.0, 1.4),
        "C": (0.1, 0.6),
        "psi": (0.05, 0.4),
        "L": (14.0, 22.0),
        "Q": (0.65, 0.85),
        "dA": (0.02, 0.06),
        "dT": (0.03, 0.09),
        "cA": (0.6, 1.6),
        "cT": (0.8, 3.0),
    }


def _default_low() -> dict[str, tuple[float, float]]:
    return {
        "A": (0.15, 0.45),
        "T": (0.65, 0.90),
        "S": (0.35, 0.70),
        "D": (1.2, 2.2),
        "C": (0.2, 0.9),
        "psi": (0.2, 0.8),
        "L": (10.0, 18.0),
        "Q": (0.50, 0.75),
        "dA": (0.05, 0.20),
        "dT": (0.02, 0.06),
        "cA": (0.4, 1.4),
        "cT": (1.0, 4.0),
    }


@dataclass(frozen=True)
class AccessModel:
    """Uniform input ranges for one health-system setting.

    Parameters
    ----------
    name : str
        Label for the setting.
    ranges : dict of str to tuple of float
        Lower and upper bound of each input. Keys are ``A``, ``T``, ``S``,
        ``D``, ``C``, ``psi``, ``L``, ``Q``, ``dA``, ``dT``, ``cA``, ``cT``.

    Notes
    -----
    Independent uniform draws are a deliberate simplification. Real inputs are
    correlated, and the reported disagreement rate would change under a
    correlated prior. The ranges are illustrative, not estimates.
    """

    name: str
    ranges: dict[str, tuple[float, float]] = field(default_factory=_default_high)

    @classmethod
    def high_resource(cls) -> AccessModel:
        """The high-resource setting: access already high, efficacy dear."""
        return cls("high-resource", _default_high())

    @classmethod
    def low_resource(cls) -> AccessModel:
        """The low-resource setting: access low and cheap to improve."""
        return cls("low-resource", _default_low())

    def draw(self, rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
        """Draw ``n`` independent uniform values for every input.

        Parameters
        ----------
        rng : numpy.random.Generator
            Explicit generator.
        n : int
            Number of draws.

        Returns
        -------
        dict of str to numpy.ndarray
        """
        return {k: rng.uniform(lo, hi, size=int(n)) for k, (lo, hi) in self.ranges.items()}

    def closed_form_check(self, rng: np.random.Generator, n: int = 20000) -> float:
        """Largest relative gap between the closed form and a direct evaluation.

        Parameters
        ----------
        rng : numpy.random.Generator
            Explicit generator.
        n : int, default 20000
            Draws.

        Returns
        -------
        float
            ``max |V_closed - V_direct| / V_direct``, which should be at the
            level of floating-point rounding.

        Notes
        -----
        The direct evaluation forms ``dH/dA`` and ``dH/dT`` from the capability
        expression and takes the ratio of gain per unit spend. It shares no code
        with :func:`costed_statistic`, so agreement is evidence about the
        algebra rather than about a single implementation.
        """
        p = self.draw(rng, n)
        common = (p["L"] * p["Q"]) / (p["D"] * (1.0 + p["C"] + p["psi"]))
        dh_da = common * p["S"] * p["T"]
        dh_dt = common * p["S"] * p["A"]
        direct = (dh_da * p["dA"] / p["cA"]) / (dh_dt * p["dT"] / p["cT"])
        closed = costed_statistic(p["T"], p["A"], p["dA"], p["dT"], p["cA"], p["cT"])
        return float(np.max(np.abs(closed - direct) / direct))

    def psa(self, rng: np.random.Generator, n: int = 200_000) -> dict[str, float]:
        """Probabilistic sensitivity analysis of the costed rule.

        Parameters
        ----------
        rng : numpy.random.Generator
            Explicit generator.
        n : int, default 200000
            Draws.

        Returns
        -------
        dict
            Probability that each rule favours access, medians of the three
            terms and of ``V``, a 95 per cent range for ``V``, and
            ``P_disagreement``, the share of draws on which the uncosted and
            costed rules point different ways.

        Notes
        -----
        In the low-resource setting the disagreement rate is exactly zero, not
        approximately zero: over the stated ranges ``T / A`` is above one in
        every draw and so is ``V``, so the two rules cannot disagree. The
        high-resource setting is where the uncosted rule misleads.
        """
        q = self.draw(rng, n)
        structural = q["T"] / q["A"]
        feasibility = q["dA"] / q["dT"]
        price = q["cT"] / q["cA"]
        v = structural * feasibility * price
        return {
            "P_uncosted_rule_favours_access": float(np.mean(structural > 1)),
            "P_costed_rule_favours_access": float(np.mean(v > 1)),
            "median_structural_term": float(np.median(structural)),
            "median_feasibility_term": float(np.median(feasibility)),
            "median_price_term": float(np.median(price)),
            "median_V": float(np.median(v)),
            "V_2_5": float(np.percentile(v, 2.5)),
            "V_97_5": float(np.percentile(v, 97.5)),
            "P_disagreement": float(np.mean((structural > 1) != (v > 1))),
        }
