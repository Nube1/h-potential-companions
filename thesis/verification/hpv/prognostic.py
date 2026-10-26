"""Survival, discrimination, calibration and net benefit: Chapters 3 and 6."""
from __future__ import annotations
import math
from typing import Sequence


def hazard_ratio_to_risk(baseline_risk: float, hr: float,
                         sd_from_mean: float = 1.0) -> float:
    """Convert a hazard ratio into an absolute risk under a constant hazard.

    S = exp(-H0), so H0 = -log(1 - baseline_risk); the person's cumulative
    hazard is H0 * hr**sd_from_mean.
    """
    if not 0.0 < baseline_risk < 1.0:
        raise ValueError("baseline risk must lie strictly inside (0,1)")
    H0 = -math.log(1.0 - baseline_risk)
    return 1.0 - math.exp(-H0 * hr ** sd_from_mean)


def number_needed(risk_a: float, risk_b: float) -> float:
    """One event avoided per this many people, from an absolute difference."""
    d = abs(risk_a - risk_b)
    if d == 0:
        raise ZeroDivisionError("no risk difference")
    return 1.0 / d


def cumulative_incidence_competing(rate_event: float, rate_death: float,
                                   t: float) -> float:
    """True cumulative incidence of the event with a competing death, under
    constant cause-specific rates."""
    tot = rate_event + rate_death
    return (rate_event / tot) * (1.0 - math.exp(-tot * t))


def km_complement(rate_event: float, t: float) -> float:
    """What 1 - Kaplan-Meier reports when the competing death is censored.
    Always at least the true cumulative incidence, and the gap widens with
    the competing rate."""
    return 1.0 - math.exp(-rate_event * t)


def net_benefit(tp: int, fp: int, n: int, threshold: float) -> float:
    """Vickers and Elkin decision-curve net benefit."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie strictly inside (0,1)")
    return tp / n - (fp / n) * (threshold / (1.0 - threshold))


def net_benefit_treat_all(prevalence: float, threshold: float) -> float:
    return prevalence - (1 - prevalence) * (threshold / (1 - threshold))


def attenuated_coefficient(beta: float, reliability: float) -> float:
    """Classical-error attenuation: beta_naive ~= rho_H * beta."""
    if not 0.0 < reliability <= 1.0:
        raise ValueError("reliability must lie in (0,1]")
    return reliability * beta
