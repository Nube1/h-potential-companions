"""Stage-two prognostic modelling on an estimated factor score.

The problem
-----------
Stage one estimates a factor score; stage two puts that score into a Cox model.
The score is not the latent construct: it is a noisy estimate whose correlation
with the truth is the factor-score determinacy ``rho_D``. This module supplies
the pieces needed to quantify what that does to the hazard ratio, the standard
error and the power of a study.

Analytic expectation
--------------------
For the regression (Thurstone) score with ``Var(zeta) = 1`` the score has
variance ``rho_D^2``, and after standardisation ``zeta_hat / rho_D`` has unit
variance and covariance ``rho_D`` with ``zeta``. Under a proportional-hazards
model in the true score with small to moderate effects, the coefficient
recovered on the standardised estimated score is attenuated toward the null by
approximately ``rho_D``:

    beta_observed ~= rho_D * beta_true.

This is the classical regression-dilution result. It is an approximation over
finite follow-up, not an identity: conditioning on a mismeasured covariate
induces frailty, so the conditional log hazard ratio drifts further toward the
null as follow-up lengthens, at order ``beta_true^2``. The whole-pipeline
simulation is what establishes that the approximation is close enough at the
effect sizes at issue.

Cost in events
--------------
Matching the noncentrality of the score test between an oracle design and a
score-based design gives ``d_score / d_oracle ~= 1 / rho_D^2``, bounded below by
``(1 + c) / c``. At the ceiling ``rho_D = 0.877819`` that is 1.298, and at the
planned design ``rho_D = 0.8361`` it is 1.431. The inflation cannot be
engineered away by writing more items.

Limitations
-----------
Everything assumes a single standardised covariate, non-differential
measurement error, and the normal approximation for the score test. With
covariate adjustment the inflation applies to the partial information and the
constant changes; the direction and the bound do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .measurement import (
    MeasurementModel,
    fit_cfa,
    regression_scores,
    rho_determinacy,
    simulate_indicators,
)

__all__ = [
    "CoxFit",
    "PipelineReplicate",
    "attenuated_beta",
    "cox_fit",
    "events_inflation",
    "pipeline_replicate",
    "required_events",
    "simulate_survival",
]

#: Two-sided normal critical value at the 5 per cent level.
Z_CRIT: float = 1.959964


@dataclass(frozen=True)
class CoxFit:
    """A single-covariate Cox fit.

    Attributes
    ----------
    beta : float
        Log hazard ratio per unit of the covariate.
    se : float
        Standard error from the inverse observed information.
    iterations : int
        Newton-Raphson steps taken.
    converged : bool
        Whether the step size fell below the tolerance.
    """

    beta: float
    se: float
    iterations: int
    converged: bool

    @property
    def hazard_ratio(self) -> float:
        """``exp(beta)``."""
        return float(np.exp(self.beta))

    @property
    def significant(self) -> bool:
        """Whether the Wald test rejects at the two-sided 5 per cent level."""
        if not np.isfinite(self.beta) or not np.isfinite(self.se) or self.se <= 0:
            return False
        return bool(abs(self.beta / self.se) > Z_CRIT)


def cox_fit(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    tol: float = 1e-9,
    maxit: int = 100,
) -> CoxFit:
    """Newton-Raphson maximisation of the Cox partial likelihood.

    Parameters
    ----------
    x : numpy.ndarray
        Single covariate, shape ``(n,)``.
    time : numpy.ndarray
        Observed follow-up time, shape ``(n,)``.
    event : numpy.ndarray
        Event indicator, 1 for an event and 0 for censoring.
    tol : float, default 1e-9
        Stop when the Newton step is smaller than this.
    maxit : int, default 100
        Maximum iterations.

    Returns
    -------
    CoxFit

    Notes
    -----
    This is the partial likelihood with Breslow's handling of ties, not an
    approximation to it. Writing the risk-set sums as reverse cumulative sums
    over time-ordered data makes each iteration linear in ``n``:

        U(b)   = sum_{i: d_i=1} [ x_i - S1_i / S0_i ]
        I(b)   = sum_{i: d_i=1} [ S2_i / S0_i - (S1_i / S0_i)^2 ]

    with ``S0, S1, S2`` the risk-set sums of ``exp(b x)``, ``x exp(b x)`` and
    ``x^2 exp(b x)``. The standard error is ``sqrt(1 / I(b_hat))``.

    A difference of group means, ``mean(x | event) - mean(x | no event)``, is
    not this estimator and is biased: at a true coefficient of ``-0.12`` on the
    simulation used in this package the difference of means returns ``-0.1393``
    where the partial likelihood returns ``-0.1228``.

    Returns ``beta = nan`` when there is no event or the information is not
    positive, rather than a number that looks like an estimate.

    Raises
    ------
    ValueError
        If the three arrays have different lengths.
    """
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    if not (x.shape == time.shape == event.shape):
        raise ValueError("x, time and event must have the same shape")
    order = np.argsort(time, kind="stable")
    x, time, event = x[order], time[order], event[order]
    died = event == 1
    if not np.any(died):
        return CoxFit(float("nan"), float("nan"), 0, False)

    b = 0.0
    info = 0.0
    converged = False
    used = 0
    for it in range(1, maxit + 1):
        used = it
        eb = np.exp(b * x)
        s0 = np.cumsum(eb[::-1])[::-1]
        s1 = np.cumsum((eb * x)[::-1])[::-1]
        s2 = np.cumsum((eb * x * x)[::-1])[::-1]
        ratio = s1[died] / s0[died]
        score = float(np.sum(x[died] - ratio))
        info = float(np.sum(s2[died] / s0[died] - ratio**2))
        if info <= 0:
            return CoxFit(float("nan"), float("nan"), it, False)
        step = score / info
        b += step
        if abs(step) < tol:
            converged = True
            break
    return CoxFit(float(b), float(np.sqrt(1.0 / info)), used, converged)


def simulate_survival(
    linear_predictor: np.ndarray,
    event_fraction: float,
    rng: np.random.Generator,
    baseline_rate: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exponential survival times with administrative censoring.

    Parameters
    ----------
    linear_predictor : numpy.ndarray
        ``beta * covariate``, shape ``(n,)``.
    event_fraction : float
        Target proportion observed to have an event. The censoring time is the
        corresponding quantile of the simulated event times, so the realised
        fraction equals the target up to the quantile's discreteness.
    rng : numpy.random.Generator
        Explicit generator.
    baseline_rate : float, default 1.0
        Baseline hazard of the exponential model.

    Returns
    -------
    time : numpy.ndarray
        ``min(event time, censoring time)``.
    event : numpy.ndarray
        1 where the event was observed.
    """
    if not 0.0 < event_fraction <= 1.0:
        raise ValueError("event_fraction must lie in (0, 1]")
    lin = np.asarray(linear_predictor, dtype=float)
    t_event = rng.exponential(1.0 / (baseline_rate * np.exp(lin)))
    tau = float(np.quantile(t_event, event_fraction))
    time = np.minimum(t_event, tau)
    event = (t_event <= tau).astype(int)
    return time, event


def attenuated_beta(beta_true: float, rho_d: float) -> float:
    """Coefficient expected on the standardised estimated score.

    Parameters
    ----------
    beta_true : float
        Log hazard ratio per SD of the true construct.
    rho_d : float
        Factor-score determinacy.

    Returns
    -------
    float
        ``rho_d * beta_true``, the first-order regression-dilution prediction.
    """
    return float(rho_d) * float(beta_true)


def events_inflation(rho_d: float) -> float:
    """Events needed relative to an oracle study, ``1 / rho_D^2``.

    Parameters
    ----------
    rho_d : float
        Factor-score determinacy, strictly between 0 and 1.

    Returns
    -------
    float
        The inflation factor.

    Examples
    --------
    >>> round(events_inflation(0.877819), 4)
    1.2977
    >>> round(events_inflation(0.8361), 4)
    1.4305
    """
    rho = float(rho_d)
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho_d must lie in (0, 1]")
    return 1.0 / rho**2


def required_events(
    beta: float, power: float = 0.90, alpha: float = 0.05, rho_d: float = 1.0
) -> float:
    """Events needed to detect a log hazard ratio on a standardised covariate.

    Parameters
    ----------
    beta : float
        Log hazard ratio per SD of the true construct.
    power : float, default 0.90
        Target power.
    alpha : float, default 0.05
        Two-sided level.
    rho_d : float, default 1.0
        Determinacy of the covariate actually used. The default is the oracle
        study in which the construct is observed.

    Returns
    -------
    float
        Expected number of events, ``(z_{1-a/2} + z_power)^2 / (rho_D beta)^2``.

    Notes
    -----
    Uses the standard single-covariate result that the information about
    ``beta`` from ``d`` events is approximately ``d``. Fractional; round up in
    a protocol.
    """
    if beta == 0.0:
        raise ValueError("beta must be non-zero")
    z_a = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_b = float(stats.norm.ppf(power))
    return float((z_a + z_b) ** 2 / (float(rho_d) * float(beta)) ** 2)


@dataclass(frozen=True)
class PipelineReplicate:
    """One whole-pipeline replicate: indicators, factor model, score, Cox model.

    Attributes
    ----------
    n : int
        Analytic sample size.
    endpoint : str
        Label of the endpoint simulated.
    beta_oracle, se_oracle : float
        Cox fit on the standardised true construct.
    beta_observed, se_observed : float
        Cox fit on the standardised estimated score.
    rho_hat : float
        Determinacy implied by the fitted measurement model.
    events : int
        Observed events.
    significant, significant_oracle : bool
        Wald tests at the two-sided 5 per cent level.
    """

    n: int
    endpoint: str
    beta_oracle: float
    se_oracle: float
    beta_observed: float
    se_observed: float
    rho_hat: float
    events: int
    significant: bool
    significant_oracle: bool


def pipeline_replicate(
    n: int,
    endpoint: str,
    event_fraction: float,
    model: MeasurementModel,
    beta_true: float,
    rng: np.random.Generator,
    per_domain: int = 4,
    start_loading: float = 0.9,
) -> PipelineReplicate | None:
    """Simulate the complete two-stage pipeline once.

    Parameters
    ----------
    n : int
        Analytic sample size.
    endpoint : str
        Label carried into the result.
    event_fraction : float
        Target proportion of observed events.
    model : MeasurementModel
        Population measurement model.
    beta_true : float
        Log hazard ratio per SD of the true construct.
    rng : numpy.random.Generator
        Explicit generator.
    per_domain : int, default 4
        Indicators in each domain.
    start_loading : float, default 0.9
        Starting value for the free loadings in stage one.

    Returns
    -------
    PipelineReplicate or None
        ``None`` when the sample covariance is singular, the factor fit is
        inadmissible, or the Cox fit fails. A dropped replicate is dropped, not
        replaced by a filled-in value.

    Notes
    -----
    Both stages are run on the same simulated people, which is the point: the
    oracle fit and the score fit see the same survival data and differ only in
    the covariate. The score is standardised before entering the Cox model, so
    the attenuation is a ratio of comparable coefficients.
    """
    zeta, _, y = simulate_indicators(model, int(n), rng)
    time, event = simulate_survival(float(beta_true) * zeta, float(event_fraction), rng)

    oracle = cox_fit((zeta - zeta.mean()) / zeta.std(), time, event)
    if not np.isfinite(oracle.beta):
        return None

    s_matrix = np.cov(y, rowvar=False)
    sign, _ = np.linalg.slogdet(s_matrix)
    if sign <= 0:
        return None
    fit = fit_cfa(s_matrix, model.n_domains, per_domain, start_loading=start_loading, ftol=1e-11)
    if not (np.all(fit.theta > 1e-4) and np.all(fit.psi > 1e-4)):
        return None
    cov_zy = fit.lambda_matrix @ fit.gamma
    score = regression_scores(y, fit.sigma, cov_zy)
    rho_hat = rho_determinacy(fit.lambda_matrix, fit.gamma, fit.psi, fit.theta)
    z_hat = (score - score.mean()) / score.std()

    observed = cox_fit(z_hat, time, event)
    if not np.isfinite(observed.beta):
        return None
    return PipelineReplicate(
        n=int(n),
        endpoint=str(endpoint),
        beta_oracle=oracle.beta,
        se_oracle=oracle.se,
        beta_observed=observed.beta,
        se_observed=observed.se,
        rho_hat=float(rho_hat),
        events=int(event.sum()),
        significant=observed.significant,
        significant_oracle=oracle.significant,
    )
