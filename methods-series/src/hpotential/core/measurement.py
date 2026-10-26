"""Second-order measurement model: determinacy, its ceiling, and identification.

The quantity
------------
Factor-score determinacy is the correlation between the true factor and its
estimated score,

.. math::
    \\rho_D = \\sqrt{\\operatorname{cov}(\\zeta, y)^\\top \\Sigma^{-1}
                     \\operatorname{cov}(y, \\zeta)},

with ``Sigma`` the model-implied covariance matrix of the indicators. It is a
population quantity. It depends on the loadings, the number of indicators and
the domain disturbances, and not on the sample size. Collecting more people
sharpens the parameter estimates; it does not make any one person's score more
determinate. A score with ``rho_D = 0.85`` is not adequate for an individual
decision however large the study was.

The ceiling
-----------
For the second-order model

    y = Lambda eta + epsilon,    eta = Gamma zeta + delta,

let the first-order factors be measured better and better by adding indicators
inside each domain. In the limit the indicators pin down ``eta`` exactly, and
the problem reduces to recovering ``zeta`` from ``eta``. By Sherman-Morrison,

.. math::
    \\rho_D^2 \\to \\gamma^\\top (\\gamma\\gamma^\\top + \\Psi)^{-1}\\gamma
    = \\frac{c}{1 + c}, \\qquad c = \\gamma^\\top \\Psi^{-1} \\gamma,

so ``rho_D`` cannot pass ``sqrt(c / (1 + c))`` however many indicators are
written. The ceiling is set by the second-order part of the model, not by the
item writing. To reach ``rho_D >= 0.90`` needs ``c >= 0.9^2 / (1 - 0.9^2) =
4.263``.

For the planned design of the theory paper, ``gamma = (0.80, 0.70, 0.62)`` and
``Psi = (0.36, 0.51, 0.62)``, giving ``c = 3.3586`` and a ceiling of
``0.877819``. Evaluating the exact expression at 400 indicators per domain with
standardised loadings of 0.85 gives ``0.877594``, which agrees to three decimal
places and is still short of 0.90.

Identification
--------------
Local identification of the second-order model holds when the Jacobian of the
map from free parameters to unique model-implied moments has full column rank
at the parameter point (Rothenberg 1971). :func:`identification_report` checks
that rank numerically and also returns the smallest singular value and the
condition number, which separate "identified" from "identified but badly
conditioned".

Estimation
----------
:func:`fit_cfa` maximises the normal-theory covariance discrepancy

    F_ML(theta) = log|Sigma(theta)| + tr(S Sigma(theta)^-1) - log|S| - p

by L-BFGS-B on an unconstrained parameterisation (log variances), and
:func:`expected_information` builds the Fisher information from central
differences of the model-implied covariance.

Limitations
-----------
Everything about the ceiling is algebra on a specified population model. It says
nothing about whether that model fits, and a determinacy computed from
estimated parameters inherits their error. The regression (Thurstone) score is
assumed; Bartlett scores have a different, generally lower, determinacy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

__all__ = [
    "GAMMA_PLANNED",
    "PSI_PLANNED",
    "CFAFit",
    "MeasurementModel",
    "balanced_loadings",
    "cfa_discrepancy",
    "determinacy_ceiling",
    "determinacy_curve",
    "expected_information",
    "fit_cfa",
    "identification_report",
    "marker_free_indices",
    "model_covariance",
    "moment_jacobian",
    "pack_parameters",
    "regression_scores",
    "required_second_order_signal",
    "rho_determinacy",
    "sample_size_replicate",
    "second_order_determinacy",
    "simulate_indicators",
    "unpack_parameters",
    "vech",
]

#: Second-order loadings of the planned design, with ``Var(zeta) = 1``.
GAMMA_PLANNED: np.ndarray = np.array([0.80, 0.70, 0.62])

#: Domain disturbance variances of the planned design.
PSI_PLANNED: np.ndarray = np.array([0.36, 0.51, 0.62])


# --------------------------------------------------------------- determinacy
def rho_determinacy(
    lambda_matrix: np.ndarray,
    gamma: np.ndarray,
    psi: np.ndarray,
    theta: np.ndarray,
) -> float:
    """Determinacy of the second-order factor score.

    Parameters
    ----------
    lambda_matrix : numpy.ndarray
        First-order loadings, indicators by domains, shape ``(p, K)``.
    gamma : numpy.ndarray
        Second-order loadings, one per domain, with ``Var(zeta) = 1``.
    psi : numpy.ndarray
        Domain disturbance variances, one per domain.
    theta : numpy.ndarray
        Indicator residual variances, one per indicator.

    Returns
    -------
    float
        ``rho_D`` in [0, 1].

    Notes
    -----
    With ``Phi = gamma gamma' + diag(psi)`` the model implies
    ``Sigma = Lambda Phi Lambda' + diag(theta)`` and
    ``cov(y, zeta) = Lambda gamma``, so
    ``rho_D^2 = (Lambda gamma)' Sigma^-1 (Lambda gamma)``. The result is
    clipped to [0, 1]: values outside come from an inadmissible model, not from
    a determinacy above one.

    Raises
    ------
    ValueError
        If the shapes disagree or a variance is not positive.
    """
    lam = np.asarray(lambda_matrix, dtype=float)
    g = np.asarray(gamma, dtype=float).ravel()
    ps = np.asarray(psi, dtype=float).ravel()
    th = np.asarray(theta, dtype=float).ravel()
    if lam.ndim != 2:
        raise ValueError("lambda_matrix must be two dimensional")
    p, k = lam.shape
    if g.size != k or ps.size != k:
        raise ValueError("gamma and psi must have one entry per domain")
    if th.size != p:
        raise ValueError("theta must have one entry per indicator")
    if np.any(ps <= 0) or np.any(th <= 0):
        raise ValueError("psi and theta must be positive")
    phi = np.outer(g, g) + np.diag(ps)
    sigma = lam @ phi @ lam.T + np.diag(th)
    cov_zy = lam @ g
    quad = float(cov_zy @ np.linalg.solve(sigma, cov_zy))
    return float(np.sqrt(np.clip(quad, 0.0, 1.0)))


def balanced_loadings(m_per_domain: int, lam_std: float, n_domains: int = 3) -> np.ndarray:
    """Block-diagonal loading matrix with equal loadings inside each domain.

    Parameters
    ----------
    m_per_domain : int
        Indicators in each domain.
    lam_std : float
        Standardised first-order loading, the same for every indicator.
    n_domains : int, default 3
        Number of first-order factors.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_domains * m_per_domain, n_domains)``.
    """
    m = int(m_per_domain)
    if m < 1:
        raise ValueError("m_per_domain must be at least 1")
    lam = np.zeros((n_domains * m, n_domains))
    for d in range(n_domains):
        lam[d * m : (d + 1) * m, d] = float(lam_std)
    return lam


def second_order_determinacy(
    m_per_domain: int,
    lam_std: float,
    gamma: np.ndarray = GAMMA_PLANNED,
    psi: np.ndarray = PSI_PLANNED,
) -> float:
    """Determinacy of a balanced design with equal loadings.

    Parameters
    ----------
    m_per_domain : int
        Indicators in each domain.
    lam_std : float
        Standardised first-order loading, the same for every indicator.
    gamma : numpy.ndarray, default :data:`GAMMA_PLANNED`
        Second-order loadings.
    psi : numpy.ndarray, default :data:`PSI_PLANNED`
        Domain disturbance variances.

    Returns
    -------
    float
        ``rho_D``.

    Notes
    -----
    Indicators are standardised, so the residual variance of an indicator in
    domain k is ``1 - lam^2 Var(eta_k)`` with ``Var(eta_k) = gamma_k^2 +
    psi_k``. Residual variances are floored at 1e-3, which matters only for
    combinations of a high loading and a high domain variance that leave no
    residual; such a design is not admissible and the floor keeps the function
    from returning a complex number instead of saying so.
    """
    m = int(m_per_domain)
    if m < 1:
        raise ValueError("m_per_domain must be at least 1")
    g = np.asarray(gamma, dtype=float).ravel()
    ps = np.asarray(psi, dtype=float).ravel()
    lam = balanced_loadings(m, lam_std, g.size)
    eta_var = g**2 + ps
    theta = np.array([1.0 - lam_std**2 * eta_var[j // m] for j in range(g.size * m)])
    theta = np.clip(theta, 1e-3, None)
    return rho_determinacy(lam, g, ps, theta)


def determinacy_ceiling(
    gamma: np.ndarray = GAMMA_PLANNED, psi: np.ndarray = PSI_PLANNED
) -> tuple[float, float]:
    """The limit of determinacy as indicators per domain grow without bound.

    Parameters
    ----------
    gamma : numpy.ndarray, default :data:`GAMMA_PLANNED`
        Second-order loadings.
    psi : numpy.ndarray, default :data:`PSI_PLANNED`
        Domain disturbance variances.

    Returns
    -------
    rho_inf : float
        ``sqrt(c / (1 + c))``.
    c : float
        The second-order signal ``gamma' Psi^-1 gamma``.

    Notes
    -----
    The bound depends only on the second-order model. No amount of item writing
    inside a domain passes it, so a determinacy target has to be met by
    strengthening the relation between the domains and the general factor, or
    by reporting domain scores instead of a single score.
    """
    g = np.asarray(gamma, dtype=float).ravel()
    ps = np.asarray(psi, dtype=float).ravel()
    if np.any(ps <= 0):
        raise ValueError("psi must be positive")
    c = float(g @ (g / ps))
    return float(np.sqrt(c / (1.0 + c))), c


def required_second_order_signal(target: float = 0.90) -> float:
    """Second-order signal ``c`` needed for a given determinacy.

    Parameters
    ----------
    target : float, default 0.90
        Desired ``rho_D``, the conventional benchmark for individual use.

    Returns
    -------
    float
        ``c = target^2 / (1 - target^2)``. For 0.90 this is 4.2632, against the
        3.3586 of the planned design.
    """
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie strictly between 0 and 1")
    return float(target**2 / (1.0 - target**2))


def determinacy_curve(
    m_values: np.ndarray | Sequence[int],
    lam_values: np.ndarray | Sequence[float],
    gamma: np.ndarray = GAMMA_PLANNED,
    psi: np.ndarray = PSI_PLANNED,
) -> pd.DataFrame:
    """Determinacy over a grid of designs.

    Parameters
    ----------
    m_values : array_like of int
        Indicators per domain to evaluate.
    lam_values : array_like of float
        Standardised loadings to evaluate.
    gamma : numpy.ndarray, default :data:`GAMMA_PLANNED`
        Second-order loadings.
    psi : numpy.ndarray, default :data:`PSI_PLANNED`
        Domain disturbance variances.

    Returns
    -------
    pandas.DataFrame
        Rows indexed by indicators per domain, one column per loading, labelled
        ``lambda=0.65`` and so on. ``frame.attrs`` carries ``ceiling`` and
        ``c``.
    """
    ms = [int(m) for m in np.asarray(m_values).ravel()]
    lams = [float(x) for x in np.asarray(lam_values).ravel()]
    data = {
        f"lambda={lam:g}": [second_order_determinacy(m, lam, gamma, psi) for m in ms]
        for lam in lams
    }
    frame = pd.DataFrame(data, index=pd.Index(ms, name="indicators_per_domain"))
    rho_inf, c = determinacy_ceiling(gamma, psi)
    frame.attrs["ceiling"] = rho_inf
    frame.attrs["c"] = c
    return frame


# ---------------------------------------------------------------- the model
@dataclass(frozen=True)
class MeasurementModel:
    """A fully specified second-order confirmatory factor model.

    Parameters
    ----------
    lambda_matrix : numpy.ndarray
        First-order loadings, shape ``(p, K)``.
    gamma : numpy.ndarray
        Second-order loadings, length ``K``, with ``Var(zeta) = 1``.
    psi : numpy.ndarray
        Domain disturbance variances, length ``K``.
    theta : numpy.ndarray
        Indicator residual variances, length ``p``.

    Notes
    -----
    The dataclass is frozen and its arrays are marked read-only, so a model can
    be shared between analyses without one of them changing it underneath the
    other. Build a variant with :meth:`balanced` or by constructing a new
    instance.
    """

    lambda_matrix: np.ndarray
    gamma: np.ndarray
    psi: np.ndarray
    theta: np.ndarray

    def __post_init__(self) -> None:
        for name in ("lambda_matrix", "gamma", "psi", "theta"):
            value = np.array(getattr(self, name), dtype=float)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.lambda_matrix.ndim != 2:
            raise ValueError("lambda_matrix must be two dimensional")
        p, k = self.lambda_matrix.shape
        if self.gamma.size != k or self.psi.size != k:
            raise ValueError("gamma and psi must have one entry per domain")
        if self.theta.size != p:
            raise ValueError("theta must have one entry per indicator")

    @classmethod
    def balanced(
        cls,
        m_per_domain: int,
        lam_std: float,
        gamma: np.ndarray = GAMMA_PLANNED,
        psi: np.ndarray = PSI_PLANNED,
        theta_scale: float = 1.0,
    ) -> MeasurementModel:
        """Balanced design with equal standardised loadings in every domain.

        Parameters
        ----------
        m_per_domain : int
            Indicators in each domain.
        lam_std : float
            Standardised first-order loading.
        gamma, psi : numpy.ndarray
            Second-order loadings and domain disturbances.
        theta_scale : float, default 1.0
            Multiplier on the residual variances. Values above one are the
            stress case: noisier indicators than the standardised design.

        Returns
        -------
        MeasurementModel
        """
        g = np.asarray(gamma, dtype=float).ravel()
        ps = np.asarray(psi, dtype=float).ravel()
        m = int(m_per_domain)
        lam = balanced_loadings(m, lam_std, g.size)
        eta_var = g**2 + ps
        theta = np.array([max(1e-3, 1.0 - lam_std**2 * eta_var[j // m]) for j in range(g.size * m)])
        return cls(lam, g, ps, theta * float(theta_scale))

    @classmethod
    def marker_scaled(
        cls,
        m_per_domain: int,
        lam_std: float,
        gamma: np.ndarray = GAMMA_PLANNED,
        psi: np.ndarray = PSI_PLANNED,
        theta_scale: float = 1.0,
        theta_floor: float = 0.15,
    ) -> MeasurementModel:
        """Population model whose first indicator in each domain loads at one.

        Parameters
        ----------
        m_per_domain : int
            Indicators in each domain.
        lam_std : float
            Loading of the non-marker indicators.
        gamma, psi : numpy.ndarray
            Second-order loadings and domain disturbances.
        theta_scale : float, default 1.0
            Multiplier on the residual variances.
        theta_floor : float, default 0.15
            Floor applied before scaling, which binds on the marker indicators
            because a loading of one leaves little residual.

        Returns
        -------
        MeasurementModel

        Notes
        -----
        Use this, not :meth:`balanced`, whenever the estimated ``gamma`` is to
        be compared with the true one. Marker scaling fixes the first loading of
        each domain to one, so a population model with every loading equal to
        ``lam`` implies a first-order factor scaled by ``1 / lam``, and the
        fitted second-order loadings come back as ``lam * gamma`` rather than
        ``gamma``. Bias and interval coverage computed against ``gamma`` under
        that mismatch measure the mismatch, not the estimator.
        """
        g = np.asarray(gamma, dtype=float).ravel()
        ps = np.asarray(psi, dtype=float).ravel()
        k, m = g.size, int(m_per_domain)
        p = k * m
        loadings = np.full(p, float(lam_std))
        loadings[::m] = 1.0
        lam = np.zeros((p, k))
        for d in range(k):
            lam[d * m : (d + 1) * m, d] = loadings[d * m : (d + 1) * m]
        eta_var = g**2 + ps
        theta = np.array(
            [
                max(theta_floor, 1.0 - loadings[j] ** 2 * eta_var[j // m]) * float(theta_scale)
                for j in range(p)
            ]
        )
        return cls(lam, g, ps, theta)

    @property
    def n_indicators(self) -> int:
        """Number of indicators ``p``."""
        return int(self.lambda_matrix.shape[0])

    @property
    def n_domains(self) -> int:
        """Number of first-order factors ``K``."""
        return int(self.lambda_matrix.shape[1])

    @property
    def phi(self) -> np.ndarray:
        """Covariance of the first-order factors, ``gamma gamma' + diag(psi)``."""
        return np.outer(self.gamma, self.gamma) + np.diag(self.psi)

    @property
    def sigma(self) -> np.ndarray:
        """Model-implied indicator covariance matrix."""
        return self.lambda_matrix @ self.phi @ self.lambda_matrix.T + np.diag(self.theta)

    @property
    def cov_zeta_y(self) -> np.ndarray:
        """Covariance between the second-order factor and the indicators."""
        return self.lambda_matrix @ self.gamma

    def determinacy(self) -> float:
        """Factor-score determinacy ``rho_D`` of this model."""
        return rho_determinacy(self.lambda_matrix, self.gamma, self.psi, self.theta)

    def ceiling(self) -> tuple[float, float]:
        """Determinacy ceiling and second-order signal of this model."""
        return determinacy_ceiling(self.gamma, self.psi)

    def simulate(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` indicator vectors from the model.

        Parameters
        ----------
        n : int
            Number of observations.
        rng : numpy.random.Generator
            Explicit generator. There is no module-level seeding anywhere in
            this package.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, p)``.
        """
        zeta, _, y = simulate_indicators(self, int(n), rng)
        del zeta
        return y


def model_covariance(model: MeasurementModel) -> np.ndarray:
    """Model-implied covariance matrix of the indicators.

    Parameters
    ----------
    model : MeasurementModel

    Returns
    -------
    numpy.ndarray
        Shape ``(p, p)``.
    """
    return model.sigma


def simulate_indicators(
    model: MeasurementModel, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the full hierarchy: ``zeta``, the domains, and the indicators.

    Parameters
    ----------
    model : MeasurementModel
        Population model.
    n : int
        Number of observations.
    rng : numpy.random.Generator
        Explicit generator.

    Returns
    -------
    zeta : numpy.ndarray
        Second-order factor, shape ``(n,)``, standard normal.
    eta : numpy.ndarray
        First-order factors, shape ``(n, K)``.
    y : numpy.ndarray
        Indicators, shape ``(n, p)``.

    Notes
    -----
    Drawing the hierarchy rather than sampling from ``Sigma`` directly keeps the
    true ``zeta`` available, which the prognostic pipeline needs for its oracle
    comparison.
    """
    zeta = rng.standard_normal(int(n))
    delta = rng.standard_normal((int(n), model.n_domains)) * np.sqrt(model.psi)
    eta = np.outer(zeta, model.gamma) + delta
    y = eta @ model.lambda_matrix.T + rng.standard_normal((int(n), model.n_indicators)) * np.sqrt(
        model.theta
    )
    return zeta, eta, y


def regression_scores(y: np.ndarray, sigma: np.ndarray, cov_zeta_y: np.ndarray) -> np.ndarray:
    """Thurstone (regression) factor scores for the second-order factor.

    Parameters
    ----------
    y : numpy.ndarray
        Indicators, shape ``(n, p)``.
    sigma : numpy.ndarray
        Indicator covariance matrix, shape ``(p, p)``.
    cov_zeta_y : numpy.ndarray
        Covariance between the factor and the indicators, shape ``(p,)``.

    Returns
    -------
    numpy.ndarray
        Scores, shape ``(n,)``, centred on the sample mean of ``y``.
    """
    centred = np.asarray(y, dtype=float) - np.mean(y, axis=0)
    return centred @ np.linalg.solve(np.asarray(sigma, dtype=float), np.asarray(cov_zeta_y))


# ------------------------------------------------------------ identification
def vech(matrix: np.ndarray) -> np.ndarray:
    """Lower-triangle half-vectorisation, including the diagonal.

    Parameters
    ----------
    matrix : numpy.ndarray
        Square symmetric matrix.

    Returns
    -------
    numpy.ndarray
        Length ``p(p+1)/2``.
    """
    m = np.asarray(matrix, dtype=float)
    return m[np.tril_indices(m.shape[0])]


def marker_free_indices(n_indicators: int, per_domain: int) -> tuple[list[int], list[int]]:
    """Marker indicators and the freely estimated loadings.

    Parameters
    ----------
    n_indicators : int
        Total indicators ``p``.
    per_domain : int
        Indicators in each domain.

    Returns
    -------
    marker : list of int
        The first indicator of each domain, whose loading is fixed to one.
    free : list of int
        Every other indicator.
    """
    marker = list(range(0, int(n_indicators), int(per_domain)))
    free = [j for j in range(int(n_indicators)) if j not in marker]
    return marker, free


def _sigma_from_vector(
    v: np.ndarray, n_indicators: int, n_domains: int, per_domain: int, free: Sequence[int]
) -> np.ndarray:
    """Model-implied covariance from a flat parameter vector under marker scaling."""
    nf = len(free)
    lam_free = v[:nf]
    g = v[nf : nf + n_domains]
    psi = v[nf + n_domains : nf + 2 * n_domains]
    theta = v[nf + 2 * n_domains :]
    lam = np.ones(n_indicators)
    lam[list(free)] = lam_free
    lam_matrix = np.zeros((n_indicators, n_domains))
    for k in range(n_domains):
        lam_matrix[k * per_domain : (k + 1) * per_domain, k] = lam[
            k * per_domain : (k + 1) * per_domain
        ]
    phi = np.outer(g, g) + np.diag(psi)
    return lam_matrix @ phi @ lam_matrix.T + np.diag(theta)


def moment_jacobian(
    v: np.ndarray,
    n_indicators: int,
    n_domains: int,
    per_domain: int,
    free: Sequence[int],
    step: float = 1e-6,
) -> np.ndarray:
    """Jacobian of the unique model-implied moments in the free parameters.

    Parameters
    ----------
    v : numpy.ndarray
        Parameter vector: free loadings, then ``gamma``, then ``psi``, then
        ``theta``.
    n_indicators, n_domains, per_domain : int
        Design constants.
    free : sequence of int
        Indices of the freely estimated loadings.
    step : float, default 1e-6
        Central-difference step.

    Returns
    -------
    numpy.ndarray
        Shape ``(p(p+1)/2, len(v))``.
    """
    v = np.asarray(v, dtype=float)
    d = v.size
    jac = np.empty((n_indicators * (n_indicators + 1) // 2, d))
    for i in range(d):
        up = v.copy()
        up[i] += step
        dn = v.copy()
        dn[i] -= step
        jac[:, i] = (
            vech(_sigma_from_vector(up, n_indicators, n_domains, per_domain, free))
            - vech(_sigma_from_vector(dn, n_indicators, n_domains, per_domain, free))
        ) / (2 * step)
    return jac


def identification_report(
    lam_std: float,
    label: str,
    gamma: np.ndarray = GAMMA_PLANNED,
    psi: np.ndarray = PSI_PLANNED,
    per_domain: int = 4,
    tol: float = 1e-10,
) -> dict[str, object]:
    """Rank check for local identification of the second-order model.

    Parameters
    ----------
    lam_std : float
        Standardised loading at which the Jacobian is evaluated.
    label : str
        Name carried into the returned record.
    gamma, psi : numpy.ndarray
        Second-order loadings and domain disturbances. Their length sets the
        number of domains, so a two-domain ``gamma`` runs the negative control.
    per_domain : int, default 4
        Indicators in each domain.
    tol : float, default 1e-10
        Singular values below ``tol`` times the largest count as zero.

    Returns
    -------
    dict
        ``label``, ``loading``, ``n_free``, ``n_moments``, ``rank``,
        ``identified``, ``smallest_singular_value``, ``condition_number``.

    Notes
    -----
    Rothenberg's criterion is local: full column rank at the parameter point
    gives identification in a neighbourhood of that point, not globally. The
    condition number is reported because a model can pass the rank test and
    still be useless in finite samples.

    Examples
    --------
    >>> report = identification_report(0.75, "planned")
    >>> report["rank"], report["n_free"], report["identified"]
    (27, 27, True)
    """
    g = np.asarray(gamma, dtype=float).ravel()
    ps = np.asarray(psi, dtype=float).ravel()
    n_domains = g.size
    p = n_domains * int(per_domain)
    _, free = marker_free_indices(p, int(per_domain))
    theta = np.array(
        [
            max(1e-3, 1.0 - lam_std**2 * (g[j // per_domain] ** 2 + ps[j // per_domain]))
            for j in range(p)
        ]
    )
    v = np.concatenate([np.full(len(free), float(lam_std)), g, ps, theta])
    jac = moment_jacobian(v, p, n_domains, int(per_domain), free)
    s = np.linalg.svd(jac, compute_uv=False)
    rank = int(np.sum(s > s[0] * tol))
    return {
        "label": label,
        "loading": float(lam_std),
        "n_free": int(v.size),
        "n_moments": int(jac.shape[0]),
        "rank": rank,
        "identified": bool(rank == v.size),
        "smallest_singular_value": float(s[-1]),
        "condition_number": float(s[0] / s[-1]),
    }


# ------------------------------------------------------------------ ML fitting
def pack_parameters(
    lam_free: np.ndarray, gamma: np.ndarray, psi: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """Flatten the free parameters onto the unconstrained scale.

    Variances are carried as logarithms so the optimiser cannot walk into a
    negative variance.
    """
    return np.concatenate(
        [
            np.asarray(lam_free, dtype=float),
            np.asarray(gamma, dtype=float),
            np.log(np.asarray(psi, dtype=float)),
            np.log(np.asarray(theta, dtype=float)),
        ]
    )


def unpack_parameters(
    v: np.ndarray, n_free: int, n_domains: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_parameters`."""
    v = np.asarray(v, dtype=float)
    lam_free = v[:n_free]
    gamma = v[n_free : n_free + n_domains]
    psi = np.exp(v[n_free + n_domains : n_free + 2 * n_domains])
    theta = np.exp(v[n_free + 2 * n_domains :])
    return lam_free, gamma, psi, theta


def _sigma_marker(
    lam_free: np.ndarray,
    gamma: np.ndarray,
    psi: np.ndarray,
    theta: np.ndarray,
    n_indicators: int,
    n_domains: int,
    per_domain: int,
    free: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    lam = np.ones(n_indicators)
    lam[list(free)] = lam_free
    lam_matrix = np.zeros((n_indicators, n_domains))
    for k in range(n_domains):
        lam_matrix[k * per_domain : (k + 1) * per_domain, k] = lam[
            k * per_domain : (k + 1) * per_domain
        ]
    phi = np.outer(gamma, gamma) + np.diag(psi)
    return lam_matrix @ phi @ lam_matrix.T + np.diag(theta), lam_matrix


def cfa_discrepancy(
    v: np.ndarray,
    s_matrix: np.ndarray,
    logdet_s: float,
    n_indicators: int,
    n_domains: int,
    per_domain: int,
    free: Sequence[int],
) -> float:
    """Normal-theory ML discrepancy ``F_ML``.

    Parameters
    ----------
    v : numpy.ndarray
        Packed parameter vector.
    s_matrix : numpy.ndarray
        Sample covariance matrix.
    logdet_s : float
        ``log|S|``, passed in because it does not change across iterations.
    n_indicators, n_domains, per_domain : int
        Design constants.
    free : sequence of int
        Freely estimated loading indices.

    Returns
    -------
    float
        ``log|Sigma| + tr(S Sigma^-1) - log|S| - p``, or ``1e10`` when
        ``Sigma`` is not positive definite, which steers the optimiser away
        rather than raising inside the objective.
    """
    lam_free, gamma, psi, theta = unpack_parameters(v, len(free), n_domains)
    sigma, _ = _sigma_marker(lam_free, gamma, psi, theta, n_indicators, n_domains, per_domain, free)
    try:
        chol = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        return 1e10
    logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    return float(logdet + np.trace(s_matrix @ np.linalg.inv(sigma)) - logdet_s - n_indicators)


@dataclass(frozen=True)
class CFAFit:
    """Result of one confirmatory factor fit.

    Attributes
    ----------
    converged : bool
        Whether the optimiser reported success (or hit the iteration limit,
        which L-BFGS-B flags separately and which is treated as convergence
        when the gradient is small).
    admissible : bool
        Whether every variance is positive and no loading is absurd.
    lambda_free, gamma, psi, theta : numpy.ndarray
        Estimates.
    lambda_matrix : numpy.ndarray
        Assembled loading matrix including the fixed markers.
    sigma : numpy.ndarray
        Model-implied covariance at the estimates.
    discrepancy : float
        ``F_ML`` at the solution.
    """

    converged: bool
    admissible: bool
    lambda_free: np.ndarray
    gamma: np.ndarray
    psi: np.ndarray
    theta: np.ndarray
    lambda_matrix: np.ndarray
    sigma: np.ndarray
    discrepancy: float
    packed: np.ndarray

    def determinacy(self) -> float:
        """Determinacy implied by the fitted parameters."""
        return rho_determinacy(self.lambda_matrix, self.gamma, self.psi, self.theta)


def fit_cfa(
    s_matrix: np.ndarray,
    n_domains: int = 3,
    per_domain: int = 4,
    start_loading: float = 0.8,
    maxiter: int = 2000,
    ftol: float = 1e-12,
    gtol: float = 1e-9,
) -> CFAFit:
    """Fit the second-order model by normal-theory maximum likelihood.

    Parameters
    ----------
    s_matrix : numpy.ndarray
        Sample covariance matrix, shape ``(p, p)``.
    n_domains : int, default 3
        Number of first-order factors.
    per_domain : int, default 4
        Indicators in each domain.
    start_loading : float, default 0.8
        Starting value for the free loadings.
    maxiter, ftol, gtol : float
        L-BFGS-B controls.

    Returns
    -------
    CFAFit

    Notes
    -----
    The scale of each first-order factor is fixed by a unit marker loading on
    its first indicator. Variances are estimated on the log scale. No
    constraint forces the solution to be admissible, because the rate of
    inadmissible solutions is one of the quantities the sample-size study
    reports.
    """
    s_matrix = np.asarray(s_matrix, dtype=float)
    p = s_matrix.shape[0]
    _, free = marker_free_indices(p, per_domain)
    sign, logdet_s = np.linalg.slogdet(s_matrix)
    if sign <= 0:
        raise ValueError("sample covariance matrix is not positive definite")
    start = pack_parameters(
        np.full(len(free), float(start_loading)),
        np.full(n_domains, 0.7),
        np.full(n_domains, 0.5),
        np.full(p, 0.5),
    )
    res = minimize(
        cfa_discrepancy,
        start,
        args=(s_matrix, logdet_s, p, n_domains, per_domain, free),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol},
    )
    converged = bool(res.success or res.status == 1)
    lam_free, gamma, psi, theta = unpack_parameters(res.x, len(free), n_domains)
    sigma, lam_matrix = _sigma_marker(lam_free, gamma, psi, theta, p, n_domains, per_domain, free)
    admissible = bool(
        converged and np.all(theta > 1e-4) and np.all(psi > 1e-4) and np.all(np.abs(lam_free) < 5)
    )
    return CFAFit(
        converged=converged,
        admissible=admissible,
        lambda_free=lam_free,
        gamma=gamma,
        psi=psi,
        theta=theta,
        lambda_matrix=lam_matrix,
        sigma=sigma,
        discrepancy=float(res.fun),
        packed=res.x,
    )


def expected_information(
    packed: np.ndarray,
    n: int,
    n_indicators: int,
    n_domains: int,
    per_domain: int,
    free: Sequence[int],
    step: float = 1e-5,
) -> np.ndarray:
    """Expected (Fisher) information for the normal-theory ML discrepancy.

    Parameters
    ----------
    packed : numpy.ndarray
        Parameter vector on the packed scale.
    n : int
        Sample size.
    n_indicators, n_domains, per_domain : int
        Design constants.
    free : sequence of int
        Freely estimated loading indices.
    step : float, default 1e-5
        Central-difference step on the packed scale.

    Returns
    -------
    numpy.ndarray
        ``I_ij = (n/2) tr(Sigma^-1 dSigma_i Sigma^-1 dSigma_j)``.

    Notes
    -----
    Central differences on the model-implied covariance are far cheaper than a
    numerical Hessian of ``F_ML`` and better behaved. Standard errors follow as
    the square roots of the diagonal of the inverse.
    """
    packed = np.asarray(packed, dtype=float)

    def sigma_at(vec: np.ndarray) -> np.ndarray:
        lam_free, gamma, psi, theta = unpack_parameters(vec, len(free), n_domains)
        sigma, _ = _sigma_marker(
            lam_free, gamma, psi, theta, n_indicators, n_domains, per_domain, free
        )
        return sigma

    sigma = sigma_at(packed)
    s_inv = np.linalg.inv(sigma)
    d = packed.size
    derivs = np.empty((d, n_indicators, n_indicators))
    for i in range(d):
        up = packed.copy()
        up[i] += step
        dn = packed.copy()
        dn[i] -= step
        derivs[i] = (sigma_at(up) - sigma_at(dn)) / (2 * step)
    a = np.einsum("ij,kjl->kil", s_inv, derivs)
    return 0.5 * n * np.einsum("kij,lji->kl", a, a)


def sample_size_replicate(
    n: int,
    model: MeasurementModel,
    rng: np.random.Generator,
    gamma_true: np.ndarray = GAMMA_PLANNED,
    per_domain: int = 4,
    start_loading: float = 0.8,
) -> dict[str, object] | None:
    """One replicate of the sample-size study: simulate, fit, summarise.

    Parameters
    ----------
    n : int
        Sample size.
    model : MeasurementModel
        Population model to draw from.
    rng : numpy.random.Generator
        Explicit generator.
    gamma_true : numpy.ndarray
        True second-order loadings, used for bias and coverage.
    per_domain : int, default 4
        Indicators in each domain.
    start_loading : float, default 0.8
        Starting value for the free loadings.

    Returns
    -------
    dict or None
        ``None`` when the sample covariance matrix is singular. Otherwise
        ``converged``, ``admissible`` and, for admissible fits, ``gamma``,
        ``se_gamma``, ``cover`` and ``rho``.

    Notes
    -----
    Coverage is the proportion of replicates whose Wald interval, built from
    the expected information, contains the true loading. Nothing is hardcoded:
    a replicate that does not converge contributes a zero to the convergence
    rate and nothing else.

    The fit uses marker scaling, so ``model`` must be built with
    :meth:`MeasurementModel.marker_scaled` for the reported bias and coverage to
    be about the estimator rather than about a scale mismatch.
    """
    p, k = model.n_indicators, model.n_domains
    _, free = marker_free_indices(p, per_domain)
    chol = np.linalg.cholesky(model.sigma)
    x = rng.standard_normal((int(n), p)) @ chol.T
    s_matrix = np.cov(x, rowvar=False, bias=False)
    sign, _ = np.linalg.slogdet(s_matrix)
    if sign <= 0:
        return None
    fit = fit_cfa(s_matrix, k, per_domain, start_loading=start_loading)
    out: dict[str, object] = {
        "n": int(n),
        "converged": int(fit.converged),
        "admissible": int(fit.admissible),
    }
    if not fit.admissible:
        return out
    try:
        info = expected_information(fit.packed, int(n), p, k, per_domain, free)
        acov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return out
    se = np.sqrt(np.clip(np.diag(acov), 0.0, None))
    se_gamma = se[len(free) : len(free) + k]
    gamma_true = np.asarray(gamma_true, dtype=float)
    out.update(
        gamma=fit.gamma.tolist(),
        se_gamma=se_gamma.tolist(),
        cover=[
            (
                int(abs(fit.gamma[j] - gamma_true[j]) <= 1.959964 * se_gamma[j])
                if se_gamma[j] > 0
                else 0
            )
            for j in range(k)
        ],
        rho=fit.determinacy(),
    )
    return out
