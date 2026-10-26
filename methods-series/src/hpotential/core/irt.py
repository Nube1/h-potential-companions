"""Samejima's graded response model, in numpy and scipy alone.

Indexing convention, stated once and used everywhere
----------------------------------------------------
An item ``j`` has ``K_j`` ordered categories, coded ``0, 1, ..., K_j - 1`` in
the data. It has one discrimination ``a_j > 0`` and ``K_j - 1`` thresholds. The
thresholds are written ``b_{j2} < b_{j3} < ... < b_{jK}``: the subscript names
the category the respondent moves *into*, so there is no threshold ``b_{j1}``
(nobody crosses into the lowest category) and the count is ``K_j - 1``.

The boundary, or cumulative, functions are

.. math::
    P^*_{jk}(\\theta) = P(Y_j \\ge k \\mid \\theta)
    = \\Lambda[a_j(\\theta - b_{jk})], \\quad k = 2, \\ldots, K_j,

with the two conventions ``P^*_{j1} = 1`` and ``P^*_{j,K+1} = 0``, where
``\\Lambda`` is the logistic function. Category probabilities are differences of
neighbouring boundary functions,

.. math::
    P_{jk}(\\theta) = P^*_{jk}(\\theta) - P^*_{j,k+1}(\\theta),

so they are non-negative because the thresholds are ordered, and they sum to 1
by telescoping. The draft this module replaces declared ``K - 1`` thresholds
``b_1 ... b_{K-1}`` and then wrote the top category as
``P(Y = K) = \\Lambda[a(\\theta - b_K)]``, which needs a ``b_K`` the declaration
does not provide. The convention above is the one used in the code, the tests
and the documentation.

Estimation
----------
Marginal maximum likelihood by EM. The ability distribution is fixed at
``theta ~ N(0, 1)``, which identifies the scale, and the marginal likelihood is
integrated by Gauss-Hermite quadrature. The E step forms the posterior over
quadrature nodes for each person; the M step maximises the expected complete
data log-likelihood for each item separately, with ``a`` held positive by
estimating ``log a`` and the thresholds held ordered by estimating the first
threshold and log gaps.

There is no R dependency. The draft called ``mirt`` through ``rpy2``, which
means a reviewer needs R, the ``mirt`` package and a working ``rpy2`` build
before any of this runs. The estimates here are checked against simulation
rather than against another package.

Limitations
-----------
Standard errors for item parameters are not computed; the module returns point
estimates and the marginal log-likelihood. The model assumes one latent
dimension, local independence given theta, and monotone boundary functions. It
does not check dimensionality, which is the factor model's job, and it will
happily return parameters for items that violate local independence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

__all__ = [
    "GradedResponseModel",
    "GRMItem",
    "category_probabilities",
    "expected_score",
    "item_information",
    "simulate_grm",
    "standard_error",
    "test_information",
]


def _logistic(x: np.ndarray) -> np.ndarray:
    """Logistic function, written to avoid overflow for large |x|."""
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _boundaries(theta: np.ndarray, a: float, b: np.ndarray) -> np.ndarray:
    """Boundary functions ``P*`` including both conventions.

    Returns an array of shape ``(n_theta, K + 1)`` whose first column is 1 and
    whose last column is 0.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    b = np.asarray(b, dtype=float)
    inner = _logistic(a * (theta[:, None] - b[None, :]))
    ones = np.ones((theta.size, 1))
    zeros = np.zeros((theta.size, 1))
    return np.hstack([ones, inner, zeros])


def category_probabilities(theta: np.ndarray | float, a: float, b: Sequence[float]) -> np.ndarray:
    """Category probabilities of a graded response item.

    Parameters
    ----------
    theta : float or numpy.ndarray
        Latent trait values.
    a : float
        Discrimination, positive.
    b : sequence of float
        The ``K - 1`` thresholds ``b_2 < ... < b_K``, strictly increasing.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_theta, K)``. Row ``i`` holds ``P(Y = 0), ..., P(Y = K - 1)``
        at ``theta[i]`` and sums to 1 up to floating point.

    Raises
    ------
    ValueError
        If ``a`` is not positive or the thresholds are not strictly increasing.

    Examples
    --------
    >>> import numpy as np
    >>> p = category_probabilities(np.array([-1.0, 0.0, 1.0]), 1.4, [-1.0, 0.0, 1.0])
    >>> np.allclose(p.sum(axis=1), 1.0)
    True
    """
    a = float(a)
    b_arr = np.asarray(b, dtype=float)
    if a <= 0:
        raise ValueError("discrimination a must be positive")
    if b_arr.ndim != 1 or b_arr.size < 1:
        raise ValueError("b must be a one dimensional array of K-1 thresholds")
    if np.any(np.diff(b_arr) <= 0):
        raise ValueError(f"thresholds must be strictly increasing, got {b_arr}")
    star = _boundaries(np.atleast_1d(theta), a, b_arr)
    return star[:, :-1] - star[:, 1:]


def item_information(theta: np.ndarray | float, a: float, b: Sequence[float]) -> np.ndarray:
    """Fisher information of one item at given trait values.

    Parameters
    ----------
    theta : float or numpy.ndarray
        Latent trait values.
    a : float
        Discrimination.
    b : sequence of float
        Thresholds ``b_2 < ... < b_K``.

    Returns
    -------
    numpy.ndarray
        Information at each ``theta``, shape ``(n_theta,)``.

    Notes
    -----
    ``I_j(theta) = sum_k [dP_jk/dtheta]^2 / P_jk``, with
    ``dP*_jk/dtheta = a P*_jk (1 - P*_jk)`` and the derivative of a category
    probability the difference of two neighbouring boundary derivatives.
    Categories with probability below machine tolerance are skipped rather than
    dividing by zero; their contribution to the sum is of the same order.
    """
    theta_arr = np.atleast_1d(np.asarray(theta, dtype=float))
    a = float(a)
    b_arr = np.asarray(b, dtype=float)
    star = _boundaries(theta_arr, a, b_arr)
    dstar = a * star * (1.0 - star)  # zero at both boundary conventions
    probs = star[:, :-1] - star[:, 1:]
    dprobs = dstar[:, :-1] - dstar[:, 1:]
    safe = probs > 1e-12
    contrib = np.zeros_like(probs)
    contrib[safe] = dprobs[safe] ** 2 / probs[safe]
    return contrib.sum(axis=1)


def test_information(
    theta: np.ndarray | float,
    a: Sequence[float],
    b: Sequence[Sequence[float]],
) -> np.ndarray:
    """Test information, the sum of item information.

    Parameters
    ----------
    theta : float or numpy.ndarray
        Latent trait values.
    a : sequence of float
        One discrimination per item.
    b : sequence of sequence of float
        One threshold vector per item.

    Returns
    -------
    numpy.ndarray
        Test information at each ``theta``.

    Notes
    -----
    Information adds across items only under local independence. When items
    share a stem or a context, the sum overstates the precision of the test.
    """
    theta_arr = np.atleast_1d(np.asarray(theta, dtype=float))
    total = np.zeros_like(theta_arr, dtype=float)
    for a_j, b_j in zip(a, b, strict=True):
        total += item_information(theta_arr, a_j, b_j)
    return total


def standard_error(information: np.ndarray) -> np.ndarray:
    """Standard error of a maximum likelihood trait estimate.

    Parameters
    ----------
    information : numpy.ndarray
        Test information at one or more trait values.

    Returns
    -------
    numpy.ndarray
        ``1 / sqrt(I)``, with ``inf`` where information is zero.

    Notes
    -----
    This is the asymptotic standard error of the maximum likelihood estimate.
    An expected a posteriori score borrows from the prior and has a smaller
    posterior standard deviation, which is not the same quantity; the two
    should not be swapped in a report.
    """
    info = np.asarray(information, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(info > 0, 1.0 / np.sqrt(np.where(info > 0, info, 1.0)), np.inf)


def expected_score(theta: np.ndarray | float, a: float, b: Sequence[float]) -> np.ndarray:
    """Expected raw category score of an item.

    Parameters
    ----------
    theta : float or numpy.ndarray
        Latent trait values.
    a : float
        Discrimination.
    b : sequence of float
        Thresholds.

    Returns
    -------
    numpy.ndarray
        ``sum_k k P_k(theta)`` with categories coded ``0 ... K - 1``.
    """
    probs = category_probabilities(theta, a, b)
    return probs @ np.arange(probs.shape[1], dtype=float)


def simulate_grm(
    n_persons: int,
    a: Sequence[float],
    b: Sequence[Sequence[float]],
    seed: int | None = None,
    theta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate responses from a graded response model.

    Parameters
    ----------
    n_persons : int
        Number of respondents.
    a : sequence of float
        One discrimination per item.
    b : sequence of sequence of float
        One increasing threshold vector per item.
    seed : int, optional
        Seed for the random generator.
    theta : numpy.ndarray, optional
        Trait values to use. Drawn from ``N(0, 1)`` when omitted.

    Returns
    -------
    responses : numpy.ndarray
        Integer array of shape ``(n_persons, n_items)``, coded ``0 ... K - 1``.
    theta : numpy.ndarray
        The trait values used.

    Notes
    -----
    Synthetic data only. Nothing generated here describes any person.
    """
    rng = np.random.default_rng(seed)
    if theta is None:
        theta = rng.normal(size=int(n_persons))
    theta = np.asarray(theta, dtype=float)
    responses = np.empty((theta.size, len(list(a))), dtype=int)
    for j, (a_j, b_j) in enumerate(zip(a, b, strict=True)):
        probs = category_probabilities(theta, a_j, b_j)
        cumulative = np.cumsum(probs, axis=1)
        draws = rng.random(theta.size)[:, None]
        responses[:, j] = (draws > cumulative).sum(axis=1)
    return responses, theta


@dataclass
class GRMItem:
    """Estimated parameters of one item.

    Attributes
    ----------
    name : str
        Item label.
    a : float
        Discrimination.
    b : numpy.ndarray
        Thresholds ``b_2 < ... < b_K``.
    n_categories : int
        ``K``, one more than the number of thresholds.
    """

    name: str
    a: float
    b: np.ndarray
    n_categories: int

    def probabilities(self, theta: np.ndarray | float) -> np.ndarray:
        """Category probabilities at ``theta``. See :func:`category_probabilities`."""
        return category_probabilities(theta, self.a, self.b)

    def information(self, theta: np.ndarray | float) -> np.ndarray:
        """Item information at ``theta``. See :func:`item_information`."""
        return item_information(theta, self.a, self.b)


@dataclass
class GradedResponseModel:
    """Fit Samejima's graded response model by marginal maximum likelihood.

    Parameters
    ----------
    n_quadrature : int, default 41
        Gauss-Hermite nodes. Below about 21 the marginal likelihood is
        noticeably biased; above about 61 nothing changes and the fit is
        slower.
    max_em_iter : int, default 500
        Largest number of EM iterations.
    tol : float, default 1e-5
        Convergence tolerance on the change in marginal log-likelihood.
    max_a : float, default 6.0
        Upper bound on discrimination. Items answered almost deterministically
        push ``a`` towards infinity; the bound keeps the fit finite and its
        being hit is reported in :attr:`boundary_items`.

    Attributes
    ----------
    items : list of GRMItem
        Estimated parameters, filled by :meth:`fit`.
    loglik : float
        Marginal log-likelihood at the solution.
    n_iter : int
        EM iterations used.
    converged : bool
        Whether the tolerance was met before ``max_em_iter``.
    boundary_items : list of str
        Items whose discrimination hit ``max_a``.

    Examples
    --------
    >>> import numpy as np
    >>> a = [1.4, 1.1, 1.7]
    >>> b = [[-1.0, 0.2, 1.1], [-1.4, 0.0, 1.3], [-0.8, 0.4, 1.5]]
    >>> x, _ = simulate_grm(400, a, b, seed=1)
    >>> model = GradedResponseModel().fit(x)
    >>> model.converged
    True

    Limitations
    -----------
    EM finds a local maximum. Sparse categories, small samples and items with
    little discrimination can leave thresholds poorly determined; collapse
    categories with very few responses before fitting rather than trusting a
    threshold estimated from a handful of people.
    """

    n_quadrature: int = 41
    max_em_iter: int = 500
    tol: float = 1e-5
    max_a: float = 6.0

    items: list[GRMItem] = field(default_factory=list)
    loglik: float = float("nan")
    n_iter: int = 0
    converged: bool = False
    boundary_items: list[str] = field(default_factory=list)
    item_names: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ fit
    def fit(self, data: np.ndarray | pd.DataFrame) -> GradedResponseModel:
        """Estimate item parameters from response data.

        Parameters
        ----------
        data : numpy.ndarray or pandas.DataFrame
            Persons by items, coded ``0 ... K_j - 1`` per item. Missing values
            may be ``NaN``; they are skipped person by person, which assumes
            they are missing at random given the responses observed.

        Returns
        -------
        GradedResponseModel
            The fitted model, for chaining.

        Raises
        ------
        ValueError
            If an item has fewer than two observed categories, or if the
            observed codes of an item are not the consecutive integers
            ``0 ... K - 1``.
        """
        x, names = _as_response_matrix(data)
        n_cats = _category_counts(x, names)
        self.item_names = names
        nodes, weights = _gauss_hermite_normal(self.n_quadrature)

        params = [_starting_values(x[:, j], n_cats[j]) for j in range(x.shape[1])]
        loglik_old = -np.inf
        self.converged = False
        for iteration in range(1, self.max_em_iter + 1):
            log_lik_nodes = _log_likelihood_nodes(x, params, nodes)
            marginal, posterior = _posterior(log_lik_nodes, weights)
            loglik = float(marginal.sum())
            counts = _expected_counts(x, posterior, n_cats)
            params = [
                _maximise_item(counts[j], nodes, params[j], self.max_a) for j in range(x.shape[1])
            ]
            self.n_iter = iteration
            if np.isfinite(loglik_old) and abs(loglik - loglik_old) < self.tol:
                self.converged = True
                loglik_old = loglik
                break
            loglik_old = loglik

        log_lik_nodes = _log_likelihood_nodes(x, params, nodes)
        marginal, _ = _posterior(log_lik_nodes, weights)
        self.loglik = float(marginal.sum())
        self.items = [
            GRMItem(
                name=names[j],
                a=float(p[0]),
                b=np.asarray(p[1], dtype=float),
                n_categories=n_cats[j],
            )
            for j, p in enumerate(params)
        ]
        self.boundary_items = [it.name for it in self.items if it.a >= self.max_a - 1e-6]
        return self

    # -------------------------------------------------------------- outputs
    @property
    def a(self) -> np.ndarray:
        """Discriminations, one per item."""
        return np.array([it.a for it in self.items], dtype=float)

    @property
    def b(self) -> list[np.ndarray]:
        """Threshold vectors, one per item."""
        return [it.b for it in self.items]

    def parameter_frame(self) -> pd.DataFrame:
        """Estimated parameters as a table.

        Returns
        -------
        pandas.DataFrame
            Indexed by item, with ``a`` and one column per threshold, named
            ``b2``, ``b3``, ... following the module convention. Items with
            fewer categories leave later columns empty.
        """
        rows = []
        for it in self.items:
            row: dict[str, object] = {"item": it.name, "a": it.a, "n_categories": it.n_categories}
            for k, value in enumerate(it.b, start=2):
                row[f"b{k}"] = float(value)
            rows.append(row)
        return pd.DataFrame(rows).set_index("item")

    def information(self, theta: np.ndarray | float) -> np.ndarray:
        """Test information at ``theta``. See :func:`test_information`."""
        return test_information(theta, self.a, self.b)

    def standard_error(self, theta: np.ndarray | float) -> np.ndarray:
        """Standard error of measurement at ``theta``. See :func:`standard_error`."""
        return standard_error(self.information(theta))

    def score(self, data: np.ndarray | pd.DataFrame) -> pd.DataFrame:
        """Expected a posteriori trait scores.

        Parameters
        ----------
        data : numpy.ndarray or pandas.DataFrame
            Persons by items, coded as for :meth:`fit`.

        Returns
        -------
        pandas.DataFrame
            ``theta_eap`` and ``se_eap``, the posterior mean and standard
            deviation over the quadrature grid.

        Notes
        -----
        EAP scores shrink towards the prior mean. That is what makes them
        stable, and it also means group differences computed from EAP scores
        are attenuated; use plausible values or a latent regression when the
        target is a group comparison.
        """
        if not self.items:
            raise RuntimeError("fit the model before scoring")
        x, _ = _as_response_matrix(data)
        nodes, weights = _gauss_hermite_normal(self.n_quadrature)
        params = [(it.a, it.b) for it in self.items]
        log_lik_nodes = _log_likelihood_nodes(x, params, nodes)
        _, posterior = _posterior(log_lik_nodes, weights)
        mean = posterior @ nodes
        var = posterior @ (nodes**2) - mean**2
        return pd.DataFrame({"theta_eap": mean, "se_eap": np.sqrt(np.clip(var, 0, None))})


# --------------------------------------------------------------------------
# estimation internals
# --------------------------------------------------------------------------
def _as_response_matrix(data: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return responses as a float matrix plus item names."""
    if isinstance(data, pd.DataFrame):
        names = [str(c) for c in data.columns]
        values = data.to_numpy(dtype=float)
    else:
        values = np.asarray(data, dtype=float)
        names = [f"item{j + 1}" for j in range(values.shape[1])]
    if values.ndim != 2:
        raise ValueError("data must be two dimensional (persons by items)")
    return values, names


def _category_counts(x: np.ndarray, names: Sequence[str]) -> list[int]:
    """Number of categories per item, checking the coding."""
    counts = []
    for j, name in enumerate(names):
        col = x[:, j]
        observed = np.unique(col[~np.isnan(col)])
        if observed.size < 2:
            raise ValueError(f"item {name!r} has fewer than two observed categories")
        if not np.allclose(observed, np.round(observed)):
            raise ValueError(f"item {name!r} has non-integer response codes")
        expected = np.arange(observed.max() + 1)
        if not np.array_equal(observed.astype(int), expected):
            raise ValueError(
                f"item {name!r} must be coded 0..K-1 with no empty category; observed "
                f"{observed.astype(int).tolist()}. Collapse sparse categories first."
            )
        counts.append(int(observed.max()) + 1)
    return counts


def _gauss_hermite_normal(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature nodes and weights for a standard normal ability density."""
    x, w = np.polynomial.hermite.hermgauss(int(n_points))
    return np.sqrt(2.0) * x, w / np.sqrt(np.pi)


def _starting_values(column: np.ndarray, n_cats: int) -> tuple[float, np.ndarray]:
    """Crude but ordered starting values from the observed frequencies."""
    obs = column[~np.isnan(column)].astype(int)
    counts = np.bincount(obs, minlength=n_cats).astype(float)
    cumulative = np.cumsum(counts[::-1])[::-1] / counts.sum()  # P(Y >= k)
    p_star = np.clip(cumulative[1:], 0.02, 0.98)
    b = np.log(p_star / (1.0 - p_star)) * -1.0
    b = np.sort(b)
    gaps = np.diff(b)
    if gaps.size and gaps.min() < 1e-3:
        b = b[0] + np.cumsum(np.concatenate([[0.0], np.maximum(gaps, 0.05)]))
    return 1.0, b


def _pack(a: float, b: np.ndarray) -> np.ndarray:
    """Parameters on an unconstrained scale: ``log a``, ``b_2``, ``log gaps``."""
    gaps = np.diff(b)
    return np.concatenate([[np.log(a)], [b[0]], np.log(np.clip(gaps, 1e-6, None))])


def _unpack(vec: np.ndarray) -> tuple[float, np.ndarray]:
    """Inverse of :func:`_pack`, guaranteeing ``a > 0`` and ordered thresholds."""
    a = float(np.exp(vec[0]))
    b = np.concatenate([[vec[1]], vec[1] + np.cumsum(np.exp(vec[2:]))])
    return a, b


def _log_likelihood_nodes(
    x: np.ndarray,
    params: Sequence[tuple[float, np.ndarray]],
    nodes: np.ndarray,
) -> np.ndarray:
    """Log-likelihood of every person at every quadrature node."""
    n_persons = x.shape[0]
    out = np.zeros((n_persons, nodes.size))
    for j, (a_j, b_j) in enumerate(params):
        probs = np.clip(category_probabilities(nodes, a_j, b_j), 1e-300, None)  # (Q, K)
        col = x[:, j]
        observed = ~np.isnan(col)
        codes = np.zeros(n_persons, dtype=int)
        codes[observed] = col[observed].astype(int)
        contribution = np.log(probs.T)[codes]  # (n_persons, Q)
        out[observed] += contribution[observed]
    return out


def _posterior(log_lik_nodes: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Marginal log-likelihood per person and the posterior over nodes."""
    shifted = log_lik_nodes - log_lik_nodes.max(axis=1, keepdims=True)
    unnormalised = np.exp(shifted) * weights[None, :]
    total = unnormalised.sum(axis=1)
    marginal = np.log(total) + log_lik_nodes.max(axis=1)
    return marginal, unnormalised / total[:, None]


def _expected_counts(
    x: np.ndarray, posterior: np.ndarray, n_cats: Sequence[int]
) -> list[np.ndarray]:
    """Expected number of people at each node giving each category."""
    counts = []
    for j, k_j in enumerate(n_cats):
        col = x[:, j]
        observed = ~np.isnan(col)
        table = np.zeros((posterior.shape[1], k_j))
        codes = col[observed].astype(int)
        post = posterior[observed]
        for k in range(k_j):
            mask = codes == k
            if mask.any():
                table[:, k] = post[mask].sum(axis=0)
        counts.append(table)
    return counts


def _maximise_item(
    counts: np.ndarray,
    nodes: np.ndarray,
    start: tuple[float, np.ndarray],
    max_a: float,
) -> tuple[float, np.ndarray]:
    """M step for one item: maximise the expected complete-data log-likelihood."""
    x0 = _pack(start[0], np.asarray(start[1], dtype=float))

    def negative_loglik(vec: np.ndarray) -> float:
        a, b = _unpack(vec)
        a = min(a, max_a)
        probs = np.clip(category_probabilities(nodes, a, b), 1e-12, None)
        return float(-np.sum(counts * np.log(probs)))

    bounds = (
        [(np.log(0.05), np.log(max_a))]
        + [(-10.0, 10.0)]
        + [(np.log(1e-3), np.log(20.0))] * (x0.size - 2)
    )
    result = optimize.minimize(
        negative_loglik,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 400, "ftol": 1e-10},
    )
    a, b = _unpack(result.x if result.success or np.isfinite(result.fun) else x0)
    return min(a, max_a), b
