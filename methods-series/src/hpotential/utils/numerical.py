"""Small numerical helpers used by more than one analysis.

Nothing here is specific to the H-potential. The functions exist so that the
convergence studies in the dynamical paper and the checks in the tests measure
orders the same way rather than each fitting its own line.
"""

from __future__ import annotations

import numpy as np

__all__ = ["convergence_order", "pairwise_orders", "richardson", "safe_cholesky"]


def convergence_order(steps: np.ndarray, errors: np.ndarray) -> float:
    """Observed order of accuracy, as the slope of ``log e`` on ``log h``.

    Parameters
    ----------
    steps : array_like
        Step sizes or cell widths, strictly positive.
    errors : array_like
        Errors at those steps, strictly positive.

    Returns
    -------
    float
        The fitted slope.

    Notes
    -----
    A least-squares slope over the whole grid, not a single ratio, so one noisy
    point does not set the answer. It assumes the errors are in the asymptotic
    regime: include a step so coarse that the leading term does not dominate, or
    so fine that rounding does, and the fit will say so by drifting away from
    the theoretical order.

    Examples
    --------
    >>> h = np.array([0.4, 0.2, 0.1, 0.05])
    >>> round(convergence_order(h, 3.0 * h**2), 6)
    2.0
    """
    h = np.asarray(steps, dtype=float)
    e = np.asarray(errors, dtype=float)
    if h.shape != e.shape:
        raise ValueError("steps and errors must have the same shape")
    if h.size < 2:
        raise ValueError("need at least two points to fit an order")
    if np.any(h <= 0) or np.any(e <= 0):
        raise ValueError("steps and errors must be positive to take logarithms")
    return float(np.polyfit(np.log(h), np.log(e), 1)[0])


def pairwise_orders(steps: np.ndarray, errors: np.ndarray) -> np.ndarray:
    """Order estimated from each consecutive pair of refinements.

    Parameters
    ----------
    steps, errors : array_like
        As for :func:`convergence_order`.

    Returns
    -------
    numpy.ndarray
        One estimate per pair, ``log(e_i / e_{i+1}) / log(h_i / h_{i+1})``.
        Useful for seeing whether the order is settling or drifting.
    """
    h = np.asarray(steps, dtype=float)
    e = np.asarray(errors, dtype=float)
    if h.size < 2:
        raise ValueError("need at least two points")
    return np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])


def richardson(coarse: float, fine: float, ratio: float, order: int) -> float:
    """Richardson extrapolation of two solutions to a finer answer.

    Parameters
    ----------
    coarse : float
        Solution at step ``h``.
    fine : float
        Solution at step ``h / ratio``.
    ratio : float
        Refinement ratio, greater than one.
    order : int
        Expected order of accuracy of the scheme.

    Returns
    -------
    float
        ``fine + (fine - coarse) / (ratio^order - 1)``.

    Notes
    -----
    Valid only when both solutions are in the asymptotic regime and the scheme
    really has the stated order. Extrapolating a scheme whose order is wrong
    produces a confident wrong answer, so check the order first with
    :func:`convergence_order`.

    Examples
    --------
    >>> round(richardson(1.04, 1.01, 2.0, 2), 6)
    1.0
    """
    if ratio <= 1:
        raise ValueError("ratio must exceed one")
    return float(fine + (fine - coarse) / (ratio**order - 1.0))


def safe_cholesky(matrix: np.ndarray, jitter: float = 1e-10, max_tries: int = 8) -> np.ndarray:
    """Cholesky factor, adding a small ridge if the matrix is barely indefinite.

    Parameters
    ----------
    matrix : array_like
        Symmetric matrix.
    jitter : float, default 1e-10
        First ridge tried, scaled by the mean diagonal. Each retry multiplies it
        by ten.
    max_tries : int, default 8
        Number of ridges tried before giving up.

    Returns
    -------
    numpy.ndarray
        Lower-triangular factor ``L`` with ``L L' ~= matrix``.

    Raises
    ------
    numpy.linalg.LinAlgError
        If no ridge in the sequence makes the matrix factorisable. A matrix that
        needs a ridge of ``1e-3`` is not nearly positive definite, and the
        caller should be told rather than handed a factor of something else.

    Notes
    -----
    The ridge is reported nowhere, so use this only where a covariance is
    theoretically positive definite and the failure is rounding. Do not use it
    to rescue an inadmissible model fit; the inadmissibility is the result.
    """
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")
    a = 0.5 * (a + a.T)
    try:
        return np.linalg.cholesky(a)
    except np.linalg.LinAlgError:
        pass
    scale = float(np.mean(np.diag(a)))
    ridge = jitter * (scale if scale > 0 else 1.0)
    eye = np.eye(a.shape[0])
    for _ in range(int(max_tries)):
        try:
            return np.linalg.cholesky(a + ridge * eye)
        except np.linalg.LinAlgError:
            ridge *= 10.0
    raise np.linalg.LinAlgError("matrix is not close to positive definite")
