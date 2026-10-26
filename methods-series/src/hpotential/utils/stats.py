"""Summary statistics used across the analyses.

Small, explicit and seeded. Every function that draws random numbers takes a
generator; none of them touches the global numpy random state.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "binned_conditional_mean",
    "bootstrap_ci",
    "credible_interval",
    "relative_bias",
    "summarise",
]


def credible_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Equal-tailed interval of a sample.

    Parameters
    ----------
    values : array_like
        Draws.
    level : float, default 0.95
        Coverage, strictly between 0 and 1.

    Returns
    -------
    lower, upper : float

    Examples
    --------
    >>> lo, hi = credible_interval(np.arange(101.0))
    >>> round(lo, 1), round(hi, 1)
    (2.5, 97.5)
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    tail = (1.0 - level) / 2.0
    v = np.asarray(values, dtype=float)
    return float(np.percentile(v, 100 * tail)), float(np.percentile(v, 100 * (1 - tail)))


def binned_conditional_mean(
    x: np.ndarray, y: np.ndarray, n_bins: int = 120, smooth: bool = True
) -> np.ndarray:
    """``E[y | x]`` by equal-count binning, optionally smoothed.

    Parameters
    ----------
    x : array_like
        Conditioning variable.
    y : array_like
        Response.
    n_bins : int, default 120
        Number of equal-count bins. Fewer bins mean more smoothing and more
        bias; more bins mean a noisier curve. The value used in the published
        value-of-information analysis is 120 on 100000 draws.
    smooth : bool, default True
        Apply a ``(0.25, 0.5, 0.25)`` kernel to the bin means.

    Returns
    -------
    numpy.ndarray
        Fitted values at each ``x``, in the original order.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if int(n_bins) < 1:
        raise ValueError("n_bins must be at least 1")
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    edges = np.linspace(0, xs.size, int(n_bins) + 1).astype(int)
    centres = np.array([xs[a:b].mean() for a, b in zip(edges[:-1], edges[1:], strict=False)])
    means = np.array([ys[a:b].mean() for a, b in zip(edges[:-1], edges[1:], strict=False)])
    if smooth and means.size >= 3:
        kernel = np.array([0.25, 0.5, 0.25])
        means = np.convolve(np.pad(means, 1, mode="edge"), kernel, mode="valid")
    out = np.empty_like(x)
    out[order] = np.interp(xs, centres, means)
    return out


def bootstrap_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    statistic=np.mean,
    n_boot: int = 2000,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a statistic.

    Parameters
    ----------
    values : array_like
        Sample.
    rng : numpy.random.Generator
        Explicit generator. The interval is reproducible for a given generator
        state.
    statistic : callable, default numpy.mean
        Applied to each resample.
    n_boot : int, default 2000
        Resamples.
    level : float, default 0.95
        Coverage.

    Returns
    -------
    lower, upper : float

    Notes
    -----
    The percentile bootstrap, not the bias-corrected accelerated one. It is
    adequate for the near-symmetric statistics used here and is stated plainly
    rather than implied; for a skewed statistic prefer BCa.
    """
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        raise ValueError("need at least two observations")
    idx = rng.integers(0, v.size, size=(int(n_boot), v.size))
    draws = np.array([statistic(v[row]) for row in idx])
    return credible_interval(draws, level)


def relative_bias(estimates: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Mean relative bias of an estimator, one entry per parameter.

    Parameters
    ----------
    estimates : array_like
        Shape ``(n_replicates, n_parameters)``.
    truth : array_like
        Shape ``(n_parameters,)``, with no zero entries.

    Returns
    -------
    numpy.ndarray
        ``(mean(estimate) - truth) / truth``.
    """
    est = np.atleast_2d(np.asarray(estimates, dtype=float))
    true = np.asarray(truth, dtype=float).ravel()
    if est.shape[1] != true.size:
        raise ValueError("estimates must have one column per parameter")
    if np.any(true == 0):
        raise ValueError("relative bias is undefined at a true value of zero")
    return (est.mean(axis=0) - true) / true


def summarise(values: np.ndarray, level: float = 0.95) -> dict[str, float]:
    """Mean, standard deviation, median and an equal-tailed interval.

    Parameters
    ----------
    values : array_like
        Sample.
    level : float, default 0.95
        Coverage of the reported interval.

    Returns
    -------
    dict
        ``mean``, ``sd``, ``median``, ``lower``, ``upper``, ``n``.
    """
    v = np.asarray(values, dtype=float)
    lower, upper = credible_interval(v, level)
    return {
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
        "median": float(np.median(v)),
        "lower": lower,
        "upper": upper,
        "n": int(v.size),
    }
