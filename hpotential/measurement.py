"""Gaussian posterior scoring for an externally specified measurement model.

This is not CFA estimation or a NHATS analysis: loadings, latent covariance
and residual variances are inputs. No outcome variables are accepted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["GaussianMeasurement", "anchored"]

FloatArray = NDArray[np.float64]


class GaussianMeasurement:
    """Linear factor model ``x = intercept + loadings @ eta + e``.

    ``eta ~ N(0, phi)`` and ``e ~ N(0, diag(uniqueness))``. With every
    parameter fixed in advance, the posterior of ``eta`` given the observed
    indicators is Gaussian and computed exactly.

    Parameters
    ----------
    names:
        Indicator names, one per row of ``loadings``. Scoring input must use
        exactly these names.
    loadings:
        ``(p, k)`` matrix for ``p`` indicators and ``k`` latent domains.
    phi:
        ``(k, k)`` positive-definite latent covariance.
    uniqueness:
        ``(p,)`` positive residual variances.
    intercept:
        ``(p,)`` indicator intercepts; zero if omitted.
    """

    def __init__(
        self,
        names: Sequence[str],
        loadings: ArrayLike,
        phi: ArrayLike,
        uniqueness: ArrayLike,
        intercept: ArrayLike | None = None,
    ) -> None:
        self.names = tuple(names)
        self.loadings = np.asarray(loadings, dtype=float)
        self.phi = np.asarray(phi, dtype=float)
        self.uniqueness = np.asarray(uniqueness, dtype=float)
        self.intercept = (
            np.zeros(len(self.names)) if intercept is None else np.asarray(intercept, dtype=float)
        )

        if self.loadings.ndim != 2:
            raise ValueError("loadings must be a (p, k) matrix")
        p, k = self.loadings.shape
        if p != len(self.names) or len(set(self.names)) != p:
            raise ValueError("Need one unique name per loading row")
        if self.phi.shape != (k, k):
            raise ValueError(f"phi must be ({k}, {k}), got {self.phi.shape}")
        if self.uniqueness.shape != (p,) or self.intercept.shape != (p,):
            raise ValueError(f"uniqueness and intercept must have shape ({p},)")
        for label, arr in (
            ("loadings", self.loadings),
            ("phi", self.phi),
            ("uniqueness", self.uniqueness),
            ("intercept", self.intercept),
        ):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{label} contains nonfinite values")
        if not np.allclose(self.phi, self.phi.T) or np.any(np.linalg.eigvalsh(self.phi) <= 0):
            raise ValueError("phi must be symmetric positive definite")
        if np.any(self.uniqueness <= 0):
            raise ValueError("Residual variances must be positive")

    @property
    def n_domains(self) -> int:
        return self.phi.shape[0]

    def _posterior(self, observed: NDArray[np.bool_]) -> tuple[FloatArray, FloatArray]:
        """Gain matrix and posterior covariance for one missingness pattern."""
        lam = self.loadings[observed]
        lam_phi = lam @ self.phi
        sigma = lam_phi @ lam.T + np.diag(self.uniqueness[observed])
        # Solve rather than invert the observed covariance.
        gain = np.linalg.solve(sigma, lam_phi).T
        cov = self.phi - gain @ lam_phi
        return gain, (cov + cov.T) / 2

    def score(self, indicators: Mapping[str, ArrayLike]) -> tuple[FloatArray, FloatArray]:
        """Posterior means ``(n, k)`` and covariances ``(n, k, k)``.

        ``NaN`` marks a missing indicator. A row with no observed indicator
        gets the prior mean (zero) and prior covariance ``phi``.

        The mapping must contain exactly the model's indicator names, so an
        accidental outcome column fails instead of entering the model.
        """
        if set(indicators) != set(self.names):
            extra = sorted(set(indicators) - set(self.names))
            missing = sorted(set(self.names) - set(indicators))
            raise ValueError(
                "Input columns must equal the frozen indicators "
                f"(extra: {extra}, missing: {missing})"
            )
        x = np.column_stack([np.asarray(indicators[n], dtype=float) for n in self.names])
        if np.any(np.isinf(x)):
            raise ValueError("Infinite indicator values")

        n, k = x.shape[0], self.n_domains
        means = np.zeros((n, k))
        covariances = np.broadcast_to(self.phi, (n, k, k)).copy()
        observed = ~np.isnan(x)
        # Rows sharing a missingness pattern share one gain matrix.
        patterns, inverse = np.unique(observed, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        for j, pattern in enumerate(patterns):
            if not pattern.any():
                continue
            rows = inverse == j
            gain, cov = self._posterior(pattern)
            centred = x[np.ix_(rows, pattern)] - self.intercept[pattern]
            means[rows] = centred @ gain.T
            covariances[rows] = cov
        return means, covariances

    def implied_determinacy(self) -> FloatArray:
        """Model-implied correlation between each domain and its complete-data score.

        A property of the assumed parameters, not empirical validation.
        """
        _, cov = self._posterior(np.ones(len(self.names), dtype=bool))
        return np.sqrt(np.maximum(0.0, 1 - np.diag(cov) / np.diag(self.phi)))


def anchored(
    value: ArrayLike, lower: float, upper: float, higher_is_better: bool = True
) -> FloatArray:
    """Map ``[lower, upper]`` to ``[0, 1]`` with anchors fixed in advance.

    Values outside the anchors raise instead of being extrapolated or clipped.
    ``NaN`` (missing) passes through unchanged.
    """
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("Anchors must be finite with lower < upper")
    x = np.asarray(value, dtype=float)
    if np.any(np.isinf(x)) or np.any((x < lower) | (x > upper)):
        raise ValueError("Values outside the frozen anchors")
    z = (x - lower) / (upper - lower)
    return z if higher_is_better else 1 - z
