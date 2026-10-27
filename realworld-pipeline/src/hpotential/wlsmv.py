"""WLSMV: diagonally weighted least squares on polychoric and polyserial correlations.

The ML fit in ``cfa`` treats every indicator as interval-scaled. WLSMV treats
an ordinal indicator as a coarsened normal variable with thresholds, which is
the estimator Chapter 5 specifies as the check on ML. The steps are:

1. Classify each indicator: ordinal if it has at most ``max_categories``
   distinct (rounded) values, otherwise continuous.
2. Estimate thresholds from the marginal proportions, then the correlation of
   every pair: polychoric for two ordinal indicators (maximum likelihood with
   thresholds fixed), polyserial for ordinal with continuous (the two-step
   estimator of Olsson, Drasgow and Dorans 1982), Pearson otherwise.
3. Estimate the asymptotic covariance Gamma of the correlations. This module
   uses the bootstrap; Mplus and lavaan use the analytic estimating-equation
   form. Both are consistent; the bootstrap is simpler to verify.
4. Fit the correlation structure Lambda Phi Lambda' (unit diagonal) by
   minimising (s - sigma)' W (s - sigma), W = diag(Gamma)^-1.
5. Robust (sandwich) standard errors, and the scaled-and-shifted chi-square
   of Asparouhov and Muthen (2010) for the model and the independence
   baseline, from which CFI, TLI and RMSEA follow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm

_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)
_BIG = 8.0  # thresholds beyond +/- 8 SD are treated as infinite


# ----------------------------------------------------------- bivariate normal


def bvn_cdf(h, k, rho):
    """P(X <= h, Y <= k) for standard bivariate normal with correlation rho.

    Uses Plackett's identity, d Phi2 / d rho = phi2(h, k; rho), integrated
    from 0 to rho by Gauss-Legendre quadrature. Vectorised over h and k.
    """
    h = np.clip(np.asarray(h, float), -_BIG, _BIG)
    k = np.clip(np.asarray(k, float), -_BIG, _BIG)
    base = norm.cdf(h) * norm.cdf(k)
    if rho == 0:
        return base
    r = 0.5 * rho * (_GL_X + 1)  # nodes on [0, rho]
    w = 0.5 * rho * _GL_W
    hh, kk = h[..., None], k[..., None]
    one = 1 - r**2
    dens = np.exp(-(hh**2 - 2 * r * hh * kk + kk**2) / (2 * one)) / (2 * np.pi * np.sqrt(one))
    return base + (dens * w).sum(axis=-1)


# --------------------------------------------------------------- correlations


def _weights(n, w):
    w = np.ones(n) if w is None else np.asarray(w, float)
    return w / w.sum()


def thresholds(x, w=None):
    """Thresholds of an ordinal variable coded as integers (with -inf, +inf ends)."""
    w = _weights(len(x), w)
    cats = np.unique(x)
    cum = np.cumsum([w[x == c].sum() for c in cats])[:-1]
    tau = norm.ppf(np.clip(cum, 1e-6, 1 - 1e-6))
    return cats, np.concatenate([[-np.inf], tau, [np.inf]])


def polychoric(x, y, w=None) -> float:
    """Two-step polychoric correlation (thresholds from the margins)."""
    w = _weights(len(x), w)
    cx, tx = thresholds(x, w)
    cy, ty = thresholds(y, w)
    ix = np.searchsorted(cx, x)
    iy = np.searchsorted(cy, y)
    counts = np.zeros((len(cx), len(cy)))
    np.add.at(counts, (ix, iy), w)
    a_lo, a_hi = tx[:-1][:, None], tx[1:][:, None]
    b_lo, b_hi = ty[:-1][None, :], ty[1:][None, :]

    def nll(rho):
        p = (
            bvn_cdf(a_hi, b_hi, rho)
            - bvn_cdf(a_lo, b_hi, rho)
            - bvn_cdf(a_hi, b_lo, rho)
            + bvn_cdf(a_lo, b_lo, rho)
        )
        return -(counts * np.log(np.clip(p, 1e-12, None))).sum()

    return float(minimize_scalar(nll, bounds=(-0.995, 0.995), method="bounded").x)


def polyserial(x_ord, y, w=None) -> float:
    """Two-step polyserial correlation: r_xy * sd(x) / sum phi(tau)."""
    w = _weights(len(x_ord), w)
    _, tau = thresholds(x_ord, w)
    mx, my = w @ x_ord, w @ y
    sx = np.sqrt(w @ (x_ord - mx) ** 2)
    sy = np.sqrt(w @ (y - my) ** 2)
    r = (w @ ((x_ord - mx) * (y - my))) / (sx * sy)
    return float(np.clip(r * sx / norm.pdf(tau[1:-1]).sum(), -0.995, 0.995))


def pearson(x, y, w=None) -> float:
    w = _weights(len(x), w)
    mx, my = w @ x, w @ y
    return float((w @ ((x - mx) * (y - my))) / np.sqrt((w @ (x - mx) ** 2) * (w @ (y - my) ** 2)))


def classify(X, max_categories: int = 7) -> np.ndarray:
    """True for ordinal columns: at most ``max_categories`` distinct observed values.

    Classify on the observed data; imputed values are fractional and would
    make every column look continuous. Pass the result to ``fit_wlsmv``.
    """
    X = np.asarray(X, float)
    return np.array(
        [len(np.unique(X[~np.isnan(X[:, j]), j])) <= max_categories for j in range(X.shape[1])]
    )


def correlations(X, ordinal, w=None) -> np.ndarray:
    """The p(p-1)/2 correlations (upper triangle, row order) of mixed data."""
    X = np.asarray(X, float)
    p = X.shape[1]
    Z = X.copy()
    Z[:, ordinal] = np.round(Z[:, ordinal])
    out = []
    for i in range(p):
        for j in range(i + 1, p):
            if ordinal[i] and ordinal[j]:
                out.append(polychoric(Z[:, i], Z[:, j], w))
            elif ordinal[i]:
                out.append(polyserial(Z[:, i], Z[:, j], w))
            elif ordinal[j]:
                out.append(polyserial(Z[:, j], Z[:, i], w))
            else:
                out.append(pearson(Z[:, i], Z[:, j], w))
    return np.array(out)


# -------------------------------------------------------------------- the fit


@dataclass
class WLSMVResult:
    factors: list
    assign: np.ndarray
    ordinal: np.ndarray
    n: int
    loadings: np.ndarray
    loadings_se: np.ndarray
    factor_corr: np.ndarray
    chi2: float
    df: int
    chi2_base: float
    df_base: int
    srmr: float

    @property
    def cfi(self):
        d = max(self.chi2 - self.df, 0)
        d0 = max(self.chi2_base - self.df_base, d, 0)
        return float(1 - d / d0) if d0 > 0 else 1.0

    @property
    def tli(self):
        r0 = self.chi2_base / self.df_base
        return float((r0 - self.chi2 / self.df) / (r0 - 1))

    @property
    def rmsea(self):
        return float(np.sqrt(max(self.chi2 - self.df, 0) / (self.df * (self.n - 1))))

    def fit_indices(self):
        return {
            "chi2": self.chi2,
            "df": self.df,
            "cfi": self.cfi,
            "tli": self.tli,
            "rmsea": self.rmsea,
            "srmr": self.srmr,
        }


def _scaled_shifted(T, U, Gamma, d):
    """Asparouhov and Muthen (2010) scaled-and-shifted statistic."""
    UG = U @ Gamma
    tr1, tr2 = np.trace(UG), np.trace(UG @ UG)
    a = np.sqrt(d / tr2)
    return float(a * T + d - a * tr1)


def fit_wlsmv(
    X, pattern: dict, w=None, reps: int = 200, seed: int = 0, max_categories=7, ordinal=None
):
    """Fit the correlated-factor model by WLSMV. ``pattern`` maps factor to columns.

    ``ordinal`` marks the ordinal columns; by default they are those with at most
    ``max_categories`` distinct values. Ordinal columns are rounded to integers.
    """
    X = np.asarray(X, float)
    n, p = X.shape
    factors = list(pattern)
    k = len(factors)
    assign = np.empty(p, int)
    for fi, f in enumerate(factors):
        assign[list(pattern[f])] = fi
    ordinal = classify(X, max_categories) if ordinal is None else np.asarray(ordinal, bool)
    s = correlations(X, ordinal, w)
    rng = np.random.default_rng(seed)
    wv = np.ones(n) if w is None else np.asarray(w, float)
    boots = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        try:
            boots.append(correlations(X[idx], ordinal, wv[idx]))
        except (ValueError, FloatingPointError):
            continue
    Gamma = n * np.cov(np.array(boots), rowvar=False)
    W = np.diag(1 / np.clip(np.diag(Gamma), 1e-10, None))
    iu = np.triu_indices(p, 1)
    ic = np.triu_indices(k, 1)

    def sigma(theta):
        L = np.zeros((p, k))
        L[np.arange(p), assign] = theta[:p]
        C = np.eye(k)
        C[ic] = np.tanh(theta[p:])
        C[(ic[1], ic[0])] = C[ic]
        return (L @ C @ L.T)[iu]

    def F(theta):
        r = s - sigma(theta)
        return r @ W @ r

    theta0 = np.r_[np.full(p, 0.6), np.full(len(ic[0]), np.arctanh(0.3))]
    res = minimize(F, theta0, method="BFGS")
    th = res.x
    q = len(th)
    D = np.zeros((len(s), q))
    for j in range(q):
        e = np.zeros(q)
        e[j] = 1e-6
        D[:, j] = (sigma(th + e) - sigma(th - e)) / 2e-6
    A = np.linalg.inv(D.T @ W @ D)
    acov = A @ D.T @ W @ Gamma @ W @ D @ A / n
    d = len(s) - q
    U = W - W @ D @ A @ D.T @ W
    T = n * res.fun
    T_adj = _scaled_shifted(T, U, Gamma, d)
    T0 = n * s @ W @ s
    T0_adj = _scaled_shifted(T0, W, Gamma, len(s))
    C = np.eye(k)
    C[ic] = np.tanh(th[p:])
    C[(ic[1], ic[0])] = C[ic]
    return WLSMVResult(
        factors=factors,
        assign=assign,
        ordinal=ordinal,
        n=n,
        loadings=th[:p],
        loadings_se=np.sqrt(np.clip(np.diag(acov)[:p], 0, None)),
        factor_corr=C,
        chi2=T_adj,
        df=int(d),
        chi2_base=T0_adj,
        df_base=len(s),
        srmr=float(np.sqrt(np.mean((s - sigma(th)) ** 2))),
    )


__all__ = ["WLSMVResult", "bvn_cdf", "fit_wlsmv", "polychoric", "polyserial"]
