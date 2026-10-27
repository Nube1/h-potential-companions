"""Confirmatory factor analysis by maximum likelihood, with multi-group invariance.

The measurement model of Chapter 3 is a correlated three-domain model: each
indicator loads on one domain, the domains correlate, and residuals are
independent. This module fits it by maximum likelihood on (optionally
survey-weighted) moments, and reports what Chapter 5 needs:

* fit: chi-square, CFI, TLI, RMSEA and SRMR against the independence model;
* standardised loadings with standard errors from the observed information;
* the second-order model, which with three domains is just identified at the
  structural level, so its loadings follow from the factor correlations;
* reliability (coefficient omega, average variance extracted), HTMT, and the
  Fornell-Larcker comparison;
* regression factor scores and their determinacy, for the domains and for the
  general factor;
* configural, metric and scalar invariance across groups.

Latent variances are fixed at one in the first group (and in every group for
the configural model); latent means are zero in the first group.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

LEVELS = ("configural", "metric", "scalar")


# ------------------------------------------------------------------ moments


def moments(X, w=None):
    """Mean vector and covariance (divisor: sum of weights) of complete data."""
    X = np.asarray(X, float)
    w = np.ones(len(X)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    m = w @ X
    d = X - m
    return m, (d * w[:, None]).T @ d


# --------------------------------------------------------------- parameters


@dataclass
class _Layout:
    p: int
    k: int
    assign: np.ndarray  # factor index of each indicator
    groups: int
    level: str
    slices: dict = field(default_factory=dict)
    size: int = 0

    def __post_init__(self):
        G, p, k = self.groups, self.p, self.k
        nc = k * (k - 1) // 2
        pos = 0

        def take(name, n):
            nonlocal pos
            self.slices[name] = slice(pos, pos + n)
            pos += n

        shared_load = self.level != "configural"
        for g in range(G):
            if g == 0 or not shared_load:
                take(f"lam{g}", p)
            take(f"th{g}", p)
            take(f"cor{g}", nc)
            if g > 0 and shared_load:
                take(f"var{g}", k)
            if g == 0 or self.level != "scalar":
                take(f"nu{g}", p)
            if g > 0 and self.level == "scalar":
                take(f"kap{g}", k)
        self.size = pos

    def unpack(self, v):
        out = []
        iu = np.triu_indices(self.k, 1)
        for g in range(self.groups):
            lam_v = v[self.slices[f"lam{g}" if f"lam{g}" in self.slices else "lam0"]]
            L = np.zeros((self.p, self.k))
            L[np.arange(self.p), self.assign] = lam_v
            C = np.eye(self.k)
            C[iu] = np.tanh(v[self.slices[f"cor{g}"]])
            C[(iu[1], iu[0])] = C[iu]
            sd = np.exp(0.5 * v[self.slices[f"var{g}"]]) if f"var{g}" in self.slices else 1.0
            Phi = C * np.outer(sd, sd) if np.ndim(sd) else C
            th = np.exp(v[self.slices[f"th{g}"]])
            nu = v[self.slices[f"nu{g}" if f"nu{g}" in self.slices else "nu0"]]
            kap = v[self.slices[f"kap{g}"]] if f"kap{g}" in self.slices else np.zeros(self.k)
            out.append((L, Phi, th, nu, kap))
        return out

    def start(self, means):
        v = np.zeros(self.size)
        for name, s in self.slices.items():
            g = int(name[3:]) if name[:3] in ("lam", "cor", "var", "kap") else int(name[2:])
            if name.startswith("lam"):
                v[s] = 0.7
            elif name.startswith("th"):
                v[s] = np.log(0.5)
            elif name.startswith("cor"):
                v[s] = np.arctanh(0.3)
            elif name.startswith("nu"):
                v[s] = means[g]
        return v


def _implied(L, Phi, th, nu, kap):
    return L @ Phi @ L.T + np.diag(th), nu + L @ kap


def _discrepancy(S, m, Sig, mu):
    sign, logdet = np.linalg.slogdet(Sig)
    if sign <= 0:
        return np.inf
    inv = np.linalg.inv(Sig)
    d = m - mu
    p = len(m)
    return logdet + np.trace(S @ inv) + d @ inv @ d - np.linalg.slogdet(S)[1] - p


# ------------------------------------------------------------------- result


@dataclass
class CFAResult:
    names: list
    factors: list
    assign: np.ndarray
    level: str
    n: np.ndarray
    params: np.ndarray
    layout: _Layout
    cov: np.ndarray
    chi2: float
    df: int
    chi2_base: float
    df_base: int
    srmr: float
    S: list
    m: list

    # ---- fit
    @property
    def rmsea(self):
        N = self.n.sum()
        return float(np.sqrt(max(self.chi2 - self.df, 0) / (self.df * (N - 1)))) if self.df else 0.0

    @property
    def cfi(self):
        d = max(self.chi2 - self.df, 0)
        d0 = max(self.chi2_base - self.df_base, d, 0)
        return float(1 - d / d0) if d0 > 0 else 1.0

    @property
    def tli(self):
        r0 = self.chi2_base / self.df_base
        r = self.chi2 / self.df if self.df else 1.0
        return float((r0 - r) / (r0 - 1))

    def fit_indices(self):
        return {
            "chi2": self.chi2,
            "df": self.df,
            "cfi": self.cfi,
            "tli": self.tli,
            "rmsea": self.rmsea,
            "srmr": self.srmr,
        }

    # ---- parameters (first group)
    def group(self, g=0):
        return self.layout.unpack(self.params)[g]

    def _delta(self, f):
        """Value and delta-method standard error of a scalar function of the parameters."""
        v0 = f(self.params)
        J = np.zeros(len(self.params))
        for i in range(len(self.params)):
            h = 1e-5 * max(1.0, abs(self.params[i]))
            up, dn = self.params.copy(), self.params.copy()
            up[i] += h
            dn[i] -= h
            J[i] = (f(up) - f(dn)) / (2 * h)
        return float(v0), float(np.sqrt(max(J @ self.cov @ J, 0)))

    def standardised_loadings(self, g=0):
        """(estimate, standard error) for each indicator."""
        out = []
        for j in range(len(self.names)):

            def f(v, j=j):
                L, Phi, th, nu, kap = self.layout.unpack(v)[g]
                Sig, _ = _implied(L, Phi, th, nu, kap)
                f_ = self.assign[j]
                return L[j, f_] * np.sqrt(Phi[f_, f_]) / np.sqrt(Sig[j, j])

            out.append(self._delta(f))
        return out

    def factor_correlations(self, g=0):
        L, Phi, *_ = self.group(g)
        d = np.sqrt(np.diag(Phi))
        return Phi / np.outer(d, d)

    def second_order(self):
        """Loadings of the domains on a general factor, with standard errors.

        With three first-order factors the structural part is just identified:
        gamma_1 = sqrt(phi_12 phi_13 / phi_23), and so on. Undefined when a
        correlation is not positive.
        """
        if len(self.factors) != 3:
            raise ValueError("closed form needs exactly three domains")
        out = []
        for a, b, c in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):

            def f(v, a=a, b=b, c=c):
                L, Phi, *_ = self.layout.unpack(v)[0]
                d = np.sqrt(np.diag(Phi))
                R = Phi / np.outer(d, d)
                x = R[a, b] * R[a, c] / R[b, c]
                return np.sqrt(x) if x > 0 else np.nan

            out.append(self._delta(f))
        return out

    def second_order_admissible(self) -> bool:
        """True when every second-order loading is defined and below one.

        A loading of one or more implies a negative residual variance for that
        domain (a Heywood case); a non-positive factor correlation leaves the
        loadings undefined. Either way no general factor is retained (H2).
        """
        gam = np.array([e for e, _ in self.second_order()])
        return bool(np.all(np.isfinite(gam)) and np.all(gam < 1))

    def reliability(self):
        """Omega and AVE per domain, first group, standardised metric."""
        L, Phi, th, *_ = self.group(0)
        std = np.array([e for e, _ in self.standardised_loadings()])
        res = {}
        for fi, name in enumerate(self.factors):
            idx = self.assign == fi
            lam = L[idx, fi] * np.sqrt(Phi[fi, fi])
            omega = lam.sum() ** 2 / (lam.sum() ** 2 + th[idx].sum())
            res[name] = {"omega": float(omega), "ave": float(np.mean(std[idx] ** 2))}
        return res

    def htmt(self):
        S = self.S[0]
        d = np.sqrt(np.diag(S))
        R = np.abs(S / np.outer(d, d))
        k = len(self.factors)
        out = np.full((k, k), np.nan)
        for a in range(k):
            for b in range(a + 1, k):
                ia, ib = np.flatnonzero(self.assign == a), np.flatnonzero(self.assign == b)
                hetero = R[np.ix_(ia, ib)].mean()

                def mono(ix):
                    iu = np.triu_indices(len(ix), 1)
                    return R[np.ix_(ix, ix)][iu].mean()

                out[a, b] = out[b, a] = hetero / np.sqrt(mono(ia) * mono(ib))
        return out

    # ---- scores
    def _weights(self, g=0):
        L, Phi, th, nu, kap = self.group(g)
        Sig, mu = _implied(L, Phi, th, nu, kap)
        inv = np.linalg.inv(Sig)
        return L, Phi, inv, mu, kap

    def scores(self, X, g=0):
        """Regression (Thurstone) factor scores for the domains."""
        L, Phi, inv, mu, kap = self._weights(g)
        return kap + (np.asarray(X, float) - mu) @ inv @ L @ Phi

    def general_scores(self, X):
        """Regression scores for the second-order general factor (NaN if inadmissible)."""
        if not self.second_order_admissible():
            return np.full(len(X), np.nan)
        gam = np.array([e for e, _ in self.second_order()])
        L, Phi, inv, mu, _ = self._weights(0)
        return (np.asarray(X, float) - mu) @ inv @ L @ gam

    def determinacy(self):
        L, Phi, inv, *_ = self._weights(0)
        M = Phi @ L.T @ inv @ L @ Phi
        dom = np.sqrt(np.diag(M) / np.diag(Phi))
        gam = np.array([e for e, _ in self.second_order()])
        gen = (
            float(np.sqrt(gam @ L.T @ inv @ L @ gam)) if self.second_order_admissible() else np.nan
        )
        return {**dict(zip(self.factors, dom.tolist(), strict=True)), "general": gen}


# ---------------------------------------------------------------------- fit


def fit_cfa(
    groups, pattern: dict, weights=None, level: str = "configural", se: bool = True
) -> CFAResult:
    """Fit the CFA to one or more groups of complete data.

    ``groups`` is a list of (n x p) arrays (or one array). ``pattern`` maps each
    factor name to its column indices, in the column order of the arrays.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    if not isinstance(groups, list | tuple):
        groups = [groups]
        weights = [weights]
    weights = weights or [None] * len(groups)
    factors = list(pattern)
    p = groups[0].shape[1]
    assign = np.empty(p, int)
    for fi, name in enumerate(factors):
        assign[list(pattern[name])] = fi
    mom = [moments(X, w) for X, w in zip(groups, weights, strict=True)]
    m = [a for a, _ in mom]
    S = [b for _, b in mom]
    n = np.array([len(X) for X in groups], float)
    N = n.sum()
    lay = _Layout(p, len(factors), assign, len(groups), level)

    def obj(v):
        total = 0.0
        for g, (L, Phi, th, nu, kap) in enumerate(lay.unpack(v)):
            Sig, mu = _implied(L, Phi, th, nu, kap)
            total += n[g] / N * _discrepancy(S[g], m[g], Sig, mu)
        return total if np.isfinite(total) else 1e10

    best = None
    lam_idx = np.concatenate(
        [np.arange(s.start, s.stop) for nm, s in lay.slices.items() if nm.startswith("lam")]
    )
    for scale in (1.0, 0.5):
        v0 = lay.start(m)
        v0[lam_idx] *= scale
        res = minimize(obj, v0, method="L-BFGS-B", options={"maxiter": 5000, "maxfun": 200000})
        if best is None or res.fun < best.fun:
            best = res
    v = best.x
    # Standard errors: cov = (2/N) H^-1, H the Hessian of the fit function.
    q = len(v)
    H = np.full((q, q), np.nan) if not se else np.zeros((q, q))
    h = 1e-4
    f0 = obj(v)
    for i in range(q if se else 0):
        for j in range(i, q):
            e_i = np.zeros(q)
            e_j = np.zeros(q)
            e_i[i] = h
            e_j[j] = h
            if i == j:
                H[i, i] = (obj(v + e_i) - 2 * f0 + obj(v - e_i)) / h**2
            else:
                H[i, j] = H[j, i] = (
                    obj(v + e_i + e_j)
                    - obj(v + e_i - e_j)
                    - obj(v - e_i + e_j)
                    + obj(v - e_i - e_j)
                ) / (4 * h**2)
    try:
        cov = 2 / N * np.linalg.inv(H) if se else np.full((q, q), np.nan)
    except np.linalg.LinAlgError:
        cov = np.full((q, q), np.nan)
    chi2 = N * best.fun
    df = len(groups) * (p * (p + 1) // 2 + p) - q
    base = sum(
        n[g] * (np.log(np.diag(S[g])).sum() - np.linalg.slogdet(S[g])[1])
        for g in range(len(groups))
    )
    df_base = len(groups) * p * (p - 1) // 2
    srmr = 0.0
    for g, (L, Phi, th, nu, kap) in enumerate(lay.unpack(v)):
        Sig, _ = _implied(L, Phi, th, nu, kap)
        d = np.sqrt(np.diag(S[g]))
        r = (S[g] - Sig) / np.outer(d, d)
        il = np.tril_indices(p)
        srmr += n[g] / N * np.sqrt(np.mean(r[il] ** 2))
    return CFAResult(
        names=[str(i) for i in range(p)],
        factors=factors,
        assign=assign,
        level=level,
        n=n,
        params=v,
        layout=lay,
        cov=cov,
        chi2=float(chi2),
        df=int(df),
        chi2_base=float(base),
        df_base=int(df_base),
        srmr=float(srmr),
        S=S,
        m=m,
    )


def invariance(groups, pattern, weights=None) -> dict:
    """Configural, metric and scalar fits across groups, with changes in fit."""
    fits = {lvl: fit_cfa(list(groups), pattern, weights, lvl) for lvl in LEVELS}
    rows = {}
    prev = None
    for lvl in LEVELS:
        f = fits[lvl]
        row = f.fit_indices()
        if prev is not None:
            row["delta_cfi"] = row["cfi"] - prev.cfi
            row["delta_rmsea"] = row["rmsea"] - prev.rmsea
            row["delta_chi2"] = row["chi2"] - prev.chi2
            row["delta_df"] = row["df"] - prev.df
        rows[lvl] = row
        prev = f
    return rows


# ------------------------------------------------------------ Rubin pooling


def rubin(estimates, variances) -> dict:
    """Pool m estimates and their sampling variances by Rubin's rules."""
    q = np.asarray(estimates, float)
    u = np.asarray(variances, float)
    m = len(q)
    qbar = q.mean(axis=0)
    ubar = u.mean(axis=0)
    b = q.var(axis=0, ddof=1) if m > 1 else np.zeros_like(qbar)
    t = ubar + (1 + 1 / m) * b
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (1 + 1 / m) * b / ubar
        dof = np.where(b > 0, (m - 1) * (1 + 1 / r) ** 2, np.inf)
    return {"estimate": qbar, "se": np.sqrt(t), "df": dof, "fmi_ratio": r}


__all__ = ["CFAResult", "fit_cfa", "invariance", "moments", "rubin"]
