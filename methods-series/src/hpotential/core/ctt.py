"""Classical test theory and exploratory factor analysis.

The module holds the summaries a first look at item responses calls for:
Cronbach's alpha, McDonald's omega for a congeneric model, the
Kaiser-Meyer-Olkin index, Bartlett's test of sphericity, item-total statistics,
and exploratory factor analysis with promax rotation.

What alpha does and does not say
--------------------------------
Alpha is a lower bound on reliability under strictly parallel or at least
tau-equivalent items. It is not a test of unidimensionality. A two-factor set of
items with a moderate correlation between factors can return alpha above 0.9,
and a genuinely unidimensional set of four items can return alpha below 0.7
because alpha grows with the number of items. Reporting alpha alone does not
support a claim that the items measure one thing; the factor model does that
work, and omega reports reliability under a model that allows loadings to
differ.

Extraction here is dependency-free: single-factor and multi-factor solutions use
principal axis factoring on the correlation matrix, with promax obtained from a
varimax start. When ``factor_analyzer`` is installed (the ``full`` extra), pass
``backend="factor_analyzer"`` to :func:`efa` to cross-check.

Limitations
-----------
Everything here treats responses as continuous. For ordinal items with few
categories, and for skewed items, a polychoric correlation matrix and a
categorical estimator are the right tools; the functions accept a precomputed
correlation matrix so that a polychoric matrix can be passed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import optimize, special

__all__ = [
    "ClassicalItemAnalysis",
    "EFAResult",
    "bartlett_sphericity",
    "cronbach_alpha",
    "efa",
    "item_total_statistics",
    "kmo",
    "mcdonalds_omega",
]

ArrayLike = np.ndarray | pd.DataFrame


def _as_matrix(data: ArrayLike) -> np.ndarray:
    """Return a complete-case float matrix of persons by items."""
    values = (
        data.to_numpy(dtype=float)
        if isinstance(data, pd.DataFrame)
        else np.asarray(data, dtype=float)
    )
    if values.ndim != 2:
        raise ValueError("data must be two dimensional (persons by items)")
    values = values[~np.isnan(values).any(axis=1)]
    if values.shape[0] < 3:
        raise ValueError("need at least three complete cases")
    if values.shape[1] < 2:
        raise ValueError("need at least two items")
    return values


def cronbach_alpha(data: ArrayLike, ddof: int = 1) -> float:
    """Cronbach's alpha for a set of items.

    Parameters
    ----------
    data : numpy.ndarray or pandas.DataFrame
        Persons by items. Rows with any missing value are dropped.
    ddof : int, default 1
        Denominator adjustment passed to the variance.

    Returns
    -------
    float
        ``alpha = k / (k - 1) * (1 - sum(item variances) / variance of total)``.

    Notes
    -----
    An earlier draft defined this as a method whose signature omitted ``self``,
    so the first item argument was bound to the instance and the call failed
    with a type error before computing anything. It is a plain function here.

    Assumptions and limits: alpha assumes items are at least tau-equivalent and
    errors are uncorrelated. It is a lower bound on reliability when those hold
    and can be an overestimate when errors correlate, for instance across items
    sharing a stem. Alpha rises with the number of items whatever the structure,
    so it says nothing about unidimensionality. Report omega alongside it.
    """
    values = _as_matrix(data)
    k = values.shape[1]
    item_var = values.var(axis=0, ddof=ddof).sum()
    total_var = values.sum(axis=1).var(ddof=ddof)
    if total_var <= 0:
        raise ValueError("total score has zero variance; alpha is undefined")
    return float(k / (k - 1) * (1.0 - item_var / total_var))


def item_total_statistics(data: ArrayLike) -> pd.DataFrame:
    """Item-level statistics for a scale.

    Parameters
    ----------
    data : numpy.ndarray or pandas.DataFrame
        Persons by items.

    Returns
    -------
    pandas.DataFrame
        Indexed by item, with ``mean``, ``sd``, ``corrected_item_total_r`` (the
        correlation of the item with the sum of the other items) and
        ``alpha_if_dropped``.

    Notes
    -----
    The corrected correlation uses the rest score, not the total, because an
    item correlates with itself through the total and the uncorrected figure is
    inflated, badly so on short scales.
    """
    values = _as_matrix(data)
    names = (
        list(data.columns)
        if isinstance(data, pd.DataFrame)
        else [f"item{j + 1}" for j in range(values.shape[1])]
    )
    rows = []
    for j in range(values.shape[1]):
        rest = np.delete(values, j, axis=1).sum(axis=1)
        r = float(np.corrcoef(values[:, j], rest)[0, 1])
        if values.shape[1] > 2:
            alpha_drop = cronbach_alpha(np.delete(values, j, axis=1))
        else:
            alpha_drop = float("nan")
        rows.append(
            {
                "item": names[j],
                "mean": float(values[:, j].mean()),
                "sd": float(values[:, j].std(ddof=1)),
                "corrected_item_total_r": r,
                "alpha_if_dropped": alpha_drop,
            }
        )
    return pd.DataFrame(rows).set_index("item")


def _correlation(data: ArrayLike | None, corr: np.ndarray | None) -> np.ndarray:
    """Return a correlation matrix from data or pass one through."""
    if corr is not None:
        r = np.asarray(corr, dtype=float)
        if r.ndim != 2 or r.shape[0] != r.shape[1]:
            raise ValueError("corr must be square")
        return r
    if data is None:
        raise ValueError("supply data or corr")
    values = _as_matrix(data)
    return np.corrcoef(values, rowvar=False)


def _principal_axis(
    corr: np.ndarray, n_factors: int, n_iter: int = 100, tol: float = 1e-7
) -> np.ndarray:
    """Principal axis factoring of a correlation matrix.

    Communalities start at the squared multiple correlations and are iterated
    until they stop moving. Returns a ``(p, n_factors)`` loading matrix.
    """
    p = corr.shape[0]
    reduced = corr.copy()
    inv = np.linalg.pinv(corr)
    comm = 1.0 - 1.0 / np.clip(np.diag(inv), 1e-8, None)
    comm = np.clip(comm, 1e-3, 0.999)
    loadings = np.zeros((p, n_factors))
    for _ in range(n_iter):
        np.fill_diagonal(reduced, comm)
        vals, vecs = np.linalg.eigh(reduced)
        idx = np.argsort(vals)[::-1][:n_factors]
        lam = np.clip(vals[idx], 0.0, None)
        loadings = vecs[:, idx] * np.sqrt(lam)
        new_comm = np.clip((loadings**2).sum(axis=1), 1e-3, 0.999)
        if np.max(np.abs(new_comm - comm)) < tol:
            comm = new_comm
            break
        comm = new_comm
    # fix reflection so each factor loads mostly positive
    for f in range(n_factors):
        if loadings[:, f].sum() < 0:
            loadings[:, f] *= -1
    return loadings


def _congeneric_loadings(corr: np.ndarray) -> np.ndarray:
    """Single-factor loadings fitted by least squares on the off-diagonal.

    The congeneric model says ``R = lambda lambda' + diag(theta)``. Only the
    off-diagonal entries carry information about lambda, so they are the ones
    fitted; the diagonal is reproduced by construction.
    """
    p = corr.shape[0]
    off = ~np.eye(p, dtype=bool)
    start = np.clip(np.sqrt(np.clip(np.abs(corr[off]).mean(), 1e-3, 0.99)), 0.2, 0.9)
    x0 = np.full(p, start)

    def residual(lam: np.ndarray) -> np.ndarray:
        return (np.outer(lam, lam) - corr)[off]

    fit = optimize.least_squares(residual, x0, method="lm", max_nfev=20000)
    lam = fit.x
    if lam.sum() < 0:
        lam = -lam
    return np.clip(lam, -0.999, 0.999)


def mcdonalds_omega(
    data: ArrayLike | None = None,
    corr: np.ndarray | None = None,
) -> float:
    """McDonald's omega total for a one-factor congeneric model.

    Parameters
    ----------
    data : numpy.ndarray or pandas.DataFrame, optional
        Persons by items. Ignored when ``corr`` is given.
    corr : numpy.ndarray, optional
        A precomputed correlation matrix, for instance polychoric.

    Returns
    -------
    float
        ``omega = (sum lambda)^2 / ((sum lambda)^2 + sum theta)`` with
        ``theta_i = 1 - lambda_i^2`` on the standardised scale, matching the
        expression in the theory paper.

    Notes
    -----
    Omega allows loadings to differ across items, which alpha does not, so the
    two separate when items are unequally related to the factor. Omega is only
    as good as the one-factor model: if the items are not unidimensional, omega
    total still returns a number and that number is not the reliability of a
    single score. Check the factor solution before quoting it.
    """
    r = _correlation(data, corr)
    lam = _congeneric_loadings(r)
    theta = np.clip(1.0 - lam**2, 1e-6, None)
    num = float(lam.sum() ** 2)
    return float(num / (num + float(theta.sum())))


def kmo(data: ArrayLike | None = None, corr: np.ndarray | None = None) -> dict[str, object]:
    """Kaiser-Meyer-Olkin measure of sampling adequacy.

    Parameters
    ----------
    data : numpy.ndarray or pandas.DataFrame, optional
        Persons by items.
    corr : numpy.ndarray, optional
        A precomputed correlation matrix.

    Returns
    -------
    dict
        ``kmo_total`` and ``kmo_per_item`` (a numpy array).

    Notes
    -----
    KMO compares the correlations with the partial correlations. Values near 1
    mean the items share variance that a factor model can pick up; below about
    0.5 the correlation pattern is too diffuse for factoring to be sensible. The
    index is descriptive, not a test, and it does not tell you how many factors
    there are.
    """
    r = _correlation(data, corr)
    inv = np.linalg.pinv(r)
    d = np.sqrt(np.diag(inv))
    partial = -inv / np.outer(d, d)
    np.fill_diagonal(partial, 0.0)
    r_off = r.copy()
    np.fill_diagonal(r_off, 0.0)
    sq_r = r_off**2
    sq_p = partial**2
    total = float(sq_r.sum() / (sq_r.sum() + sq_p.sum()))
    per_item = sq_r.sum(axis=0) / (sq_r.sum(axis=0) + sq_p.sum(axis=0))
    return {"kmo_total": total, "kmo_per_item": per_item}


def bartlett_sphericity(
    data: ArrayLike | None = None,
    corr: np.ndarray | None = None,
    n_obs: int | None = None,
) -> dict[str, float]:
    """Bartlett's test that the correlation matrix is an identity.

    Parameters
    ----------
    data : numpy.ndarray or pandas.DataFrame, optional
        Persons by items. Supplies ``n_obs`` when it is not given.
    corr : numpy.ndarray, optional
        A precomputed correlation matrix. Requires ``n_obs``.
    n_obs : int, optional
        Number of complete cases.

    Returns
    -------
    dict
        ``chi2``, ``df`` and ``p_value``.

    Notes
    -----
    ``chi2 = -[(n - 1) - (2p + 5) / 6] log|R|`` on ``p(p - 1) / 2`` degrees of
    freedom. The test is a floor, not evidence for a factor model: with a few
    hundred respondents it rejects the identity for almost any real item set, so
    a small p value says only that the items are not mutually unrelated.
    """
    if corr is not None and n_obs is None:
        raise ValueError("supply n_obs with corr")
    if corr is None:
        values = _as_matrix(data)
        n_obs = values.shape[0]
        r = np.corrcoef(values, rowvar=False)
    else:
        r = np.asarray(corr, dtype=float)
    p = r.shape[0]
    sign, logdet = np.linalg.slogdet(r)
    if sign <= 0:
        raise ValueError("correlation matrix is not positive definite")
    chi2 = -((n_obs - 1) - (2 * p + 5) / 6.0) * logdet
    df = p * (p - 1) // 2
    return {
        "chi2": float(chi2),
        "df": int(df),
        "p_value": float(special.gammaincc(df / 2.0, max(chi2, 0.0) / 2.0)),
    }


def _varimax(loadings: np.ndarray, tol: float = 1e-8, n_iter: int = 500) -> np.ndarray:
    """Varimax rotation of a loading matrix, Kaiser normalised."""
    p, k = loadings.shape
    if k < 2:
        return loadings.copy()
    h = np.sqrt((loadings**2).sum(axis=1))
    h[h == 0] = 1.0
    lam = loadings / h[:, None]
    rot = np.eye(k)
    d_old = 0.0
    for _ in range(n_iter):
        z = lam @ rot
        u, s, vt = np.linalg.svd(lam.T @ (z**3 - z @ np.diag((z**2).sum(axis=0)) / p))
        rot = u @ vt
        d = float(s.sum())
        if d_old != 0.0 and abs(d - d_old) / d < tol:
            break
        d_old = d
    return (lam @ rot) * h[:, None]


def _promax(loadings: np.ndarray, kappa: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Promax rotation from a varimax start.

    Returns the pattern matrix and the factor correlation matrix. Promax is
    oblique: with correlated domains, as the H-potential model expects, an
    orthogonal rotation forces cross-loadings that are artefacts of the
    constraint rather than features of the items.
    """
    rotated = _varimax(loadings)
    if rotated.shape[1] < 2:
        return rotated, np.ones((1, 1))
    # target matrix: sign preserved, magnitudes raised to the power kappa
    target = np.sign(rotated) * np.abs(rotated) ** kappa
    coef = np.linalg.lstsq(rotated, target, rcond=None)[0]
    scale = np.diag(np.linalg.pinv(coef.T @ coef))
    coef = coef @ np.diag(np.sqrt(np.clip(scale, 1e-12, None)))
    pattern = rotated @ coef
    phi = np.linalg.pinv(coef.T @ coef)
    d = np.sqrt(np.diag(phi))
    phi = phi / np.outer(d, d)
    for f in range(pattern.shape[1]):
        if pattern[:, f].sum() < 0:
            pattern[:, f] *= -1
            phi[f, :] *= -1
            phi[:, f] *= -1
    np.fill_diagonal(phi, 1.0)
    return pattern, phi


@dataclass
class EFAResult:
    """Output of an exploratory factor analysis.

    Attributes
    ----------
    loadings : pandas.DataFrame
        Pattern matrix, items by factors. With an oblique rotation these are
        partial coefficients, not correlations with the factor.
    communalities : pandas.Series
        Shared variance per item, from the unrotated solution.
    factor_correlations : pandas.DataFrame
        Correlations among factors. The identity under an orthogonal rotation.
    eigenvalues : numpy.ndarray
        Eigenvalues of the correlation matrix, for a scree plot.
    rotation : str
        Name of the rotation applied.
    n_factors : int
        Number of factors extracted.
    backend : str
        ``"hpotential"`` or ``"factor_analyzer"``.
    """

    loadings: pd.DataFrame
    communalities: pd.Series
    factor_correlations: pd.DataFrame
    eigenvalues: np.ndarray
    rotation: str
    n_factors: int
    backend: str


def efa(
    data: ArrayLike | None = None,
    n_factors: int = 3,
    rotation: Literal["promax", "varimax", "none"] = "promax",
    corr: np.ndarray | None = None,
    item_names: list[str] | None = None,
    backend: Literal["hpotential", "factor_analyzer"] = "hpotential",
) -> EFAResult:
    """Exploratory factor analysis by principal axis factoring.

    Parameters
    ----------
    data : numpy.ndarray or pandas.DataFrame, optional
        Persons by items.
    n_factors : int, default 3
        Number of factors to extract. Three matches the hypothesised domain
        structure; run alternatives before settling on it.
    rotation : {"promax", "varimax", "none"}, default "promax"
        Promax is oblique and is the default because the domains are expected
        to correlate.
    corr : numpy.ndarray, optional
        A precomputed correlation matrix, for instance polychoric.
    item_names : list of str, optional
        Row labels. Taken from ``data`` when it is a frame.
    backend : {"hpotential", "factor_analyzer"}, default "hpotential"
        ``"factor_analyzer"`` requires the ``full`` extra and is offered as a
        cross-check.

    Returns
    -------
    EFAResult
        Loadings, communalities, factor correlations and eigenvalues.

    Notes
    -----
    Exploratory analysis generates hypotheses. It does not confirm a structure,
    and a clean three-factor solution on the sample that suggested three factors
    is not evidence of anything. The confirmatory model belongs in a separate
    sample.
    """
    if backend == "factor_analyzer":
        return _efa_factor_analyzer(data, n_factors, rotation, item_names)
    r = _correlation(data, corr)
    names = item_names or (
        list(data.columns)
        if isinstance(data, pd.DataFrame)
        else [f"item{j + 1}" for j in range(r.shape[0])]
    )
    if n_factors < 1 or n_factors >= r.shape[0]:
        raise ValueError("n_factors must be between 1 and the number of items minus one")
    eigenvalues = np.sort(np.linalg.eigvalsh(r))[::-1]
    unrotated = _principal_axis(r, n_factors)
    communalities = (unrotated**2).sum(axis=1)
    if rotation == "promax":
        pattern, phi = _promax(unrotated)
    elif rotation == "varimax":
        pattern, phi = _varimax(unrotated), np.eye(n_factors)
    else:
        pattern, phi = unrotated, np.eye(n_factors)
    cols = [f"F{f + 1}" for f in range(n_factors)]
    return EFAResult(
        loadings=pd.DataFrame(pattern, index=names, columns=cols),
        communalities=pd.Series(communalities, index=names, name="communality"),
        factor_correlations=pd.DataFrame(phi, index=cols, columns=cols),
        eigenvalues=eigenvalues,
        rotation=rotation,
        n_factors=int(n_factors),
        backend="hpotential",
    )


def _efa_factor_analyzer(
    data: ArrayLike | None,
    n_factors: int,
    rotation: str,
    item_names: list[str] | None,
) -> EFAResult:
    """Delegate to ``factor_analyzer`` when it is installed."""
    try:
        from factor_analyzer import FactorAnalyzer
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "backend='factor_analyzer' needs the optional extra: pip install 'hpotential[full]'"
        ) from exc
    if data is None:
        raise ValueError("the factor_analyzer backend needs raw data")
    values = _as_matrix(data)
    names = item_names or (
        list(data.columns)
        if isinstance(data, pd.DataFrame)
        else [f"item{j + 1}" for j in range(values.shape[1])]
    )
    rot = None if rotation == "none" else rotation
    # minres converges to the same solution as iterated principal axis
    # factoring. method="principal" would extract principal components
    # instead, which give larger loadings and are not a like-for-like check.
    model = FactorAnalyzer(n_factors=n_factors, rotation=rot, method="minres")
    try:
        model.fit(values)
    except TypeError as exc:  # pragma: no cover - version clash, not our code
        raise RuntimeError(
            "the installed factor_analyzer is not compatible with the installed "
            f"scikit-learn: {exc}. Use the default backend."
        ) from exc
    cols = [f"F{f + 1}" for f in range(n_factors)]
    phi = model.phi_ if getattr(model, "phi_", None) is not None else np.eye(n_factors)
    return EFAResult(
        loadings=pd.DataFrame(model.loadings_, index=names, columns=cols),
        communalities=pd.Series(model.get_communalities(), index=names, name="communality"),
        factor_correlations=pd.DataFrame(np.asarray(phi), index=cols, columns=cols),
        eigenvalues=np.asarray(model.get_eigenvalues()[0]),
        rotation=rotation,
        n_factors=int(n_factors),
        backend="factor_analyzer",
    )


class ClassicalItemAnalysis:
    """A first pass over a set of item responses.

    Parameters
    ----------
    data : numpy.ndarray or pandas.DataFrame
        Persons by items. Rows with any missing value are dropped, and the
        number kept is reported by :attr:`n_obs`.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> theta = rng.normal(size=300)
    >>> x = theta[:, None] * 0.8 + rng.normal(size=(300, 6))
    >>> analysis = ClassicalItemAnalysis(x)
    >>> 0.0 < analysis.alpha() < 1.0
    True

    Limitations
    -----------
    Every statistic here describes the sample in hand under a continuous
    measurement model. None of them establishes that the items measure one
    construct, and none survives a change of population without checking.
    """

    def __init__(self, data: ArrayLike) -> None:
        self.data = data
        self._values = _as_matrix(data)
        self._names = (
            list(data.columns)
            if isinstance(data, pd.DataFrame)
            else [f"item{j + 1}" for j in range(self._values.shape[1])]
        )

    @property
    def n_obs(self) -> int:
        """Complete cases used."""
        return int(self._values.shape[0])

    @property
    def n_items(self) -> int:
        """Number of items."""
        return int(self._values.shape[1])

    def alpha(self) -> float:
        """Cronbach's alpha. See :func:`cronbach_alpha`."""
        return cronbach_alpha(self._values)

    def omega(self) -> float:
        """McDonald's omega total. See :func:`mcdonalds_omega`."""
        return mcdonalds_omega(self._values)

    def item_statistics(self) -> pd.DataFrame:
        """Item-total statistics. See :func:`item_total_statistics`."""
        return item_total_statistics(pd.DataFrame(self._values, columns=self._names))

    def factorability(self) -> dict[str, object]:
        """KMO and Bartlett together.

        Returns
        -------
        dict
            ``kmo_total``, ``kmo_per_item``, ``bartlett_chi2``, ``bartlett_df``
            and ``bartlett_p``.
        """
        k = kmo(self._values)
        b = bartlett_sphericity(self._values)
        return {
            "kmo_total": k["kmo_total"],
            "kmo_per_item": k["kmo_per_item"],
            "bartlett_chi2": b["chi2"],
            "bartlett_df": b["df"],
            "bartlett_p": b["p_value"],
        }

    def efa(self, n_factors: int = 3, rotation: str = "promax", **kwargs) -> EFAResult:
        """Exploratory factor analysis. See :func:`efa`."""
        return efa(
            pd.DataFrame(self._values, columns=self._names),
            n_factors=n_factors,
            rotation=rotation,  # type: ignore[arg-type]
            **kwargs,
        )

    def summary(self) -> pd.Series:
        """One-line summary of the scale.

        Returns
        -------
        pandas.Series
            ``n_obs``, ``n_items``, ``alpha``, ``omega``, ``kmo`` and the
            Bartlett p value.
        """
        fact = self.factorability()
        return pd.Series(
            {
                "n_obs": self.n_obs,
                "n_items": self.n_items,
                "alpha": self.alpha(),
                "omega": self.omega(),
                "kmo": fact["kmo_total"],
                "bartlett_p": fact["bartlett_p"],
            },
            name="classical_summary",
        )
