"""Differential item functioning by ordinal logistic regression.

The three nested models
-----------------------
For an ordinal item ``Y`` with categories ``0 ... K - 1``, a matching variable
``M`` and a group indicator ``G``, fit

    M1 (base)         logit P(Y <= c) = gamma_c - beta1 M
    M2 (uniform)      logit P(Y <= c) = gamma_c - beta1 M - beta2 G
    M3 (non-uniform)  logit P(Y <= c) = gamma_c - beta1 M - beta2 G - beta3 M G

and compare them with likelihood-ratio tests: M2 against M1 on 1 degree of
freedom for uniform DIF, M3 against M2 on 1 degree of freedom for non-uniform
DIF, and M3 against M1 on 2 degrees of freedom for either. This is the model
written in the theory paper, equation (a_dif).

The sign convention, spelled out
--------------------------------
The linear predictor enters with a MINUS: ``logit P(Y <= c) = gamma_c - x'b``.
This is the parameterisation ``statsmodels.miscmodels.ordinal_model.OrderedModel``
uses, and it is the one used here, so the two agree without translation. The
consequence is worth stating plainly, because it is easy to misread: a positive
coefficient means a higher value of that covariate pushes responses towards
HIGHER categories. A positive ``beta2`` means the focal group answers higher
than the reference group at the same matched level. Some texts write the model
with a plus, in which case every coefficient reported here changes sign.

Three corrections to the draft
------------------------------
1. Matching variable. The default is the rest score, the total over all other
   items, not the total score. Including the studied item means matching partly
   on the difference under test, which costs power. In the simulations in
   ``tests/test_dif.py``, with uniform DIF planted on one item of a three-item
   scale, the likelihood-ratio statistic for that item falls from 183 to 152
   when the match changes from the rest score to the total, and the gap closes
   as the scale lengthens, because one item is a smaller share of a longer
   total. The contaminated total also carries the DIF onto the other items: on
   a six-item scale the clean items pick up a mean group coefficient of 0.25
   under total matching against 0.18 under rest matching, when the true value
   for all of them is zero. Pass ``matching="total"`` to get the old
   behaviour.
2. Effect size. McFadden's change in R-squared is computed against the
   intercept-only model M0:

       Delta R^2 = [loglik(M3) - loglik(M1)] / (-loglik(M0)),

   which is the difference of the two McFadden R-squared values, since
   ``R^2_m = 1 - loglik(Mm) / loglik(M0)``. The draft divided by
   ``loglik(M1)``, which is not McFadden's R-squared and carries the wrong
   sign: log-likelihoods are negative, so dividing a positive improvement by a
   negative reference returns a negative effect size.
3. Classification. An item is flagged only when the likelihood-ratio test is
   significant AND ``Delta R^2 >= 0.03``. Significance alone flags trivial
   differences in any sample large enough, which is how DIF screening produces
   long lists of items nobody can interpret.

Estimation
----------
A proportional-odds model fitted with scipy, so statsmodels is not required.
Thresholds are kept ordered by estimating the first one and log gaps. When
statsmodels is installed, :func:`fit_ordinal_statsmodels` runs the same model
through ``OrderedModel`` as a cross-check; the tests compare the two
log-likelihoods and skip when it is absent.

Limitations
-----------
The observed matching variable contains measurement error, so this is a test
conditional on an imperfect proxy for the trait; with unreliable scales it can
flag items that function identically at equal true trait. It assumes
proportional odds. It does not adjust for multiplicity across items, and DIF
detected is not DIF explained: a flagged item needs review of its wording and
of what it assumes about the respondent's circumstances.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import optimize, special

__all__ = [
    "DIFResult",
    "OrdinalFit",
    "dif_scan",
    "fit_ordinal_statsmodels",
    "fit_proportional_odds",
    "ordinal_logistic_dif",
    "rest_score",
]

#: Effect-size cuts on McFadden's change in R-squared, in the tradition of
#: Zumbo and of Jodoin and Gierl. Below 0.03 the difference is treated as
#: negligible however small the p value.
NEGLIGIBLE_MAX = 0.03
MODERATE_MAX = 0.07


@dataclass
class OrdinalFit:
    """A fitted proportional-odds model.

    Attributes
    ----------
    loglik : float
        Log-likelihood at the solution.
    thresholds : numpy.ndarray
        The ``K - 1`` cut points ``gamma_c``, increasing.
    beta : numpy.ndarray
        Slope coefficients, in the order of the design columns, under the
        convention ``logit P(Y <= c) = gamma_c - x'beta``.
    names : list of str
        Names of the design columns.
    n_params : int
        Number of free parameters.
    converged : bool
        Whether the optimiser reported success.
    """

    loglik: float
    thresholds: np.ndarray
    beta: np.ndarray
    names: list[str]
    n_params: int
    converged: bool

    def coefficient(self, name: str) -> float:
        """Return one slope by name."""
        return float(self.beta[self.names.index(name)])


def _prepare(y: np.ndarray | pd.Series, x: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate an ordinal outcome and its design matrix."""
    y_arr = np.asarray(y, dtype=float).ravel()
    if np.any(np.isnan(y_arr)):
        raise ValueError("the outcome contains missing values; drop them first")
    codes = np.unique(y_arr)
    if codes.size < 2:
        raise ValueError("the outcome must take at least two values")
    lookup = {v: i for i, v in enumerate(codes)}
    y_int = np.array([lookup[v] for v in y_arr], dtype=int)
    if x is None:
        design = np.empty((y_int.size, 0))
    else:
        design = np.asarray(x, dtype=float)
        if design.ndim == 1:
            design = design[:, None]
        if design.shape[0] != y_int.size:
            raise ValueError("design and outcome lengths differ")
        if np.any(np.isnan(design)):
            raise ValueError("the design matrix contains missing values")
    return y_int, design, int(codes.size)


def fit_proportional_odds(
    y: np.ndarray | pd.Series,
    x: np.ndarray | None = None,
    names: Sequence[str] | None = None,
) -> OrdinalFit:
    """Fit ``logit P(Y <= c) = gamma_c - x'beta`` by maximum likelihood.

    Parameters
    ----------
    y : numpy.ndarray or pandas.Series
        Ordinal outcome. Any set of ordered values; they are ranked to
        consecutive codes internally.
    x : numpy.ndarray, optional
        Design matrix without an intercept, which the thresholds supply.
        ``None`` fits the intercept-only model M0.
    names : sequence of str, optional
        Column names for reporting.

    Returns
    -------
    OrdinalFit
        Log-likelihood, thresholds, slopes and convergence.

    Notes
    -----
    Thresholds are estimated as a first cut point plus log gaps, so the
    ordering constraint holds by construction and the optimiser is
    unconstrained. Note the minus sign in the linear predictor; see the module
    docstring.

    Limitations
    -----------
    Proportional odds is an assumption, not a result. When it fails badly, a
    single slope per covariate is a summary of effects that differ by category,
    and the likelihood-ratio tests built on it lose their meaning.
    """
    y_int, design, n_cats = _prepare(y, x)
    n_beta = design.shape[1]
    col_names = list(names) if names is not None else [f"x{i + 1}" for i in range(n_beta)]
    if len(col_names) != n_beta:
        raise ValueError("names must match the number of design columns")

    n = y_int.size
    n_thresholds = n_cats - 1
    rows = np.arange(n)

    def unpack(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        first = vec[0]
        gaps = np.exp(vec[1 : n_cats - 1])
        gamma = np.concatenate([[first], first + np.cumsum(gaps)])
        return gamma, vec[n_cats - 1 :]

    def _pieces(vec: np.ndarray):
        """Upper and lower cumulative probabilities of the observed category."""
        gamma, beta = unpack(vec)
        eta = design @ beta if n_beta else np.zeros(n)
        cum = special.expit(gamma[None, :] - eta[:, None])  # P(Y <= c), c = 0..K-2
        padded = np.hstack([np.zeros((n, 1)), cum, np.ones((n, 1))])
        upper = padded[rows, y_int + 1]
        lower = padded[rows, y_int]
        probs = np.clip(upper - lower, 1e-12, None)
        return gamma, beta, cum, upper, lower, probs

    def negative_loglik(vec: np.ndarray) -> float:
        *_, probs = _pieces(vec)
        return float(-np.sum(np.log(probs)))

    def gradient(vec: np.ndarray) -> np.ndarray:
        _, _, _, upper, lower, probs = _pieces(vec)
        # densities at the two boundaries; zero at the conventions F = 0 and 1
        f_upper = np.where(y_int < n_thresholds, upper * (1.0 - upper), 0.0)
        f_lower = np.where(y_int > 0, lower * (1.0 - lower), 0.0)
        d_gamma = np.zeros(n_thresholds)
        inside_upper = y_int < n_thresholds
        inside_lower = y_int > 0
        np.add.at(d_gamma, y_int[inside_upper], (f_upper / probs)[inside_upper])
        np.add.at(d_gamma, y_int[inside_lower] - 1, -(f_lower / probs)[inside_lower])
        d_beta = -design.T @ ((f_upper - f_lower) / probs) if n_beta else np.zeros(0)
        # chain rule for the first cut point and the log gaps
        d_first = d_gamma.sum()
        gaps = np.exp(vec[1 : n_cats - 1])
        tail = np.cumsum(d_gamma[::-1])[::-1][1:] if n_thresholds > 1 else np.zeros(0)
        d_gaps = gaps * tail
        return -np.concatenate([[d_first], d_gaps, d_beta])

    # start from the marginal cumulative proportions and zero slopes
    counts = np.bincount(y_int, minlength=n_cats).astype(float)
    cum_prop = np.clip(np.cumsum(counts)[:-1] / counts.sum(), 1e-3, 1 - 1e-3)
    gamma0 = special.logit(cum_prop)
    gaps0 = np.log(np.clip(np.diff(gamma0), 1e-3, None)) if n_cats > 2 else np.array([])
    x0 = np.concatenate([[gamma0[0]], gaps0, np.zeros(n_beta)])

    result = optimize.minimize(
        negative_loglik,
        x0,
        jac=gradient,
        method="BFGS",
        options={"maxiter": 2000, "gtol": 1e-6},
    )
    gamma, beta = unpack(result.x)
    return OrdinalFit(
        loglik=float(-result.fun),
        thresholds=gamma,
        beta=beta,
        names=col_names,
        n_params=int(result.x.size),
        converged=bool(result.success),
    )


def fit_ordinal_statsmodels(
    y: np.ndarray | pd.Series, x: np.ndarray | None = None
) -> dict[str, object]:
    """Fit the same model with ``statsmodels`` as a cross-check.

    Parameters
    ----------
    y : numpy.ndarray or pandas.Series
        Ordinal outcome.
    x : numpy.ndarray, optional
        Design matrix without an intercept.

    Returns
    -------
    dict
        ``loglik``, ``params`` and ``model``.

    Raises
    ------
    ImportError
        If statsmodels is not installed. It is an optional extra:
        ``pip install 'hpotential[full]'``.

    Notes
    -----
    ``OrderedModel`` uses the same minus-sign convention as this module, so the
    slope estimates are directly comparable rather than mirrored.
    """
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "statsmodels is an optional extra: pip install 'hpotential[full]'"
        ) from exc
    y_int, design, _ = _prepare(y, x)
    if design.shape[1] == 0:
        design = np.zeros((y_int.size, 0))
    model = OrderedModel(y_int, design, distr="logit")
    res = model.fit(method="bfgs", disp=False)
    return {"loglik": float(res.llf), "params": np.asarray(res.params), "model": res}


def rest_score(responses: np.ndarray | pd.DataFrame, item_index: int) -> np.ndarray:
    """Total over all items except the studied one.

    Parameters
    ----------
    responses : numpy.ndarray or pandas.DataFrame
        Persons by items.
    item_index : int
        Position of the studied item.

    Returns
    -------
    numpy.ndarray
        The rest score per person.

    Notes
    -----
    The rest score is the default matching variable. Matching on the total
    score includes the studied item on both sides of the comparison, which
    shrinks any group difference on that item towards zero.
    """
    values = (
        responses.to_numpy(dtype=float)
        if isinstance(responses, pd.DataFrame)
        else np.asarray(responses, dtype=float)
    )
    return np.delete(values, item_index, axis=1).sum(axis=1)


@dataclass
class DIFResult:
    """Result of testing one item.

    Attributes
    ----------
    item : str
        Item label.
    chi2_uniform, chi2_nonuniform, chi2_total : float
        Likelihood-ratio statistics for M2 vs M1, M3 vs M2 and M3 vs M1.
    df_uniform, df_nonuniform, df_total : int
        Degrees of freedom: 1, 1 and 2.
    p_uniform, p_nonuniform, p_total : float
        The matching p values.
    delta_r2 : float
        McFadden's change in R-squared between M3 and M1, against M0.
    delta_r2_uniform : float
        The same quantity for M2 against M1, so that a uniform effect can be
        sized on its own.
    beta_group : float
        Group coefficient in M2, under the minus-sign convention.
    beta_interaction : float
        Interaction coefficient in M3.
    classification : str
        ``"none"``, ``"uniform"``, ``"non-uniform"`` or ``"mixed"``.
    magnitude : str
        ``"negligible"``, ``"moderate"`` or ``"large"``, from ``delta_r2``.
    flagged : bool
        True when the test is significant and ``delta_r2 >= 0.03``.
    n : int
        Complete cases used.
    matching : str
        ``"rest"`` or ``"total"``.
    loglik : dict
        Log-likelihoods of M0, M1, M2 and M3.
    """

    item: str
    chi2_uniform: float
    chi2_nonuniform: float
    chi2_total: float
    df_uniform: int
    df_nonuniform: int
    df_total: int
    p_uniform: float
    p_nonuniform: float
    p_total: float
    delta_r2: float
    delta_r2_uniform: float
    beta_group: float
    beta_interaction: float
    classification: str
    magnitude: str
    flagged: bool
    n: int
    matching: str
    loglik: dict[str, float]

    def to_series(self) -> pd.Series:
        """Return the result as a row for a summary table."""
        return pd.Series(
            {
                "item": self.item,
                "n": self.n,
                "matching": self.matching,
                "chi2_total": self.chi2_total,
                "p_total": self.p_total,
                "chi2_uniform": self.chi2_uniform,
                "p_uniform": self.p_uniform,
                "chi2_nonuniform": self.chi2_nonuniform,
                "p_nonuniform": self.p_nonuniform,
                "delta_r2": self.delta_r2,
                "beta_group": self.beta_group,
                "beta_interaction": self.beta_interaction,
                "classification": self.classification,
                "magnitude": self.magnitude,
                "flagged": self.flagged,
            }
        )


def _chi2_sf(stat: float, df: int) -> float:
    """Upper tail of the chi-squared distribution."""
    return float(special.gammaincc(df / 2.0, max(stat, 0.0) / 2.0))


def _magnitude(delta_r2: float) -> str:
    """Label an effect size on McFadden's change in R-squared."""
    if delta_r2 < NEGLIGIBLE_MAX:
        return "negligible"
    if delta_r2 < MODERATE_MAX:
        return "moderate"
    return "large"


def ordinal_logistic_dif(
    item: np.ndarray | pd.Series,
    group: np.ndarray | pd.Series,
    match: np.ndarray | pd.Series,
    item_name: str = "item",
    alpha: float = 0.05,
    matching: Literal["rest", "total"] = "rest",
    standardise_match: bool = True,
) -> DIFResult:
    """Test one item for uniform and non-uniform DIF.

    Parameters
    ----------
    item : numpy.ndarray or pandas.Series
        Ordinal responses to the studied item.
    group : numpy.ndarray or pandas.Series
        Two-level group indicator. Coded 0 for the reference group and 1 for
        the focal group; any two distinct values are accepted and the lower is
        taken as the reference.
    match : numpy.ndarray or pandas.Series
        Matching variable, normally the rest score. See :func:`rest_score`.
    item_name : str, default "item"
        Label for reporting.
    alpha : float, default 0.05
        Significance level for the likelihood-ratio tests.
    matching : {"rest", "total"}, default "rest"
        Recorded in the result so a table says which was used. It does not
        change the computation; pass the matching variable you mean.
    standardise_match : bool, default True
        Centre and scale the matching variable. This leaves the tests
        unchanged and makes the group coefficient readable as the group
        difference at the mean match rather than at a match of zero.

    Returns
    -------
    DIFResult
        Statistics, effect size and classification.

    Notes
    -----
    Rows with any missing value are dropped. The classification requires both a
    significant likelihood-ratio test and ``Delta R^2 >= 0.03``; ``"mixed"``
    means both the uniform and the non-uniform tests were significant.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> g = rng.integers(0, 2, n)
    >>> m = rng.normal(size=n)
    >>> y = np.clip(np.round(m + 2.5), 0, 4).astype(int)
    >>> res = ordinal_logistic_dif(y, g, m, item_name="demo")
    >>> res.classification
    'none'
    """
    y = np.asarray(item, dtype=float).ravel()
    g = np.asarray(group).ravel()
    m = np.asarray(match, dtype=float).ravel()
    if not (y.size == g.size == m.size):
        raise ValueError("item, group and match must have the same length")
    keep = ~(pd.isna(y) | pd.isna(m) | pd.isna(g))
    y, g, m = y[keep], g[keep], m[keep]
    levels = np.unique(g)
    if levels.size != 2:
        raise ValueError(f"group must have exactly two levels, found {levels.size}")
    g_bin = (g == levels[1]).astype(float)
    if standardise_match:
        sd = m.std(ddof=1)
        m = (m - m.mean()) / (sd if sd > 0 else 1.0)

    m0 = fit_proportional_odds(y, None)
    m1 = fit_proportional_odds(y, m[:, None], names=["match"])
    m2 = fit_proportional_odds(y, np.column_stack([m, g_bin]), names=["match", "group"])
    m3 = fit_proportional_odds(
        np.asarray(y),
        np.column_stack([m, g_bin, m * g_bin]),
        names=["match", "group", "match:group"],
    )

    chi2_u = 2.0 * (m2.loglik - m1.loglik)
    chi2_nu = 2.0 * (m3.loglik - m2.loglik)
    chi2_t = 2.0 * (m3.loglik - m1.loglik)
    p_u, p_nu, p_t = _chi2_sf(chi2_u, 1), _chi2_sf(chi2_nu, 1), _chi2_sf(chi2_t, 2)

    null = -m0.loglik
    delta_r2 = (m3.loglik - m1.loglik) / null if null > 0 else float("nan")
    delta_r2_u = (m2.loglik - m1.loglik) / null if null > 0 else float("nan")

    sig_u, sig_nu = p_u < alpha, p_nu < alpha
    big = delta_r2 >= NEGLIGIBLE_MAX
    if not big or not (sig_u or sig_nu):
        classification = "none"
    elif sig_u and sig_nu:
        classification = "mixed"
    elif sig_nu:
        classification = "non-uniform"
    else:
        classification = "uniform"

    return DIFResult(
        item=item_name,
        chi2_uniform=float(chi2_u),
        chi2_nonuniform=float(chi2_nu),
        chi2_total=float(chi2_t),
        df_uniform=1,
        df_nonuniform=1,
        df_total=2,
        p_uniform=float(p_u),
        p_nonuniform=float(p_nu),
        p_total=float(p_t),
        delta_r2=float(delta_r2),
        delta_r2_uniform=float(delta_r2_u),
        beta_group=float(m2.coefficient("group")),
        beta_interaction=float(m3.coefficient("match:group")),
        classification=classification,
        magnitude=_magnitude(float(delta_r2)),
        flagged=classification != "none",
        n=int(y.size),
        matching=matching,
        loglik={"M0": m0.loglik, "M1": m1.loglik, "M2": m2.loglik, "M3": m3.loglik},
    )


def dif_scan(
    responses: np.ndarray | pd.DataFrame,
    group: np.ndarray | pd.Series,
    matching: Literal["rest", "total"] = "rest",
    alpha: float = 0.05,
    item_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Test every item in a scale.

    Parameters
    ----------
    responses : numpy.ndarray or pandas.DataFrame
        Persons by items.
    group : numpy.ndarray or pandas.Series
        Two-level group indicator.
    matching : {"rest", "total"}, default "rest"
        ``"rest"`` matches each item on the sum of the other items, which is
        the default and the defensible choice. ``"total"`` matches on the sum
        of all items including the studied one, and biases the test towards
        finding no DIF.
    alpha : float, default 0.05
        Significance level.
    item_names : sequence of str, optional
        Item labels. Taken from ``responses`` when it is a frame.

    Returns
    -------
    pandas.DataFrame
        One row per item, indexed by item, from :meth:`DIFResult.to_series`.

    Notes
    -----
    No multiplicity adjustment is applied. With 30 items at alpha 0.05 roughly
    one or two items are expected to reach significance with no DIF present,
    which is one reason the effect-size requirement is part of the flag. If a
    controlled family-wise or false-discovery rate is wanted, apply it to the
    ``p_total`` column and say so in the report.

    A purified scan, where flagged items are removed from the matching variable
    and the scan is repeated, is often recommended. It is not done here because
    it needs a stopping rule and an anchor policy that should be prespecified
    rather than chosen by a function default.
    """
    values = (
        responses.to_numpy(dtype=float)
        if isinstance(responses, pd.DataFrame)
        else np.asarray(responses, dtype=float)
    )
    names = (
        list(item_names)
        if item_names is not None
        else (
            [str(c) for c in responses.columns]
            if isinstance(responses, pd.DataFrame)
            else [f"item{j + 1}" for j in range(values.shape[1])]
        )
    )
    rows = []
    for j, name in enumerate(names):
        match = rest_score(values, j) if matching == "rest" else np.nansum(values, axis=1)
        rows.append(
            ordinal_logistic_dif(
                values[:, j], group, match, item_name=name, alpha=alpha, matching=matching
            ).to_series()
        )
    return pd.DataFrame(rows).set_index("item")
