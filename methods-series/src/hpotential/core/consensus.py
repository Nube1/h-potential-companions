"""Modified Delphi consensus for item retention, and its error rates.

The rule
--------
Panellists rate the relevance of each item on a 9-point scale. An item is
retained when all three of the following hold:

    median >= 7,   IQR <= ``iqr_threshold``,   I-CVI >= 0.78,

where the item-level content validity index is the proportion of panellists
rating the item 7 or higher.

Why the default IQR threshold is 2.0 and not 1.5
------------------------------------------------
The threshold is a constructor argument because it drives the behaviour of the
whole rule, and the value usually quoted in the literature is not defensible on
integer 9-point data. The operating characteristics were simulated in
``hpotential.analyses.paper3_consensus`` (published results in
``tests/fixtures/p3_delphi_oc.json``), generating
ratings as ``x = clip(round(N(mu, sigma)), 1, 9)`` and repeating 20000 panels
per point:

* On integer data a threshold of 1.5 behaves almost identically to 1.0, because
  the IQR of integer ratings moves in steps. At panel size 30, for an item with
  true endorsement 0.90, the false-drop rate is 0.694 at a threshold of 1.5 and
  0.707 at 1.0 once panel disagreement reaches sigma = 1.2. It stays above half
  across the range studied: 0.580 and 0.564 at sigma = 1.5, 0.570 and 0.564 at
  sigma = 2.4.
* The false-drop rate rises with panel size, from 0.38 at n = 12 to 0.81 at
  n = 60. More panellists make the rule stricter, not better. The reason is
  visible in which criterion fails: among dropped items at n = 30 the IQR fails
  in 99 percent of cases and the median in none. The rule as usually written is
  a rule about panel agreement, not about content validity.
* At a threshold of 2.0 the false-drop rate for the same item is 0.023 at
  sigma = 1.2, 0.026 at sigma = 1.5 and 0.147 at sigma = 2.4.

So 2.0 is the default here. Set ``iqr_threshold=1.5`` to reproduce the rule as
usually specified, and expect the behaviour above.

One consequence of the default should be said out loud. On a 1 to 9 scale with
a relevance cut of 7, an item that meets I-CVI >= 0.78 has its lower quartile
at 7 or above and its upper quartile at 9 or below, so its IQR cannot exceed 2.
The dispersion criterion at a threshold of 2.0 is therefore never the binding
one: the rule reduces to median and endorsement, and the IQR is reported as a
description of the panel rather than as a gate. That is the intended
behaviour, since dispersion at 1.5 was binding on almost every dropped item.
It is checked over a million simulated panels in
``tests/test_core_consensus.py``, and proved exhaustively for panels up to 14
by :func:`iqr_lemma_violations`. A
panel that wants dispersion to bind should say so and set the threshold to 1.5
or 1.0 with the false-drop rates above in view.

What this module does not do
----------------------------
It applies a decision rule to ratings and reports what the rule did, including
which criterion bound each non-retained item. It does not run a Delphi, does not
weight panellists, and cannot tell whether a panel understood an item. No Delphi
has been run for this series; every figure quoted above is a property of the
stated rule under a stated generative model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import special, stats

__all__ = [
    "DelphiConsensusEngine",
    "DelphiRule",
    "endorsement_probability",
    "iqr_lemma_violations",
    "kendalls_w",
    "mu_for_endorsement",
    "operating_characteristics",
    "retention_masks",
    "scale_cvi",
    "simulate_panel",
    "simulate_ratings",
    "stability",
]

_DECISION_RETAIN = "RETAIN"
_DECISION_DROP = "DROP"


@dataclass(frozen=True)
class DelphiRule:
    """The three-part retention rule.

    Parameters
    ----------
    median_min : float
        Smallest acceptable median rating.
    iqr_threshold : float
        Largest acceptable interquartile range.
    cvi_min : float
        Smallest acceptable item-level content validity index.
    relevance_cut : int
        A rating at or above this counts as relevant when forming the I-CVI.
    scale : tuple of int
        Lowest and highest valid rating, inclusive.
    """

    median_min: float = 7.0
    iqr_threshold: float = 2.0
    cvi_min: float = 0.78
    relevance_cut: int = 7
    scale: tuple[int, int] = (1, 9)


class DelphiConsensusEngine:
    """Apply the retention rule to a panel's ratings.

    Parameters
    ----------
    iqr_threshold : float, default 2.0
        Largest acceptable interquartile range. See the module docstring for
        why the default is 2.0 rather than the 1.5 usually quoted.
    median_min : float, default 7.0
        Smallest acceptable median.
    cvi_min : float, default 0.78
        Smallest acceptable I-CVI. The conventional value for panels of six or
        more raters.
    relevance_cut : int, default 7
        Ratings at or above this count as relevant.
    scale : tuple of int, default (1, 9)
        Valid rating range, inclusive. Ratings outside it raise.

    Notes
    -----
    Quartiles use linear interpolation (``numpy.percentile`` default), matching
    ``hpotential.analyses.paper3_consensus``. Other quartile definitions shift the IQR on
    small panels, which changes retention; the definition is part of the rule
    and must be reported with it.

    Limitations
    -----------
    The engine treats panellists as exchangeable within a group and ratings as
    independent given the item. Real panels talk to each other between rounds,
    which is the point of a Delphi, so round 2 ratings are not independent
    replicates of round 1.
    """

    def __init__(
        self,
        iqr_threshold: float = 2.0,
        median_min: float = 7.0,
        cvi_min: float = 0.78,
        relevance_cut: int = 7,
        scale: tuple[int, int] = (1, 9),
    ) -> None:
        if not scale[0] < scale[1]:
            raise ValueError("scale must be (low, high) with low < high")
        if not scale[0] <= relevance_cut <= scale[1]:
            raise ValueError("relevance_cut must lie inside the scale")
        if iqr_threshold < 0:
            raise ValueError("iqr_threshold must not be negative")
        if not 0.0 <= cvi_min <= 1.0:
            raise ValueError("cvi_min must lie in [0, 1]")
        self.rule = DelphiRule(
            median_min=float(median_min),
            iqr_threshold=float(iqr_threshold),
            cvi_min=float(cvi_min),
            relevance_cut=int(relevance_cut),
            scale=(int(scale[0]), int(scale[1])),
        )

    # ------------------------------------------------------------------ core
    def item_statistics(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """Summarise each item and apply the rule.

        Parameters
        ----------
        ratings : pandas.DataFrame
            Rows are panellists, columns are items, entries are integer ratings
            on the scale. Missing values are dropped item by item, and the
            denominator ``n_raters`` reports what was left.

        Returns
        -------
        pandas.DataFrame
            Indexed by item, with columns ``n_raters``, ``median``, ``q1``,
            ``q3``, ``iqr``, ``i_cvi``, ``kappa_star``, ``ok_median``,
            ``ok_iqr``, ``ok_cvi``, ``binding``, ``decision``. ``binding`` names
            the criteria that failed, joined by ``+``, and is empty for retained
            items. ``kappa_star`` is the chance-corrected agreement of the
            manuscripts, which assumes equiprobable chance ratings and is
            reported for completeness rather than used in the rule.
        """
        frame = _check_ratings(ratings, self.rule.scale)
        rule = self.rule
        rows: list[dict[str, object]] = []
        for item in frame.columns:
            x = frame[item].to_numpy(dtype=float)
            x = x[~np.isnan(x)]
            if x.size == 0:
                raise ValueError(f"item {item!r} has no ratings")
            q1, q3 = np.percentile(x, [25, 75])
            median = float(np.median(x))
            iqr = float(q3 - q1)
            n_rel = int(np.sum(x >= rule.relevance_cut))
            i_cvi = n_rel / x.size
            ok_median = median >= rule.median_min
            ok_iqr = iqr <= rule.iqr_threshold
            ok_cvi = i_cvi >= rule.cvi_min
            binding = "+".join(
                name
                for name, ok in (("median", ok_median), ("IQR", ok_iqr), ("CVI", ok_cvi))
                if not ok
            )
            rows.append(
                {
                    "item": item,
                    "n_raters": int(x.size),
                    "median": median,
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": iqr,
                    "i_cvi": float(i_cvi),
                    "kappa_star": _kappa_star(n_rel, x.size, i_cvi),
                    "ok_median": bool(ok_median),
                    "ok_iqr": bool(ok_iqr),
                    "ok_cvi": bool(ok_cvi),
                    "binding": binding,
                    "decision": _DECISION_RETAIN if binding == "" else _DECISION_DROP,
                }
            )
        return pd.DataFrame(rows).set_index("item")

    def scale_cvi(self, ratings: pd.DataFrame) -> dict[str, float]:
        """Scale-level content validity indices.

        Parameters
        ----------
        ratings : pandas.DataFrame
            Panellists by items.

        Returns
        -------
        dict
            ``s_cvi_ave``, the mean I-CVI over items, and ``s_cvi_ua``, the
            proportion of items every panellist rated as relevant (universal
            agreement). S-CVI/UA falls quickly as the panel grows, because one
            dissenting rater is enough to lose an item.
        """
        return scale_cvi(ratings, relevance_cut=self.rule.relevance_cut, scale=self.rule.scale)

    def kendalls_w(self, ratings: pd.DataFrame) -> dict[str, float]:
        """Kendall's coefficient of concordance with the tie correction.

        Parameters
        ----------
        ratings : pandas.DataFrame
            Panellists by items. Rows with any missing rating are dropped,
            because ranking needs a complete row.

        Returns
        -------
        dict
            See :func:`kendalls_w`.
        """
        return kendalls_w(ratings)

    # ------------------------------------------------------- the whole panel
    def analyse(
        self,
        ratings: pd.DataFrame,
        groups: pd.Series | Sequence[str] | None = None,
        priority_group: str | None = None,
    ) -> DelphiResult:
        """Run the rule, the scale indices, concordance and the safeguard.

        Parameters
        ----------
        ratings : pandas.DataFrame
            Panellists by items.
        groups : pandas.Series or sequence of str, optional
            Stakeholder group of each panellist, aligned to the rows of
            ``ratings``. When given, I-CVI is also reported per group.
        priority_group : str, optional
            The named group whose endorsement triggers the safeguard, for
            example ``"patient"``. Requires ``groups``.

        Returns
        -------
        DelphiResult
            Items, per-group indices, scale indices, concordance and flags.
        """
        items = self.item_statistics(ratings)
        by_group = None
        flags = None
        if groups is not None:
            by_group = self.group_cvi(ratings, groups)
            if priority_group is not None:
                flags = self.priority_flags(items, by_group, priority_group)
        return DelphiResult(
            rule=self.rule,
            items=items,
            scale_indices=self.scale_cvi(ratings),
            concordance=self.kendalls_w(ratings),
            group_cvi=by_group,
            priority_flags=flags,
        )

    def group_cvi(self, ratings: pd.DataFrame, groups: pd.Series | Sequence[str]) -> pd.DataFrame:
        """I-CVI computed separately within each stakeholder group.

        Parameters
        ----------
        ratings : pandas.DataFrame
            Panellists by items.
        groups : pandas.Series or sequence of str
            Group label per panellist, aligned to the rows of ``ratings``.

        Returns
        -------
        pandas.DataFrame
            Items as rows, groups as columns, I-CVI as entries, with the group
            denominators kept in ``frame.attrs["n_raters"]``.
        """
        frame = _check_ratings(ratings, self.rule.scale)
        labels = _check_groups(groups, frame)
        out: dict[str, pd.Series] = {}
        sizes: dict[str, int] = {}
        for label in pd.unique(labels):
            block = frame.loc[labels == label]
            rel = (block >= self.rule.relevance_cut).sum(axis=0)
            out[str(label)] = rel / block.notna().sum(axis=0)
            sizes[str(label)] = int(block.shape[0])
        result = pd.DataFrame(out)
        result.index.name = "item"
        result.attrs["n_raters"] = sizes
        return result

    def priority_flags(
        self,
        items: pd.DataFrame,
        group_cvi: pd.DataFrame,
        priority_group: str,
    ) -> pd.DataFrame:
        """Flag items a named subgroup endorses but the pooled rule drops.

        Parameters
        ----------
        items : pandas.DataFrame
            Output of :meth:`item_statistics`.
        group_cvi : pandas.DataFrame
            Output of :meth:`group_cvi`.
        priority_group : str
            Column of ``group_cvi`` to protect, for example ``"patient"``.

        Returns
        -------
        pandas.DataFrame
            One row per flagged item, with the subgroup I-CVI, the pooled
            I-CVI, the criterion that bound and the group denominator. An empty
            frame with the same columns when nothing is flagged.

        Notes
        -----
        A pooled rule can lose an item that one stakeholder group endorses
        strongly, because pooling averages the group away. Simulated in
        ``hpotential.analyses.paper3_consensus``: an item the patients rate highly
        (mu = 8.3, sigma = 1.2) survives the pooled rule with probability 0.9
        percent on a panel of 6 patients and 24 clinicians, 4.9 percent at 12
        to 18, and 8.6 percent at 15 to 15, when clinician endorsement is 0.50.
        Even at parity the patients cannot carry an item alone.

        The flag is a prompt for the panel to look again, not a decision. It
        says the pooled rule and one group disagree, and that this should be
        recorded rather than absorbed.
        """
        if priority_group not in group_cvi.columns:
            raise KeyError(f"group {priority_group!r} not present; have {list(group_cvi.columns)}")
        sizes = group_cvi.attrs.get("n_raters", {})
        rows = []
        for item in items.index:
            sub = float(group_cvi.loc[item, priority_group])
            retained = items.loc[item, "decision"] == _DECISION_RETAIN
            if sub >= self.rule.cvi_min and not retained:
                rows.append(
                    {
                        "item": item,
                        "priority_group": priority_group,
                        "n_group": sizes.get(priority_group, np.nan),
                        "group_i_cvi": sub,
                        "pooled_i_cvi": float(items.loc[item, "i_cvi"]),
                        "pooled_decision": items.loc[item, "decision"],
                        "binding": items.loc[item, "binding"],
                    }
                )
        columns = [
            "item",
            "priority_group",
            "n_group",
            "group_i_cvi",
            "pooled_i_cvi",
            "pooled_decision",
            "binding",
        ]
        return pd.DataFrame(rows, columns=columns)


@dataclass
class DelphiResult:
    """Everything one round of the rule produced.

    Attributes
    ----------
    rule : DelphiRule
        The rule that was applied, so a table can be read years later.
    items : pandas.DataFrame
        Per-item statistics and decisions.
    scale_indices : dict
        S-CVI/Ave and S-CVI/UA.
    concordance : dict
        Kendall's W and its test.
    group_cvi : pandas.DataFrame or None
        I-CVI by stakeholder group, when groups were supplied.
    priority_flags : pandas.DataFrame or None
        Safeguard flags, when a priority group was named.
    """

    rule: DelphiRule
    items: pd.DataFrame
    scale_indices: dict[str, float]
    concordance: dict[str, float]
    group_cvi: pd.DataFrame | None = None
    priority_flags: pd.DataFrame | None = None

    @property
    def retained(self) -> list[str]:
        """Item identifiers the rule retained."""
        return list(self.items.index[self.items["decision"] == _DECISION_RETAIN])

    def binding_counts(self) -> pd.Series:
        """Count how often each criterion bound among non-retained items.

        Returns
        -------
        pandas.Series
            Counts for ``median``, ``IQR`` and ``CVI``. An item failing two
            criteria counts once against each.
        """
        dropped = self.items[self.items["decision"] != _DECISION_RETAIN]
        counts = {
            "median": int((~dropped["ok_median"]).sum()),
            "IQR": int((~dropped["ok_iqr"]).sum()),
            "CVI": int((~dropped["ok_cvi"]).sum()),
        }
        return pd.Series(counts, name="n_items_bound")


# --------------------------------------------------------------------------
# module-level functions
# --------------------------------------------------------------------------
def scale_cvi(
    ratings: pd.DataFrame,
    relevance_cut: int = 7,
    scale: tuple[int, int] = (1, 9),
) -> dict[str, float]:
    """Scale-level content validity indices.

    Parameters
    ----------
    ratings : pandas.DataFrame
        Panellists by items.
    relevance_cut : int, default 7
        Ratings at or above this count as relevant.
    scale : tuple of int, default (1, 9)
        Valid rating range, inclusive.

    Returns
    -------
    dict
        ``s_cvi_ave`` is the mean of the item-level indices. ``s_cvi_ua`` is the
        proportion of items rated relevant by every panellist. ``n_items`` and
        ``n_raters`` give the denominators.

    Notes
    -----
    S-CVI/Ave and S-CVI/UA answer different questions and are not
    interchangeable. UA is severe: with 30 panellists and independent ratings,
    a single dissent loses the item, so a low UA on a large panel says more
    about the panel size than about the instrument.
    """
    frame = _check_ratings(ratings, scale)
    relevant = frame >= relevance_cut
    denom = frame.notna().sum(axis=0)
    i_cvi = relevant.sum(axis=0) / denom
    ua = relevant.all(axis=0)
    return {
        "s_cvi_ave": float(i_cvi.mean()),
        "s_cvi_ua": float(ua.mean()),
        "n_items": int(frame.shape[1]),
        "n_raters": int(frame.shape[0]),
    }


def kendalls_w(ratings: pd.DataFrame | np.ndarray) -> dict[str, float]:
    """Kendall's coefficient of concordance, corrected for ties.

    Parameters
    ----------
    ratings : pandas.DataFrame or numpy.ndarray
        Rows are raters (M of them), columns are items (J of them). Rows with
        any missing value are dropped, because a rank needs a complete row.

    Returns
    -------
    dict
        ``W`` in [0, 1]; ``W_uncorrected`` computed without the tie term, for
        comparison; ``chi2 = M (J - 1) W`` with ``df = J - 1`` and its ``p``
        value; ``n_raters``; ``n_items``; ``tie_correction``, the value of
        ``sum_i T_i``.

    Notes
    -----
    Ratings are converted to midranks within each rater. With ranks summed over
    raters to give ``R_j``, and ``S = sum_j (R_j - mean R)^2``,

    .. math::
        W = \\frac{12 S}{M^2 (J^3 - J) - M \\sum_i T_i},
        \\qquad T_i = \\sum_g (t_{ig}^3 - t_{ig}),

    where ``t_ig`` is the size of the g-th tie group in rater i. Dropping the
    tie term understates W, sometimes badly: a 9-point scale applied to 20
    items forces ties, and a panel in perfect agreement then scores below 1.

    Limitations
    -----------
    The chi-squared approximation needs roughly J > 7 items. On short item sets
    read W as a descriptive statistic and ignore the p value. W measures whether
    raters order items the same way, not whether they rate them highly: a panel
    that agrees every item is irrelevant has W near 1.
    """
    values = _ranking_matrix(ratings)
    n_raters, n_items = values.shape
    if n_items < 2:
        raise ValueError("Kendall's W needs at least two items")
    if n_raters < 2:
        raise ValueError("Kendall's W needs at least two raters")

    ranks = np.vstack([_midranks(row) for row in values])
    rank_sums = ranks.sum(axis=0)
    s = float(np.sum((rank_sums - rank_sums.mean()) ** 2))

    tie_term = float(sum(_tie_term(row) for row in values))
    denom_uncorrected = n_raters**2 * (n_items**3 - n_items)
    denom = denom_uncorrected - n_raters * tie_term
    if denom <= 0:
        # Every rater tied every item: no ordering information at all.
        return {
            "W": float("nan"),
            "W_uncorrected": 0.0,
            "chi2": float("nan"),
            "df": n_items - 1,
            "p_value": float("nan"),
            "n_raters": n_raters,
            "n_items": n_items,
            "tie_correction": tie_term,
        }
    w = 12.0 * s / denom
    w_unc = 12.0 * s / denom_uncorrected
    chi2 = n_raters * (n_items - 1) * w
    df = n_items - 1
    return {
        "W": float(w),
        "W_uncorrected": float(w_unc),
        "chi2": float(chi2),
        "df": int(df),
        "p_value": float(special.gammaincc(df / 2.0, chi2 / 2.0)),
        "n_raters": int(n_raters),
        "n_items": int(n_items),
        "tie_correction": tie_term,
    }


def stability(
    round1: pd.DataFrame,
    round2: pd.DataFrame,
    relevance_cut: int = 7,
    scale: tuple[int, int] = (1, 9),
    tolerance: float = 0.15,
) -> pd.DataFrame:
    """Compare two Delphi rounds item by item.

    Parameters
    ----------
    round1, round2 : pandas.DataFrame
        Panellists by items, one frame per round. The panels need not be the
        same size; attrition is reported. Only items present in both rounds are
        compared.
    relevance_cut : int, default 7
        Ratings at or above this count as relevant.
    scale : tuple of int, default (1, 9)
        Valid rating range, inclusive.
    tolerance : float, default 0.15
        Largest absolute change in I-CVI treated as stable.

    Returns
    -------
    pandas.DataFrame
        Indexed by item, with ``i_cvi_r1``, ``i_cvi_r2``, ``delta_i_cvi``
        (absolute), ``median_r1``, ``median_r2``, ``delta_median`` and
        ``stable``. ``frame.attrs["attrition"]`` holds ``1 - N2 / N1``.

    Notes
    -----
    An earlier draft took both rounds and used only the second, so the reported
    concordance described one round while claiming to describe change. The two
    questions are kept apart here: :func:`kendalls_w` takes a single round, and
    this function takes two.

    Limitations
    -----------
    Stability of an index is not agreement. Two rounds can produce the same
    I-CVI with different panellists changing their minds in opposite
    directions. Where panellist identifiers are available, compare individuals
    rather than the index.
    """
    f1 = _check_ratings(round1, scale)
    f2 = _check_ratings(round2, scale)
    shared = [c for c in f1.columns if c in set(f2.columns)]
    if not shared:
        raise ValueError("the two rounds share no items")
    rows = []
    for item in shared:
        x1 = f1[item].dropna().to_numpy(dtype=float)
        x2 = f2[item].dropna().to_numpy(dtype=float)
        cvi1 = float(np.mean(x1 >= relevance_cut))
        cvi2 = float(np.mean(x2 >= relevance_cut))
        med1, med2 = float(np.median(x1)), float(np.median(x2))
        rows.append(
            {
                "item": item,
                "i_cvi_r1": cvi1,
                "i_cvi_r2": cvi2,
                "delta_i_cvi": abs(cvi2 - cvi1),
                "median_r1": med1,
                "median_r2": med2,
                "delta_median": med2 - med1,
                "stable": abs(cvi2 - cvi1) <= tolerance,
            }
        )
    out = pd.DataFrame(rows).set_index("item")
    n1, n2 = f1.shape[0], f2.shape[0]
    out.attrs["attrition"] = float(1.0 - n2 / n1) if n1 else float("nan")
    out.attrs["n_round1"] = int(n1)
    out.attrs["n_round2"] = int(n2)
    return out


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _check_ratings(ratings: pd.DataFrame, scale: tuple[int, int]) -> pd.DataFrame:
    """Validate a ratings frame and return it as floats."""
    if not isinstance(ratings, pd.DataFrame):
        ratings = pd.DataFrame(np.asarray(ratings))
    if ratings.shape[0] < 1 or ratings.shape[1] < 1:
        raise ValueError("ratings must have at least one rater and one item")
    frame = ratings.apply(pd.to_numeric, errors="coerce").astype(float)
    values = frame.to_numpy()
    finite = values[~np.isnan(values)]
    if finite.size and (finite.min() < scale[0] or finite.max() > scale[1]):
        raise ValueError(
            f"ratings outside the scale {scale}: observed " f"[{finite.min():g}, {finite.max():g}]"
        )
    return frame


def _check_groups(groups: pd.Series | Sequence[str], frame: pd.DataFrame) -> pd.Series:
    """Align group labels to the rows of a ratings frame."""
    if isinstance(groups, pd.Series):
        labels = (
            groups.reindex(frame.index)
            if groups.index.equals(frame.index)
            else pd.Series(groups.to_numpy(), index=frame.index)
        )
    else:
        labels = pd.Series(list(groups), index=frame.index)
    if len(labels) != frame.shape[0]:
        raise ValueError("groups must have one label per panellist")
    if labels.isna().any():
        raise ValueError("every panellist needs a group label")
    return labels.astype(str)


def _ranking_matrix(ratings: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Return a complete-case float matrix of raters by items."""
    values = (
        ratings.to_numpy(dtype=float)
        if isinstance(ratings, pd.DataFrame)
        else np.asarray(ratings, dtype=float)
    )
    if values.ndim != 2:
        raise ValueError("ratings must be two dimensional (raters by items)")
    keep = ~np.isnan(values).any(axis=1)
    return values[keep]


def _midranks(row: np.ndarray) -> np.ndarray:
    """Ranks of one rater's ratings, ties sharing their mean rank."""
    order = np.argsort(row, kind="mergesort")
    ranks = np.empty(row.size, dtype=float)
    ranks[order] = np.arange(1, row.size + 1, dtype=float)
    # average within tie groups
    values = row[order]
    start = 0
    for i in range(1, row.size + 1):
        if i == row.size or values[i] != values[start]:
            if i - start > 1:
                idx = order[start:i]
                ranks[idx] = ranks[idx].mean()
            start = i
    return ranks


def _tie_term(row: np.ndarray) -> float:
    """``sum_g (t^3 - t)`` over the tie groups of one rater."""
    _, counts = np.unique(row, return_counts=True)
    counts = counts[counts > 1].astype(float)
    return float(np.sum(counts**3 - counts))


def _kappa_star(n_relevant: int, n_raters: int, i_cvi: float) -> float:
    """Chance-corrected agreement as written in the manuscripts.

    ``P_c = C(N, a) 2^-N`` and ``kappa* = (I-CVI - P_c) / (1 - P_c)``. The
    chance model assumes a panellist is equally likely to call an item relevant
    or not, which is generous to the index; it is reported, not used.
    """
    if n_raters == 0:
        return float("nan")
    log_pc = (
        special.gammaln(n_raters + 1)
        - special.gammaln(n_relevant + 1)
        - special.gammaln(n_raters - n_relevant + 1)
        - n_raters * np.log(2.0)
    )
    p_c = float(np.exp(log_pc))
    if p_c >= 1.0:
        return float("nan")
    return float((i_cvi - p_c) / (1.0 - p_c))


def simulate_panel(
    n_raters: int,
    item_means: Iterable[float],
    sigma: float = 1.2,
    scale: tuple[int, int] = (1, 9),
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic panel ratings for testing and examples.

    Parameters
    ----------
    n_raters : int
        Panel size.
    item_means : iterable of float
        One latent mean per item, on the rating scale.
    sigma : float, default 1.2
        Panel disagreement, the standard deviation before rounding.
    scale : tuple of int, default (1, 9)
        Valid rating range, inclusive.
    seed : int, optional
        Seed for the random generator.

    Returns
    -------
    pandas.DataFrame
        Panellists by items, integer ratings.

    Notes
    -----
    The generative model is ``x = clip(round(N(mu, sigma)), low, high)``, the
    same one used in ``hpotential.analyses.paper3_consensus``. It is a convenience for
    exercising the code; it is not a model of how experts think.
    """
    rng = np.random.default_rng(seed)
    mus = np.asarray(list(item_means), dtype=float)
    draws = rng.normal(mus, sigma, size=(int(n_raters), mus.size))
    values = np.clip(np.rint(draws), scale[0], scale[1]).astype(int)
    return pd.DataFrame(
        values,
        columns=[f"item{j + 1:02d}" for j in range(mus.size)],
        index=[f"rater{i + 1:02d}" for i in range(int(n_raters))],
    )


# ------------------------------------------------- operating characteristics
def simulate_ratings(
    rng: np.random.Generator,
    mu: float,
    sigma: float,
    shape: tuple[int, ...] | int,
    scale: tuple[int, int] = (1, 9),
) -> np.ndarray:
    """Draw integer panel ratings from the discretised normal model.

    Parameters
    ----------
    rng : numpy.random.Generator
        Explicit generator.
    mu : float
        Latent standing of the item with the panel, on the rating scale.
    sigma : float
        Panel disagreement, the standard deviation before rounding.
    shape : tuple of int or int
        Output shape, normally ``(n_panels, n_raters)``.
    scale : tuple of int, default (1, 9)
        Valid rating range, inclusive.

    Returns
    -------
    numpy.ndarray
        Float array of integer-valued ratings, ``clip(round(N(mu, sigma)), lo, hi)``.

    Notes
    -----
    This is a convenience for exercising a decision rule, not a model of how
    experts think.
    """
    draws = rng.normal(float(mu), float(sigma), shape)
    return np.clip(np.rint(draws), scale[0], scale[1])


def endorsement_probability(
    mu: float, sigma: float, relevance_cut: int = 7, scale: tuple[int, int] = (1, 9)
) -> float:
    """True endorsement ``pi = P(rating >= relevance_cut)`` after rounding.

    Parameters
    ----------
    mu : float
        Latent standing of the item.
    sigma : float
        Panel disagreement before rounding.
    relevance_cut : int, default 7
        Ratings at or above this count as relevant.
    scale : tuple of int, default (1, 9)
        Valid rating range, inclusive.

    Returns
    -------
    float
        The endorsement of the ratings the panel actually reports.

    Notes
    -----
    Endorsement has to be defined on the scale the rule sees, which is the
    rounded one. Rounding sends a continuous draw to ``relevance_cut`` or above
    exactly when it is at least ``relevance_cut - 0.5``, so

        pi = 1 - Phi((cut - 0.5 - mu) / sigma).

    Clipping at either end of the scale does not move this, because the cut sits
    strictly inside the range. Defining ``pi`` on the continuous scale instead
    and then simulating rounded ratings makes the realised endorsement differ
    from the requested one: at ``sigma = 1.2`` a request for 0.90 defined
    continuously realises 0.956 after rounding.

    Examples
    --------
    >>> round(endorsement_probability(8.0379, 1.2), 4)
    0.9
    """
    lo, hi = scale
    if not lo < relevance_cut <= hi:
        raise ValueError("relevance_cut must sit inside the scale")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return float(stats.norm.sf((relevance_cut - 0.5 - float(mu)) / float(sigma)))


def mu_for_endorsement(
    pi: float, sigma: float, relevance_cut: int = 7, scale: tuple[int, int] = (1, 9)
) -> float:
    """Latent standing that gives a requested endorsement after rounding.

    Parameters
    ----------
    pi : float
        Target endorsement, strictly between 0 and 1.
    sigma : float
        Panel disagreement before rounding.
    relevance_cut : int, default 7
        Ratings at or above this count as relevant.
    scale : tuple of int, default (1, 9)
        Valid rating range, inclusive.

    Returns
    -------
    float
        ``mu = cut - 0.5 + sigma * Phi^-1(pi)``, the exact inverse of
        :func:`endorsement_probability`.

    Notes
    -----
    This is the closed form of the Monte Carlo inversion used in the published
    analysis, which reported ``mu = 8.0363`` for ``pi = 0.90`` at
    ``sigma = 1.2`` from 400000 draws per grid point. The closed form gives
    8.0379 and carries no simulation error.

    Examples
    --------
    >>> round(mu_for_endorsement(0.90, 1.2), 4)
    8.0379
    """
    if not 0.0 < pi < 1.0:
        raise ValueError("pi must lie strictly between 0 and 1")
    lo, hi = scale
    if not lo < relevance_cut <= hi:
        raise ValueError("relevance_cut must sit inside the scale")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return float(relevance_cut - 0.5 + float(sigma) * stats.norm.ppf(float(pi)))


def retention_masks(
    ratings: np.ndarray,
    median_min: float = 7.0,
    iqr_threshold: float = 1.5,
    cvi_min: float = 0.78,
    relevance_cut: int = 7,
) -> dict[str, np.ndarray]:
    """Apply the three-part rule to many simulated panels at once.

    Parameters
    ----------
    ratings : numpy.ndarray
        Shape ``(n_panels, n_raters)``.
    median_min : float, default 7.0
        Smallest acceptable median.
    iqr_threshold : float, default 1.5
        Largest acceptable interquartile range. The published operating
        characteristics use the rule as usually specified, 1.5; the reference
        engine defaults to 2.0 for the reason given in the module docstring.
    cvi_min : float, default 0.78
        Smallest acceptable I-CVI.
    relevance_cut : int, default 7
        Ratings at or above this count as relevant.

    Returns
    -------
    dict of numpy.ndarray
        ``retain``, ``ok_median``, ``ok_iqr``, ``ok_cvi``, ``median``, ``iqr``
        and ``cvi``, each of length ``n_panels``.

    Notes
    -----
    Quartiles use linear interpolation, ``numpy.percentile`` with
    ``method="linear"``, which is type 7 in the usual taxonomy. The quantile
    convention is part of the rule: under the Weibull rule the same thresholds
    give a different answer, and the dispersion lemma is false.
    """
    x = np.atleast_2d(np.asarray(ratings, dtype=float))
    median = np.median(x, axis=1)
    q1 = np.percentile(x, 25, axis=1, method="linear")
    q3 = np.percentile(x, 75, axis=1, method="linear")
    iqr = q3 - q1
    cvi = np.mean(x >= relevance_cut, axis=1)
    ok_median = median >= median_min
    ok_iqr = iqr <= iqr_threshold
    ok_cvi = cvi >= cvi_min
    return {
        "retain": ok_median & ok_iqr & ok_cvi,
        "ok_median": ok_median,
        "ok_iqr": ok_iqr,
        "ok_cvi": ok_cvi,
        "median": median,
        "iqr": iqr,
        "cvi": cvi,
    }


def operating_characteristics(
    n_panel: int,
    pi: float,
    sigma: float,
    rng: np.random.Generator,
    n_sim: int = 20000,
    median_min: float = 7.0,
    iqr_threshold: float = 1.5,
    cvi_min: float = 0.78,
    relevance_cut: int = 7,
    scale: tuple[int, int] = (1, 9),
) -> dict[str, float]:
    """Retention probability and binding criterion for one item and panel size.

    Parameters
    ----------
    n_panel : int
        Panel size.
    pi : float
        True endorsement of the item, on the rounded scale.
    sigma : float
        Panel disagreement.
    rng : numpy.random.Generator
        Explicit generator.
    n_sim : int, default 20000
        Simulated panels.
    median_min, iqr_threshold, cvi_min, relevance_cut : float
        The rule.
    scale : tuple of int, default (1, 9)
        Valid rating range.

    Returns
    -------
    dict
        ``p_retain``, ``p_drop``, ``mu``, and the share of dropped panels on
        which each criterion failed (``median_fails``, ``iqr_fails``,
        ``cvi_fails``). The shares sum to more than one when several criteria
        fail together, which is the point of reporting all three.

    Notes
    -----
    Median and interquartile range are order statistics, so the retention
    probability has no useful closed form and is obtained by simulation. The
    number reported is a property of the decision rule, not of any panel that
    has met.
    """
    mu = mu_for_endorsement(float(pi), float(sigma), relevance_cut, scale)
    x = simulate_ratings(rng, mu, sigma, (int(n_sim), int(n_panel)), scale)
    masks = retention_masks(x, median_min, iqr_threshold, cvi_min, relevance_cut)
    dropped = ~masks["retain"]
    n_dropped = int(dropped.sum())
    return {
        "mu": float(mu),
        "p_retain": float(masks["retain"].mean()),
        "p_drop": float(dropped.mean()),
        "median_fails": float(np.mean(~masks["ok_median"][dropped])) if n_dropped else 0.0,
        "iqr_fails": float(np.mean(~masks["ok_iqr"][dropped])) if n_dropped else 0.0,
        "cvi_fails": float(np.mean(~masks["ok_cvi"][dropped])) if n_dropped else 0.0,
    }


def iqr_lemma_violations(
    max_panel: int = 10,
    cvi_min: float = 0.78,
    iqr_bound: float = 2.0,
    relevance_cut: int = 7,
    scale: tuple[int, int] = (1, 9),
    method: str = "linear",
) -> dict[str, object]:
    """Exhaustive check of the dispersion lemma over every small panel.

    Parameters
    ----------
    max_panel : int, default 10
        Largest panel size enumerated. Every multiset of ratings from the scale
        is visited, so the cost grows as ``C(scale + n - 1, n)``.
    cvi_min : float, default 0.78
        I-CVI threshold. The lemma needs ``cvi_min > 0.75`` strictly.
    iqr_bound : float, default 2.0
        The bound the lemma claims.
    relevance_cut : int, default 7
        Ratings at or above this count as relevant.
    scale : tuple of int, default (1, 9)
        Valid rating range, inclusive.
    method : str, default "linear"
        Quantile convention passed to :func:`numpy.percentile`.

    Returns
    -------
    dict
        ``n_panels_checked``, ``n_passing_cvi``, ``violations``,
        ``max_iqr_among_passing``, ``worst_panel`` and ``method``.

    Notes
    -----
    The lemma: on an integer scale ``1..9`` with relevance cut 7, an item whose
    I-CVI is strictly above 0.75 has ``Q1 >= 7`` under the linear-interpolation
    rule, and ``Q3 <= 9`` because 9 is the scale maximum, so its interquartile
    range is at most 2. A dispersion limit of 2 or more can therefore never
    bind, and the rule reduces to the median and the endorsement.

    Sorted multisets suffice because every statistic in the rule is invariant to
    the order of the ratings. The convention matters: under the Weibull rule
    (type 6) the panel ``(1, 7, 9, 9, 9)`` has I-CVI 0.8 and interquartile range
    5, so the lemma is a statement about a rule and a quantile convention
    together.

    Examples
    --------
    >>> out = iqr_lemma_violations(max_panel=6)
    >>> out["violations"], out["max_iqr_among_passing"]
    (0, 2.0)
    """
    from itertools import combinations_with_replacement

    lo, hi = scale
    values = list(range(lo, hi + 1))
    checked = 0
    passing = 0
    violations = 0
    max_iqr = 0.0
    worst: tuple[int, ...] | None = None
    for n in range(1, int(max_panel) + 1):
        for panel in combinations_with_replacement(values, n):
            checked += 1
            arr = np.asarray(panel, dtype=float)
            cvi = float(np.mean(arr >= relevance_cut))
            if cvi < cvi_min:
                continue
            passing += 1
            q1 = float(np.percentile(arr, 25, method=method))
            q3 = float(np.percentile(arr, 75, method=method))
            iqr = q3 - q1
            if iqr > max_iqr:
                max_iqr, worst = iqr, panel
            if iqr > iqr_bound:
                violations += 1
    return {
        "n_panels_checked": checked,
        "n_passing_cvi": passing,
        "violations": violations,
        "max_iqr_among_passing": max_iqr,
        "worst_panel": worst,
        "method": method,
    }
