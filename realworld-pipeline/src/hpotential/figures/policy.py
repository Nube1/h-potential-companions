"""Economics and policy figures: income gradient, inequality aversion, gap sources."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import style
from .context import RunContext
from .stats import atkinson_ede, concentration_curve

GROUP = "Economics and policy"


def concentration(ctx: RunContext, index: style.FigureIndex) -> None:
    if "income" not in ctx.data:
        return
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], color="#9a9a9a", lw=0.7, linestyle=":", label="Equality")
    numbers = {}
    for i, (col, label) in enumerate([("geometric_crf", "H"), ("biomedical", "Biomedical (P)")]):
        pop, share, ci = concentration_curve(
            ctx.data[col], ctx.data["income"], ctx.data[ctx.weight]
        )
        ax.plot(pop, share, label=f"{label}: C = {ci:.3f}", **style.series(i))
        numbers[col] = round(ci, 4)
    ax.set_xlabel("Cumulative population share, poorest first")
    ax.set_ylabel("Cumulative share of the score")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    index.save(
        fig,
        "pol_01_concentration_curves",
        GROUP,
        "Concentration curves of H and of biomedical capacity against household income rank, "
        "survey-weighted. A curve above the diagonal (C < 0) means the score is concentrated among "
        "poorer people; below it (C > 0), among richer people. A larger C for H than for P shows "
        "the income gradient added by the context domains.",
        numbers,
    )


VERSIONS = [
    ("", "Preregistered scaling"),
    ("_floor", "Fixed floor (sensitivity)"),
]


def _versions(ctx: RunContext) -> list[tuple[str, str]]:
    return [(s, label) for s, label in VERSIONS if f"geometric_crf{s}" in ctx.data]


def _floor_share(ctx: RunContext) -> float:
    """Weighted share of people with H within 1e-4 of zero (preregistered scaling)."""
    at_floor = ctx.data["geometric_crf"] < 1e-4
    return float(np.average(at_floor, weights=ctx.data[ctx.weight]))


def inequality_aversion(ctx: RunContext, index: style.FigureIndex) -> None:
    eps = np.linspace(0, 3, 61)
    versions = _versions(ctx)
    fig, axes = plt.subplots(
        1, len(versions), figsize=(4.0 * len(versions), 3.2), sharey=True, squeeze=False
    )
    numbers = {}
    groups = [("All", ctx.data)]
    if "race_group" in ctx.data:
        for level, sub in ctx.data.groupby("race_group"):
            if len(sub) >= style.MIN_CELL * 5:
                groups.append((str(level), sub))
    for ax, (suffix, title) in zip(axes[0], versions):
        col = f"geometric_crf{suffix}"
        for i, (label, sub) in enumerate(groups):
            ede = [atkinson_ede(sub[col], sub[ctx.weight], e) for e in eps]
            ax.plot(eps, ede, label=label, **style.series(i))
            numbers[f"{title}/{label}"] = {
                "mean": round(ede[0], 4),
                "ede_eps1": round(ede[20], 4),
                "ede_eps2": round(ede[40], 4),
            }
        ax.set_title(title)
        ax.set_xlabel(r"Inequality aversion $\varepsilon$")
    axes[0][0].set_ylabel("Equally distributed equivalent H")
    axes[0][-1].legend(loc="upper right")
    floor = _floor_share(ctx)
    numbers["share_near_zero_preregistered"] = round(floor, 4)
    index.save(
        fig,
        "pol_02_inequality_aversion",
        GROUP,
        "Atkinson equally distributed equivalent of H as inequality aversion rises, overall and by "
        "race and ethnicity group, under the preregistered scaling (left) and with each domain "
        "mapped onto [0.05, 1] (right). At zero aversion it equals the weighted mean. Under the "
        f"preregistered scaling {floor:.1%} of the weighted population have H near zero, and "
        "above an aversion of one the EDE falls to near zero because of them; the fixed floor "
        "keeps the curve informative. Conclusions should hold under both.",
        numbers,
    )


def _gap(ctx: RunContext, suffix: str) -> dict:
    q = pd.qcut(ctx.data["income"].rank(method="first"), 5, labels=False)
    low, high = ctx.data[q == 0], ctx.data[q == 4]
    a = ctx.alpha
    parts = {}
    for label, col, coef in [
        ("Biomedical", "biomedical", 1.0),
        ("Social", "social", a),
        ("Structural", "systemic", 1 - a),
    ]:
        c = f"{col}{suffix}"
        lw = np.average(np.log(low[c]), weights=low[ctx.weight])
        hw = np.average(np.log(high[c]), weights=high[ctx.weight])
        parts[label] = coef * (hw - lw)
    h = f"geometric_crf{suffix}"
    check = np.average(np.log(high[h]), weights=high[ctx.weight]) - np.average(
        np.log(low[h]), weights=low[ctx.weight]
    )
    return {"parts": parts, "total": sum(parts.values()), "check": float(check)}


def gap_sources(ctx: RunContext, index: style.FigureIndex) -> None:
    if "income" not in ctx.data:
        return
    versions = _versions(ctx)
    fig, axes = plt.subplots(1, len(versions), figsize=(3.8 * len(versions), 3.0), squeeze=False)
    numbers = {}
    for ax, (suffix, title) in zip(axes[0], versions):
        g = _gap(ctx, suffix)
        parts, total = g["parts"], g["total"]
        start = 0.0
        pad = 0.03 * max(abs(total), max(abs(v) for v in parts.values()))
        for i, (_label, v) in enumerate(parts.items()):
            ax.bar(i, v, bottom=start if v >= 0 else start + v, **style.bar(i + 1))
            ax.text(
                i, max(start, start + v) + pad, f"{v:+.3f}", ha="center", va="bottom", fontsize=7.5
            )
            start += v
        ax.bar(len(parts), total, **style.bar(0))
        ax.text(
            len(parts), max(total, 0) + pad, f"{total:+.3f}", ha="center", va="bottom", fontsize=7.5
        )
        ax.axhline(0, color="#1a1a1a", lw=0.6)
        ax.set_ylim(min(0, total) * 1.15, max(start, total) * 1.18)
        ax.set_xticks(range(len(parts) + 1), [*parts, "Total"])
        ax.set_title(title)
        numbers[title] = {
            **{k: round(v, 5) for k, v in parts.items()},
            "total": round(total, 5),
            "direct_check": round(g["check"], 5),
        }
    axes[0][0].set_ylabel("Difference in weighted mean log H")
    index.save(
        fig,
        "pol_03_income_gap_sources",
        GROUP,
        "Exact split of the difference in weighted mean log H between the richest and poorest "
        "income fifths into biomedical, social and structural parts, using log H = log P + "
        "a log S + (1 - a) log R, under the preregistered scaling (left) and the fixed floor "
        "(right). The parts sum to the total; the split is an identity of the score, not a causal "
        "attribution. Scores near zero inflate the preregistered total, so read the shares of "
        "each part, which are more stable, alongside the totals.",
        {
            **numbers,
            "total": numbers[versions[0][1]]["total"],
            "direct_check": numbers[versions[0][1]]["direct_check"],
        },
    )


ALL = [concentration, inequality_aversion, gap_sources]
