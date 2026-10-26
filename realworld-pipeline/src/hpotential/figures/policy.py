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


def _floor_share(ctx: RunContext) -> float:
    """Weighted share of people whose H sits at the normalisation floor."""
    h = ctx.data["geometric_crf"]
    at_floor = h <= h.min() * 10
    return float(np.average(at_floor, weights=ctx.data[ctx.weight]))


def inequality_aversion(ctx: RunContext, index: style.FigureIndex) -> None:
    eps = np.linspace(0, 3, 61)
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    numbers = {}
    groups = [("All", ctx.data)]
    if "race_group" in ctx.data:
        for level, sub in ctx.data.groupby("race_group"):
            if len(sub) >= style.MIN_CELL * 5:
                groups.append((str(level), sub))
    for i, (label, sub) in enumerate(groups):
        ede = [atkinson_ede(sub["geometric_crf"], sub[ctx.weight], e) for e in eps]
        ax.plot(eps, ede, label=label, **style.series(i))
        numbers[label] = {"mean": round(ede[0], 4), "ede_eps1": round(ede[20], 4)}
    floor = _floor_share(ctx)
    numbers["share_at_floor"] = round(floor, 4)
    ax.set_xlabel(r"Inequality aversion $\varepsilon$")
    ax.set_ylabel("Equally distributed equivalent H")
    ax.legend(loc="upper right")
    index.save(
        fig,
        "pol_02_inequality_aversion",
        GROUP,
        "Atkinson equally distributed equivalent of H as inequality aversion rises, overall and by "
        "race and ethnicity group. At zero aversion it equals the weighted mean; the fall with "
        "aversion measures how unequal H is within each group. Above an aversion of one the EDE "
        "is governed by the lowest scores, so it depends on the frozen normalisation floor: "
        f"{floor:.1%} of the weighted population sit at that floor in this run.",
        numbers,
    )


def gap_sources(ctx: RunContext, index: style.FigureIndex) -> None:
    if "income" not in ctx.data:
        return
    q = pd.qcut(ctx.data["income"].rank(method="first"), 5, labels=False)
    low, high = ctx.data[q == 0], ctx.data[q == 4]
    a = ctx.alpha
    parts = {}
    for label, col, coef in [
        ("Biomedical", "biomedical", 1.0),
        ("Social", "social", a),
        ("Structural", "systemic", 1 - a),
    ]:
        lw = np.average(np.log(low[col]), weights=low[ctx.weight])
        hw = np.average(np.log(high[col]), weights=high[ctx.weight])
        parts[label] = coef * (hw - lw)
    total = sum(parts.values())
    check = np.average(np.log(high["geometric_crf"]), weights=high[ctx.weight]) - np.average(
        np.log(low["geometric_crf"]), weights=low[ctx.weight]
    )
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    start = 0.0
    pad = 0.03 * max(abs(total), max(abs(v) for v in parts.values()))
    for i, (label, v) in enumerate(parts.items()):
        ax.bar(i, v, bottom=start if v >= 0 else start + v, **style.bar(i + 1))
        top = max(start, start + v)
        ax.text(i, top + pad, f"{v:+.3f}", ha="center", va="bottom", fontsize=7.5)
        start += v
    ax.bar(len(parts), total, **style.bar(0))
    ax.text(
        len(parts), max(total, 0) + pad, f"{total:+.3f}", ha="center", va="bottom", fontsize=7.5
    )
    ax.axhline(0, color="#1a1a1a", lw=0.6)
    ax.set_ylim(min(0, total) * 1.15, max(start, total) * 1.15)
    ax.set_xticks(range(len(parts) + 1), [*parts, "Total"])
    ax.set_ylabel("Difference in weighted mean log H")
    index.save(
        fig,
        "pol_03_income_gap_sources",
        GROUP,
        "Exact split of the difference in weighted mean log H between the richest and poorest "
        "income fifths into biomedical, social and structural parts, using log H = log P + "
        "a log S + (1 - a) log R. The parts sum to the total; the decomposition is an identity of "
        "the score, not a causal attribution. Scores at the normalisation floor pull mean log H "
        "down, so the size of the total depends on that frozen floor.",
        {
            **{k: round(v, 5) for k, v in parts.items()},
            "total": round(total, 5),
            "direct_check": round(float(check), 5),
        },
    )


ALL = [concentration, inequality_aversion, gap_sources]
