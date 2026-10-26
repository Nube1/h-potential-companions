"""Medical figures: domain profiles by morbidity, age and condition."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import style
from .biostat import DOMAIN_LABELS
from .context import RunContext
from .stats import weighted_mean_ci

GROUP = "Medical"
DOMAINS = ["biomedical", "social", "systemic"]


def _morbidity_band(ctx: RunContext) -> pd.Series:
    count = ctx.data.get("comorbidity_index")
    if count is None:
        return pd.Series(np.nan, index=ctx.data.index)
    return pd.cut(
        count, [-0.5, 0.5, 1.5, 2.5, np.inf], labels=["None", "One", "Two", "Three or more"]
    )


def profile_by_morbidity(ctx: RunContext, index: style.FigureIndex) -> None:
    band = _morbidity_band(ctx)
    if band.isna().all():
        return
    levels = [lv for lv in band.cat.categories if (band == lv).sum() >= style.MIN_CELL]
    cols = [*DOMAINS, "geometric_crf"]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    width = 0.8 / len(levels)
    numbers = {}
    for j, lv in enumerate(levels):
        sub = ctx.data[band == lv]
        means = [weighted_mean_ci(sub[c], sub[ctx.weight]) for c in cols]
        x = np.arange(len(cols)) + (j - (len(levels) - 1) / 2) * width
        m = [v[0] for v in means]
        err = [[v[0] - v[1] for v in means], [v[2] - v[0] for v in means]]
        ax.bar(
            x,
            m,
            width,
            yerr=err,
            capsize=1.5,
            error_kw={"lw": 0.6},
            label=f"{lv} (n={len(sub):,})",
            **style.bar(j),
        )
        numbers[str(lv)] = dict(zip(cols, np.round(m, 4).tolist()))
    ax.set_xticks(range(len(cols)), [*[DOMAIN_LABELS[d] for d in DOMAINS], "H"])
    ax.set_ylabel("Weighted mean score")
    ax.set_ylim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.legend(title="Chronic conditions", ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    index.save(
        fig,
        "med_01_profile_by_morbidity",
        GROUP,
        "Weighted mean domain scores and H by number of the ten NHATS chronic conditions reported. "
        "Bars show approximate 95% intervals that ignore clustering. A capability gap that grows "
        "faster than the biomedical gap points to contextual conversion, not disease alone.",
        numbers,
    )


def h_by_age_sex(ctx: RunContext, index: style.FigureIndex) -> None:
    if "age" not in ctx.data or "sex" not in ctx.data:
        return
    bands = pd.cut(
        ctx.data["age"],
        [0, 69.5, 74.5, 79.5, 84.5, 89.5, 200],
        labels=["65-69", "70-74", "75-79", "80-84", "85-89", "90+"],
    )
    sexes = [s for s in ctx.data["sex"].dropna().unique()]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    numbers = {}
    width = 0.8 / max(len(sexes), 1)
    for j, sx in enumerate(sorted(sexes)):
        data, pos = [], []
        for i, b in enumerate(bands.cat.categories):
            sub = ctx.data.loc[(bands == b) & (ctx.data["sex"] == sx), "geometric_crf"].dropna()
            if len(sub) < style.MIN_CELL:
                continue
            data.append(sub.to_numpy())
            pos.append(i + (j - (len(sexes) - 1) / 2) * width)
            numbers[f"{b}/{sx}"] = round(float(sub.median()), 4)
        bp = ax.boxplot(
            data,
            positions=pos,
            widths=width * 0.85,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#1a1a1a", "lw": 1},
            whiskerprops={"lw": 0.6},
            capprops={"lw": 0.6},
        )
        for box in bp["boxes"]:
            box.set(**style.bar(j + 1))
        ax.plot(
            [],
            [],
            marker="s",
            ls="",
            mfc=style.bar(j + 1)["facecolor"],
            mec="#1a1a1a",
            label=str(sx).capitalize(),
        )
    ax.set_xticks(range(len(bands.cat.categories)), bands.cat.categories)
    ax.set_xlabel("Age band (years)")
    ax.set_ylabel("H (unweighted quartiles)")
    ax.legend(loc="upper right")
    index.save(
        fig,
        "med_02_h_by_age_sex",
        GROUP,
        "Distribution of H by age band and sex: medians, quartiles and 1.5 IQR whiskers, "
        "unweighted. Cells below the disclosure threshold are not drawn.",
        numbers,
    )


def condition_contrasts(ctx: RunContext, index: style.FigureIndex) -> None:
    conds = [c for c in ctx.data.columns if c.startswith("cond_")]
    if not conds:
        return
    rows = []
    for c in conds:
        for flag in (1.0, 0.0):
            sub = ctx.data[ctx.data[c] == flag]
            if len(sub) < style.MIN_CELL:
                continue
            rows.append(
                (
                    c[5:].replace("_", " "),
                    flag,
                    *weighted_mean_ci(sub["geometric_crf"], sub[ctx.weight]),
                    len(sub),
                )
            )
    frame = pd.DataFrame(rows, columns=["condition", "flag", "mean", "lo", "hi", "n"])
    order = frame[frame.flag == 1].sort_values("mean")["condition"].tolist()
    fig, ax = plt.subplots(figsize=(5.4, 0.3 * len(order) + 0.9))
    for i, flag in enumerate((0.0, 1.0)):
        sub = frame[frame.flag == flag].set_index("condition").reindex(order)
        y = np.arange(len(order)) + (0.12 if flag else -0.12)
        ax.errorbar(
            sub["mean"],
            y,
            xerr=[sub["mean"] - sub["lo"], sub["hi"] - sub["mean"]],
            fmt=style.MARKERS[i],
            ms=4,
            capsize=1.5,
            color=style.SHADES[i],
            mfc=style.SHADES[i] if flag else "white",
            label="Condition reported" if flag else "Not reported",
        )
    ax.set_yticks(range(len(order)), [o.capitalize() for o in order])
    ax.set_xlabel("Weighted mean H (approximate 95% interval)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    index.save(
        fig,
        "med_03_condition_contrasts",
        GROUP,
        "Weighted mean H among people who reported each chronic condition (filled) and those who "
        "did not (open). Conditions are ordered by mean H among those affected.",
        {f"{r.condition}/{int(r.flag)}": [round(r.mean, 4), int(r.n)] for r in frame.itertuples()},
    )


ALL = [profile_by_morbidity, h_by_age_sex, condition_contrasts]
