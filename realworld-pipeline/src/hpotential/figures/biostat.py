"""Biostatistics figures: cohort, missingness, survival, calibration, utility."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..evaluation import horizon_status
from . import style
from .context import RunContext
from .stats import bootstrap, n_at_risk, survival_at, weighted_km

GROUP = "Biostatistics"
MODEL_LABELS = {
    "baseline": "Age and sex",
    "additive": "Additive score",
    "geometric_crf": "Geometric CRF (H)",
    "geometric_crf_floor": "H, fixed floor",
    "frailty_index": "Frailty index",
    "comorbidity_index": "Comorbidity count",
}
DOMAIN_LABELS = {
    "biomedical": "Biomedical (P)",
    "social": "Social (S)",
    "systemic": "Structural freedom (R)",
}


def _hlabel(ctx) -> str:
    years = ctx.horizon / 365.25
    return f"{round(years):g}" if abs(years - round(years)) < 0.05 else f"{years:.1f}"


def _years(days):
    return np.asarray(days, float) / 365.25


def cohort_flow(ctx: RunContext, index: style.FigureIndex) -> None:
    steps = []
    if ctx.flow:
        steps += [
            ("Sample persons in SP file", ctx.flow["sp_file_rows"]),
            ("Positive analytic weight", ctx.flow["positive_weight"]),
            ("Status known at a follow-up", ctx.flow["with_follow_up"]),
        ]
    n = len(ctx.data)
    n_test = int((ctx.data["split"] == "test").sum())
    steps += [("Analytic cohort", n), ("Development split", n - n_test), ("Hold-out split", n_test)]
    chain, (dev, hold) = steps[:-2], steps[-2:]
    fig, ax = plt.subplots(figsize=(5.2, 0.62 * (len(chain) + 1) + 0.4))

    def box(x, y, w, label, count):
        shown = f"{count:,}" if count >= style.MIN_CELL else f"<{style.MIN_CELL}"
        ax.add_patch(plt.Rectangle((x - w / 2, y - 0.28), w, 0.56, fill=False, lw=0.7))
        ax.text(x, y, f"{label}\nn = {shown}", ha="center", va="center", fontsize=8)

    arrow = {"arrowstyle": "->", "lw": 0.7}
    top = len(chain)
    for i, (label, count) in enumerate(chain):
        y = top - i
        box(0.5, y, 0.6, label, count)
        if i < len(chain) - 1:
            ax.annotate("", (0.5, y - 0.72), (0.5, y - 0.28), arrowprops=arrow)
    for x, (label, count) in ((0.27, dev), (0.73, hold)):
        box(x, 0, 0.42, label, count)
        ax.annotate("", (x, 0.28), (0.5, 0.72), arrowprops=arrow)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.45, top + 0.45)
    ax.axis("off")
    index.save(
        fig,
        "bio_01_cohort_flow",
        GROUP,
        "Flow from the NHATS sample-person file to the analytic cohort and its random development "
        "and hold-out splits. Counts below the disclosure threshold are suppressed.",
        dict(steps),
    )


def missingness(ctx: RunContext, index: style.FigureIndex) -> None:
    frac = ctx.manifest.get("missing_fraction", {})
    indicators = [c for spec in ctx.cfg.indicators.values() for c in spec["indicators"]]
    values = [frac.get(c, 0.0) for c in indicators]
    domain_of = {c: d for d, spec in ctx.cfg.indicators.items() for c in spec["indicators"]}
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    domains = list(ctx.cfg.indicators)
    for i, (c, v) in enumerate(zip(indicators, values)):
        ax.barh(i, v, **style.bar(domains.index(domain_of[c])))
    ax.set_yticks(range(len(indicators)), [c.replace("_indicator_", " ") for c in indicators])
    ax.invert_yaxis()
    ax.set_xlabel("Share of rows missing before imputation")
    handles = [plt.Rectangle((0, 0), 1, 1, **style.bar(i)) for i in range(len(domains))]
    ax.legend(
        handles,
        [DOMAIN_LABELS.get(d, d) for d in domains],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )
    index.save(
        fig,
        "bio_02_missingness",
        GROUP,
        "Share of missing values for each indicator after NHATS special codes were set to missing. "
        "Fill pattern marks the domain.",
        dict(zip(indicators, values)),
    )


def domain_distributions(ctx: RunContext, index: style.FigureIndex) -> None:
    cols = ["biomedical", "social", "systemic", "geometric_crf"]
    labels = [*DOMAIN_LABELS.values(), r"$H = P\,S^{a}R^{1-a}$"]
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 4.2), sharex=True)
    w = ctx.data[ctx.weight].to_numpy()
    bins = np.linspace(0, 1, 26)
    stats = {}
    for i, (ax, c, label) in enumerate(zip(axes.flat, cols, labels)):
        x = ctx.data[c].to_numpy()
        ax.hist(x, bins=bins, weights=w / w.sum(), histtype="stepfilled", **style.bar(i % 3 + 1))
        m = np.average(x, weights=w)
        ax.axvline(m, **style.series(0), lw=0.9)
        ax.set_title(label)
        ax.set_ylabel("Weighted share")
        stats[c] = round(float(m), 4)
    for ax in axes[1]:
        ax.set_xlabel("Score on the unit interval")
    index.save(
        fig,
        "bio_03_domain_distributions",
        GROUP,
        "Survey-weighted distributions of the three domain scores and the geometric capability "
        "score H. The vertical line marks the weighted mean.",
        stats,
    )


def _tertiles(x: pd.Series) -> pd.Series:
    return pd.qcut(x, 3, labels=["Lowest third of H", "Middle third", "Highest third"])


def survival_by_h(ctx: RunContext, index: style.FigureIndex) -> None:
    for e_i, (endpoint, spec) in enumerate(ctx.endpoints.items()):
        frame = ctx.data.assign(group=_tertiles(ctx.data["geometric_crf"]))
        fig, (ax, tab) = plt.subplots(
            2, 1, figsize=(5.4, 4.0), gridspec_kw={"height_ratios": [4, 1]}, sharex=True
        )
        ticks = np.arange(0, ctx.horizon / 365.25 + 0.01, 1)
        summary = {}
        rows = []
        for i, (g, sub) in enumerate(frame.groupby("group", observed=True)):
            km = weighted_km(sub[spec["time"]], sub[spec["event"]], sub[ctx.weight])
            y = 1 - km["survival"] if endpoint != "mortality" else km["survival"]
            ax.step(_years(km["time"]), y, where="post", label=str(g), **style.series(i))
            s_h = survival_at(km, ctx.horizon)
            summary[str(g)] = round(1 - s_h if endpoint != "mortality" else s_h, 4)
            counts = n_at_risk(_years(sub[spec["time"]]), ticks)
            rows.append([f"{c:,}" if c >= style.MIN_CELL else f"<{style.MIN_CELL}" for c in counts])
        ax.set_ylabel("Survival" if endpoint == "mortality" else "Cumulative incidence")
        ax.legend(loc="lower left" if endpoint == "mortality" else "upper left")
        ax.set_title(f"{endpoint.capitalize()} by thirds of H (weighted Kaplan-Meier)")
        tab.axis("off")
        table = tab.table(
            cellText=rows,
            rowLabels=[str(g).split()[0] for g in summary],
            colLabels=[f"{t:g} y" for t in ticks],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        for cell in table.get_celld().values():
            cell.set_linewidth(0)
        ax.set_xlabel("")
        tab.set_xlabel("Years since baseline interview (numbers at risk below)")
        name = f"bio_{4 + e_i:02d}_{'km' if endpoint == 'mortality' else 'cuminc'}_{endpoint}"
        index.save(
            fig,
            name,
            GROUP,
            f"{'Survival' if endpoint == 'mortality' else 'Cumulative incidence of first ' + endpoint} "
            "by thirds of the capability score, survey-weighted Kaplan-Meier estimates, with "
            "unweighted numbers at risk. Event times are interval midpoints between annual rounds.",
            summary,
        )


def _observed_risk(time, event, weight, horizon) -> float:
    return 1 - survival_at(weighted_km(time, event, weight), horizon)


def calibration(ctx: RunContext, index: style.FigureIndex) -> None:
    test = ctx.test()
    endpoints = list(ctx.endpoints)
    fig, axes = plt.subplots(1, len(endpoints), figsize=(3.1 * len(endpoints), 3.1), squeeze=False)
    numbers = {}
    for ax, endpoint in zip(axes[0], endpoints):
        spec = ctx.endpoints[endpoint]
        top = 0.0
        for i, model in enumerate(["baseline", "geometric_crf"]):
            risk = test[ctx.risk_column(endpoint, model)]
            bins = pd.qcut(risk.rank(method="first"), 10, labels=False)
            pred, obs = [], []
            for b in range(10):
                sub = test[bins == b]
                pred.append(sub[ctx.risk_column(endpoint, model)].mean())
                obs.append(
                    _observed_risk(
                        sub[spec["time"]], sub[spec["event"]], sub[ctx.weight], ctx.horizon
                    )
                )
            ax.plot(
                pred,
                obs,
                marker=style.MARKERS[i],
                ms=3.5,
                label=MODEL_LABELS[model],
                **style.series(i),
            )
            top = max(top, max(pred), max(obs))
            numbers[f"{endpoint}/{model}"] = {
                "predicted": np.round(pred, 4).tolist(),
                "observed": np.round(obs, 4).tolist(),
            }
        ax.plot([0, top], [0, top], color="#9a9a9a", lw=0.7, linestyle=":")
        ax.set_xlabel(f"Predicted {_hlabel(ctx)}-year risk")
        ax.set_ylabel("Observed (Kaplan-Meier)")
        ax.set_title(endpoint.capitalize())
        ax.legend(loc="upper left")
    index.save(
        fig,
        "bio_06_calibration",
        GROUP,
        "Calibration in the hold-out split: mean predicted risk against Kaplan-Meier observed risk "
        "within tenths of predicted risk, for the age-and-sex model and the model adding H. "
        "The dotted line is perfect calibration.",
        numbers,
    )


def _net_benefit(test, risk, time, event, weight, horizon, thresholds):
    out = []
    for t in thresholds:
        pos = risk >= t
        share = np.average(pos, weights=test[weight])
        if pos.sum() < style.MIN_CELL:
            out.append(np.nan)
            continue
        r_pos = _observed_risk(
            test.loc[pos, time], test.loc[pos, event], test.loc[pos, weight], horizon
        )
        out.append(share * (r_pos - (1 - r_pos) * t / (1 - t)))
    return np.array(out)


def decision_curves(ctx: RunContext, index: style.FigureIndex) -> None:
    test = ctx.test()
    thresholds = np.linspace(0.02, 0.6, 30)
    endpoints = list(ctx.endpoints)
    fig, axes = plt.subplots(1, len(endpoints), figsize=(3.1 * len(endpoints), 3.1), squeeze=False)
    numbers = {}
    for ax, endpoint in zip(axes[0], endpoints):
        spec = ctx.endpoints[endpoint]
        prevalence = _observed_risk(
            test[spec["time"]], test[spec["event"]], test[ctx.weight], ctx.horizon
        )
        treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
        ax.plot(
            thresholds, treat_all, label="Treat all", color="#9a9a9a", lw=0.8, linestyle=(0, (1, 1))
        )
        ax.axhline(0, color="#1a1a1a", lw=0.6, label="Treat none")
        for i, model in enumerate(["baseline", "frailty_index", "geometric_crf"]):
            col = ctx.risk_column(endpoint, model)
            if col not in test:
                continue
            nb = _net_benefit(
                test, test[col], spec["time"], spec["event"], ctx.weight, ctx.horizon, thresholds
            )
            ax.plot(thresholds, nb, label=MODEL_LABELS[model], **style.series(i))
            numbers[f"{endpoint}/{model}"] = np.round(nb, 4).tolist()
        ax.set_ylim(-0.05, max(prevalence, 0.05) * 1.1)
        ax.set_xlabel("Threshold risk")
        ax.set_ylabel("Net benefit")
        ax.set_title(endpoint.capitalize())
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    index.save(
        fig,
        "bio_07_decision_curves",
        GROUP,
        "Decision curves in the hold-out split. Net benefit uses Kaplan-Meier risk among those "
        "above each threshold, so censoring is respected. Points with fewer than the disclosure "
        "threshold above the cut-off are dropped.",
        {"thresholds": np.round(thresholds, 3).tolist(), **numbers},
    )


def _auc(test, risk_col, spec, horizon, idx):
    sub = test.iloc[idx]
    known, y = horizon_status(sub[spec["time"]], sub[spec["event"]], horizon)
    return roc_auc_score(y[known], sub[risk_col].to_numpy()[known])


def discrimination(ctx: RunContext, index: style.FigureIndex, reps: int | None = None) -> None:
    test = ctx.test().reset_index(drop=True)
    reps = reps or int(ctx.cfg.raw["validation"].get("bootstrap_replicates", 500))
    models = [m for m in MODEL_LABELS if ctx.risk_column(next(iter(ctx.endpoints)), m) in test]
    endpoints = list(ctx.endpoints)
    fig, ax = plt.subplots(figsize=(5.4, 0.42 * len(models) * len(endpoints) + 0.8))
    numbers, y = {}, 0
    ticks, labels = [], []
    for e_i, endpoint in enumerate(endpoints):
        spec = ctx.endpoints[endpoint]
        for i, model in enumerate(models):
            col = ctx.risk_column(endpoint, model)
            point = _auc(test, col, spec, ctx.horizon, np.arange(len(test)))
            boot = bootstrap(
                lambda idx, c=col, sp=spec: _auc(test, c, sp, ctx.horizon, idx),
                len(test),
                reps,
                ctx.cfg.seed,
            )
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            ax.errorbar(
                point,
                y,
                xerr=[[point - lo], [hi - point]],
                fmt=style.MARKERS[i],
                ms=4,
                capsize=2,
                color=style.SHADES[i],
                mfc="white" if i % 2 else style.SHADES[i],
            )
            ticks.append(y)
            labels.append(f"{endpoint.capitalize()}: {MODEL_LABELS[model]}")
            numbers[f"{endpoint}/{model}"] = [round(point, 4), round(lo, 4), round(hi, 4)]
            y -= 1
        y -= 0.6
    ax.set_yticks(ticks, labels)
    ax.axvline(0.5, color="#9a9a9a", lw=0.6, linestyle=":")
    ax.set_xlabel(f"Area under the ROC curve at {_hlabel(ctx)} years (95% bootstrap interval)")
    index.save(
        fig,
        "bio_08_discrimination",
        GROUP,
        f"Hold-out discrimination of each model, with percentile intervals from {reps} bootstrap "
        "resamples of the hold-out rows (models not refitted). Status at the horizon uses people "
        "whose status is known.",
        numbers,
    )


def subgroups(ctx: RunContext, index: style.FigureIndex, reps: int = 200) -> None:
    test = ctx.test().reset_index(drop=True)
    endpoint = next(iter(ctx.endpoints))
    spec = ctx.endpoints[endpoint]
    groups = ctx.cfg.raw["validation"].get("equity_groups", [])
    rows = []
    for g in groups:
        for level, sub in test.groupby(g, dropna=True):
            known, yv = horizon_status(sub[spec["time"]], sub[spec["event"]], ctx.horizon)
            if (
                known.sum() < style.MIN_CELL
                or yv[known].sum() < style.MIN_CELL
                or (known.sum() - yv[known].sum()) < style.MIN_CELL
            ):
                continue
            sub = sub.reset_index(drop=True)
            for i, model in enumerate(["baseline", "geometric_crf"]):
                col = ctx.risk_column(endpoint, model)
                point = _auc(sub, col, spec, ctx.horizon, np.arange(len(sub)))
                boot = bootstrap(
                    lambda idx, s=sub, c=col: _auc(s, c, spec, ctx.horizon, idx),
                    len(sub),
                    reps,
                    ctx.cfg.seed,
                )
                lo, hi = np.nanpercentile(boot, [2.5, 97.5])
                rows.append((f"{g}: {level}", model, point, lo, hi, int(known.sum())))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.4, 0.32 * len(rows) + 0.8))
    labels = list(dict.fromkeys(r[0] for r in rows))
    for label_i, label in enumerate(labels):
        for r in (x for x in rows if x[0] == label):
            i = 0 if r[1] == "baseline" else 1
            y = -label_i + (0.15 if i else -0.15)
            ax.errorbar(
                r[2],
                y,
                xerr=[[r[2] - r[3]], [r[4] - r[2]]],
                fmt=style.MARKERS[i],
                ms=4,
                capsize=2,
                color=style.SHADES[i],
                mfc="white" if i == 0 else style.SHADES[i],
                label=MODEL_LABELS[r[1]] if label_i == 0 else None,
            )
    ax.set_yticks([-i for i in range(len(labels))], labels)
    ax.axvline(0.5, color="#9a9a9a", lw=0.6, linestyle=":")
    ax.set_xlabel(f"{endpoint.capitalize()}: AUC at {_hlabel(ctx)} years (95% bootstrap interval)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    index.save(
        fig,
        "bio_09_subgroup_discrimination",
        GROUP,
        "Hold-out discrimination within the predeclared equity groups, for the age-and-sex model "
        "(open markers) and the model adding H (filled). Groups with fewer than the disclosure "
        "threshold of events or non-events are not shown.",
        {f"{r[0]}/{r[1]}": [round(r[2], 4), round(r[3], 4), round(r[4], 4), r[5]] for r in rows},
    )


ALL = [
    cohort_flow,
    missingness,
    domain_distributions,
    survival_by_h,
    calibration,
    decision_curves,
    discrimination,
    subgroups,
]
