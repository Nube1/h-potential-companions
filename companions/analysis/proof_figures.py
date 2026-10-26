"""Figures that show each mathematical result of the three companion papers.

Every figure is drawn from the package functions the papers cite
(hpotential.crf and hpotential.measurement), and each one carries a numeric
check of the proposition it shows. A failed check stops the build.

    python analysis/proof_figures.py            # writes figures/
    python analysis/proof_figures.py --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hpotential.crf import Action, H, allocate, contextual_mean, ede, gap_decomposition, gradients  # noqa: E402
from hpotential.measurement import GaussianMeasurement  # noqa: E402

SHADES = ["#1a1a1a", "#2f4b7c", "#6b6b6b", "#9a9a9a", "#4f6f8f", "#c4c4c4"]
LINES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1)), (0, (1, 1))]
HATCHES = ["", "///", "...", "xxx", "\\\\\\", "ooo"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.3,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "hatch.linewidth": 0.5,
        }
    )


def s(i: int) -> dict:
    return {"color": SHADES[i % 6], "linestyle": LINES[i % 6]}


def b(i: int) -> dict:
    return {
        "facecolor": "white" if i == 0 else SHADES[(i + 2) % 6],
        "edgecolor": "#1a1a1a",
        "hatch": HATCHES[i % 6],
        "linewidth": 0.6,
    }


class Index:
    def __init__(self, out: Path):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []

    def save(self, fig, name, paper, statement, caption, check: dict) -> None:
        if not check.pop("passed"):
            raise AssertionError(f"{name}: numeric check failed: {check}")
        fig.savefig(self.out / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(self.out / f"{name}.png", metadata={"Software": "hpotential"})
        plt.close(fig)
        self.entries.append(
            {
                "name": name,
                "paper": paper,
                "statement": statement,
                "caption": caption,
                "check": check,
            }
        )

    def write(self) -> None:
        (self.out / "figure_index.json").write_text(
            json.dumps(self.entries, indent=2), encoding="utf-8"
        )
        md = [
            "# Proof figures",
            "",
            "Each figure shows one result and carries a numeric check that the build enforces.",
            "",
        ]
        for e in self.entries:
            md += [
                f"## {e['name']} (paper {e['paper']})",
                "",
                f"*Result.* {e['statement']}",
                "",
                e["caption"],
                "",
                f"![{e['name']}]({e['name']}.png)",
                "",
                "Check: " + ", ".join(f"{k} = {v}" for k, v in e["check"].items()),
                "",
            ]
        (self.out / "FIGURES.md").write_text("\n".join(md), encoding="utf-8")


# ----------------------------------------------------------------- paper 1


def level_sets(ix: Index) -> None:
    grid = np.linspace(0.001, 1, 241)
    S, R = np.meshgrid(grid, grid)
    specs = [
        (1, "Arithmetic (compensatory), $\\rho = 1$"),
        (0, "Geometric, $\\rho = 0$"),
        (-1, "Harmonic, $\\rho = -1$"),
        (-math.inf, "Bottleneck, $\\rho \\to -\\infty$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.5), sharey=True)
    levels = [0.1, 0.2, 0.3, 0.45, 0.6, 0.8]
    worst = 0.0
    for ax, (rho, title) in zip(axes, specs):
        Z = np.vectorize(lambda a, c, r=rho: contextual_mean(a, c, r))(S, R)
        cs = ax.contour(S, R, Z, levels=levels, colors="#1a1a1a", linewidths=0.7)
        ax.clabel(cs, fontsize=6, fmt="%.2f")
        ax.plot([0.45, 0.70, 0.89], [0.45, 0.20, 0.01], ls="", marker="o", ms=3, color=SHADES[1])
        ax.set_title(title)
        ax.set_xlabel("Social enablement S")
        ax.set_aspect("equal")
        if rho <= 0:
            worst = max(worst, float(np.abs(Z[0, :]).max()), float(np.abs(Z[:, 0]).max()))
    axes[0].set_ylabel("Structural freedom R")
    ix.save(
        fig,
        "proof_01_level_sets",
        1,
        "For rho <= 0 the contextual mean is zero on each axis (essential domains); only rho > 0 lets one domain compensate for the absence of the other.",
        "Level sets of the contextual mean M(S, R) with equal weights for four members of the CES family. "
        "Markers are the three Table 1 profiles with the same arithmetic mean of 0.45. Curved level sets "
        "that meet the axes at zero show that a shortfall in one domain cannot be offset by the other.",
        {
            "max |M| on the axes for rho <= 0 (grid starts at 0.001)": round(worst, 4),
            "passed": worst < 0.05 and H(1, 0, 0.9) == 0 and H(1, 0.9, 0) == 0,
        },
    )


def ces_family(ix: Index) -> None:
    rhos = np.linspace(-12, 1, 261)
    profiles = [(0.45, 0.45), (0.70, 0.20), (0.89, 0.01)]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    monotone = True
    for i, (sv, rv) in enumerate(profiles):
        m = np.array([contextual_mean(sv, rv, r) for r in rhos])
        monotone &= bool(np.all(np.diff(m) >= -1e-12))
        ax.plot(rhos, m, label=f"(S, R) = ({sv:.2f}, {rv:.2f})", **s(i))
        ax.axhline(min(sv, rv), color=SHADES[i], lw=0.5, ls=":")
    ax.axvline(0, color="#9a9a9a", lw=0.5)
    ax.set_xlabel(r"Substitution parameter $\rho$")
    ax.set_ylabel(r"Contextual mean $M_\rho(S, R)$")
    ax.legend(loc="center left")
    lim = abs(contextual_mean(0.7, 0.2, -800) - 0.2)
    ix.save(
        fig,
        "proof_02_ces_family",
        1,
        "M_rho is non-decreasing in rho, equals the geometric mean at rho = 0, the arithmetic mean at rho = 1, and tends to min(S, R) as rho -> -infinity.",
        "The CES contextual mean against rho for the three Table 1 profiles. Dotted lines mark min(S, R), the "
        "bottleneck limit. The balanced profile is flat; the more unbalanced the profile, the more the choice "
        "of rho matters.",
        {
            "monotone on the grid": monotone,
            "|M_-800(0.7, 0.2) - 0.2|": f"{lim:.2e}",
            "geometric at rho = 0": round(contextual_mean(0.7, 0.2, 0), 6),
            "passed": monotone
            and lim < 1e-2
            and abs(contextual_mean(0.7, 0.2, 0) - math.sqrt(0.14)) < 1e-12,
        },
    )


def ceiling(ix: Index) -> None:
    sv = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    top = 0.0
    for i, rv in enumerate([1.0, 0.6, 0.3, 0.1]):
        ratio = np.array([H(30, x, rv) / 30 for x in sv])
        top = max(top, ratio.max())
        ax.plot(sv, ratio, label=f"R = {rv:g}", **s(i))
    ax.axhline(1, color="#9a9a9a", lw=0.6, ls=":")
    ax.set_xlabel("Social enablement S")
    ax.set_ylabel("Realised share of capacity H / P")
    ax.legend(loc="upper left")
    ix.save(
        fig,
        "proof_03_capacity_ceiling",
        1,
        "0 <= H <= P, with H = P only at S = R = 1 and H = 0 when either context domain is zero; H/P does not depend on the unit of P.",
        "Share of biomedical capacity realised, H/P, against social enablement for four levels of structural "
        "freedom (P = 30 in arbitrary units). The ceiling at one is reached only with both domains at one.",
        {
            "max H/P": round(top, 6),
            "H(30,1,1)/30": H(30, 1, 1) / 30,
            "unit check H(30,.7,.2) = 100 H(.3,.7,.2)": math.isclose(
                H(30, 0.7, 0.2), 100 * H(0.3, 0.7, 0.2)
            ),
            "passed": top <= 1 + 1e-12
            and H(30, 0, 0.5) == 0
            and math.isclose(H(30, 0.7, 0.2), 100 * H(0.3, 0.7, 0.2)),
        },
    )


def balance_penalty(ix: Index) -> None:
    d = np.linspace(0, 0.44, 221)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for i, (rho, label) in enumerate(
        [(1, "Arithmetic"), (0, "Geometric"), (-1, "Harmonic"), (-math.inf, "Bottleneck")]
    ):
        ax.plot(d, [H(1, 0.45 + x, 0.45 - x, rho) for x in d], label=label, **s(i))
    for x, name in [(0.0, "balanced"), (0.15, "moderate"), (0.25, "unbalanced"), (0.44, "extreme")]:
        ax.plot(x, H(1, 0.45 + x, 0.45 - x), marker="o", ms=3.5, color=SHADES[1])
        ax.annotate(
            name,
            (x, H(1, 0.45 + x, 0.45 - x)),
            textcoords="offset points",
            xytext=(3, 4),
            fontsize=7,
        )
    ax.set_xlabel("Imbalance d, with S = 0.45 + d and R = 0.45 - d")
    ax.set_ylabel("H (P = 1)")
    ax.legend(loc="lower left")
    geo = [H(1, 0.45 + x, 0.45 - x) for x in d]
    ix.save(
        fig,
        "proof_04_balance_penalty",
        1,
        "At a fixed arithmetic mean of the context, the geometric score falls as the domains move apart (sqrt(0.45^2 - d^2)); the additive score does not move.",
        "H against the gap between the two context domains when their arithmetic mean is held at 0.45. "
        "Markers are the Table 1 profiles. Every non-compensatory member penalises imbalance.",
        {
            "geometric at d = 0.25": round(H(1, 0.7, 0.2), 4),
            "closed form sqrt(0.45^2 - 0.25^2)": round(math.sqrt(0.45**2 - 0.25**2), 4),
            "passed": all(np.diff(geo) <= 1e-12)
            and math.isclose(H(1, 0.7, 0.2), math.sqrt(0.45**2 - 0.25**2)),
        },
    )


def marginal_effects(ix: Index) -> None:
    sv = np.linspace(0.05, 1, 96)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    worst = 0.0
    for i, rv in enumerate([0.9, 0.5, 0.2]):
        analytic = [gradients(1, x, rv)[1] for x in sv]
        ax.plot(sv, analytic, label=f"R = {rv:g}", **s(i))
        pts = sv[::10]
        h = 1e-6
        fd = [(H(1, x + h, rv) - H(1, x - h, rv)) / (2 * h) for x in pts]
        worst = max(worst, max(abs(a - gradients(1, x, rv)[1]) for a, x in zip(fd, pts)))
        ax.plot(pts, fd, ls="", marker=MARKERS[i], ms=3.5, mfc="white", color=SHADES[i])
    ax.set_xlabel("Social enablement S")
    ax.set_ylabel(r"$\partial H / \partial S$")
    ax.legend(loc="upper right")
    ix.save(
        fig,
        "proof_05_marginal_effects",
        1,
        "dH/dS = alpha H / S: the gain from social enablement is largest where S is low and R is high (complementarity, d2H/dSdR > 0).",
        "Analytic marginal effect of social enablement (lines) and central finite differences (open markers) "
        "for three levels of structural freedom, P = 1 and alpha = 0.5. The same gain in S is worth more "
        "when structural freedom is high.",
        {"max |analytic - finite difference|": f"{worst:.1e}", "passed": worst < 1e-6},
    )


def gap_waterfall(ix: Index) -> None:
    a, c = (0.3, 0.7, 0.85), (0.3, 0.7, 0.2)
    bio, soc, st, total = gap_decomposition(a, c)
    parts = {"Biomedical": bio, "Social": soc, "Structural": st}
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    start = 0.0
    for i, (k, v) in enumerate(parts.items()):
        ax.bar(i, v, bottom=min(start, start + v), **b(i + 1))
        ax.text(i, max(start, start + v) + 0.02, f"{v:+.3f}", ha="center", fontsize=7.5)
        start += v
    ax.bar(3, total, **b(0))
    ax.text(3, total + 0.02, f"{total:+.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(4), [*parts, "Total"])
    ax.set_ylabel("Contribution to log(H_a / H_b)")
    ax.set_ylim(0, total * 1.2)
    direct = math.log(H(*a) / H(*c))
    ix.save(
        fig,
        "proof_06_gap_decomposition",
        1,
        "log(H_a/H_b) = log(P_a/P_b) + alpha log(S_a/S_b) + (1 - alpha) log(R_a/R_b), exactly, with no residual.",
        "Exact log decomposition of the gap between two profiles that differ only in structural freedom "
        "(0.85 against 0.20), from results/gap.csv. The whole gap is structural, as it should be.",
        {
            "sum of parts": round(total, 12),
            "direct log ratio": round(direct, 12),
            "passed": math.isclose(total, direct, abs_tol=1e-12),
        },
    )


# ----------------------------------------------------------------- paper 2


def _model(p_per_domain=3, loads=(0.6, 0.75, 0.9)):
    phi = np.array([[1, 0.3, 0.2], [0.3, 1, 0.4], [0.2, 0.4, 1]])
    lam = np.zeros((3 * p_per_domain, 3))
    for j in range(3 * p_per_domain):
        lam[j, j // p_per_domain] = loads[j % len(loads)]
    psi = 1 - (lam**2).sum(axis=1)
    names = [f"x{j}" for j in range(3 * p_per_domain)]
    return GaussianMeasurement(names, lam, phi, psi), names


def determinacy(ix: Index) -> None:
    loads = np.linspace(0.3, 0.95, 66)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.9))
    for i, p in enumerate([2, 3, 6]):
        det = [
            GaussianMeasurement(
                [f"x{j}" for j in range(p)], np.full((p, 1), lv), [[1]], np.full(p, 1 - lv**2)
            ).implied_determinacy()[0]
            for lv in loads
        ]
        a1.plot(loads, det, label=f"{p} indicators", **s(i))
    a1.axhline(0.8, color="#9a9a9a", lw=0.6, ls=":")
    a1.text(0.31, 0.81, "rejection threshold 0.80", fontsize=7, color="#6b6b6b")
    a1.set_xlabel("Standardised loading")
    a1.set_ylabel("Model-implied determinacy")
    a1.legend(loc="lower right")
    model, names = _model()
    sds = []
    for k in range(0, 4):
        row = {n: [1.0] for n in names}
        for j in range(k):
            row[names[3 + j]] = [np.nan]  # drop social indicators one by one
        _, cov = model.score(row)
        sds.append(math.sqrt(cov[0, 1, 1]))
    a2.bar(range(4), sds, **b(1))
    for kk, v in enumerate(sds):
        a2.text(kk, v + 0.01, f"{v:.3f}", ha="center", fontsize=7.5)
    a2.set_xticks(range(4), ["none", "one", "two", "all three"])
    a2.set_xlabel("Social indicators missing")
    a2.set_ylabel("Posterior SD of the social domain")
    one = GaussianMeasurement(["a", "b"], [[1], [1]], [[1]], [1, 1]).implied_determinacy()[0]
    ix.save(
        fig,
        "proof_07_determinacy_and_missingness",
        2,
        "With a known Gaussian model, determinacy rises with loadings and indicator count, and each missing indicator can only widen the posterior.",
        "Left: model-implied correlation between a single-factor score and its factor, by loading and number "
        "of indicators. Right: posterior standard deviation of the social domain in the three-domain "
        "synthetic model as its indicators go missing; with all three missing it borrows strength from the "
        "correlated domains and stays below the prior SD of one.",
        {
            "determinacy for two unit-loading indicators": round(one, 6),
            "sqrt(2/3)": round(math.sqrt(2 / 3), 6),
            "posterior SDs": [round(x, 4) for x in sds],
            "passed": math.isclose(one, math.sqrt(2 / 3)) and all(np.diff(sds) > 0) and sds[-1] < 1,
        },
    )


def drift(ix: Index) -> None:
    import csv

    rows = list(csv.DictReader((ROOT / "results/measurement.csv").open(encoding="utf-8")))
    shifts = np.linspace(0, 1, 11)
    model, names = _model()
    x0 = {n: [0.0] for n in names}
    base, _ = model.score(x0)
    bias = []
    for d in shifts:
        x = {n: [d if 3 <= j < 6 else 0.0] for j, n in enumerate(names)}
        m, _ = model.score(x)
        bias.append(m[0, 1] - base[0, 1])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.9))
    a1.plot(shifts, bias, **s(0))
    sim = next(r for r in rows if r["scenario"] == "social_intercept_shift" and r["domain"] == "2")
    a1.plot(
        0.5,
        float(sim["bias"]),
        marker="s",
        ms=5,
        mfc="white",
        color=SHADES[1],
        ls="",
        label="Simulation, n = 4,000",
    )
    a1.set_xlabel("Intercept shift in each social indicator")
    a1.set_ylabel("Bias of the social score")
    a1.legend(loc="upper left")
    scen = ["complete", "missing_30pct", "social_intercept_shift"]
    labels = ["Complete", "30% missing", "Social drift"]
    for i, sc in enumerate(scen):
        vals = [float(r["coverage95"]) for r in rows if r["scenario"] == sc]
        a2.bar(np.arange(3) + (i - 1) * 0.26, vals, 0.26, label=labels[i], **b(i))
    a2.axhline(0.95, color="#9a9a9a", lw=0.6, ls=":")
    a2.set_xticks(range(3), ["Biomedical", "Social", "Structural"])
    a2.set_ylabel("Coverage of nominal 95% intervals")
    a2.set_ylim(0.6, 1.0)
    a2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    analytic = float(np.interp(0.5, shifts, bias))
    ix.save(
        fig,
        "proof_08_measurement_drift",
        2,
        "A constant intercept shift moves every score by a fixed amount: correlation with the truth is unchanged while bias appears and interval coverage falls.",
        "Left: bias of the posterior social score as the intercepts of the social indicators drift, from the "
        "known-model formula (line) and from the 4,000-record simulation at a shift of 0.5 (marker). Right: "
        "coverage of 95% posterior intervals by domain and scenario, from results/measurement.csv; the "
        "dotted line is the nominal level.",
        {
            "formula bias at 0.5": round(analytic, 4),
            "simulated bias at 0.5": round(float(sim["bias"]), 4),
            "simulated social coverage under drift": float(sim["coverage95"]),
            "passed": abs(analytic - float(sim["bias"])) < 0.02 and float(sim["coverage95"]) < 0.8,
        },
    )


# ----------------------------------------------------------------- paper 3


def ede_curve(ix: Index) -> None:
    v = [0.1, 0.4, 0.9]
    eps = np.linspace(0, 8, 161)
    e = [ede(v, x) for x in eps]
    ew = [ede(v, x, [3, 1, 1]) for x in eps]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(eps, e, label="Equal weights", **s(0))
    ax.plot(eps, ew, label="Weights 3:1:1", **s(1))
    ax.axhline(np.mean(v), color="#9a9a9a", lw=0.6, ls=":")
    ax.axhline(min(v), color="#9a9a9a", lw=0.6, ls=":")
    ax.text(4, np.mean(v) - 0.025, "mean", ha="center", fontsize=7, color="#6b6b6b")
    ax.text(4, min(v) - 0.025, "minimum", ha="center", fontsize=7, color="#6b6b6b")
    ax.plot(1, ede(v, 1), marker="o", ms=3.5, color=SHADES[0])
    ax.annotate(
        "geometric mean at 1", (1, ede(v, 1)), textcoords="offset points", xytext=(5, 4), fontsize=7
    )
    ax.set_xlabel(r"Inequality aversion $\varepsilon$")
    ax.set_ylabel("Equally distributed equivalent")
    ax.set_ylim(min(v) - 0.04, np.mean(v) + 0.03)
    ax.legend(loc="center right")
    geo = (0.1 * 0.4 * 0.9) ** (1 / 3)
    ix.save(
        fig,
        "proof_09_ede_aversion",
        3,
        "The Atkinson EDE falls from the weighted mean (epsilon = 0) through the weighted geometric mean (epsilon = 1) towards the minimum as epsilon grows; any zero gives EDE = 0 for epsilon >= 1.",
        "EDE of the population (0.1, 0.4, 0.9) against inequality aversion, unweighted and with three times the "
        "weight on the worst-off person. Dotted lines are the mean and the minimum.",
        {
            "EDE at 1": round(ede(v, 1), 6),
            "geometric mean": round(geo, 6),
            "EDE([0,4], 2)": ede([0, 4], 2),
            "passed": all(np.diff(e) <= 1e-12)
            and math.isclose(ede(v, 1), geo)
            and ede([0, 4], 2) == 0,
        },
    )


def allocation(ix: Index) -> None:
    pop = [(1, 0.2, 0.2), (1, 0.8, 0.8)]
    low = Action("support_low", 1, ((0, 0.1, 0.1), (0, 0, 0)))
    high = Action("support_high", 1, ((0, 0, 0), (0, 0.2, 0.2)))
    eps = np.linspace(0, 3, 121)

    def after(a, e):
        post = [tuple(p[k] + a.changes[i][k] for k in range(3)) for i, p in enumerate(pop)]
        return ede([H(*p) for p in post], e)

    el = np.array([after(low, e) for e in eps])
    eh = np.array([after(high, e) for e in eps])
    lo, hi = 0.0, 3.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if after(high, mid) > after(low, mid) else (lo, mid)
    star = (lo + hi) / 2
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(eps, el, label="Support the worse-off (L)", **s(0))
    ax.plot(eps, eh, label="Support the better-off (H)", **s(1))
    ax.axvline(star, color="#9a9a9a", lw=0.6, ls=":")
    ax.text(star + 0.05, eh.max() * 0.98, f"reversal at {star:.3f}", fontsize=7)
    ax.set_xlabel(r"Inequality aversion $\varepsilon$")
    ax.set_ylabel("EDE of H after the action")
    ax.legend(loc="upper right")
    agree = all(
        allocate(pop, [low, high], 1, e)["selected"]
        == (["support_low"] if after(low, e) > after(high, e) else ["support_high"])
        for e in (0, 0.5, 1, 2, 3)
    )
    ix.save(
        fig,
        "proof_10_allocation_reversal",
        3,
        "With a fixed budget for one of two actions, the exact allocator's choice reverses once inequality aversion passes the point where the two EDE curves cross.",
        "EDE of H after each single action for the two-person example of Table 1 in paper 3. Below the "
        "crossing the larger total gain wins; above it the gain to the worse-off person wins. The exact "
        "enumeration in hpotential.crf.allocate picks the higher curve at every tabulated aversion.",
        {
            "crossing epsilon": round(star, 4),
            "allocator agrees with the curves": agree,
            "passed": agree and 0.5 < star < 1,
        },
    )


ALL = [
    level_sets,
    ces_family,
    ceiling,
    balance_penalty,
    marginal_effects,
    gap_waterfall,
    determinacy,
    drift,
    ede_curve,
    allocation,
]


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args(argv)
    apply_style()
    ix = Index(args.out)
    for draw in ALL:
        draw(ix)
    ix.write()
    print(f"{len(ix.entries)} proof figures in {args.out}")
    return args.out


if __name__ == "__main__":
    main()
