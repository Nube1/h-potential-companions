"""Paper 1: determinacy ceiling, identification, and sample size.

Three separate results, each producing numbers the manuscript quotes.

A. The determinacy ceiling. ``rho_D`` is a population quantity: it depends on
   the loadings, the number of indicators and the domain disturbances, and not
   on the sample size. Adding indicators inside a domain moves ``rho_D`` toward
   ``sqrt(c / (1 + c))`` from below and never past it.
B. Local identification. The manuscript's proof sketch asserts the rank
   condition; this checks it, for the planned design and for two controls, one
   of which must fail if the check is doing its job.
C. Sample size. Convergence, admissible solutions, relative bias, interval
   coverage and determinacy over a grid of sample sizes in two scenarios.

Everything here is a simulation of the planned model or algebra on it. No
participant has been measured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config.paper1 import Paper1Config
from ..core.measurement import (
    MeasurementModel,
    determinacy_ceiling,
    identification_report,
    required_second_order_signal,
    sample_size_replicate,
    second_order_determinacy,
)
from ..utils.plotting import INK, apply_style, save_figure, series_color

__all__ = ["FIGURES", "figures", "run"]

#: Figures this analysis can draw.
FIGURES: tuple[str, ...] = ("determinacy", "samplesize")


def _replicate_rng(cfg: Paper1Config, scenario_index: int, n: int, rep: int) -> np.random.Generator:
    """Generator for one replicate, derived from the configured root seed.

    Uses a seed sequence over ``(seed, scenario, n, replicate)``, so replicates
    are independent, the study is bit-reproducible, and nothing depends on the
    order in which the jobs happen to run. In particular it does not depend on
    ``hash`` of a string, which python salts per process.
    """
    return np.random.default_rng([cfg.seed, scenario_index, int(n), int(rep)])


def _determinacy_table(cfg: Paper1Config) -> dict[str, Any]:
    gamma = np.asarray(cfg.gamma)
    psi = np.asarray(cfg.psi)
    rho_inf, c = determinacy_ceiling(gamma, psi)
    table: dict[str, Any] = {
        "ceiling_rho_D": rho_inf,
        "c_gamma_Psi": c,
        "c_required_for_target": required_second_order_signal(cfg.target_determinacy),
        "target": cfg.target_determinacy,
        "numerical_check": {
            "m_per_domain": cfg.m_check,
            "lam_std": cfg.lam_check,
            "rho_D": second_order_determinacy(cfg.m_check, cfg.lam_check, gamma, psi),
        },
    }
    for lam in cfg.lam_grid:
        values = [second_order_determinacy(int(m), lam, gamma, psi) for m in cfg.m_grid]
        reached = next(
            (
                int(m)
                for m, v in zip(cfg.m_grid, values, strict=False)
                if v >= cfg.target_determinacy
            ),
            None,
        )
        table[f"lambda={lam}"] = {
            "rho_D_at_m4": second_order_determinacy(4, lam, gamma, psi),
            "rho_D_at_m8": second_order_determinacy(8, lam, gamma, psi),
            "rho_D_at_m40": second_order_determinacy(40, lam, gamma, psi),
            "indicators_per_domain_for_target": reached,
        }
    return table


def _identification(cfg: Paper1Config) -> dict[str, Any]:
    gamma = np.asarray(cfg.gamma)
    psi = np.asarray(cfg.psi)
    planned = [
        identification_report(lam, f"standardised loading {lam}", gamma, psi, cfg.per_domain)
        for lam in cfg.identification_loadings
    ]
    controls = [
        # Two indicators per domain, three domains. This one IS identified: the
        # second-order factor ties the domains together and supplies the extra
        # restrictions. Included because it is the case most often assumed to
        # fail.
        identification_report(
            0.75, "control (a): 2 indicators per domain, 3 domains", gamma, psi, 2
        ),
        # A second-order factor over only TWO domains. Here two second-order
        # loadings and two disturbances must be recovered from a single
        # between-domain covariance, so the second-order layer is not identified
        # without a further constraint. This one must fail; if it did not, the
        # rank test would not be doing its job.
        identification_report(
            0.75,
            "control (b): second-order factor over 2 domains",
            gamma[:2],
            psi[:2],
            cfg.per_domain,
        ),
    ]
    return {
        "model": "second-order CFA, 12 indicators, 3 domains, marker scaling",
        "criterion": "local identification iff rank(d sigma / d theta') = dim(theta)",
        "planned_designs": planned,
        "controls": controls,
    }


def _sample_size(cfg: Paper1Config, reps: int) -> dict[str, Any]:
    gamma = np.asarray(cfg.gamma)
    summary: dict[str, Any] = {}
    for scenario_index, (name, lam, theta_scale) in enumerate(cfg.scenarios):
        model = _population_model(cfg, lam, theta_scale)
        for n in cfg.n_grid:
            records = []
            for rep in range(int(reps)):
                rec = sample_size_replicate(
                    int(n),
                    model,
                    _replicate_rng(cfg, scenario_index, n, rep),
                    gamma_true=gamma,
                    per_domain=cfg.per_domain,
                )
                if rec is not None:
                    records.append(rec)
            summary[f"{name}|{n}"] = _summarise_records(records, gamma, len(cfg.gamma))
            summary[f"{name}|{n}"].update(scenario=name, n=int(n))
    return {"gamma_true": list(cfg.gamma), "reps": int(reps), "summary": summary}


def _population_model(cfg: Paper1Config, lam: float, theta_scale: float) -> MeasurementModel:
    """Population model of one scenario, under marker scaling.

    The first indicator of each domain carries a loading of one, so its residual
    variance is not the standardised value; it is floored at 0.15 and then
    scaled. Marker scaling in the population is what makes the reported bias and
    coverage statements about the estimator rather than about a scale mismatch
    between the simulated truth and the fitted model.
    """
    return MeasurementModel.marker_scaled(
        cfg.per_domain,
        float(lam),
        np.asarray(cfg.gamma),
        np.asarray(cfg.psi),
        theta_scale=float(theta_scale),
    )


def _summarise_records(
    records: list[dict[str, Any]], gamma_true: np.ndarray, k: int
) -> dict[str, Any]:
    if not records:
        return {"reps": 0}
    convergence = float(np.mean([r["converged"] for r in records]))
    admissible = float(np.mean([r["admissible"] for r in records]))
    good = [r for r in records if r.get("gamma") is not None]
    if not good:
        nan = [float("nan")] * k
        return {
            "reps": len(records),
            "convergence": convergence,
            "admissible": admissible,
            "rel_bias": nan,
            "max_abs_rel_bias": float("nan"),
            "coverage": nan,
            "min_coverage": float("nan"),
            "se_bias": nan,
            "max_abs_se_bias": float("nan"),
            "determinacy": float("nan"),
        }
    g = np.array([r["gamma"] for r in good])
    cover = np.array([r["cover"] for r in good])
    rel_bias = (g.mean(axis=0) - gamma_true) / gamma_true
    empirical_se = g.std(axis=0, ddof=1)
    average_se = np.array([r["se_gamma"] for r in good]).mean(axis=0)
    se_bias = (average_se - empirical_se) / empirical_se
    coverage = cover.mean(axis=0)
    return {
        "reps": len(records),
        "convergence": convergence,
        "admissible": admissible,
        "rel_bias": rel_bias.tolist(),
        "max_abs_rel_bias": float(np.max(np.abs(rel_bias))),
        "coverage": coverage.tolist(),
        "min_coverage": float(np.min(coverage)),
        "se_bias": se_bias.tolist(),
        "max_abs_se_bias": float(np.max(np.abs(se_bias))),
        "determinacy": float(np.mean([r["rho"] for r in good])),
    }


def run(
    cfg: Paper1Config | None = None,
    outdir: str | Path | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Run the paper 1 analyses.

    Parameters
    ----------
    cfg : Paper1Config, optional
        Configuration. Defaults to the published one.
    outdir : str or pathlib.Path, optional
        Directory in which to write ``paper1.json``. Nothing is written when
        this is ``None``.
    quick : bool, default False
        Cut the replicate count and the sample-size grid so the analysis
        finishes in seconds. Quick mode does not reproduce the published
        numbers; the closed-form results are unaffected because they involve no
        simulation.

    Returns
    -------
    dict
        ``determinacy``, ``identification`` and ``sample_size``.

    Notes
    -----
    Coverage is the proportion of replicates whose Wald interval, built from the
    expected information at the fitted parameters, contains the true loading. It
    is counted, never assumed: a study that reports 0.95 without counting has
    reported the nominal level, not the realised one.
    """
    cfg = cfg or Paper1Config()
    reps = 8 if quick else cfg.reps
    if quick:
        cfg = Paper1Config(
            **{
                **cfg.__dict__,
                "n_grid": (150, 400),
            }
        )
    results = {
        "determinacy": _determinacy_table(cfg),
        "identification": _identification(cfg),
        "sample_size": _sample_size(cfg, reps),
        "quick_mode": bool(quick),
    }
    if outdir is not None:
        path = Path(outdir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "paper1.json").write_text(json.dumps(results, indent=1))
    return results


def figures(
    results: dict[str, Any], outdir: str | Path, cfg: Paper1Config | None = None
) -> dict[str, Path]:
    """Draw the paper 1 figures.

    Parameters
    ----------
    results : dict
        Output of :func:`run`.
    outdir : str or pathlib.Path
        Directory for the PNG files.
    cfg : Paper1Config, optional
        Configuration used for the run.

    Returns
    -------
    dict of str to pathlib.Path
        Figure name to file written.
    """
    import matplotlib.pyplot as plt

    cfg = cfg or Paper1Config()
    apply_style()
    out = Path(outdir)
    written: dict[str, Path] = {}
    gamma, psi = np.asarray(cfg.gamma), np.asarray(cfg.psi)
    table = results["determinacy"]
    rho_inf, c = table["ceiling_rho_D"], table["c_gamma_Psi"]
    target = table["target"]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    ax = axes[0]
    ms = np.array(cfg.m_grid)
    for i, lam in enumerate(cfg.lam_grid):
        ax.plot(
            ms,
            [second_order_determinacy(int(m), lam, gamma, psi) for m in ms],
            "-",
            lw=1.4,
            color=series_color(i),
            label=rf"$\lambda={lam}$",
        )
    ax.axhline(target, ls="--", lw=0.9, color=INK)
    ax.axhline(rho_inf, ls=":", lw=1.1, color=series_color(5))
    ax.text(ms[-1], target + 0.006, r"individual-use benchmark $0.90$", ha="right", fontsize=7)
    ax.text(
        ms[0] + 0.05,
        rho_inf + 0.005,
        rf"structural ceiling $\rho_D^\infty={rho_inf:.3f}$",
        ha="left",
        fontsize=7,
        color=series_color(5),
    )
    ax.set_xscale("log")
    ax.set_xticks([3, 4, 6, 8, 12, 20, 40])
    ax.set_xticklabels(["3", "4", "6", "8", "12", "20", "40"])
    ax.set_ylim(0.70, 0.95)
    ax.set_xlabel("indicators per domain")
    ax.set_ylabel(r"factor-score determinacy $\rho_D$")
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    ax.set_title("Adding indicators cannot pass the ceiling", fontsize=8.5)

    ax2 = axes[1]
    scale = np.linspace(0.5, 3.0, 300)
    cs, rhos = [], []
    for s in scale:
        r, cc = determinacy_ceiling(np.minimum(gamma * np.sqrt(s), 0.99), psi)
        cs.append(cc)
        rhos.append(r)
    required = table["c_required_for_target"]
    ax2.plot(cs, rhos, "-", lw=1.5, color=series_color(0))
    ax2.axhline(target, ls="--", lw=0.9, color=INK)
    ax2.axvline(required, ls="--", lw=0.9, color=series_color(5))
    ax2.plot([c], [rho_inf], "o", ms=5, color=series_color(5))
    ax2.annotate(
        f"planned design\n$c={c:.2f}$",
        xy=(c, rho_inf),
        xytext=(c + 0.6, rho_inf - 0.06),
        fontsize=7,
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": INK},
    )
    ax2.text(
        required + 0.08,
        0.735,
        rf"$c=\rho^2/(1-\rho^2)={required:.2f}$",
        fontsize=7,
        color=series_color(5),
    )
    ax2.set_xlabel(r"second-order signal $c=\gamma^\top\Psi^{-1}\gamma$")
    ax2.set_ylabel(r"ceiling $\rho_D^\infty=\sqrt{c/(1+c)}$")
    ax2.set_ylim(0.70, 0.95)
    ax2.set_title("The ceiling is set by the second-order model", fontsize=8.5)
    written["determinacy"] = save_figure(fig, out / "p1_determinacy.png")

    summary = results["sample_size"]["summary"]
    ns = sorted({v["n"] for v in summary.values()})
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.9))
    for i, (name, _, _) in enumerate(cfg.scenarios):
        keys = [f"{name}|{n}" for n in ns if f"{name}|{n}" in summary]
        if not keys:
            continue
        axes[0].plot(
            ns,
            [summary[k]["admissible"] for k in keys],
            "o-",
            color=series_color(i),
            ms=3.5,
            lw=1.3,
            label=f"{name} scenario",
        )
        axes[1].plot(
            ns,
            [summary[k]["min_coverage"] for k in keys],
            "o-",
            color=series_color(i),
            ms=3.5,
            lw=1.3,
            label=f"{name} scenario",
        )
        axes[2].plot(
            ns,
            [summary[k]["determinacy"] for k in keys],
            "o-",
            color=series_color(i),
            ms=3.5,
            lw=1.3,
            label=f"{name} scenario",
        )
    axes[0].axhline(0.95, ls="--", lw=0.8, color=INK)
    axes[0].set_ylabel("admissible-solution rate")
    axes[0].set_ylim(0.4, 1.02)
    axes[1].axhline(0.95, ls="--", lw=0.8, color=INK)
    axes[1].set_ylabel("worst 95% CI coverage")
    axes[1].set_ylim(0.80, 1.0)
    axes[2].axhline(0.90, ls="--", lw=0.8, color=INK)
    axes[2].set_ylabel(r"factor-score determinacy $\rho_D$")
    axes[2].set_ylim(0.80, 1.0)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns], fontsize=7)
        ax.set_xlabel("sample size $n$")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle(
        "Simulated operating characteristics of the second-order H-potential model "
        f"({results['sample_size']['reps']} replicates per point)",
        fontsize=9,
        y=1.04,
    )
    written["samplesize"] = save_figure(fig, out / "p1_samplesize.png")
    return written
