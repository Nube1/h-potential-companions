"""Paper 2: whole-pipeline simulation for the two-stage prognostic design.

Stage one estimates a factor score; stage two puts that score into a Cox model.
The score is not the latent construct, so the stage-two coefficient is
attenuated toward the null by approximately the factor-score determinacy. This
analysis runs the complete pipeline, indicators to confirmatory factor model to
score to Cox model, and checks the analytic attenuation factor against it.

It also tabulates the events cost implied by the determinacy ceiling: matching
the noncentrality of the score test between an oracle design and a score-based
design needs ``1 / rho_D^2`` times as many events.

Nothing here uses any cohort. All data are simulated from the planned model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config.paper2 import Paper2Config
from ..core.measurement import MeasurementModel, determinacy_ceiling
from ..core.prognostic import (
    attenuated_beta,
    events_inflation,
    pipeline_replicate,
    required_events,
)
from ..utils.plotting import INK, apply_style, save_figure, series_color

__all__ = ["FIGURES", "figures", "population_model", "run"]

#: Figures this analysis can draw.
FIGURES: tuple[str, ...] = ("attenuation",)


def population_model(cfg: Paper2Config) -> MeasurementModel:
    """The frozen measurement model of the planned design.

    Parameters
    ----------
    cfg : Paper2Config
        Configuration.

    Returns
    -------
    MeasurementModel
        Balanced, with ``per_domain`` standardised loadings of ``lam_std`` in
        each domain. Its determinacy is 0.8361.
    """
    return MeasurementModel.balanced(
        cfg.per_domain, cfg.lam_std, np.asarray(cfg.gamma), np.asarray(cfg.psi)
    )


def _replicate_rng(cfg: Paper2Config, endpoint_index: int, n: int, rep: int) -> np.random.Generator:
    """Generator for one replicate, derived from the configured root seed."""
    return np.random.default_rng([cfg.seed, endpoint_index, int(n), int(rep)])


def run(
    cfg: Paper2Config | None = None,
    outdir: str | Path | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Run the paper 2 pipeline simulation.

    Parameters
    ----------
    cfg : Paper2Config, optional
        Configuration. Defaults to the published one.
    outdir : str or pathlib.Path, optional
        Directory in which to write ``paper2.json``.
    quick : bool, default False
        Cut the replicate count and the sample-size grid. Quick mode does not
        reproduce the published numbers.

    Returns
    -------
    dict
        ``rho_D_population``, ``beta_true``, ``predicted_attenuated_beta``,
        ``events_cost``, and one record per endpoint and sample size holding the
        oracle and observed coefficients, the attenuation ratio, the mean
        standard error and the two powers.

    Notes
    -----
    The attenuation ratio is the mean fitted coefficient on the estimated score
    over the mean fitted coefficient on the true construct, both from the same
    simulated people, so the ratio is not contaminated by differences in the
    survival data. The prediction it is compared against, ``rho_D``, is computed
    from the population model and not from the simulation.
    """
    cfg = cfg or Paper2Config()
    reps = 10 if quick else cfg.reps
    n_grid = (1500,) if quick else cfg.n_grid

    model = population_model(cfg)
    rho_pop = model.determinacy()
    rho_ceiling, _ = determinacy_ceiling(np.asarray(cfg.gamma), np.asarray(cfg.psi))

    results: dict[str, Any] = {
        "rho_D_population": rho_pop,
        "rho_D_ceiling": rho_ceiling,
        "beta_true": cfg.beta_true,
        "predicted_attenuated_beta": attenuated_beta(cfg.beta_true, rho_pop),
        "reps": int(reps),
        "quick_mode": bool(quick),
        "events_cost": {
            "inflation_at_ceiling": events_inflation(rho_ceiling),
            "inflation_at_planned_design": events_inflation(rho_pop),
            "events_oracle": required_events(cfg.beta_true, cfg.power, cfg.alpha, 1.0),
            "events_at_ceiling": required_events(cfg.beta_true, cfg.power, cfg.alpha, rho_ceiling),
            "events_at_planned_design": required_events(
                cfg.beta_true, cfg.power, cfg.alpha, rho_pop
            ),
        },
    }

    pooled: list[Any] = []
    for endpoint_index, (endpoint, fraction) in enumerate(cfg.event_fractions):
        for n in n_grid:
            records = []
            for rep in range(int(reps)):
                rec = pipeline_replicate(
                    int(n),
                    endpoint,
                    fraction,
                    model,
                    cfg.beta_true,
                    _replicate_rng(cfg, endpoint_index, n, rep),
                    per_domain=cfg.per_domain,
                )
                if rec is not None:
                    records.append(rec)
            if not records:
                continue
            pooled.extend(records)
            observed = np.array([r.beta_observed for r in records])
            oracle = np.array([r.beta_oracle for r in records])
            results[f"{endpoint}|{n}"] = {
                "endpoint": endpoint,
                "n": int(n),
                "reps": len(records),
                "mean_events": float(np.mean([r.events for r in records])),
                "beta_oracle": float(oracle.mean()),
                "beta_observed": float(observed.mean()),
                "attenuation_ratio": float(observed.mean() / oracle.mean()),
                "HR_observed": float(np.exp(observed.mean())),
                "HR_oracle": float(np.exp(oracle.mean())),
                "mean_se": float(np.mean([r.se_observed for r in records])),
                "mean_rho_hat": float(np.mean([r.rho_hat for r in records])),
                "power_observed": float(np.mean([r.significant for r in records])),
                "power_oracle": float(np.mean([r.significant_oracle for r in records])),
            }

    if pooled:
        # The headline attenuation is pooled over every design point. A ratio
        # computed at one sample size from 120 replicates carries a Monte Carlo
        # error of several per cent, so the per-point ratios scatter around the
        # prediction by more than the quantity being estimated. Pooling over all
        # replicates is the stable summary, and it is the one to compare with
        # rho_D.
        obs = np.array([r.beta_observed for r in pooled])
        ora = np.array([r.beta_oracle for r in pooled])
        per_point = [
            v["attenuation_ratio"]
            for v in results.values()
            if isinstance(v, dict) and "attenuation_ratio" in v
        ]
        results["attenuation_ratio_pooled"] = float(obs.mean() / ora.mean())
        results["attenuation_ratio_min"] = float(min(per_point))
        results["attenuation_ratio_max"] = float(max(per_point))
        results["n_replicates_pooled"] = int(obs.size)

    if outdir is not None:
        path = Path(outdir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "paper2.json").write_text(json.dumps(results, indent=1))
    return results


def figures(
    results: dict[str, Any], outdir: str | Path, cfg: Paper2Config | None = None
) -> dict[str, Path]:
    """Draw the paper 2 figure.

    Parameters
    ----------
    results : dict
        Output of :func:`run`.
    outdir : str or pathlib.Path
        Directory for the PNG file.
    cfg : Paper2Config, optional
        Configuration used for the run.

    Returns
    -------
    dict of str to pathlib.Path
    """
    import matplotlib.pyplot as plt

    cfg = cfg or Paper2Config()
    apply_style()
    apply_ns = sorted({v["n"] for k, v in results.items() if isinstance(v, dict) and "n" in v})
    rho_pop = results["rho_D_population"]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.95))
    ax = axes[0]
    for i, (endpoint, _) in enumerate(cfg.event_fractions):
        keys = [f"{endpoint}|{n}" for n in apply_ns if f"{endpoint}|{n}" in results]
        if not keys:
            continue
        ns = [results[k]["n"] for k in keys]
        ax.plot(
            ns,
            [results[k]["HR_observed"] for k in keys],
            "o-",
            color=series_color(i),
            ms=3.5,
            lw=1.3,
            label=f"{endpoint}, estimated score",
        )
        ax.plot(
            ns,
            [results[k]["HR_oracle"] for k in keys],
            "s--",
            color=series_color(i),
            ms=3.0,
            lw=1.0,
            alpha=0.55,
            label=f"{endpoint}, true score",
        )
    ax.axhline(np.exp(cfg.beta_true), ls=":", lw=1.0, color=INK)
    ax.set_ylabel("hazard ratio per SD")
    ax.set_title("Attenuation of the stage-2 effect", fontsize=8.5)
    ax.legend(fontsize=6.2, loc="upper right")

    ax = axes[1]
    for i, (endpoint, _) in enumerate(cfg.event_fractions):
        keys = [f"{endpoint}|{n}" for n in apply_ns if f"{endpoint}|{n}" in results]
        if not keys:
            continue
        ax.plot(
            [results[k]["n"] for k in keys],
            [results[k]["attenuation_ratio"] for k in keys],
            "o-",
            color=series_color(i),
            ms=3.5,
            lw=1.3,
            label=endpoint,
        )
    ax.axhline(rho_pop, ls="--", lw=1.0, color=series_color(5))
    ax.text(
        apply_ns[-1],
        rho_pop + 0.012,
        rf"$\rho_D={rho_pop:.3f}$",
        ha="right",
        fontsize=7.5,
        color=series_color(5),
    )
    ax.set_ylabel(r"$\hat\beta_{\rm score}/\hat\beta_{\rm true}$")
    ax.set_ylim(0.6, 1.05)
    ax.set_title("Attenuation matches determinacy", fontsize=8.5)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[2]
    for i, (endpoint, _) in enumerate(cfg.event_fractions):
        keys = [f"{endpoint}|{n}" for n in apply_ns if f"{endpoint}|{n}" in results]
        if not keys:
            continue
        ns = [results[k]["n"] for k in keys]
        ax.plot(
            ns,
            [results[k]["power_observed"] for k in keys],
            "o-",
            color=series_color(i),
            ms=3.5,
            lw=1.3,
            label=f"{endpoint}, estimated",
        )
        ax.plot(
            ns,
            [results[k]["power_oracle"] for k in keys],
            "s--",
            color=series_color(i),
            ms=3.0,
            lw=1.0,
            alpha=0.55,
            label=f"{endpoint}, true",
        )
    ax.axhline(0.90, ls=":", lw=1.0, color=INK)
    ax.set_ylabel("power at the 5% level")
    ax.set_ylim(0, 1.03)
    ax.set_title("Power cost of measurement error", fontsize=8.5)
    ax.legend(fontsize=6.2, loc="lower right")

    for ax in axes:
        ax.set_xlabel("analytic sample size $n$")
        ax.set_xticks(apply_ns)
        ax.set_xticklabels([str(n) for n in apply_ns], fontsize=7)
    fig.suptitle(
        "Whole-pipeline simulation: measurement error in stage 1 propagates to stage 2 "
        f"({results['reps']} replicates per point)",
        fontsize=9,
        y=1.05,
    )
    return {"attenuation": save_figure(fig, Path(outdir) / "p2_attenuation.png")}
