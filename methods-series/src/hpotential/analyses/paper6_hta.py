"""Paper 6: probabilistic analysis, value of information, and outcome currency.

The case is a hypothetical durable therapy for severe haemophilia compared with
established prophylaxis. Every input is synthetic and chosen only to demonstrate
the workflow. Nothing here is evidence about any real product.

Outputs
-------
1. Distributions of incremental cost, incremental QALYs, incremental biomedical
   potential and net monetary benefit, under three structural scenarios.
2. Per-person and population expected value of perfect information.
3. Expected value of partial perfect information for each uncertain input, by
   the single-parameter regression method of Strong, Oakley and Brennan.
4. How often a decision taken on biomedical potential disagrees with one taken
   on QALYs, when the biomedical potential uses an unvalidated quality mapping.
   That is the paper's central caution, stated as a number rather than as an
   assertion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config.paper6 import Paper6Config
from ..core.hta import HTAModel
from ..utils.plotting import INK, apply_style, save_figure, series_color

__all__ = ["FIGURES", "build_model", "figures", "run"]

#: Figures this analysis can draw.
FIGURES: tuple[str, ...] = ("voi",)

#: Human-readable names for the six uncertain inputs.
INPUT_LABELS: dict[str, str] = {
    "q_gain": "utility gain",
    "waning": "durability",
    "access": "implementation gap",
    "price": "therapy price",
    "proph": "prophylaxis cost",
    "L0": "survival horizon",
}


def build_model(cfg: Paper6Config) -> HTAModel:
    """Assemble the model carrying the configured constants."""
    return HTAModel(
        threshold=cfg.threshold,
        population=cfg.population,
        horizon_years=cfg.horizon_years,
        discount=cfg.discount,
        comparator_quality=cfg.comparator_quality,
        price_median=cfg.price_median,
        prophylaxis_median=cfg.prophylaxis_median,
        bmp_quality_multiplier=cfg.bmp_quality_multiplier,
        max_years=cfg.max_years,
    )


def run(
    cfg: Paper6Config | None = None,
    outdir: str | Path | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Run the paper 6 early-assessment analysis.

    Parameters
    ----------
    cfg : Paper6Config, optional
        Configuration. Defaults to the published one.
    outdir : str or pathlib.Path, optional
        Directory in which to write ``paper6.json``.
    quick : bool, default False
        Cut the number of draws to 10000. Quick mode does not reproduce the
        published numbers; the EVPPI estimates in particular are noticeably
        noisier, because each conditional mean is then built from far fewer
        observations per bin.

    Returns
    -------
    dict
        Configuration constants, a disclaimer, and one record per scenario.

    Notes
    -----
    The scenarios are drawn from a single generator in a fixed order, so the
    whole analysis is reproducible from ``cfg.seed`` alone.
    """
    cfg = cfg or Paper6Config()
    draws = 10_000 if quick else cfg.draws
    model = build_model(cfg)
    rng = np.random.default_rng(cfg.seed)

    results: dict[str, Any] = {
        "lambda": cfg.threshold,
        "population": cfg.population,
        "horizon_years": cfg.horizon_years,
        "discount": cfg.discount,
        "draws": int(draws),
        "quick_mode": bool(quick),
        "disclaimer": "All inputs synthetic; illustrates workflow only.",
        "scenarios": {},
    }
    for scenario in cfg.scenarios:
        result = model.run_scenario(rng, scenario, draws)
        summary = model.summarise(result, cfg.evppi_bins)
        summary["bmp_to_qaly_ratio"] = float(
            summary["mean_incremental_bmp"] / summary["mean_incremental_qaly"]
        )
        results["scenarios"][scenario] = summary

    if outdir is not None:
        path = Path(outdir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "paper6.json").write_text(json.dumps(results, indent=1))
    return results


def figures(
    results: dict[str, Any], outdir: str | Path, cfg: Paper6Config | None = None
) -> dict[str, Path]:
    """Draw the paper 6 figure.

    Parameters
    ----------
    results : dict
        Output of :func:`run`.
    outdir : str or pathlib.Path
        Directory for the PNG file.
    cfg : Paper6Config, optional
        Configuration used for the run.

    Returns
    -------
    dict of str to pathlib.Path
    """
    import matplotlib.pyplot as plt

    cfg = cfg or Paper6Config()
    apply_style()
    model = build_model(cfg)
    rng = np.random.default_rng(cfg.seed)
    draws = int(results["draws"])
    names = list(results["scenarios"])

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.05))
    for i, scenario in enumerate(names):
        result = model.run_scenario(rng, scenario, draws)
        axes[0].hist(
            result.inb_qaly / 1000,
            bins=160,
            histtype="step",
            lw=1.3,
            density=True,
            color=series_color(i),
            label=scenario,
        )
    axes[0].axvline(0, ls="--", lw=1.0, color=INK)
    axes[0].set_xlabel("incremental net benefit (thousands)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Decision uncertainty by scenario", fontsize=8.5)
    axes[0].legend(fontsize=7)
    axes[0].set_xlim(-400, 400)

    ax = axes[1]
    central = results["scenarios"].get("central", results["scenarios"][names[0]])
    keys = [k for k in INPUT_LABELS if k in central["EVPPI"]]
    ax.barh(
        [INPUT_LABELS[k] for k in keys],
        [max(central["EVPPI"][k], 0.0) for k in keys],
        color=series_color(0),
        edgecolor="none",
    )
    ax.axvline(central["per_person_EVPI"], ls="--", lw=1.0, color=series_color(5))
    ax.text(
        central["per_person_EVPI"],
        -0.45,
        " per-person EVPI",
        fontsize=6.8,
        color=series_color(5),
        va="bottom",
    )
    ax.set_xlabel("EVPPI per person (currency units)")
    ax.set_title("What is worth researching (central)", fontsize=8.5)

    ax = axes[2]
    xs = np.arange(len(names))
    width = 0.36
    ax.bar(
        xs - width / 2,
        [results["scenarios"][n]["P_cost_effective_QALY"] for n in names],
        width=width,
        color=series_color(0),
        label="QALY basis",
        edgecolor="none",
    )
    ax.bar(
        xs + width / 2,
        [results["scenarios"][n]["P_cost_effective_BMP"] for n in names],
        width=width,
        color=series_color(1),
        label="BMP basis",
        edgecolor="none",
    )
    for i, name in enumerate(names):
        ax.text(
            i,
            1.02,
            f"disagree\n{results['scenarios'][name]['P_decision_disagreement'] * 100:.0f}%",
            ha="center",
            fontsize=6.4,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel("probability the therapy is favoured")
    ax.set_ylim(0, 1.25)
    ax.set_title("BMP is not interchangeable with QALYs", fontsize=8.5)
    ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(
        "Probabilistic analysis, value of information, and outcome-currency check "
        "(synthetic inputs)",
        fontsize=9,
        y=1.05,
    )
    return {"voi": save_figure(fig, Path(outdir) / "p6_voi.png")}
