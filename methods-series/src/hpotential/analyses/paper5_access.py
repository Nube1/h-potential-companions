"""Paper 5: from an uncosted derivative ratio to a costed decision rule.

The theorem gives the uncosted marginal-effect ratio ``dH/dA : dH/dT = T/A``.
That is not an allocation rule. A decision maker choosing between an access
programme and an efficacy programme compares gain per unit spend, which gives

    V = (T/A) * (Delta_A / Delta_T) * (c_T / c_A) > 1,

a product of a structural term, a feasibility term and a price term. The
uncosted rule ``T > A`` is the special case in which the last two are one.

This analysis
-------------
1. Verifies the closed form against a direct numerical evaluation of the ratio
   of gain per unit spend.
2. Runs a probabilistic sensitivity analysis over every input in a
   high-resource and a low-resource setting, and reports how often the costed
   and uncosted rules disagree.
3. Maps the indifference surface ``V = 1``.

All inputs are illustrative ranges, not estimates from any health system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config.paper5 import Paper5Config
from ..core.access import AccessModel
from ..utils.plotting import INK, apply_style, save_figure, series_color

__all__ = ["FIGURES", "figures", "models", "run"]

#: Figures this analysis can draw.
FIGURES: tuple[str, ...] = ("costed",)


def models(cfg: Paper5Config) -> dict[str, AccessModel]:
    """The two configured settings, keyed by name."""
    available = {
        "high-resource": AccessModel.high_resource(),
        "low-resource": AccessModel.low_resource(),
    }
    return {name: available[name] for name in cfg.settings}


def run(
    cfg: Paper5Config | None = None,
    outdir: str | Path | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Run the paper 5 costed-frontier analysis.

    Parameters
    ----------
    cfg : Paper5Config, optional
        Configuration. Defaults to the published one.
    outdir : str or pathlib.Path, optional
        Directory in which to write ``paper5.json``.
    quick : bool, default False
        Cut the number of draws. Quick mode does not reproduce the published
        numbers, though the closed-form check is unaffected in substance because
        it is a statement about floating-point agreement rather than about a
        sample.

    Returns
    -------
    dict
        ``closed_form_check`` and ``psa`` by setting.

    Notes
    -----
    The closed-form check evaluates the two sides through separate code paths,
    so agreement is evidence about the algebra of the theorem rather than about
    a single implementation of it.
    """
    cfg = cfg or Paper5Config()
    n_psa = 20_000 if quick else cfg.n_psa
    n_check = 2_000 if quick else cfg.n_check
    rng = np.random.default_rng(cfg.seed)
    setting_models = models(cfg)

    check_model = setting_models.get("low-resource") or next(iter(setting_models.values()))
    max_rel = check_model.closed_form_check(rng, n_check)

    results: dict[str, Any] = {
        "quick_mode": bool(quick),
        "n_psa": int(n_psa),
        "closed_form_check": {
            "max_relative_discrepancy": max_rel,
            "note": (
                "V = (T/A)(dA/dT)(cT/cA) reproduces " "(dH/dA dA / cA)/(dH/dT dT / cT) exactly"
            ),
        },
        "psa": {name: model.psa(rng, n_psa) for name, model in setting_models.items()},
    }

    if outdir is not None:
        path = Path(outdir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "paper5.json").write_text(json.dumps(results, indent=1))
    return results


def figures(
    results: dict[str, Any], outdir: str | Path, cfg: Paper5Config | None = None
) -> dict[str, Path]:
    """Draw the paper 5 figure.

    Parameters
    ----------
    results : dict
        Output of :func:`run`.
    outdir : str or pathlib.Path
        Directory for the PNG file.
    cfg : Paper5Config, optional
        Configuration used for the run.

    Returns
    -------
    dict of str to pathlib.Path
    """
    import matplotlib.pyplot as plt

    cfg = cfg or Paper5Config()
    apply_style()
    rng = np.random.default_rng(cfg.seed + 1)
    setting_models = models(cfg)
    n_draw = min(int(results["n_psa"]), 200_000)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))

    for i, (name, model) in enumerate(setting_models.items()):
        q = model.draw(rng, n_draw)
        v = (q["T"] / q["A"]) * (q["dA"] / q["dT"]) * (q["cT"] / q["cA"])
        axes[0].hist(
            np.log10(v),
            bins=140,
            histtype="step",
            lw=1.3,
            color=series_color(i),
            density=True,
            label=name,
        )
    axes[0].axvline(0, ls="--", lw=1.0, color=INK)
    axes[0].set_xlabel(r"$\log_{10} V$   (access favoured when $V>1$)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Costed decision statistic", fontsize=8.5)
    axes[0].legend(fontsize=7)
    axes[0].set_xlim(-1.5, 2.5)

    ax = axes[1]
    labels = [
        "structural\n$T/A$",
        "feasibility\n$\\Delta_A/\\Delta_T$",
        "price\n$c_T/c_A$",
        "combined\n$V$",
    ]
    width = 0.36
    xs = np.arange(4)
    for i, (name, record) in enumerate(results["psa"].items()):
        values = [
            record["median_structural_term"],
            record["median_feasibility_term"],
            record["median_price_term"],
            record["median_V"],
        ]
        ax.bar(
            xs + (i - 0.5) * width,
            values,
            width=width,
            color=series_color(i),
            label=name,
            edgecolor="none",
        )
    ax.axhline(1.0, ls="--", lw=1.0, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylabel("median value (log scale)")
    ax.set_title("Where the decision comes from", fontsize=8.5)
    ax.legend(fontsize=7)

    ax = axes[2]
    access = np.linspace(cfg.access_grid_lo, cfg.access_grid_hi, cfg.access_grid_n)
    for i, ratio in enumerate(cfg.cost_ratios):
        ax.plot(
            access,
            (cfg.efficacy_fixed / access) * ratio,
            lw=1.4,
            color=series_color(i),
            label=rf"$(\Delta_A/\Delta_T)(c_T/c_A)={ratio}$",
        )
    ax.axhline(1.0, ls="--", lw=1.0, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(0.06, 30)
    ax.set_xlabel("access $A$")
    ax.set_ylabel("$V$")
    ax.set_title(rf"Indifference at $V=1$ (efficacy $T={cfg.efficacy_fixed}$)", fontsize=8.5)
    ax.legend(fontsize=6.4, loc="upper right")

    fig.suptitle(
        "Costed access-versus-efficacy comparison: the uncosted ratio is one factor of three",
        fontsize=9,
        y=1.05,
    )
    return {"costed": save_figure(fig, Path(outdir) / "p5_costed.png")}
