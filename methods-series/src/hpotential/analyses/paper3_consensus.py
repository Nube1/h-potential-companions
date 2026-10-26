"""Paper 3: operating characteristics of the modified Delphi retention rule.

A consensus rule is a decision procedure and therefore has error rates. On a
9-point relevance scale an item is retained when all three hold:

    median >= 7,   IQR <= 1.5,   I-CVI = P(rating >= 7) >= 0.78.

Median and interquartile range are order statistics, so the retention
probability is obtained by simulation rather than in closed form. Panellist
ratings are generated from a discretised normal,
``x = clip(round(N(mu, sigma)), 1, 9)``, which is the graded-response idea
applied to expert judgement: ``mu`` is the item's standing with the panel and
``sigma`` is how much panellists disagree about it.

Reported here
-------------
1. False-drop and false-keep rates at reference items, by panel size, and which
   of the three criteria binds when an item is dropped.
2. The same false-drop rate as panel disagreement grows, under four candidate
   dispersion thresholds.
3. A stakeholder split: the probability that an item the patients endorse
   strongly survives a pooled rule as clinician support falls.
4. An exhaustive check of the dispersion lemma, which says a threshold of 2 can
   never bind.

No Delphi has been run. These are properties of the stated decision rule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config.paper3 import Paper3Config
from ..core.consensus import (
    endorsement_probability,
    iqr_lemma_violations,
    mu_for_endorsement,
    operating_characteristics,
    retention_masks,
    simulate_ratings,
)
from ..utils.plotting import INK, apply_style, save_figure, series_color

__all__ = ["FIGURES", "figures", "run"]

#: Figures this analysis can draw.
FIGURES: tuple[str, ...] = ("operating_characteristics",)


def _rule_kwargs(cfg: Paper3Config, iqr_threshold: float | None = None) -> dict[str, Any]:
    return {
        "median_min": cfg.median_min,
        "iqr_threshold": cfg.iqr_threshold if iqr_threshold is None else iqr_threshold,
        "cvi_min": cfg.cvi_min,
        "relevance_cut": cfg.relevance_cut,
    }


def run(
    cfg: Paper3Config | None = None,
    outdir: str | Path | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Run the paper 3 operating-characteristics study.

    Parameters
    ----------
    cfg : Paper3Config, optional
        Configuration. Defaults to the published one.
    outdir : str or pathlib.Path, optional
        Directory in which to write ``paper3.json``.
    quick : bool, default False
        Cut the number of simulated panels and the size of the exhaustive lemma
        check. Quick mode does not reproduce the published numbers.

    Returns
    -------
    dict
        ``rule``, ``operating_characteristics`` by panel size,
        ``iqr_threshold_diagnostic``, ``patient_endorsed_item`` and
        ``dispersion_lemma``.

    Notes
    -----
    The latent standing that gives a requested endorsement is obtained in closed
    form from :func:`hpotential.core.consensus.mu_for_endorsement`, which
    inverts the endorsement of the *rounded* ratings. Defining the endorsement
    on the continuous scale and then simulating rounded ratings would make the
    realised endorsement differ from the requested one: at ``sigma = 1.2``, a
    request for 0.90 defined continuously realises 0.956, and every error rate
    computed from it would describe a different item.
    """
    cfg = cfg or Paper3Config()
    n_sim = 2000 if quick else cfg.n_sim
    lemma_max = 6 if quick else cfg.lemma_max_panel
    rng = np.random.default_rng(cfg.seed)

    results: dict[str, Any] = {
        "rule": {
            "scale": f"{cfg.scale[0]}-{cfg.scale[1]}",
            "median_min": cfg.median_min,
            "iqr_max": cfg.iqr_threshold,
            "i_cvi_min": cfg.cvi_min,
        },
        "generative_model": "x = clip(round(N(mu, sigma)), 1, 9)",
        "n_sim": int(n_sim),
        "quick_mode": bool(quick),
        "reference_items": {
            "mu_for_pi_0_90": mu_for_endorsement(
                cfg.pi_good, cfg.sigma_tight, cfg.relevance_cut, cfg.scale
            ),
            "mu_for_pi_0_65": mu_for_endorsement(
                cfg.pi_marginal, cfg.sigma_tight, cfg.relevance_cut, cfg.scale
            ),
            "sigma": cfg.sigma_tight,
        },
    }

    # ---- 1. error rates and binding criterion by panel size ----------------
    by_panel: dict[str, Any] = {}
    for n_panel in cfg.panel_sizes:
        good = operating_characteristics(
            n_panel,
            cfg.pi_good,
            cfg.sigma_tight,
            rng,
            n_sim,
            scale=cfg.scale,
            **_rule_kwargs(cfg),
        )
        marginal = operating_characteristics(
            n_panel,
            cfg.pi_marginal,
            cfg.sigma_tight,
            rng,
            n_sim,
            scale=cfg.scale,
            **_rule_kwargs(cfg),
        )
        by_panel[str(n_panel)] = {
            "false_drop_at_pi_0_90": good["p_drop"],
            "false_keep_at_pi_0_65": marginal["p_retain"],
            "binding_when_dropped": {
                "median_fails": good["median_fails"],
                "iqr_fails": good["iqr_fails"],
                "cvi_fails": good["cvi_fails"],
            },
        }
    results["operating_characteristics"] = by_panel

    # ---- 2. the dispersion threshold is what binds -------------------------
    diagnostic: dict[str, list[float]] = {}
    for threshold in cfg.iqr_candidates:
        rates = []
        for sigma in cfg.sigma_grid:
            oc = operating_characteristics(
                cfg.diagnostic_panel,
                cfg.pi_good,
                float(sigma),
                rng,
                n_sim,
                scale=cfg.scale,
                **_rule_kwargs(cfg, threshold),
            )
            rates.append(oc["p_drop"])
        diagnostic[str(threshold)] = rates
    results["iqr_threshold_diagnostic"] = {
        "sigmas": list(cfg.sigma_grid),
        "panel_size": cfg.diagnostic_panel,
        "false_drop_rate_by_threshold": diagnostic,
        "note": (
            "False-drop rate for an item with pi=0.90 at the stated panel size, as "
            "panel disagreement sigma grows, under four candidate IQR thresholds."
        ),
    }

    # ---- 3. stakeholder split ----------------------------------------------
    clinician_pi = np.linspace(0.05, 0.98, 41)
    stakeholder: dict[str, Any] = {}
    curves: dict[str, np.ndarray] = {}
    for n_patients, n_clinicians in cfg.stakeholder_splits:
        probabilities = []
        for pi_clin in clinician_pi:
            mu_clin = mu_for_endorsement(
                float(pi_clin), cfg.sigma_tight, cfg.relevance_cut, cfg.scale
            )
            patients = simulate_ratings(
                rng, cfg.patient_mu, cfg.sigma_tight, (n_sim, n_patients), cfg.scale
            )
            clinicians = simulate_ratings(
                rng, mu_clin, cfg.sigma_tight, (n_sim, n_clinicians), cfg.scale
            )
            masks = retention_masks(
                np.concatenate([patients, clinicians], axis=1), **_rule_kwargs(cfg)
            )
            probabilities.append(float(masks["retain"].mean()))
        probabilities_arr = np.asarray(probabilities)
        curves[f"{n_patients}:{n_clinicians}"] = probabilities_arr
        stakeholder[f"{n_patients}:{n_clinicians}"] = {
            "P_retained_at_clinician_pi_0_50": float(
                np.interp(0.50, clinician_pi, probabilities_arr)
            ),
            "clinician_pi_needed_for_P_0_80": (
                float(np.interp(0.80, probabilities_arr, clinician_pi))
                if probabilities_arr.max() >= 0.80
                else None
            ),
        }
    results["patient_endorsed_item"] = {
        "patient_mu": cfg.patient_mu,
        "patient_pi": endorsement_probability(
            cfg.patient_mu, cfg.sigma_tight, cfg.relevance_cut, cfg.scale
        ),
        "clinician_pi_grid": clinician_pi.tolist(),
        "curves": {k: v.tolist() for k, v in curves.items()},
        "note": (
            "Probability that an item the patients rate highly survives the pooled "
            "rule as clinician endorsement falls."
        ),
        "results": stakeholder,
    }

    # ---- 4. the dispersion lemma -------------------------------------------
    results["dispersion_lemma"] = {
        "linear": iqr_lemma_violations(
            lemma_max, cfg.cvi_min, 2.0, cfg.relevance_cut, cfg.scale, "linear"
        ),
        "weibull": iqr_lemma_violations(
            lemma_max, cfg.cvi_min, 2.0, cfg.relevance_cut, cfg.scale, "weibull"
        ),
    }

    if outdir is not None:
        path = Path(outdir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "paper3.json").write_text(json.dumps(results, indent=1))
    return results


def figures(
    results: dict[str, Any], outdir: str | Path, cfg: Paper3Config | None = None
) -> dict[str, Path]:
    """Draw the paper 3 figure.

    Parameters
    ----------
    results : dict
        Output of :func:`run`.
    outdir : str or pathlib.Path
        Directory for the PNG file.
    cfg : Paper3Config, optional
        Configuration used for the run.

    Returns
    -------
    dict of str to pathlib.Path
    """
    import matplotlib.pyplot as plt

    cfg = cfg or Paper3Config()
    apply_style()
    rng = np.random.default_rng(cfg.seed + 1)
    n_sim = int(results["n_sim"])
    panels = [int(k) for k in results["operating_characteristics"]]

    fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.05))

    # retention curves against true endorsement
    ax = axes[0]
    pis = np.linspace(0.05, 0.995, 40)
    for i, n_panel in enumerate(panels):
        values = []
        for pi in pis:
            mu = mu_for_endorsement(float(pi), cfg.sigma_tight, cfg.relevance_cut, cfg.scale)
            x = simulate_ratings(rng, mu, cfg.sigma_tight, (n_sim, n_panel), cfg.scale)
            values.append(float(retention_masks(x, **_rule_kwargs(cfg))["retain"].mean()))
        ax.plot(pis, values, "-", lw=1.4, color=series_color(i), label=f"$n={n_panel}$")
    wide = []
    for pi in pis:
        mu = mu_for_endorsement(float(pi), cfg.sigma_wide, cfg.relevance_cut, cfg.scale)
        x = simulate_ratings(rng, mu, cfg.sigma_wide, (n_sim, 30), cfg.scale)
        wide.append(float(retention_masks(x, **_rule_kwargs(cfg))["retain"].mean()))
    ax.plot(pis, wide, "--", lw=1.3, color=series_color(5), label="$n=30$, wide spread")
    ax.axvline(cfg.cvi_min, ls=":", lw=1.0, color=INK)
    ax.set_xlabel(r"true endorsement $\pi=P(\mathrm{rating}\geq7)$")
    ax.set_ylabel("probability item is retained")
    ax.set_title("Retention curve, 9-point rule", fontsize=8.5)
    ax.legend(fontsize=6.4, loc="upper left")

    # error rates by panel size
    ax = axes[1]
    oc = results["operating_characteristics"]
    ax.plot(
        panels,
        [oc[str(n)]["false_drop_at_pi_0_90"] for n in panels],
        "o-",
        ms=4,
        lw=1.4,
        color=series_color(0),
        label=r"false drop, $\pi=0.90$",
    )
    ax.plot(
        panels,
        [oc[str(n)]["false_keep_at_pi_0_65"] for n in panels],
        "s-",
        ms=4,
        lw=1.4,
        color=series_color(1),
        label=r"false keep, $\pi=0.65$",
    )
    ax.axhline(0.10, ls="--", lw=0.9, color=INK)
    ax.set_xlabel("panel size $n$")
    ax.set_ylabel("error probability")
    ax.set_xticks(panels)
    ax.set_ylim(-0.03, 1.0)
    ax.set_title(rf"As specified: IQR $\leq{cfg.iqr_threshold}$", fontsize=8.5)
    ax.legend(fontsize=7, loc="center right")

    # dispersion threshold diagnostic
    ax = axes[2]
    diag = results["iqr_threshold_diagnostic"]
    sigmas = diag["sigmas"]
    for i, (threshold, rates) in enumerate(diag["false_drop_rate_by_threshold"].items()):
        ax.plot(sigmas, rates, "-", lw=1.4, color=series_color(i), label=rf"IQR $\leq{threshold}$")
    ax.axhline(0.10, ls="--", lw=0.9, color=INK)
    ax.set_xlabel(r"panel disagreement $\sigma$")
    ax.set_ylabel(r"false-drop rate, $\pi=0.90$")
    ax.set_ylim(-0.03, 1.0)
    ax.set_title("The dispersion threshold is what binds", fontsize=8.5)
    ax.legend(fontsize=6.6, loc="lower right")

    # stakeholder split
    ax = axes[3]
    split = results["patient_endorsed_item"]
    grid = split["clinician_pi_grid"]
    for i, (label, curve) in enumerate(split["curves"].items()):
        patients, clinicians = label.split(":")
        ax.plot(
            grid,
            curve,
            "-",
            lw=1.4,
            color=series_color(i),
            label=f"{patients} patients : {clinicians} clinicians",
        )
    ax.axhline(0.80, ls="--", lw=0.9, color=INK)
    ax.set_xlabel(r"clinician endorsement $\pi_{\rm clin}$")
    ax.set_ylabel("probability item is retained")
    ax.set_title("A patient-endorsed item can still be lost", fontsize=8.5)
    ax.legend(fontsize=6.4, loc="upper left")

    fig.suptitle(
        "Operating characteristics of the 9-point Delphi retention rule "
        rf"(median $\geq{cfg.median_min:g}$, IQR $\leq{cfg.iqr_threshold:g}$, "
        rf"I-CVI $\geq{cfg.cvi_min}$; {n_sim:,} panels per point)",
        fontsize=9,
        y=1.05,
    )
    return {"operating_characteristics": save_figure(fig, Path(outdir) / "p3_delphi_oc.png")}
