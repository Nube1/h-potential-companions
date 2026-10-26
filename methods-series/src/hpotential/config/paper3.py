"""Published parameters for paper 3: the modified Delphi retention rule."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Paper3Config"]


@dataclass(frozen=True)
class Paper3Config:
    """Constants of the consensus operating-characteristics study.

    Parameters
    ----------
    scale : tuple of int
        Rating scale, inclusive.
    median_min, iqr_threshold, cvi_min, relevance_cut : float
        The rule as usually specified. ``iqr_threshold`` is 1.5 here because
        that is the rule whose error rates the paper reports; the reference
        engine defaults to 2.0 for the reason given in
        :mod:`hpotential.core.consensus`.
    panel_sizes : tuple of int
        Panel sizes studied.
    n_sim : int
        Simulated panels per point.
    sigma_tight, sigma_wide : float
        Panel disagreement in the two regimes.
    pi_good, pi_marginal : float
        Endorsement of the reference strong item and the reference marginal
        item.
    iqr_candidates : tuple of float
        Dispersion thresholds compared in the diagnostic.
    sigma_grid : tuple of float
        Panel disagreements over which the diagnostic is run.
    diagnostic_panel : int
        Panel size for the diagnostic.
    stakeholder_splits : tuple of tuple of int
        Patients-to-clinicians splits in the stakeholder analysis.
    patient_mu : float
        Latent standing of the item among patients.
    lemma_max_panel : int
        Largest panel size in the exhaustive dispersion-lemma check.
    seed : int
        Root seed.
    """

    scale: tuple[int, int] = (1, 9)
    median_min: float = 7.0
    iqr_threshold: float = 1.5
    cvi_min: float = 0.78
    relevance_cut: int = 7
    panel_sizes: tuple[int, ...] = (12, 20, 30, 45, 60)
    n_sim: int = 20000
    sigma_tight: float = 1.2
    sigma_wide: float = 2.0
    pi_good: float = 0.90
    pi_marginal: float = 0.65
    iqr_candidates: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5)
    sigma_grid: tuple[float, ...] = tuple(round(0.5 + 0.1 * i, 3) for i in range(20))
    diagnostic_panel: int = 30
    stakeholder_splits: tuple[tuple[int, int], ...] = ((6, 24), (12, 18), (15, 15))
    patient_mu: float = 8.3
    lemma_max_panel: int = 10
    seed: int = 20260815
