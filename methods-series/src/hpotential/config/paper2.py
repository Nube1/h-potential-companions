"""Published parameters for paper 2: the two-stage prognostic pipeline."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Paper2Config"]


@dataclass(frozen=True)
class Paper2Config:
    """Constants of the whole-pipeline simulation.

    Parameters
    ----------
    gamma, psi : tuple of float
        The frozen measurement model carried over from paper 1.
    lam_std : float
        Standardised first-order loading of the planned design. With four
        indicators per domain this gives ``rho_D = 0.8361``.
    per_domain : int
        Indicators in each domain.
    beta_true : float
        Log hazard ratio per SD of the true construct. Protective and modest.
    n_grid : tuple of int
        Analytic sample sizes.
    event_fractions : tuple of tuple
        Endpoint name and the proportion of people observed to have an event.
    reps : int
        Replicates per design point.
    seed : int
        Root seed.
    ceiling_rho : float
        Determinacy ceiling, used for the events-inflation table.
    """

    gamma: tuple[float, ...] = (0.80, 0.70, 0.62)
    psi: tuple[float, ...] = (0.36, 0.51, 0.62)
    lam_std: float = 0.75
    per_domain: int = 4
    beta_true: float = -0.12
    n_grid: tuple[int, ...] = (1500, 2500, 4000, 6000)
    event_fractions: tuple[tuple[str, float], ...] = (
        ("mortality", 0.22),
        ("hospitalisation", 0.38),
    )
    reps: int = 120
    seed: int = 20260815
    power: float = 0.90
    alpha: float = 0.05
