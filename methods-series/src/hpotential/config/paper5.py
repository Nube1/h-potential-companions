"""Published parameters for paper 5: the costed access-versus-efficacy rule."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Paper5Config"]


@dataclass(frozen=True)
class Paper5Config:
    """Constants of the costed-frontier analysis.

    Parameters
    ----------
    n_psa : int
        Draws in the probabilistic sensitivity analysis.
    n_check : int
        Draws in the closed-form check.
    settings : tuple of str
        Names of the two health-system settings, whose input ranges live in
        :class:`hpotential.core.access.AccessModel`.
    efficacy_fixed : float
        Efficacy level at which the indifference surface is drawn.
    cost_ratios : tuple of float
        Values of ``(Delta_A / Delta_T)(c_T / c_A)`` on the indifference plot.
    seed : int
        Root seed.
    """

    n_psa: int = 200_000
    n_check: int = 20_000
    settings: tuple[str, ...] = ("high-resource", "low-resource")
    efficacy_fixed: float = 0.80
    cost_ratios: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    access_grid_lo: float = 0.10
    access_grid_hi: float = 0.95
    access_grid_n: int = 260
    seed: int = 20260815
