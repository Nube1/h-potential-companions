"""Published parameters for paper 1: the measurement model and its determinacy.

Every field is the value used to produce the numbers in the manuscript. Change a
field and the analysis changes with it; the defaults are the published design.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Paper1Config"]


@dataclass(frozen=True)
class Paper1Config:
    """Design constants of the second-order H-potential model.

    Parameters
    ----------
    gamma : tuple of float
        Second-order loadings, one per domain, with ``Var(zeta) = 1``.
    psi : tuple of float
        Domain disturbance variances.
    n_indicators : int
        Total indicators.
    per_domain : int
        Indicators in each domain.
    target_determinacy : float
        The conventional benchmark for individual use.
    m_grid : tuple of int
        Indicators per domain over which the determinacy curve is drawn.
    lam_grid : tuple of float
        Standardised loadings for the determinacy curve.
    lam_check : float
        Loading used for the numerical check against the ceiling.
    m_check : int
        Indicators per domain for that check.
    identification_loadings : tuple of float
        Loadings at which the Jacobian rank is evaluated.
    n_grid : tuple of int
        Sample sizes in the simulation study.
    reps : int
        Replicates per design point.
    seed : int
        Root seed. Every replicate derives its own generator from it, so the
        study is reproducible and the replicates are independent.
    scenarios : dict
        Loading and residual-variance multiplier of each scenario.
    """

    gamma: tuple[float, ...] = (0.80, 0.70, 0.62)
    psi: tuple[float, ...] = (0.36, 0.51, 0.62)
    n_indicators: int = 12
    per_domain: int = 4
    target_determinacy: float = 0.90
    m_grid: tuple[int, ...] = tuple(range(3, 41))
    lam_grid: tuple[float, ...] = (0.55, 0.65, 0.75, 0.85)
    lam_check: float = 0.85
    m_check: int = 400
    identification_loadings: tuple[float, ...] = (0.55, 0.62, 0.75, 0.90)
    n_grid: tuple[int, ...] = (150, 250, 400, 600, 900, 1400)
    reps: int = 150
    seed: int = 20260814
    scenarios: tuple[tuple[str, float, float], ...] = (
        ("central", 0.90, 1.00),
        ("stress", 0.62, 1.45),
    )

    @property
    def n_domains(self) -> int:
        """Number of first-order factors."""
        return len(self.gamma)
