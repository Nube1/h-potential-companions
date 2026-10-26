"""Published parameters for paper 4: the continuous-time model."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Paper4Config"]


@dataclass(frozen=True)
class Paper4Config:
    """Constants of the numerical verification and identifiability study.

    Parameters
    ----------
    a, b, sigma : float
        Process parameters of ``dX = (a - bX) dt + sigma dW``.
    x0 : float
        Initial state.
    t_end : float
        Final time of the convergence studies.
    rk4_steps : tuple of float
        Step sizes for the deterministic convergence study.
    em_step_counts : tuple of int
        Step counts for the stochastic convergence study. Each must divide
        ``em_fine_steps``.
    em_paths : int
        Paths per step size.
    em_fine_steps : int
        Steps in the reference solution.
    fp_grids : tuple of int
        Cell counts for the Fokker-Planck grid-refinement study.
    fp_domain : tuple of float
        Domain edges.
    fp_t_end : float
        Final time, well past the relaxation time ``1 / b`` so that the
        remaining error is discretisation and not leftover transient.
    fp_cfl : float
        Fraction of the stability limit taken as the time step.
    obs_sd : float
        True observation standard deviation in the identifiability study.
    sigma_grid : tuple of float
        Grid over which the profile likelihood is computed.
    n_subjects : int
        Subjects whose profiles are summed.
    designs : tuple of tuple
        Label, number of visits, and number of indicators per visit.
    seed : int
        Root seed.
    """

    a: float = 1.2
    b: float = 0.6
    sigma: float = 0.45
    x0: float = 3.0
    t_end: float = 6.0
    rk4_steps: tuple[float, ...] = (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125)
    em_step_counts: tuple[int, ...] = (75, 150, 300, 600, 1200)
    em_paths: int = 4000
    em_fine_steps: int = 9600
    fp_grids: tuple[int, ...] = (100, 200, 400, 800)
    fp_domain: tuple[float, float] = (-2.0, 8.0)
    fp_t_end: float = 25.0
    fp_cfl: float = 0.4
    obs_sd: float = 0.35
    sigma_grid_lo: float = 0.12
    sigma_grid_hi: float = 1.20
    sigma_grid_n: int = 40
    n_subjects: int = 60
    designs: tuple[tuple[str, int, int], ...] = (
        ("sparse, 1 indicator", 5, 1),
        ("sparse, 3 indicators", 5, 3),
        ("dense, 3 indicators", 25, 3),
    )
    seed: int = 20260815
