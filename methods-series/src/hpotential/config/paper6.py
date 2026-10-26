"""Published parameters for paper 6: the illustrative early assessment."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Paper6Config"]


@dataclass(frozen=True)
class Paper6Config:
    """Constants of the probabilistic analysis and value-of-information study.

    Every input is synthetic. The numbers demonstrate a workflow and say nothing
    about any real product.

    Parameters
    ----------
    threshold : float
        Willingness to pay per QALY, in a generic currency unit.
    population : int
        People eligible over the decision horizon.
    horizon_years : int
        Years the decision stands.
    discount : float
        Annual discount rate.
    draws : int
        Monte Carlo draws per scenario.
    comparator_quality : float
        Health-related quality on established prophylaxis.
    price_median, prophylaxis_median : float
        Medians of the two lognormal cost inputs.
    bmp_quality_multiplier : float
        Share of the utility gain carried into biomedical potential.
    max_years : int
        Length of the annual cycle.
    evppi_bins : int
        Equal-count bins for the conditional means behind EVPPI.
    scenarios : tuple of str
        Scenario names.
    seed : int
        Root seed.
    """

    threshold: float = 30_000.0
    population: int = 1_800
    horizon_years: int = 10
    discount: float = 0.03
    draws: int = 100_000
    comparator_quality: float = 0.72
    price_median: float = 260_000.0
    prophylaxis_median: float = 21_000.0
    bmp_quality_multiplier: float = 0.75
    max_years: int = 50
    evppi_bins: int = 120
    scenarios: tuple[str, ...] = ("optimistic", "central", "limited")
    seed: int = 20260815
