"""Early health technology assessment with value of information.

The case
--------
A hypothetical durable therapy for severe haemophilia compared with established
prophylaxis. Every input is synthetic and chosen only to demonstrate a workflow.
Nothing here is evidence about any real product.

What is computed
----------------
* Incremental discounted QALYs, incremental biomedical potential, and
  incremental cost per treated person, under three structural scenarios that
  differ in how fast the treatment effect wanes and how much trial efficacy
  survives routine implementation.
* Incremental net monetary benefit at a willingness-to-pay threshold, and the
  probability the therapy is favoured.
* Per-person and population expected value of perfect information.
* Expected value of partial perfect information for each input, by the
  single-parameter regression method of Strong, Oakley and Brennan:
  ``EVPPI_i = E[max(g_i, 0)] - max(E[INB], 0)`` with
  ``g_i(theta_i) = E[INB | theta_i]`` estimated by a smoothed conditional mean.
* How often a decision taken on biomedical potential disagrees with one taken
  on QALYs. That is the central caution of the paper, stated as a number.

Limitations
-----------
The coupling between benefit and cost matters: the therapy displaces
prophylaxis only while its effect lasts, so durability drives both sides. A
single deterministic score would hide that. The biomedical potential used here
applies a bespoke quality multiplier standing in for a preference-based utility
and carries no waning term, which is exactly why it is not interchangeable with
a QALY.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["HTAModel", "HTAResult", "evppi", "evpi", "smooth_conditional_mean"]


def smooth_conditional_mean(x: np.ndarray, y: np.ndarray, n_bins: int = 120) -> np.ndarray:
    """``E[y | x]`` by equal-count binning and a light three-point smoother.

    Parameters
    ----------
    x : numpy.ndarray
        Conditioning variable.
    y : numpy.ndarray
        Response.
    n_bins : int, default 120
        Number of equal-count bins.

    Returns
    -------
    numpy.ndarray
        Fitted conditional mean at each ``x``, in the original order.

    Notes
    -----
    Equal-count bins rather than equal-width bins, so the estimate does not
    degrade in the tails of a skewed input. The smoother is a fixed
    ``(0.25, 0.5, 0.25)`` kernel over bin means with edge padding.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    edges = np.linspace(0, xs.size, int(n_bins) + 1).astype(int)
    centres = np.array([xs[a:b].mean() for a, b in zip(edges[:-1], edges[1:], strict=False)])
    means = np.array([ys[a:b].mean() for a, b in zip(edges[:-1], edges[1:], strict=False)])
    kernel = np.array([0.25, 0.5, 0.25])
    smoothed = np.convolve(np.pad(means, 1, mode="edge"), kernel, mode="valid")
    out = np.empty_like(x)
    out[order] = np.interp(xs, centres, smoothed)
    return out


def evpi(inb: np.ndarray) -> float:
    """Per-person expected value of perfect information.

    Parameters
    ----------
    inb : numpy.ndarray
        Incremental net monetary benefit, one value per draw.

    Returns
    -------
    float
        ``E[max(INB, 0)] - max(E[INB], 0)``, which is zero only when the sign of
        the net benefit is the same in every draw.
    """
    inb = np.asarray(inb, dtype=float)
    return float(np.mean(np.maximum(inb, 0.0)) - max(inb.mean(), 0.0))


def evppi(inb: np.ndarray, theta: np.ndarray, n_bins: int = 120) -> float:
    """Expected value of partial perfect information for one input.

    Parameters
    ----------
    inb : numpy.ndarray
        Incremental net monetary benefit.
    theta : numpy.ndarray
        The input to be learned perfectly.
    n_bins : int, default 120
        Bins for the conditional mean.

    Returns
    -------
    float
        ``E_theta[max(E[INB | theta], 0)] - max(E[INB], 0)``.

    Notes
    -----
    A small negative value is possible and is reported as computed. It is
    estimation error in the conditional mean, not a negative value of
    information, and it flags an input that carries essentially none. Clipping
    it to zero silently would hide how noisy the estimate is.
    """
    base = max(float(np.mean(inb)), 0.0)
    g = smooth_conditional_mean(theta, inb, n_bins)
    return float(np.mean(np.maximum(g, 0.0)) - base)


def _default_scenarios() -> dict[str, dict[str, tuple[float, float]]]:
    return {
        "optimistic": {"waning": (0.005, 0.035), "access": (0.80, 0.98)},
        "central": {"waning": (0.030, 0.100), "access": (0.55, 0.85)},
        "limited": {"waning": (0.080, 0.200), "access": (0.30, 0.60)},
    }


@dataclass(frozen=True)
class HTAResult:
    """Per-draw outputs of one scenario."""

    incremental_qaly: np.ndarray
    incremental_bmp: np.ndarray
    incremental_cost: np.ndarray
    inb_qaly: np.ndarray
    inb_bmp: np.ndarray
    inputs: dict[str, np.ndarray]


@dataclass(frozen=True)
class HTAModel:
    """The synthetic early-assessment model.

    Parameters
    ----------
    threshold : float, default 30000.0
        Willingness to pay per QALY, in a generic currency unit.
    population : int, default 1800
        People eligible over the decision horizon.
    horizon_years : int, default 10
        Years the decision stands.
    discount : float, default 0.03
        Annual discount rate.
    comparator_quality : float, default 0.72
        Health-related quality on established prophylaxis.
    price_median : float, default 260000.0
        One-off price of the durable therapy.
    prophylaxis_median : float, default 21000.0
        Annual cost of established prophylaxis.
    bmp_quality_multiplier : float, default 0.75
        Share of the utility gain carried into biomedical potential by the
        bespoke quality mapping.
    max_years : int, default 50
        Length of the annual cycle.
    scenarios : dict
        Ranges for the two structural inputs in each scenario.
    """

    threshold: float = 30_000.0
    population: int = 1_800
    horizon_years: int = 10
    discount: float = 0.03
    comparator_quality: float = 0.72
    price_median: float = 260_000.0
    prophylaxis_median: float = 21_000.0
    bmp_quality_multiplier: float = 0.75
    max_years: int = 50
    scenarios: dict[str, dict[str, tuple[float, float]]] = field(default_factory=_default_scenarios)

    def sample(self, rng: np.random.Generator, scenario: str, n: int) -> dict[str, np.ndarray]:
        """Draw the six uncertain inputs for one scenario.

        Parameters
        ----------
        rng : numpy.random.Generator
            Explicit generator.
        scenario : str
            One of the configured scenario names.
        n : int
            Draws.

        Returns
        -------
        dict of str to numpy.ndarray
            ``L0`` remaining years, ``q_gain`` utility gain, ``waning``
            durability, ``access`` implementation gap, ``price``, ``proph``.

        Notes
        -----
        The utility gain and the durability share a latent treatment-quality
        factor ``z``, so they are not sampled independently: a therapy that
        helps more also tends to last longer. Sampling them independently would
        understate the spread of net benefit and overstate the value of
        learning either one alone.
        """
        if scenario not in self.scenarios:
            raise KeyError(f"unknown scenario {scenario!r}")
        spec = self.scenarios[scenario]
        n = int(n)
        l0 = rng.normal(28.0, 3.5, n).clip(8, 48)
        z = rng.standard_normal(n)
        q_gain = (0.080 + 0.030 * z + rng.normal(0, 0.014, n)).clip(-0.02, 0.22)
        waning = (rng.uniform(*spec["waning"], n) * np.exp(-0.35 * z)).clip(1e-4, 0.6)
        access = rng.uniform(*spec["access"], n)
        price = rng.lognormal(np.log(self.price_median), 0.30, n)
        proph = rng.lognormal(np.log(self.prophylaxis_median), 0.25, n)
        return {
            "L0": l0,
            "q_gain": q_gain,
            "waning": waning,
            "access": access,
            "price": price,
            "proph": proph,
        }

    def evaluate(self, p: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per treated person: discounted QALY gain, BMP gain, incremental cost.

        Parameters
        ----------
        p : dict of str to numpy.ndarray
            Inputs as returned by :meth:`sample`.

        Returns
        -------
        d_qaly, d_bmp, d_cost : numpy.ndarray

        Notes
        -----
        Incremental cost is the one-off price minus the prophylaxis spending the
        therapy displaces. That offset lasts only as long as the effect does, so
        durability drives both the benefit and the cost side.

        Biomedical potential is computed as the paper defines it: ``L * Q``,
        undiscounted, with a bespoke quality multiplier standing in for a
        preference-based utility, and no waning term. The three differences from
        the QALY calculation, discounting, the utility scale, and durability,
        are what drive the decisions apart.
        """
        t = np.arange(0, int(self.max_years))[:, None]
        disc = 1.0 / (1 + self.discount) ** t
        alive = (t < p["L0"][None, :]).astype(float)
        retained = np.exp(-p["waning"][None, :] * t)

        d_qaly = p["access"] * ((p["q_gain"][None, :] * retained * alive * disc).sum(axis=0))
        avoided = p["proph"] * ((retained * alive * disc).sum(axis=0))
        d_cost = p["price"] - avoided

        bmp0 = p["L0"] * self.comparator_quality
        bmp1 = p["L0"] * (self.comparator_quality + p["q_gain"] * self.bmp_quality_multiplier)
        d_bmp = p["access"] * (bmp1 - bmp0)
        return d_qaly, d_bmp, d_cost

    def run_scenario(self, rng: np.random.Generator, scenario: str, n: int = 100_000) -> HTAResult:
        """Sample and evaluate one scenario.

        Parameters
        ----------
        rng : numpy.random.Generator
            Explicit generator.
        scenario : str
            Scenario name.
        n : int, default 100000
            Draws.

        Returns
        -------
        HTAResult
        """
        p = self.sample(rng, scenario, n)
        d_qaly, d_bmp, d_cost = self.evaluate(p)
        return HTAResult(
            incremental_qaly=d_qaly,
            incremental_bmp=d_bmp,
            incremental_cost=d_cost,
            inb_qaly=self.threshold * d_qaly - d_cost,
            inb_bmp=self.threshold * d_bmp - d_cost,
            inputs=p,
        )

    def population_multiplier(self) -> float:
        """Discounted number of people the decision covers.

        Returns
        -------
        float
            ``sum_t (population / horizon) / (1 + discount)^t`` over the
            horizon.
        """
        annual = self.population / self.horizon_years
        return float(sum(annual / (1 + self.discount) ** t for t in range(self.horizon_years)))

    def summarise(self, result: HTAResult, n_bins: int = 120) -> dict[str, object]:
        """Decision summary, EVPI and EVPPI for one scenario.

        Parameters
        ----------
        result : HTAResult
            Output of :meth:`run_scenario`.
        n_bins : int, default 120
            Bins for the EVPPI conditional means.

        Returns
        -------
        dict
            Means of the incremental quantities, the two probabilities of
            being cost-effective, the disagreement rate, per-person and
            population EVPI, EVPPI by input, and each EVPPI as a share of EVPI.
        """
        inb = result.inb_qaly
        per_person = evpi(inb)
        evppis = {k: evppi(inb, v, n_bins) for k, v in result.inputs.items()}
        return {
            "mean_incremental_qaly": float(result.incremental_qaly.mean()),
            "mean_incremental_bmp": float(result.incremental_bmp.mean()),
            "mean_incremental_cost": float(result.incremental_cost.mean()),
            "mean_INB": float(inb.mean()),
            "P_cost_effective_QALY": float(np.mean(inb > 0)),
            "P_cost_effective_BMP": float(np.mean(result.inb_bmp > 0)),
            "P_decision_disagreement": float(np.mean((inb > 0) != (result.inb_bmp > 0))),
            "per_person_EVPI": per_person,
            "population_EVPI": float(per_person * self.population_multiplier()),
            "EVPPI": {k: float(v) for k, v in evppis.items()},
            "EVPPI_share_of_EVPI": {
                k: (float(max(v, 0.0) / per_person) if per_person > 0 else None)
                for k, v in evppis.items()
            },
        }
