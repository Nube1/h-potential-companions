"""Continuous-time model: solvers, their convergence, and identifiability.

The state equation is the scalar Ornstein-Uhlenbeck process

    dX = (a - b X) dt + sigma(X) dW,   sigma(X) = sigma  or  sigma X,

whose deterministic part has the exact solution ``a/b + (x0 - a/b) e^{-bt}`` and
whose additive-noise stationary law is ``N(a/b, sigma^2 / (2b))``.

What is verified here
---------------------
* :func:`rk4` on the deterministic equation, against the exact solution, to
  confirm fourth-order global accuracy.
* :func:`euler_maruyama` strong and weak order. The general globally-Lipschitz
  bound is strong order one half, but additive noise attains order one; a paper
  quoting a strong order must say which case it means.
* :class:`FokkerPlanck`, a conservative finite-volume scheme with zero-flux
  boundaries. Mass conservation is a property of the scheme, so the solver never
  renormalises the density. Renormalising every step would make the
  mass-conservation check vacuous: it would report machine epsilon whatever the
  flux computation did.
* :func:`kalman_loglik` and :func:`profile_sigma`, which say when process
  diffusion and observation error separate and when they do not.

Limitations
-----------
The schemes are explicit, so the Fokker-Planck time step obeys both a diffusive
and an advective stability limit; :func:`stable_time_step` computes it. Nothing
here has been fitted to a person, and the observation model is not validated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

__all__ = [
    "FokkerPlanck",
    "FokkerPlanckSolution",
    "LinearSDE",
    "convergence_order",
    "euler_maruyama",
    "kalman_loglik",
    "profile_sigma",
    "rk4",
    "simulate_state",
    "stable_time_step",
    "strong_error_study",
    "weak_error_study",
]


@dataclass(frozen=True)
class LinearSDE:
    """Scalar Ornstein-Uhlenbeck process ``dX = (a - bX) dt + sigma(X) dW``.

    Parameters
    ----------
    a : float
        Drift intercept. The mean reverts to ``a / b``.
    b : float
        Reversion rate, strictly positive.
    sigma : float
        Diffusion coefficient.
    multiplicative : bool, default False
        When true the diffusion is ``sigma * X``, otherwise the constant
        ``sigma``.
    """

    a: float = 1.2
    b: float = 0.6
    sigma: float = 0.45
    multiplicative: bool = False

    def __post_init__(self) -> None:
        if self.b <= 0:
            raise ValueError("b must be positive for a mean-reverting process")
        if self.sigma < 0:
            raise ValueError("sigma must not be negative")

    @property
    def mean_reversion_level(self) -> float:
        """``a / b``, the long-run mean."""
        return self.a / self.b

    def drift(self, x: np.ndarray | float) -> np.ndarray | float:
        """Drift ``a - b x``."""
        return self.a - self.b * x

    def diffusion(self, x: np.ndarray | float) -> np.ndarray | float:
        """Diffusion coefficient at ``x``."""
        return self.sigma * x if self.multiplicative else self.sigma

    def exact_solution(self, t: float, x0: float) -> float:
        """Solution of the deterministic equation at time ``t``.

        Parameters
        ----------
        t : float
            Time.
        x0 : float
            Initial state.

        Returns
        -------
        float
            ``a/b + (x0 - a/b) exp(-b t)``.
        """
        return float(
            self.mean_reversion_level + (x0 - self.mean_reversion_level) * np.exp(-self.b * t)
        )

    def stationary_pdf(self, x: np.ndarray) -> np.ndarray:
        """Stationary density of the additive-noise process.

        Parameters
        ----------
        x : numpy.ndarray
            Points at which to evaluate.

        Returns
        -------
        numpy.ndarray
            Density of ``N(a/b, sigma^2 / (2b))``.

        Raises
        ------
        ValueError
            If the process has multiplicative noise, whose stationary law is
            not this normal.
        """
        if self.multiplicative:
            raise ValueError("the normal stationary law holds only for additive noise")
        return np.asarray(
            stats.norm.pdf(
                np.asarray(x, dtype=float),
                loc=self.mean_reversion_level,
                scale=np.sqrt(self.sigma**2 / (2 * self.b)),
            )
        )


def rk4(f: Callable[[float, float], float], x0: float, t_end: float, h: float) -> float:
    """Classical fourth-order Runge-Kutta, returning the state at ``t_end``.

    Parameters
    ----------
    f : callable
        Right-hand side ``f(t, x)``.
    x0 : float
        Initial state.
    t_end : float
        Final time.
    h : float
        Step size. The number of steps is ``round(t_end / h)``, so choose ``h``
        that divides ``t_end``.

    Returns
    -------
    float
        The state at ``t_end``.

    Notes
    -----
    The global error is ``O(h^4)``. On the linear test problem of this module
    the observed order is 4.05 over steps from 0.4 down to 0.0125; the small
    excess over 4 is the higher-order term, not noise, because the calculation
    is deterministic.

    Examples
    --------
    >>> sde = LinearSDE()
    >>> value = rk4(lambda t, x: sde.drift(x), 3.0, 6.0, 0.05)
    >>> abs(value - sde.exact_solution(6.0, 3.0)) < 1e-8
    True
    """
    n = int(round(t_end / h))
    if n < 1:
        raise ValueError("h must be smaller than t_end")
    x, t = float(x0), 0.0
    for _ in range(n):
        k1 = f(t, x)
        k2 = f(t + h / 2, x + h * k1 / 2)
        k3 = f(t + h / 2, x + h * k2 / 2)
        k4 = f(t + h, x + h * k3)
        x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h
    return float(x)


def euler_maruyama(
    sde: LinearSDE,
    x0: float,
    t_end: float,
    h: float,
    n_paths: int,
    rng: np.random.Generator | None = None,
    base_increments: np.ndarray | None = None,
) -> np.ndarray:
    """Euler-Maruyama paths, returning the state at ``t_end``.

    Parameters
    ----------
    sde : LinearSDE
        The process.
    x0 : float
        Initial state, shared by every path.
    t_end : float
        Final time.
    h : float
        Step size, which must divide ``t_end``.
    n_paths : int
        Number of paths.
    rng : numpy.random.Generator, optional
        Generator for fresh Brownian increments. Required when
        ``base_increments`` is not given.
    base_increments : numpy.ndarray, optional
        Fine Brownian increments, shape ``(n_fine, n_paths)``, coarsened by
        summing consecutive blocks. Reusing one Brownian path across step sizes
        is what makes a strong-error comparison meaningful; drawing fresh noise
        at each step size measures only Monte Carlo scatter.

    Returns
    -------
    numpy.ndarray
        Terminal states, shape ``(n_paths,)``.

    Raises
    ------
    ValueError
        If neither ``rng`` nor ``base_increments`` is supplied, or if the fine
        grid does not divide evenly into the coarse one.
    """
    n = int(round(t_end / h))
    if n < 1:
        raise ValueError("h must be smaller than t_end")
    if base_increments is None:
        if rng is None:
            raise ValueError("supply either rng or base_increments")
        incs = rng.standard_normal((n, int(n_paths))) * np.sqrt(h)
    else:
        base_increments = np.asarray(base_increments, dtype=float)
        n_fine = base_increments.shape[0]
        if n_fine % n != 0:
            raise ValueError("the fine grid must divide evenly into the coarse grid")
        k = n_fine // n
        incs = base_increments.reshape(n, k, base_increments.shape[1]).sum(axis=1)
    x = np.full(incs.shape[1], float(x0))
    for i in range(n):
        x = x + sde.drift(x) * h + sde.diffusion(x) * incs[i]
    return x


def convergence_order(steps: np.ndarray, errors: np.ndarray) -> float:
    """Slope of ``log error`` on ``log step``.

    Parameters
    ----------
    steps : numpy.ndarray
        Step sizes.
    errors : numpy.ndarray
        Errors at those steps, all strictly positive.

    Returns
    -------
    float
        The fitted order.
    """
    steps = np.asarray(steps, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if np.any(errors <= 0):
        raise ValueError("errors must be positive to take logarithms")
    return float(np.polyfit(np.log(steps), np.log(errors), 1)[0])


def strong_error_study(
    sde: LinearSDE,
    x0: float,
    t_end: float,
    n_steps: list[int],
    n_paths: int,
    rng: np.random.Generator,
    n_fine: int = 9600,
) -> dict[str, object]:
    """Strong convergence of Euler-Maruyama against a fine reference path.

    Parameters
    ----------
    sde : LinearSDE
        The process.
    x0 : float
        Initial state.
    t_end : float
        Final time.
    n_steps : list of int
        Coarse step counts, each of which must divide ``n_fine``.
    n_paths : int
        Paths per step size.
    rng : numpy.random.Generator
        Explicit generator.
    n_fine : int, default 9600
        Steps in the reference solution.

    Returns
    -------
    dict
        ``steps``, ``strong_errors``, ``strong_order`` and ``mc_se``, the last
        being the Monte Carlo standard error of the terminal mean at each step.

    Notes
    -----
    The reference and every coarse solution are driven by the same Brownian
    increments, so the difference is discretisation error and not sampling
    noise. Expect order one for additive noise and one half for multiplicative
    noise; that difference is the whole reason both cases are reported.
    """
    h_fine = t_end / n_fine
    fine = rng.standard_normal((n_fine, int(n_paths))) * np.sqrt(h_fine)
    reference = euler_maruyama(sde, x0, t_end, h_fine, n_paths, base_increments=fine)
    steps, errors, mc_se = [], [], []
    for count in n_steps:
        h = t_end / count
        xs = euler_maruyama(sde, x0, t_end, h, n_paths, base_increments=fine)
        steps.append(float(h))
        errors.append(float(np.sqrt(np.mean((xs - reference) ** 2))))
        mc_se.append(float(xs.std(ddof=1) / np.sqrt(int(n_paths))))
    return {
        "steps": steps,
        "strong_errors": errors,
        "strong_order": convergence_order(np.array(steps), np.array(errors)),
        "mc_se": mc_se,
    }


def weak_error_study(
    sde: LinearSDE, x0: float, t_end: float, n_steps: list[int]
) -> dict[str, object]:
    """Weak convergence in the mean, computed without Monte Carlo noise.

    Parameters
    ----------
    sde : LinearSDE
        The process.
    x0 : float
        Initial state.
    t_end : float
        Final time.
    n_steps : list of int
        Step counts.

    Returns
    -------
    dict
        ``steps``, ``weak_errors`` and ``weak_order``.

    Notes
    -----
    Because ``E[sigma(X) dW] = 0`` for both the additive and the multiplicative
    case, the Euler mean obeys the deterministic recursion
    ``m <- m + (a - b m) h`` exactly. The weak error in the mean can therefore
    be evaluated in closed form. Estimating it from a few thousand paths would
    be swamped: the sampling error is an order of magnitude larger than the bias
    it is meant to measure.
    """
    exact = sde.exact_solution(t_end, x0)
    steps, errors = [], []
    for count in n_steps:
        h = t_end / count
        m = float(x0)
        for _ in range(int(count)):
            m = m + sde.drift(m) * h
        steps.append(float(h))
        errors.append(float(abs(m - exact)))
    return {
        "steps": steps,
        "weak_errors": errors,
        "weak_order": convergence_order(np.array(steps), np.array(errors)),
    }


def stable_time_step(
    sde: LinearSDE, dx: float, x_lo: float, x_hi: float, safety: float = 0.4
) -> float:
    """Largest stable time step for the explicit finite-volume scheme.

    Parameters
    ----------
    sde : LinearSDE
        The process.
    dx : float
        Cell width.
    x_lo, x_hi : float
        Domain edges.
    safety : float, default 0.4
        Fraction of the limit to take.

    Returns
    -------
    float
        ``safety * min(dx^2 / (2 D), dx / max|mu|)`` with ``D = sigma^2 / 2``.
    """
    diffusivity = 0.5 * sde.sigma**2
    mu_max = max(abs(sde.drift(x_lo)), abs(sde.drift(x_hi)))
    return float(safety * min(dx * dx / (2 * diffusivity), dx / mu_max))


@dataclass(frozen=True)
class FokkerPlanckSolution:
    """Result of a Fokker-Planck run.

    Attributes
    ----------
    centres : numpy.ndarray
        Cell centres.
    density : numpy.ndarray
        Density at the final time.
    mass : numpy.ndarray
        Total mass after every step, starting from the initial condition.
    min_density : float
        Smallest density seen at any time, the non-negativity check.
    dx, dt : float
        Cell width and time step.
    """

    centres: np.ndarray
    density: np.ndarray
    mass: np.ndarray
    min_density: float
    dx: float
    dt: float

    @property
    def mass_drift(self) -> float:
        """``|mass(T) - mass(0)|``, which the scheme conserves to rounding."""
        return float(abs(self.mass[-1] - self.mass[0]))

    def l1_error(self, exact: np.ndarray) -> float:
        """``L1`` distance to a reference density on the same grid."""
        return float(np.sum(np.abs(self.density - np.asarray(exact, dtype=float))) * self.dx)


class FokkerPlanck:
    """Conservative finite-volume solver for the Fokker-Planck equation.

    Parameters
    ----------
    sde : LinearSDE
        The process whose density is propagated.
    x_lo, x_hi : float
        Domain edges. Zero-flux (reflecting) boundaries are imposed there, so
        the domain must be wide enough that the true density is negligible at
        the edges.

    Notes
    -----
    The equation solved is

        dp/dt = -d/dx[ mu(x) p ] + D d2p/dx2,   D = sigma^2 / 2,

    discretised in flux form. The advective flux is upwinded and the diffusive
    flux is central; both are evaluated at cell faces, and the two boundary
    fluxes are set to zero. Because the update is the difference of face fluxes,
    the sum of ``p * dx`` changes only by floating-point rounding: mass
    conservation is a property of the discretisation and is therefore worth
    checking.

    The density is never renormalised. Rescaling ``p`` to unit mass after every
    step would make the mass-conservation check vacuous, since it would report
    machine epsilon whatever the flux computation did, and it would hide a sign
    error in the flux behind a correct-looking total.

    The scheme is explicit, so the step must respect
    :func:`stable_time_step`. Upwinding keeps the density non-negative under
    that limit.

    Limitations
    -----------
    First-order accurate in space because of the upwind flux, which is what the
    observed ``L1`` grid order near 1 reflects. A second-order flux limiter
    would converge faster and is not implemented.
    """

    def __init__(self, sde: LinearSDE, x_lo: float = -2.0, x_hi: float = 8.0) -> None:
        if x_hi <= x_lo:
            raise ValueError("x_hi must exceed x_lo")
        if sde.multiplicative:
            raise ValueError("this solver assumes state-independent diffusion")
        self.sde = sde
        self.x_lo = float(x_lo)
        self.x_hi = float(x_hi)

    def grid(self, nx: int) -> tuple[np.ndarray, np.ndarray, float]:
        """Cell centres, cell faces and cell width for ``nx`` cells."""
        dx = (self.x_hi - self.x_lo) / int(nx)
        centres = self.x_lo + dx * (np.arange(int(nx)) + 0.5)
        faces = self.x_lo + dx * np.arange(int(nx) + 1)
        return centres, faces, dx

    def solve(
        self,
        nx: int,
        n_steps: int,
        t_end: float,
        x0: float = 3.0,
        initial_sd: float = 0.35,
    ) -> FokkerPlanckSolution:
        """Propagate an initial density to ``t_end``.

        Parameters
        ----------
        nx : int
            Number of cells.
        n_steps : int
            Number of time steps. Use :func:`stable_time_step` to choose it.
        t_end : float
            Final time. Run well past the relaxation time ``1 / b`` if the
            stationary density is the target, so the remaining error is
            discretisation and not leftover transient.
        x0 : float, default 3.0
            Centre of the initial Gaussian.
        initial_sd : float, default 0.35
            Standard deviation of the initial Gaussian.

        Returns
        -------
        FokkerPlanckSolution
        """
        centres, faces, dx = self.grid(nx)
        dt = float(t_end) / int(n_steps)
        mu_f = self.sde.drift(faces)
        diffusivity = 0.5 * self.sde.sigma**2

        p = np.exp(-0.5 * ((centres - float(x0)) / float(initial_sd)) ** 2)
        p /= p.sum() * dx
        mass = [float(p.sum() * dx)]
        min_density = float(p.min())
        for _ in range(int(n_steps)):
            padded = np.concatenate([[p[0]], p, [p[-1]]])
            advective = np.where(mu_f > 0, mu_f * padded[:-1], mu_f * padded[1:])
            diffusive = -diffusivity * (padded[1:] - padded[:-1]) / dx
            flux = advective + diffusive
            flux[0] = flux[-1] = 0.0
            p = p - dt * (flux[1:] - flux[:-1]) / dx
            mass.append(float(p.sum() * dx))
            min_density = min(min_density, float(p.min()))
        return FokkerPlanckSolution(
            centres=centres,
            density=p,
            mass=np.asarray(mass),
            min_density=min_density,
            dx=dx,
            dt=dt,
        )


# ------------------------------------------------------ state-space filtering
def kalman_loglik(
    y: np.ndarray,
    times: np.ndarray,
    a: float,
    b: float,
    sigma: float,
    obs_sd: float,
    m0: float | None = None,
    v0: float | None = None,
) -> float:
    """Exact Kalman log-likelihood for a latent OU state observed with error.

    Parameters
    ----------
    y : numpy.ndarray
        Observations, shape ``(n_times,)`` or ``(n_times, n_indicators)``.
    times : numpy.ndarray
        Observation times, ascending and possibly irregular.
    a, b, sigma : float
        Process parameters.
    obs_sd : float
        Observation standard deviation, shared by the indicators.
    m0, v0 : float, optional
        Prior mean and variance. Default to the stationary law
        ``N(a/b, sigma^2 / (2b))``.

    Returns
    -------
    float
        Log-likelihood.

    Notes
    -----
    Between observations the state is propagated with the exact OU transition,
    ``m <- a/b + (m - a/b) e^{-b dt}`` and
    ``v <- v e^{-2b dt} + sigma^2 (1 - e^{-2b dt}) / (2b)``, so irregular
    spacing costs nothing in accuracy. Multiple indicators at one visit are
    conditionally independent given the state and are absorbed one at a time.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    times = np.asarray(times, dtype=float)
    if y.shape[0] != times.size:
        raise ValueError("y must have one row per observation time")
    n_ind = y.shape[1]
    level = a / b
    m = level if m0 is None else float(m0)
    v = sigma**2 / (2 * b) if v0 is None else float(v0)
    ll = 0.0
    prev = times[0]
    for i, t in enumerate(times):
        dt = t - prev
        if dt > 0:
            decay = np.exp(-b * dt)
            m = level + (m - level) * decay
            v = v * decay**2 + sigma**2 / (2 * b) * (1 - decay**2)
        prev = t
        for j in range(n_ind):
            s = v + obs_sd**2
            resid = y[i, j] - m
            ll += -0.5 * (np.log(2 * np.pi * s) + resid * resid / s)
            gain = v / s
            m = m + gain * resid
            v = v * (1 - gain)
    return float(ll)


def simulate_state(
    sde: LinearSDE,
    times: np.ndarray,
    rng: np.random.Generator,
    obs_sd: float,
    n_indicators: int = 1,
) -> np.ndarray:
    """Simulate one subject's noisy observations of the latent OU state.

    Parameters
    ----------
    sde : LinearSDE
        The process, which must have additive noise.
    times : numpy.ndarray
        Observation times.
    rng : numpy.random.Generator
        Explicit generator.
    obs_sd : float
        Observation standard deviation.
    n_indicators : int, default 1
        Indicators measured at each visit.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_times, n_indicators)``.
    """
    if sde.multiplicative:
        raise ValueError("the exact OU transition assumes additive noise")
    times = np.asarray(times, dtype=float)
    level = sde.mean_reversion_level
    stationary_sd = sde.sigma / np.sqrt(2 * sde.b)
    state = np.empty(times.size)
    current = rng.normal(level, stationary_sd)
    prev = times[0]
    for i, t in enumerate(times):
        dt = t - prev
        if dt > 0:
            decay = np.exp(-sde.b * dt)
            current = (
                level
                + (current - level) * decay
                + rng.normal(0.0, np.sqrt(sde.sigma**2 / (2 * sde.b) * (1 - decay**2)))
            )
        prev = t
        state[i] = current
    return state[:, None] + rng.standard_normal((times.size, int(n_indicators))) * obs_sd


def profile_sigma(
    y: np.ndarray,
    times: np.ndarray,
    sde: LinearSDE,
    grid: np.ndarray,
    obs_sd_bounds: tuple[float, float] = (0.02, 3.0),
) -> np.ndarray:
    """Profile log-likelihood over the process diffusion.

    Parameters
    ----------
    y : numpy.ndarray
        Observations for one subject.
    times : numpy.ndarray
        Observation times.
    sde : LinearSDE
        Supplies ``a`` and ``b``, which are held fixed.
    grid : numpy.ndarray
        Values of ``sigma`` to profile over.
    obs_sd_bounds : tuple of float, default (0.02, 3.0)
        Search range for the observation standard deviation.

    Returns
    -------
    numpy.ndarray
        Profile log-likelihood on ``grid``, shifted so its maximum is zero.

    Notes
    -----
    At each ``sigma`` the observation standard deviation is maximised out on the
    log scale by bounded scalar search. A 95 per cent profile interval is the
    set where the curve is above ``-1.92``. A flat profile means the two
    variances are not separated by the design, which is a statement about the
    design, not about the optimiser.
    """
    grid = np.asarray(grid, dtype=float)
    profile = np.empty(grid.size)
    for i, sg in enumerate(grid):
        res = minimize_scalar(
            lambda log_sd, sg=sg: -kalman_loglik(y, times, sde.a, sde.b, sg, float(np.exp(log_sd))),
            bounds=(np.log(obs_sd_bounds[0]), np.log(obs_sd_bounds[1])),
            method="bounded",
        )
        profile[i] = -res.fun
    return profile - profile.max()
