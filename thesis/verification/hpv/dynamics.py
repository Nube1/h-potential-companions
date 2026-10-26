"""The disease-independent ODE, SDE and Fokker-Planck layers of Appendix B."""
from __future__ import annotations
import math
import numpy as np


def michaelis_menten(T, Vmax, Km):
    return Vmax * T / (Km + T)


def hill(x, K, n):
    return x ** n / (K ** n + x ** n)


def drift(X, *, Vmax, Km, T, k_inh, Y, gamma, Kd, n, Z, k_elim):
    """Right-hand side of eq. (B.1)."""
    return (michaelis_menten(T, Vmax, Km)
            - k_inh * Y * X
            + gamma * hill(Z, Kd, n)
            - k_elim * X)


def equilibrium(*, Vmax, Km, T, k_inh, Y, gamma, Kd, n, Z, k_elim):
    """Proposition B.2: the unique stable fixed point, input over output."""
    num = michaelis_menten(T, Vmax, Km) + gamma * hill(Z, Kd, n)
    den = k_inh * Y + k_elim
    if den <= 0:
        raise ValueError("denominator must be positive for a stable point")
    return num / den


def lipschitz_constant(*, k_inh, Y, k_elim):
    """Proposition B.1: |d(rhs)/dX| = k_inh*Y + k_elim."""
    return k_inh * Y + k_elim


def integrate_ode(x0, t_end, params, dt=1e-3):
    """Fixed-step RK4 on eq. (B.1)."""
    n = int(round(t_end / dt))
    x = float(x0)
    for _ in range(n):
        k1 = drift(x, **params)
        k2 = drift(x + 0.5 * dt * k1, **params)
        k3 = drift(x + 0.5 * dt * k2, **params)
        k4 = drift(x + dt * k3, **params)
        x += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return x


def euler_maruyama(x0, t_end, params, sigma, dt=1e-3, paths=1, seed=0):
    """Eq. (B.2). Returns the terminal states of every path."""
    rng = np.random.default_rng(seed)
    n = int(round(t_end / dt))
    x = np.full(paths, float(x0))
    sq = math.sqrt(dt)
    for _ in range(n):
        mu = np.array([drift(xi, **params) for xi in x])
        x = x + mu * dt + sigma * sq * rng.standard_normal(paths)
    return x


def ou_mean(x0, a, b, t):
    """Proposition B.4: E[X(t)] for dX = (a - bX)dt + sigma dW."""
    return a / b + (x0 - a / b) * math.exp(-b * t)


def fokker_planck_step(rho, x, mu, D, dt):
    """One backward-Euler step of eq. (B.3) on a conservative finite-volume
    grid with reflecting (zero-flux) boundaries. Returns the new density.

    The assembled operator has zero column sums, so total probability is
    conserved exactly up to linear-solve error (Proposition B.5).
    """
    n = len(x)
    dx = x[1] - x[0]
    A = np.zeros((n, n))
    for i in range(n):
        # flux across the face to the right of cell i
        if i < n - 1:
            mu_f = 0.5 * (mu[i] + mu[i + 1])
            D_f = 0.5 * (D[i] + D[i + 1])
            up = i if mu_f >= 0 else i + 1          # upwind advection
            # d(rho)/dt = -A rho, so the cell losing the flux takes +mu/dx.
            A[i, up] += mu_f / dx
            A[i + 1, up] -= mu_f / dx
            A[i, i] += 0.5 * D_f / dx ** 2
            A[i, i + 1] -= 0.5 * D_f / dx ** 2
            A[i + 1, i + 1] += 0.5 * D_f / dx ** 2
            A[i + 1, i] -= 0.5 * D_f / dx ** 2
    M = np.eye(n) + dt * A
    return np.linalg.solve(M, rho), A
