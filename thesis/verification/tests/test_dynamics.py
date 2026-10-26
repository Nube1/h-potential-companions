"""Appendix B: the ODE backbone, the SDE, and the Fokker-Planck layer."""
import math
import numpy as np
import pytest
from hpv import dynamics as dyn

P = dict(Vmax=100.0, Km=10.0, T=20.0, k_inh=0.05, Y=2.0,
         gamma=30.0, Kd=50.0, n=2.0, Z=80.0, k_elim=0.8)


def test_proposition_existence_lipschitz_constant():
    """Proposition B.1: the right-hand side is globally Lipschitz in X with
    constant k_inh*Y + k_elim, so a unique solution exists and is stable."""
    L = dyn.lipschitz_constant(k_inh=P["k_inh"], Y=P["Y"], k_elim=P["k_elim"])
    assert L == pytest.approx(0.05 * 2.0 + 0.8)
    eps = 1e-6
    for X in (0.0, 5.0, 50.0, 500.0):
        num = (dyn.drift(X + eps, **P) - dyn.drift(X, **P)) / eps
        assert num == pytest.approx(-L, abs=1e-4)


def test_proposition_trajectories_converge_not_diverge():
    """The corrected reading of Proposition B.1: because the derivative is
    negative, two solutions starting apart close as exp(-Lt). An error in the
    assumed starting factor level is forgotten, and forgotten faster when the
    inhibitor titre is higher."""
    L = dyn.lipschitz_constant(k_inh=P["k_inh"], Y=P["Y"], k_elim=P["k_elim"])
    a = dyn.integrate_ode(2.0, 5.0, P, dt=1e-3)
    b = dyn.integrate_ode(3.0, 5.0, P, dt=1e-3)
    assert abs(a - b) == pytest.approx(1.0 * math.exp(-L * 5.0), rel=1e-3)
    assert abs(a - b) < 1.0
    hi = dict(P, Y=20.0)
    L_hi = dyn.lipschitz_constant(k_inh=hi["k_inh"], Y=hi["Y"], k_elim=hi["k_elim"])
    assert L_hi > L
    a2 = dyn.integrate_ode(2.0, 5.0, hi, dt=1e-3)
    b2 = dyn.integrate_ode(3.0, 5.0, hi, dt=1e-3)
    assert abs(a2 - b2) < abs(a - b)


def test_proposition_equilibrium_matches_the_solver():
    """Proposition B.2: the closed-form fixed point is what the integrator
    converges to from any starting value."""
    star = dyn.equilibrium(**P)
    for x0 in (0.0, 1.0, 200.0):
        assert dyn.integrate_ode(x0, 60.0, P, dt=1e-3) == pytest.approx(star, rel=1e-6)
    assert dyn.drift(star, **P) == pytest.approx(0.0, abs=1e-9)


def test_proposition_sde_reduces_to_ode_as_noise_vanishes():
    """Proposition B.3: the stochastic model contains the deterministic one.
    The discrepancy falls monotonically with sigma."""
    ref = dyn.integrate_ode(10.0, 5.0, P, dt=1e-3)
    gaps = []
    for sigma in (0.5, 0.1, 0.02):
        paths = dyn.euler_maruyama(10.0, 5.0, P, sigma, dt=1e-3, paths=200, seed=7)
        gaps.append(abs(paths.mean() - ref))
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[-1] < 0.05


def test_proposition_population_consistency_affine_drift():
    """Proposition B.4: for dX = (a - bX)dt + sigma dW the mean follows the
    deterministic law exactly."""
    a, b, sigma, x0, t = 4.0, 0.8, 0.6, 1.0, 3.0
    lin = dict(Vmax=a, Km=1e-12, T=1.0, k_inh=b, Y=1.0,
               gamma=0.0, Kd=1.0, n=1.0, Z=0.0, k_elim=0.0)
    analytic = dyn.ou_mean(x0, a, b, t)
    paths = dyn.euler_maruyama(x0, t, lin, sigma, dt=5e-4, paths=4000, seed=11)
    assert paths.mean() == pytest.approx(analytic, abs=0.03)


def test_the_average_patient_is_not_a_patient():
    """The reading attached to Proposition B.4. Two cohorts with the same mean
    can differ completely in time spent below a protective threshold, because
    that quantity is a non-linear function of the path."""
    thr = 3.0
    steady = dyn.euler_maruyama(4.0, 2.0, dict(P, Vmax=0, gamma=0, k_inh=0,
                                               k_elim=1e-9), 0.05,
                                dt=1e-3, paths=2000, seed=3)
    swingy = dyn.euler_maruyama(4.0, 2.0, dict(P, Vmax=0, gamma=0, k_inh=0,
                                               k_elim=1e-9), 1.2,
                                dt=1e-3, paths=2000, seed=3)
    assert steady.mean() == pytest.approx(swingy.mean(), abs=0.15)
    assert (steady < thr).mean() < (swingy < thr).mean()
    assert (swingy < thr).mean() > 0.15


def test_proposition_fokker_planck_conserves_mass():
    """Proposition B.5: with reflecting boundaries the assembled operator has
    zero column sums, so total probability is conserved."""
    x = np.linspace(0.0, 10.0, 201)
    dx = x[1] - x[0]
    rho = np.exp(-((x - 5.0) ** 2) / 0.5)
    rho /= rho.sum() * dx
    mu = 0.4 - 0.1 * x
    D = np.full_like(x, 0.3)
    mass = [rho.sum() * dx]
    for _ in range(50):
        rho, A = dyn.fokker_planck_step(rho, x, mu, D, dt=0.02)
        mass.append(rho.sum() * dx)
    assert abs(A.sum(axis=0)).max() < 1e-10      # zero column sums
    assert max(abs(m - mass[0]) for m in mass) < 1e-10


def test_fokker_planck_separates_drift_from_diffusion():
    """The reading attached to Proposition B.5. Raising everyone's dose moves
    the histogram; reducing variability narrows it. The two terms of the
    equation are the two things a policy can do, and they are not the same
    thing. A service with a good mean and a fat left tail needs the second."""
    x = np.linspace(0.0, 10.0, 401)
    dx = x[1] - x[0]
    base = np.exp(-((x - 5.0) ** 2) / 0.5); base /= base.sum() * dx
    mean = lambda r: float((x * r).sum() * dx)
    var = lambda r: float(((x - mean(r)) ** 2 * r).sum() * dx)

    def run(mu, D, dt=0.02, steps=50):
        r = base.copy()
        for _ in range(steps):
            r, _ = dyn.fokker_planck_step(r, x, mu, D, dt)
        return r

    still = run(np.zeros_like(x), np.full_like(x, 0.05))
    drifted = run(np.full_like(x, 0.5), np.full_like(x, 0.05))
    diffused = run(np.zeros_like(x), np.full_like(x, 0.8))

    # drift moves the centre and leaves the width almost alone
    assert mean(drifted) == pytest.approx(mean(still) + 0.5, abs=1e-3)
    assert var(drifted) == pytest.approx(var(still), rel=0.1)
    # diffusion widens the distribution and leaves the centre alone
    assert mean(diffused) == pytest.approx(mean(still), abs=1e-6)
    assert var(diffused) > 3 * var(still)


def test_time_varying_score_is_differentiable():
    """Proposition B.6: the product rule on L(t)Q(t)S(t) - B(t)."""
    L = lambda t: 10.0 - 0.1 * t
    Q = lambda t: 0.8 - 0.01 * t
    S = lambda t: 0.5 + 0.02 * t
    B = lambda t: 0.2 + 0.005 * t
    Hf = lambda t: L(t) * Q(t) * S(t) - B(t)
    t, h = 3.0, 1e-6
    numeric = (Hf(t + h) - Hf(t - h)) / (2 * h)
    analytic = (-0.1 * Q(t) * S(t) + L(t) * (-0.01) * S(t)
                + L(t) * Q(t) * 0.02 - 0.005)
    assert numeric == pytest.approx(analytic, rel=1e-6)


def test_worked_example_trough_level_for_mr_b():
    """Worked Example B.2: the fixed point read as a trough. Numerator over
    denominator, with the inhibitor sitting in the denominator, so doubling
    the denominator exactly halves the trough."""
    f = dyn.michaelis_menten(P["T"], P["Vmax"], P["Km"])
    h = P["gamma"] * dyn.hill(P["Z"], P["Kd"], P["n"])
    assert f == pytest.approx(66.667, abs=1e-3)
    assert h == pytest.approx(21.573, abs=1e-3)
    den = P["k_inh"] * P["Y"] + P["k_elim"]
    assert den == pytest.approx(0.9)
    star = dyn.equilibrium(**P)
    assert star == pytest.approx(98.044, abs=1e-3)
    assert star == pytest.approx((f + h) / den)
    # a titre that doubles the denominator halves the trough exactly
    hi = dict(P, Y=20.0)
    assert dyn.lipschitz_constant(k_inh=hi["k_inh"], Y=hi["Y"],
                                  k_elim=hi["k_elim"]) == pytest.approx(1.8)
    assert dyn.equilibrium(**hi) == pytest.approx(star / 2.0)


def test_worked_example_gronwall_two_assay_readings():
    """Worked Example B.1: a true starting activity of 2 against a reported 3.
    The gap closes as exp(-Lt) rather than growing, and closes faster when the
    titre is higher, because the titre enters the Lipschitz constant."""
    L = dyn.lipschitz_constant(k_inh=P["k_inh"], Y=P["Y"], k_elim=P["k_elim"])
    assert L == pytest.approx(0.9)
    gaps = {}
    for t in (1.0, 2.0, 5.0, 10.0):
        a = dyn.integrate_ode(2.0, t, P, dt=1e-3)
        b = dyn.integrate_ode(3.0, t, P, dt=1e-3)
        gaps[t] = abs(a - b)
        assert gaps[t] == pytest.approx(math.exp(-L * t), rel=1e-3)
    assert gaps[1.0] == pytest.approx(0.407, abs=1e-3)
    assert gaps[5.0] == pytest.approx(0.0111, abs=1e-4)
    assert gaps[10.0] == pytest.approx(0.000123, abs=1e-6)
    hi = dict(P, Y=20.0)
    a2 = dyn.integrate_ode(2.0, 5.0, hi, dt=1e-3)
    b2 = dyn.integrate_ode(3.0, 5.0, hi, dt=1e-3)
    assert abs(a2 - b2) == pytest.approx(0.000123, abs=1e-6)
    assert abs(a2 - b2) < gaps[5.0] / 50


def test_worked_example_drift_and_diffusion_in_numbers():
    """Worked Example B.3: the two terms of the Fokker-Planck equation are the
    two things a policy can do, and they move different summaries. Drift moves
    the mean and leaves the variance alone; diffusion does the reverse. Only
    the tail mass below a protective level distinguishes them."""
    x = np.linspace(0.0, 10.0, 401)
    dx = x[1] - x[0]
    base = np.exp(-((x - 5.0) ** 2) / 0.5)
    base /= base.sum() * dx
    mean = lambda r: float((x * r).sum() * dx)
    var = lambda r: float(((x - mean(r)) ** 2 * r).sum() * dx)
    tail = lambda r: float(r[x < 4.5].sum() * dx)

    def run(mu, D, dt=0.02, steps=50):
        r = base.copy()
        for _ in range(steps):
            r, _ = dyn.fokker_planck_step(r, x, mu, D, dt)
        return r

    zero = np.zeros_like(x)
    still = run(zero, np.full_like(x, 0.05))
    drifted = run(np.full_like(x, 0.5), np.full_like(x, 0.05))
    diffused = run(zero, np.full_like(x, 0.8))

    assert mean(base) == pytest.approx(5.0, abs=1e-9)
    assert var(base) == pytest.approx(0.25, abs=1e-4)
    assert tail(base) == pytest.approx(0.153, abs=1e-3)

    assert mean(still) == pytest.approx(5.0, abs=1e-6)
    assert var(still) == pytest.approx(0.300, abs=1e-3)
    assert tail(still) == pytest.approx(0.175, abs=1e-3)

    assert mean(drifted) == pytest.approx(5.5, abs=1e-3)
    assert var(drifted) == pytest.approx(0.318, abs=2e-3)
    assert tail(drifted) == pytest.approx(0.036, abs=1e-3)

    assert mean(diffused) == pytest.approx(5.0, abs=1e-6)
    assert var(diffused) == pytest.approx(1.05, abs=1e-3)
    assert tail(diffused) == pytest.approx(0.308, abs=1e-3)


def test_worked_example_rk4_order_of_convergence():
    """Worked Example D.2: halving the step on a problem with a closed-form
    solution. The observed error ratios approach 16 from above; they are not
    equal to 16, and quoting 16 as a result rather than a limit would be
    wrong."""
    a, b, x0, t = 4.0, 0.8, 1.0, 3.0
    lin = dict(Vmax=a, Km=1e-12, T=1.0, k_inh=b, Y=1.0,
               gamma=0.0, Kd=1.0, n=1.0, Z=0.0, k_elim=0.0)
    exact = dyn.ou_mean(x0, a, b, t)
    assert exact == pytest.approx(4.63713, abs=1e-5)
    errs = [abs(dyn.integrate_ode(x0, t, lin, dt=dt) - exact)
            for dt in (0.25, 0.125, 0.0625, 0.03125)]
    assert errs[0] == pytest.approx(1.372e-5, rel=1e-2)
    assert errs[-1] == pytest.approx(2.899e-9, rel=1e-2)
    ratios = [p / q for p, q in zip(errs, errs[1:])]
    assert ratios[0] == pytest.approx(17.40, abs=0.05)
    assert ratios[1] == pytest.approx(16.68, abs=0.05)
    assert ratios[2] == pytest.approx(16.31, abs=0.05)
    # monotonically approaching the fourth-order limit from above
    assert ratios == sorted(ratios, reverse=True)
    assert all(r > 16.0 for r in ratios)


def test_worked_example_apd_order_of_convergence():
    """Worked Example D.2: fixed-step RK4 error ratios of about 18.6, 17.3 and
    16.6 under successive step halving, approaching the fourth-order value of
    16 from above."""
    ref = dyn.integrate_ode(0.0, 4.0, P, dt=1e-5)
    errs = [abs(dyn.integrate_ode(0.0, 4.0, P, dt=dt) - ref)
            for dt in (0.4, 0.2, 0.1, 0.05)]
    ratios = [a / b for a, b in zip(errs, errs[1:])]
    assert ratios[0] == pytest.approx(18.6, abs=0.3)
    assert ratios[1] == pytest.approx(17.3, abs=0.3)
    assert ratios[2] == pytest.approx(16.6, abs=0.3)
    # monotonically approaching 16 from above
    assert ratios == sorted(ratios, reverse=True)
    assert all(r > 16.0 for r in ratios)
    assert errs == sorted(errs, reverse=True)
