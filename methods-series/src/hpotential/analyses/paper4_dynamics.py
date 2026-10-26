"""Paper 4: numerical verification and practical identifiability.

Four checks, each producing a number the manuscript quotes.

A. Runge-Kutta 4 on the linear ODE, against the exact solution.
B. Euler-Maruyama strong and weak convergence, run twice: additive noise
   (constant diffusion) and multiplicative noise (``sigma X``). The general
   globally-Lipschitz bound is strong order one half, but additive noise attains
   order one, so a manuscript quoting a strong order has to say which case it
   means.
C. Fokker-Planck finite-volume solver: mass conservation over time,
   non-negativity, and grid convergence of the stationary density against the
   analytic ``N(a/b, sigma^2 / (2b))`` limit. The density is never renormalised,
   so the mass check measures the scheme rather than the rescaling.
D. Practical identifiability of the state-space model by profile likelihood. An
   exact Kalman filter gives the likelihood for a latent OU state observed with
   error; process diffusion is profiled against observation error under three
   designs, to show when the two variances separate and when they do not.

All results are simulations of the equations in the paper. No participant data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config.paper4 import Paper4Config
from ..core.dynamical import (
    FokkerPlanck,
    LinearSDE,
    profile_sigma,
    rk4,
    simulate_state,
    stable_time_step,
    strong_error_study,
    weak_error_study,
)
from ..utils.numerical import convergence_order
from ..utils.plotting import INK, apply_style, save_figure, series_color

__all__ = ["FIGURES", "figures", "run"]

#: Figures this analysis can draw.
FIGURES: tuple[str, ...] = ("verification",)


def run(
    cfg: Paper4Config | None = None,
    outdir: str | Path | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Run the paper 4 verification study.

    Parameters
    ----------
    cfg : Paper4Config, optional
        Configuration. Defaults to the published one.
    outdir : str or pathlib.Path, optional
        Directory in which to write ``paper4.json``.
    quick : bool, default False
        Coarsen the grids and cut the number of paths and subjects. Quick mode
        does not reproduce the published numbers, and the observed orders in
        particular become unreliable because the coarse grids are not in the
        asymptotic regime.

    Returns
    -------
    dict
        ``parameters``, ``rk4``, ``euler_maruyama``, ``fokker_planck`` and
        ``identifiability``.

    Notes
    -----
    The Fokker-Planck time step is set by :func:`stable_time_step`, which honours
    both the diffusive and the advective limit of the explicit scheme. Choosing
    the step by the diffusive limit alone gives an unstable run at coarse grids,
    where the drift is large at the domain edges.
    """
    cfg = cfg or Paper4Config()
    sde = LinearSDE(cfg.a, cfg.b, cfg.sigma, multiplicative=False)
    results: dict[str, Any] = {
        "parameters": {"a": cfg.a, "b": cfg.b, "x0": cfg.x0, "sigma": cfg.sigma, "T": cfg.t_end},
        "quick_mode": bool(quick),
    }

    # ---- A. RK4 -------------------------------------------------------------
    steps = cfg.rk4_steps[:3] if quick else cfg.rk4_steps
    exact = sde.exact_solution(cfg.t_end, cfg.x0)
    errors = [abs(rk4(lambda t, x: sde.drift(x), cfg.x0, cfg.t_end, h) - exact) for h in steps]
    results["rk4"] = {
        "steps": list(steps),
        "errors": errors,
        "observed_order": convergence_order(np.array(steps), np.array(errors)),
    }

    # ---- B. Euler-Maruyama --------------------------------------------------
    n_paths = 400 if quick else cfg.em_paths
    counts = list(cfg.em_step_counts[:3] if quick else cfg.em_step_counts)
    em: dict[str, Any] = {}
    for index, multiplicative in enumerate((False, True)):
        process = LinearSDE(cfg.a, cfg.b, cfg.sigma, multiplicative=multiplicative)
        rng = np.random.default_rng([cfg.seed, index])
        strong = strong_error_study(
            process, cfg.x0, cfg.t_end, counts, n_paths, rng, cfg.em_fine_steps
        )
        weak = weak_error_study(process, cfg.x0, cfg.t_end, counts)
        name = "multiplicative_noise" if multiplicative else "additive_noise"
        em[name] = {**strong, **weak}
    results["euler_maruyama"] = em

    # ---- C. Fokker-Planck ---------------------------------------------------
    grids = cfg.fp_grids[:2] if quick else cfg.fp_grids
    t_end = 5.0 if quick else cfg.fp_t_end
    fp = FokkerPlanck(sde, *cfg.fp_domain)
    fp_results: dict[str, Any] = {}
    widths, l1 = [], []
    for nx in grids:
        _, _, dx = fp.grid(int(nx))
        dt_cap = stable_time_step(sde, dx, *cfg.fp_domain, safety=cfg.fp_cfl)
        n_steps = int(np.ceil(t_end / dt_cap))
        solution = fp.solve(int(nx), n_steps, t_end, x0=cfg.x0)
        error = solution.l1_error(sde.stationary_pdf(solution.centres))
        widths.append(dx)
        l1.append(error)
        fp_results[str(nx)] = {
            "mass_drift": solution.mass_drift,
            "min_density": solution.min_density,
            "L1_error_vs_stationary": error,
            "dx": solution.dx,
            "dt": solution.dt,
        }
    fp_results["grid_convergence_order_L1"] = convergence_order(np.array(widths), np.array(l1))
    results["fokker_planck"] = fp_results

    # ---- D. practical identifiability --------------------------------------
    n_subjects = 6 if quick else cfg.n_subjects
    n_grid_points = 12 if quick else cfg.sigma_grid_n
    grid = np.linspace(cfg.sigma_grid_lo, cfg.sigma_grid_hi, n_grid_points)
    ident: dict[str, Any] = {}
    curves: dict[str, list[float]] = {}
    for design_index, (name, n_visits, n_indicators) in enumerate(cfg.designs):
        times = np.linspace(0, cfg.t_end, int(n_visits))
        rng = np.random.default_rng([cfg.seed, 5, design_index])
        total = np.zeros_like(grid)
        for _ in range(int(n_subjects)):
            y = simulate_state(sde, times, rng, cfg.obs_sd, int(n_indicators))
            total += profile_sigma(y, times, sde, grid)
        total -= total.max()
        curves[name] = total.tolist()
        inside = grid[total >= -1.92]
        ident[name] = {
            "n_visits": int(n_visits),
            "n_indicators": int(n_indicators),
            "n_subjects": int(n_subjects),
            "sigma_hat": float(grid[int(np.argmax(total))]),
            "interval": [float(inside.min()), float(inside.max())] if inside.size else None,
            "interval_width": float(inside.max() - inside.min()) if inside.size else None,
        }
    results["identifiability"] = ident
    results["identifiability_curves"] = {"sigma_grid": grid.tolist(), "profiles": curves}

    if outdir is not None:
        path = Path(outdir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "paper4.json").write_text(json.dumps(results, indent=1))
    return results


def figures(
    results: dict[str, Any], outdir: str | Path, cfg: Paper4Config | None = None
) -> dict[str, Path]:
    """Draw the paper 4 figure.

    Parameters
    ----------
    results : dict
        Output of :func:`run`.
    outdir : str or pathlib.Path
        Directory for the PNG file.
    cfg : Paper4Config, optional
        Configuration used for the run.

    Returns
    -------
    dict of str to pathlib.Path
    """
    import matplotlib.pyplot as plt

    cfg = cfg or Paper4Config()
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 2.95))

    ax = axes[0]
    steps = np.array(results["rk4"]["steps"])
    errors = np.array(results["rk4"]["errors"])
    ax.loglog(steps, errors, "o-", ms=4, lw=1.3, color=series_color(0), label="RK4")
    ax.loglog(steps, errors[0] * (steps / steps[0]) ** 4, "--", lw=0.9, color=INK, label="slope 4")
    ax.set_xlabel("step $h$")
    ax.set_ylabel("global error at $T$")
    ax.set_title(f"RK4: observed order {results['rk4']['observed_order']:.2f}", fontsize=8.5)
    ax.legend(fontsize=7)

    ax = axes[1]
    add = results["euler_maruyama"]["additive_noise"]
    mult = results["euler_maruyama"]["multiplicative_noise"]
    h_a = np.array(add["steps"])
    s_a = np.array(add["strong_errors"])
    ax.loglog(
        h_a,
        s_a,
        "o-",
        ms=4,
        lw=1.3,
        color=series_color(0),
        label=f"additive, order {add['strong_order']:.2f}",
    )
    ax.loglog(
        np.array(mult["steps"]),
        np.array(mult["strong_errors"]),
        "s-",
        ms=4,
        lw=1.3,
        color=series_color(1),
        label=f"multiplicative, order {mult['strong_order']:.2f}",
    )
    ax.loglog(h_a, s_a[0] * (h_a / h_a[0]) ** 0.5, ":", lw=0.9, color=INK, label="slope 1/2")
    ax.loglog(h_a, s_a[0] * (h_a / h_a[0]) ** 1.0, "--", lw=0.9, color=INK, label="slope 1")
    ax.set_xlabel("step $h$")
    ax.set_ylabel("strong error")
    ax.set_title("Euler--Maruyama strong order", fontsize=8.5)
    ax.legend(fontsize=6.3)

    ax = axes[2]
    fp = results["fokker_planck"]
    grids = [k for k in fp if k != "grid_convergence_order_L1"]
    widths = np.array([fp[k]["dx"] for k in grids])
    l1 = np.array([fp[k]["L1_error_vs_stationary"] for k in grids])
    ax.loglog(
        widths, l1, "o-", ms=4, lw=1.3, color=series_color(0), label=r"$\|p_h-p_\infty\|_{1}$"
    )
    ax.loglog(widths, l1[0] * (widths / widths[0]) ** 1.0, "--", lw=0.9, color=INK, label="slope 1")
    ax.set_xlabel(r"cell width $\Delta x$")
    ax.set_ylabel(r"$L^1$ error")
    ax.set_title(f"Fokker--Planck: order {fp['grid_convergence_order_L1']:.2f}", fontsize=8.5)
    ax.legend(fontsize=7)

    ax = axes[3]
    curves = results["identifiability_curves"]
    grid = np.array(curves["sigma_grid"])
    for i, (name, profile) in enumerate(curves["profiles"].items()):
        ax.plot(grid, profile, "-", lw=1.4, color=series_color(i), label=name)
    ax.axhline(-1.92, ls="--", lw=0.9, color=INK)
    ax.axvline(cfg.sigma, ls=":", lw=1.0, color=series_color(5))
    ax.set_ylim(-12, 0.6)
    ax.set_xlabel(r"process diffusion $\sigma$")
    ax.set_ylabel("profile log-likelihood")
    ax.set_title("Practical identifiability", fontsize=8.5)
    ax.legend(fontsize=6.3, loc="lower right")

    fig.suptitle(
        "Numerical verification and identifiability checks for the dynamical model",
        fontsize=9,
        y=1.05,
    )
    return {"verification": save_figure(fig, Path(outdir) / "p4_verification.png")}
