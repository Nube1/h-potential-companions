"""Build the H-potential Colab notebook series with nbformat.

Run: python3 build_notebooks.py
Writes eight .ipynb files into notebooks/ next to this script.
"""

from __future__ import annotations

import pathlib

import nbformat as nbf

OUT = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
ORG = "Nube1"
REPO = "h-potential-companions"
SUBDIR = "methods-series"  # folder of this package inside the repository


# --------------------------------------------------------------- helpers
def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    cell = nbf.v4.new_code_cell(text.strip("\n"))
    cell["execution_count"] = None
    cell["outputs"] = []
    return cell


def badge(name: str) -> str:
    return (
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"(https://colab.research.google.com/github/{ORG}/{REPO}/blob/main/{SUBDIR}/notebooks/{name}.ipynb)"
    )


STATUS = """
## Status, before anything else

- **Every datum here is synthetic.** Nothing in this notebook was measured on a person.
- **Papers 1 to 3 are protocols.** They say what will be done, not what was found. No
  participant has been recruited and no Delphi panel has met.
- **Papers 4 to 6 are method illustrations.** Their inputs show a workflow. They are not
  estimates from any health system and not claims about any product.
- **No figure below is evidence about patients.** Each number is a property of a stated
  model or a stated decision rule, and nothing more.
"""

SETUP_CODE = f"""
# ---------------------------------------------------------------- setup
# Repository holding this package. The repository is private: in Colab, sign
# in to GitHub or clone it yourself before running.
GITHUB_OWNER = "{ORG}"
REPO = "{REPO}"
SUBDIR = "{SUBDIR}"

import sys

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    !pip install -q "hpotential[full] @ git+https://github.com/{{GITHUB_OWNER}}/{{REPO}}.git#subdirectory={{SUBDIR}}"
else:
    # Running from a clone: install the checkout this notebook sits in.
    import importlib.util
    import pathlib
    import subprocess

    if importlib.util.find_spec("hpotential") is None:
        for parent in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:
            if (parent / "src" / "hpotential").exists():
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "-e", str(parent)], check=True
                )
                break
        else:
            raise RuntimeError(
                "hpotential is not importable and no local checkout was found. "
                "Run this in Colab with access to the repository, or clone it."
            )

import hpotential

print(f"hpotential {{hpotential.__version__}}")
print(f"running in Colab: {{IN_COLAB}}")
"""

PLOT_CODE = '''
# Importing hpotential selects the Agg backend, because the library writes figures to
# files. Put the plots back into the page, then borrow the series figure style.
%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hpotential.utils.plotting import GRID, INK, apply_style, series_color

apply_style()
plt.rcParams["figure.dpi"] = 110  # 200 is for print; this is for a screen

try:
    import ipywidgets as widgets
    from IPython.display import display

    HAVE_WIDGETS = True
except Exception:  # Colab sometimes ships without a working widget stack
    HAVE_WIDGETS = False


def explore(fn, controls, defaults):
    """Draw the static case first, then attach sliders if ipywidgets works.

    The static call means the notebook still teaches something when it is read on
    GitHub, where no widget ever runs.
    """
    fn(**defaults)
    if HAVE_WIDGETS:
        display(widgets.interactive(fn, **controls))
    else:
        print("ipywidgets is unavailable here, so the static case above is all there is.")


print("plots inline, style applied, widgets:", HAVE_WIDGETS)
'''


def flag_code(full_run: bool = False, full_note: str = "") -> str:
    return f'''
# ---------------------------------------------------------------- run size
FULL_RUN = {full_run}   # True reproduces the published values. {full_note}


def banner():
    """Print at the top of every results cell. It has to be impossible to miss."""
    if FULL_RUN:
        print("=" * 74)
        print("FULL RUN: published replicate counts. These are the published values.")
        print("=" * 74)
    else:
        print("=" * 74)
        print("REDUCED RUN: cut replicate counts for speed.")
        print("The numbers below are NOT the published values. Set FULL_RUN = True.")
        print("=" * 74)


banner()
'''


NOT_ESTABLISH_TAIL = """
---

## What this does not establish
"""


def write(name: str, cells: list[nbf.NotebookNode]) -> pathlib.Path:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "colab": {"provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
            "mimetype": "text/x-python",
            "file_extension": ".py",
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
        },
    }
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    path = OUT / f"{name}.ipynb"
    OUT.mkdir(parents=True, exist_ok=True)
    # Keep cell ids stable across rebuilds so the diff shows only real changes:
    # reuse the ids of the existing file, position by position, else number them.
    old_ids = []
    if path.exists():
        old_ids = [c.get("id") for c in nbf.read(str(path), as_version=4)["cells"]]
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = old_ids[i] if i < len(old_ids) and old_ids[i] else f"{name}-{i:02d}"
    nbf.write(nb, str(path))
    return path


def head(
    name: str, title: str, blurb: str, runtime: str, full_note: str = "", full_run: bool = False
) -> list[nbf.NotebookNode]:
    return [
        md(f"{badge(name)}\n\n# {title}\n\n{blurb.strip()}"),
        md(STATUS),
        md(
            "## Setup\n\n"
            "The cell below installs the package straight from GitHub when this runs in "
            "Colab, and installs the local checkout when it does not. It prints the "
            f"resolved version so you know what you are running.\n\n{runtime}"
        ),
        code(SETUP_CODE),
        code(PLOT_CODE),
        code(flag_code(full_run, full_note)),
    ]


# ======================================================================= 00
def build_quickstart() -> pathlib.Path:
    name = "00_quickstart"
    cells = head(
        name,
        "H-potential in fifteen minutes",
        """
Six papers, six headline results, one pass. Each result gets about ten lines of code and
a plain statement of what it means. The notebooks numbered 01 to 06 take one result each
and do the work properly; this one is the map.
""",
        runtime=(
            "**Runtime.** Reduced run: under one minute. Full run: about four minutes, "
            "almost all of it papers 1 and 2."
        ),
        full_note="Full run here takes roughly four minutes.",
    )

    cells += [
        md(r"""
## 1. Paper 1: the determinacy ceiling

A second-order factor model puts indicators under domains and domains under one general
construct $\zeta$. The factor score $\widehat\zeta$ correlates with the truth at

$$\rho_D = \operatorname{corr}(\zeta, \widehat\zeta),$$

and no matter how many indicators are written, that correlation stops at

$$\rho_D^{\infty} = \sqrt{\frac{c}{1+c}}, \qquad c = \gamma^{\top}\Psi^{-1}\gamma .$$

The ceiling is set by the second-order part of the model, not by the item writing.
"""),
        code("""
banner()

from hpotential import determinacy_ceiling
from hpotential.core.measurement import required_second_order_signal, second_order_determinacy

rho_inf, c = determinacy_ceiling()
print(f"second-order signal c   = {c:.6f}")
print(f"ceiling rho_D_inf       = {rho_inf:.6f}")
print(f"at 400 indicators/domain= {second_order_determinacy(400, 0.85):.6f}   (still below)")
print(f"c needed for rho_D=0.90 = {required_second_order_signal(0.90):.4f}   (the design has 3.36)")
"""),
        md("""
**What it means.** The planned design cannot deliver a score good enough for a decision
about one person, and writing more items will not fix it.
**What it does not mean.** It is not a statement that the construct is unmeasurable. Raise
the second-order loadings, or cut the domain disturbances, and the ceiling rises.
"""),
        md(r"""
## 2. Paper 2: attenuation and the price in events

Fit a hazard model on the score rather than on the construct and the coefficient shrinks,
$\beta_{\text{obs}} \approx \rho_D\,\beta_{\text{true}}$. Matching the power of an oracle
study then needs $1/\rho_D^2$ times as many events.
"""),
        code("""
banner()

from hpotential.analyses.paper2_prognostic import population_model
from hpotential.config import Paper2Config
from hpotential.core.prognostic import attenuated_beta, events_inflation, required_events

cfg = Paper2Config()
rho_planned = population_model(cfg).determinacy()
print(f"rho_D of the planned design  {rho_planned:.4f}")
print(f"beta_true {cfg.beta_true:+.3f}  ->  expected fitted beta {attenuated_beta(cfg.beta_true, rho_planned):+.4f}")
for label, rho in [("oracle", 1.0), ("at the ceiling", rho_inf), ("planned design", rho_planned)]:
    print(f"events for 90% power, {label:<15} {required_events(cfg.beta_true, 0.90, 0.05, rho):7.0f}"
          f"   inflation {events_inflation(rho):.4f}")
"""),
        md("""
**What it means.** 947 events is a floor. No instrument built on this second-order
structure beats it, so the extra 30 per cent over the oracle study is a design cost to
budget for, not a defect to engineer away.
**What it does not mean.** It does not say the study is infeasible.
"""),
        md(r"""
## 3. Paper 3: what the Delphi rule actually does

The rule keeps an item when the median is at least 7, the interquartile range is at most
$\delta$, and the content-validity index is at least 0.78. At $\delta = 1.5$ the
dispersion limit is what does the dropping, and it drops good items more often as the
panel grows.
"""),
        code("""
banner()

from hpotential.core.consensus import operating_characteristics

n_sim = 20000 if FULL_RUN else 4000
rng = np.random.default_rng(20260815)
rows = []
for n_panel in (12, 20, 30, 45, 60):
    oc = operating_characteristics(n_panel, 0.90, 1.2, rng, n_sim, iqr_threshold=1.5)
    rows.append({"panel": n_panel, "false drop": oc["p_drop"], "IQR was binding": oc["iqr_fails"]})
pd.DataFrame(rows).round(3)
"""),
        md("""
**What it means.** A rule meant to protect the item set is throwing well-endorsed items
away, and the larger the panel the worse it gets. The dispersion limit is the cause.
**What it does not mean.** These are properties of the rule under a stated rating model.
No panel has met, so this is not a finding about any expert group.
"""),
        md(r"""
## 4. Paper 4: the solvers do what they claim

A continuous-time state model needs solvers that are checked, not asserted. Three checks:
fourth-order accuracy for RK4, strong order one for Euler-Maruyama with additive noise and
one half with multiplicative noise, and a Fokker-Planck scheme that conserves mass without
being renormalised.
"""),
        code("""
banner()

from hpotential.core.dynamical import (
    FokkerPlanck, LinearSDE, rk4, stable_time_step, strong_error_study,
)
from hpotential.utils.numerical import convergence_order

sde = LinearSDE(1.2, 0.6, 0.45)
steps = (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125)
exact = sde.exact_solution(6.0, 3.0)
err = [abs(rk4(lambda t, x: sde.drift(x), 3.0, 6.0, h) - exact) for h in steps]
print(f"RK4 observed order {convergence_order(np.array(steps), np.array(err)):.3f}")

paths = 4000 if FULL_RUN else 800
for mult in (False, True):
    s = strong_error_study(LinearSDE(1.2, 0.6, 0.45, multiplicative=mult), 3.0, 6.0,
                           [75, 150, 300, 600, 1200], paths, np.random.default_rng(7), 9600)
    print(f"Euler-Maruyama strong order, {'multiplicative' if mult else 'additive':<14} "
          f"{s['strong_order']:.3f}")

fp = FokkerPlanck(sde, -2.0, 8.0)
_, _, dx = fp.grid(200)
sol = fp.solve(200, int(np.ceil(25.0 / stable_time_step(sde, dx, -2.0, 8.0, 0.4))), 25.0, x0=3.0)
print(f"Fokker-Planck mass drift {sol.mass_drift:.3e}   with no renormalisation")
"""),
        md("""
**What it means.** The orders come out where the theory says, and the density solver keeps
its mass to machine precision because the scheme is conservative, not because anything was
rescaled afterwards.
**What it does not mean.** A verified solver says nothing about whether the model is the
right model.
"""),
        md(r"""
## 5. Paper 5: costing the access-versus-efficacy choice

The uncosted comparison says spend on access when $T > A$. Put price and feasibility back
in and the rule becomes

$$V=\frac{T}{A}\cdot\frac{\Delta_A}{\Delta_T}\cdot\frac{c_T}{c_A} > 1 .$$

Two of the three factors are the ones the uncosted rule throws away.
"""),
        code("""
banner()

from hpotential.core.access import AccessModel

draws = 200_000 if FULL_RUN else 40_000
rng = np.random.default_rng(20260815)
rows = []
for model in (AccessModel.high_resource(), AccessModel.low_resource()):
    psa = model.psa(rng, draws)
    rows.append({"setting": model.name, "median V": psa["median_V"],
                 "uncosted favours access": psa["P_uncosted_rule_favours_access"],
                 "costed favours access": psa["P_costed_rule_favours_access"],
                 "rules disagree": psa["P_disagreement"]})
pd.DataFrame(rows).round(3)
"""),
        md("""
**What it means.** In the high-resource setting the two rules point different ways on about
half the draws. The uncosted rule is not a safe shortcut there. In the low-resource setting
they never disagree, so the shortcut costs nothing.
**What it does not mean.** The input ranges are illustrative. The disagreement rate is a
property of those ranges, not a measurement of any health system.
"""),
        md(r"""
## 6. Paper 6: early assessment, and what it is worth learning

Incremental net monetary benefit at a threshold $\lambda$ is
$\text{INB} = \lambda\,\Delta\text{QALY} - \Delta\text{cost}$. Expected value of perfect
information is $E[\max(\text{INB},0)] - \max(E[\text{INB}],0)$: what you would pay to stop
guessing.
"""),
        code("""
banner()

from hpotential.core.hta import HTAModel

draws = 100_000 if FULL_RUN else 20_000
model = HTAModel()
rng = np.random.default_rng(20260815)
rows = []
for scenario in ("optimistic", "central", "limited"):
    s = model.summarise(model.run_scenario(rng, scenario, draws))
    rows.append({"scenario": scenario, "P(cost-effective, QALY)": s["P_cost_effective_QALY"],
                 "P(cost-effective, BMP)": s["P_cost_effective_BMP"],
                 "decisions disagree": s["P_decision_disagreement"],
                 "per-person EVPI": s["per_person_EVPI"],
                 "BMP / QALY gain": s["mean_incremental_bmp"] / s["mean_incremental_qaly"]})
pd.DataFrame(rows).round(3)
"""),
        md("""
**What it means.** Swap the outcome from QALYs to biomedical potential and the incremental
gain is about 1.9 times larger in the central scenario, which moves the decision. Two
outcome measures are not interchangeable just because both go up when health improves.
**What it does not mean.** Every input is synthetic. This is a workflow, not an appraisal.
"""),
        md("## The six results on one page"),
        code("""
banner()

fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.0))

ax = axes[0, 0]
ms = np.arange(3, 61)
for i, lam in enumerate((0.55, 0.65, 0.75, 0.85)):
    ax.plot(ms, [second_order_determinacy(int(m), lam) for m in ms], lw=1.3,
            color=series_color(i), label=rf"$\\lambda$={lam}")
ax.axhline(rho_inf, ls=":", lw=1.2, color=series_color(5))
ax.axhline(0.90, ls="--", lw=0.9, color=INK)
ax.set_xlabel("indicators per domain"); ax.set_ylabel(r"$\\rho_D$")
ax.set_title("1. the ceiling holds", fontsize=9); ax.legend(fontsize=6.5, ncol=2)

ax = axes[0, 1]
rhos = np.linspace(0.6, 1.0, 200)
ax.plot(rhos, [required_events(cfg.beta_true, 0.90, 0.05, r) for r in rhos], lw=1.5,
        color=series_color(0))
for r, lab in [(1.0, "oracle"), (rho_inf, "ceiling"), (rho_planned, "planned")]:
    ax.plot([r], [required_events(cfg.beta_true, 0.90, 0.05, r)], "o", ms=5,
            color=series_color(5))
    ax.annotate(lab, (r, required_events(cfg.beta_true, 0.90, 0.05, r)), fontsize=7,
                xytext=(-6, 8), textcoords="offset points", ha="right")
ax.set_xlabel(r"$\\rho_D$"); ax.set_ylabel("events for 90% power")
ax.set_title("2. the price in events", fontsize=9)

ax = axes[0, 2]
panels = [12, 20, 30, 45, 60]
drops = [operating_characteristics(n, 0.90, 1.2, np.random.default_rng(n), n_sim,
                                   iqr_threshold=1.5)["p_drop"] for n in panels]
ax.plot(panels, drops, "o-", lw=1.4, ms=4, color=series_color(1))
ax.set_xlabel("panel size"); ax.set_ylabel("false-drop rate"); ax.set_ylim(0, 1)
ax.set_title("3. the rule drops good items", fontsize=9)

ax = axes[1, 0]
s_add = strong_error_study(LinearSDE(1.2, 0.6, 0.45), 3.0, 6.0, [75, 150, 300, 600, 1200],
                           paths, np.random.default_rng(7), 9600)
s_mul = strong_error_study(LinearSDE(1.2, 0.6, 0.45, multiplicative=True), 3.0, 6.0,
                           [75, 150, 300, 600, 1200], paths, np.random.default_rng(7), 9600)
ax.loglog(s_add["steps"], s_add["strong_errors"], "o-", ms=4, color=series_color(0),
          label=f"additive, {s_add['strong_order']:.2f}")
ax.loglog(s_mul["steps"], s_mul["strong_errors"], "s-", ms=4, color=series_color(1),
          label=f"multiplicative, {s_mul['strong_order']:.2f}")
ax.set_xlabel("step $h$"); ax.set_ylabel("strong error")
ax.set_title("4. strong order, one against one half", fontsize=9); ax.legend(fontsize=7)

ax = axes[1, 1]
hi = AccessModel.high_resource().draw(np.random.default_rng(1), 20000)
v = (hi["T"] / hi["A"]) * (hi["dA"] / hi["dT"]) * (hi["cT"] / hi["cA"])
ax.hist(np.log10(v), bins=60, color=series_color(2), edgecolor="none")
ax.axvline(0.0, ls="--", lw=1.0, color=INK)
ax.set_xlabel(r"$\\log_{10} V$"); ax.set_ylabel("draws")
ax.set_title("5. $V>1$ on about half the draws", fontsize=9)

ax = axes[1, 2]
res = HTAModel().run_scenario(np.random.default_rng(3), "central", draws)
ax.scatter(res.incremental_qaly[:4000], res.incremental_cost[:4000] / 1000, s=2, alpha=0.25,
           color=series_color(3))
lam = np.linspace(0, res.incremental_qaly.max(), 50)
ax.plot(lam, 30.0 * lam, ls="--", lw=1.0, color=INK)
ax.set_xlabel("incremental QALYs"); ax.set_ylabel("incremental cost (thousands)")
ax.set_title("6. the cost-effectiveness plane", fontsize=9)

fig.tight_layout()
plt.show()
"""),
        md("""
## What each paper claims, and what it does not

| paper | the claim | the limit of the claim |
| --- | --- | --- |
| 1. Measurement | For this second-order structure $\\rho_D$ cannot pass 0.8778, and 0.90 needs $c \\ge 4.2632$ against a planned 3.3586. The model is locally identified, rank 27 of 27. | Nothing about whether the instrument will have this $c$, and nothing about whether the model fits. No item has been administered. |
| 2. Prognostic | Attenuation lands near $\\rho_D = 0.8361$ in simulation, and 90 per cent power needs about 1044 events against 730 for an oracle study. 947 is the floor at the ceiling. | A first-order regression-dilution result under non-informative measurement error. Simulated survival data, no cohort. |
| 3. Consensus | With the dispersion limit at 1.5 the false-drop rate for a well-endorsed item runs from 0.38 at 12 raters to 0.81 at 60, and dispersion is what binds on about 99 per cent of drops. At 2.0 the criterion cannot bind at all. | A property of the rule under a stated rating model and a stated quantile convention. No panel has met. |
| 4. Dynamics | RK4 order 4.05, Euler-Maruyama strong order 1.03 additive and about 0.53 multiplicative, mass conserved below 1e-15 with no renormalisation, first-order grid convergence. | Verification of solvers, not validation of the model. The parameters are illustrative. |
| 5. Access | $V$ reproduces the direct calculation to 6e-16, and the costed and uncosted rules disagree on about 48.6 per cent of high-resource draws and none of the low-resource ones. | Independent uniform inputs over illustrative ranges. $V>1$ ranks two programmes; it does not say either is worth funding. |
| 6. Early assessment | P(cost-effective on QALYs) is 0.410 in the central scenario, and the incremental biomedical potential is about 1.88 times the incremental QALY gain. | Synthetic inputs throughout. This is a workflow for an early decision, not an appraisal of a therapy. |
"""),
        md(NOT_ESTABLISH_TAIL + """
Nothing here is evidence about any patient, any clinician panel, any health system or any
product. Papers 1 to 3 are protocols and their numbers describe planned designs and stated
decision rules. Papers 4 to 6 use synthetic inputs picked to show a workflow.

A reduced run does not reproduce the published values. If `FULL_RUN` was `False` above,
treat every simulated number as an illustration of shape and sign, not of magnitude.

Nothing here says the H-potential construct exists, that the item bank measures it, or that
any of these models fits data that has not been collected.
"""),
    ]
    return write(name, cells)


# ======================================================================= 01
def build_measurement() -> pathlib.Path:
    name = "01_measurement"
    cells = head(
        name,
        "Paper 1: the determinacy ceiling, identification, and sample size",
        """
Three questions. How good can a factor score for this construct ever be? Is the model
identified? And how many people does fitting it need? The first has a closed-form answer
that no amount of item writing can move, and that is the result worth your time.
""",
        runtime=(
            "**Runtime.** Reduced run: about 30 seconds. Full run: about three minutes, "
            "nearly all of it the sample-size simulation."
        ),
        full_note="Full run here takes about three minutes.",
    )

    cells += [
        md(r"""
## The model

Indicators load on domains, domains load on one general construct:

$$y = \Lambda\eta + \varepsilon, \qquad \eta = \gamma\zeta + \delta,$$

with $\operatorname{Var}(\zeta) = 1$, $\operatorname{Var}(\delta) = \Psi$ and
$\operatorname{Var}(\varepsilon) = \Theta$. Writing $\Phi = \gamma\gamma^{\top} + \Psi$,
the model implies $\Sigma = \Lambda\Phi\Lambda^{\top} + \Theta$ and
$\operatorname{cov}(y,\zeta) = \Lambda\gamma$.

The regression (Thurstone) score has determinacy

$$\rho_D^2 = (\Lambda\gamma)^{\top}\Sigma^{-1}(\Lambda\gamma).$$

This is a population quantity. It does not depend on the sample size. Collecting more
people sharpens the estimates of $\Lambda$, $\gamma$ and $\Psi$; it does not make any one
person's score more determinate.
"""),
        md(r"""
## Deriving the ceiling

Indicators reach $\zeta$ only through $\eta$, so $\zeta \to \eta \to y$ is a Markov chain.
Everything is jointly Gaussian, so the score is $\widehat\zeta = E(\zeta \mid y)$ and
$\rho_D^2 = \operatorname{Var}\{E(\zeta \mid y)\}$. The tower property gives
$E(\zeta \mid y) = E\{E(\zeta \mid \eta) \mid y\}$, and the law of total variance gives

$$\operatorname{Var}\{E(\zeta\mid\eta)\}
 = \operatorname{Var}\{E(\zeta\mid y)\}
 + E\!\left[\operatorname{Var}\{E(\zeta\mid\eta)\mid y\}\right].$$

The last term is strictly positive whenever $\Theta \succ 0$, so no score built from the
indicators beats a score built from the domain factors themselves. And with
$\eta = \gamma\zeta + \delta$,

$$\operatorname{Var}\{E(\zeta\mid\eta)\} = \gamma^{\top}(\gamma\gamma^{\top}+\Psi)^{-1}\gamma
 = c - \frac{c^{2}}{1+c} = \frac{c}{1+c},
 \qquad c = \gamma^{\top}\Psi^{-1}\gamma,$$

by Sherman-Morrison. Hence

$$\rho_D < \rho_D^{\infty} = \sqrt{\frac{c}{1+c}},$$

approached from below as indicators per domain grow. A target $\rho_D \ge r$ needs
$c \ge r^2/(1-r^2)$, so $r = 0.90$ needs $c \ge 4.2632$.
"""),
        code("""
banner()

from hpotential.core.measurement import (
    determinacy_ceiling, determinacy_curve, required_second_order_signal, second_order_determinacy,
)

GAMMA = np.array([0.80, 0.70, 0.62])   # second-order loadings of the planned design
PSI = np.array([0.36, 0.51, 0.62])     # domain disturbance variances

rho_inf, c = determinacy_ceiling(GAMMA, PSI)
needed = required_second_order_signal(0.90)

print(f"c = gamma' Psi^-1 gamma          {c:.6f}")
print(f"ceiling  sqrt(c / (1 + c))       {rho_inf:.6f}")
print(f"c needed to reach rho_D = 0.90   {needed:.4f}")
print(f"the planned design is short by   {needed - c:.4f}")
"""),
        md("""
### Check the closed form against a direct evaluation

The ceiling is a limit. Evaluate $\\rho_D$ the long way, from the full model-implied
covariance at 400 indicators per domain, and it should sit just under the ceiling. Note
which side it falls on: the theorem says strictly below, always.
"""),
        code("""
banner()

direct = second_order_determinacy(400, 0.85, GAMMA, PSI)
print(f"closed-form ceiling                        {rho_inf:.6f}")
print(f"direct evaluation, 400 indicators/domain   {direct:.6f}")
print(f"gap                                        {rho_inf - direct:.3e}")
print(f"strictly below the ceiling, as the theorem requires: {direct < rho_inf}")
print()
print("1200 indicators. Three decimal places short of a number that is itself")
print("short of the 0.90 an individual-level decision would want.")
"""),
        md("""
### The curve, with the ceiling drawn on it

Four standardised loadings, indicators per domain on a log axis. The dotted line is the
ceiling and the dashed line is 0.90.
"""),
        code("""
banner()

ms = np.arange(3, 41)
curve = determinacy_curve(ms, (0.55, 0.65, 0.75, 0.85), GAMMA, PSI)

fig, ax = plt.subplots(figsize=(6.4, 4.0))
for i, col in enumerate(curve.columns):
    ax.plot(curve.index, curve[col], lw=1.5, color=series_color(i), label=col.replace("lambda=", r"$\\lambda$ = "))
ax.axhline(rho_inf, ls=":", lw=1.4, color=series_color(5))
ax.axhline(0.90, ls="--", lw=1.0, color=INK)
ax.text(40, rho_inf + 0.004, rf"ceiling $\\rho_D^\\infty$ = {rho_inf:.4f}", ha="right", fontsize=8,
        color=series_color(5))
ax.text(40, 0.904, "individual-use benchmark 0.90", ha="right", fontsize=8, color=INK)
ax.set_xscale("log")
ax.set_xticks([3, 4, 6, 8, 12, 20, 40])
ax.set_xticklabels(["3", "4", "6", "8", "12", "20", "40"])
ax.set_ylim(0.70, 0.95)
ax.set_xlabel("indicators per domain")
ax.set_ylabel(r"factor-score determinacy $\\rho_D$")
ax.set_title("Every curve flattens under the same line", fontsize=10)
ax.legend(fontsize=8, loc="lower right", ncol=2)
plt.show()

print("Raising the loading lifts the curve. It does not lift the line.")
"""),
        md(r"""
## The interactive part: move $\gamma$ and $\Psi$, watch the ceiling

This is the moment the paper is built around. The sliders below set the three second-order
loadings and the three domain disturbances. Watch two numbers: the ceiling, and how many
indicators per domain you would need to reach 0.90.

Try to reach 0.90 by adding indicators. You cannot. The only way is up the $\gamma$
sliders or down the $\Psi$ ones, because those are the only things $c$ depends on.

For speed the search over indicator counts uses a $K \times K$ reduction of the same
algebra rather than inverting a large covariance matrix. The next cell checks that the
reduction agrees with the library function before anything is built on it.
"""),
        code("""
banner()


def rho_reduced(m, lam, gamma, psi):
    \"\"\"rho_D for a balanced design, computed on K x K matrices.

    Woodbury on Sigma = Lambda Phi Lambda' + Theta turns the p x p inverse into a
    K x K one, with W = Lambda' Theta^-1 Lambda = diag(m lam^2 / theta_k):

        rho_D^2 = gamma' W gamma - gamma' W (Phi^-1 + W)^-1 W gamma.

    Same quantity, cheap enough to sit behind a slider.
    \"\"\"
    gamma = np.asarray(gamma, float)
    psi = np.asarray(psi, float)
    theta = np.clip(1.0 - lam**2 * (gamma**2 + psi), 1e-3, None)
    w = np.diag(m * lam**2 / theta)
    phi = np.outer(gamma, gamma) + np.diag(psi)
    inner = np.linalg.inv(np.linalg.inv(phi) + w)
    q = gamma @ w @ gamma - gamma @ w @ inner @ w @ gamma
    return float(np.sqrt(np.clip(q, 0.0, 1.0)))


print(f"{'m':>6} {'library':>12} {'reduction':>12} {'difference':>12}")
for m in (1, 4, 8, 40, 400):
    a = second_order_determinacy(m, 0.85, GAMMA, PSI)
    b = rho_reduced(m, 0.85, GAMMA, PSI)
    print(f"{m:>6} {a:>12.9f} {b:>12.9f} {abs(a - b):>12.2e}")
"""),
        code("""
banner()


def indicators_for_target(lam, gamma, psi, target=0.90, m_max=100_000):
    \"\"\"Smallest indicators per domain reaching the target, or None if unreachable.\"\"\"
    ceiling, _ = determinacy_ceiling(np.asarray(gamma), np.asarray(psi))
    if ceiling <= target:
        return None
    lo, hi = 1, 2
    while rho_reduced(hi, lam, gamma, psi) < target:
        hi *= 2
        if hi > m_max:
            return None
    while lo < hi:
        mid = (lo + hi) // 2
        if rho_reduced(mid, lam, gamma, psi) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def ceiling_panel(g1=0.80, g2=0.70, g3=0.62, p1=0.36, p2=0.51, p3=0.62, lam=0.75):
    gamma = np.array([g1, g2, g3])
    psi = np.array([p1, p2, p3])
    ceiling, c_now = determinacy_ceiling(gamma, psi)
    m_needed = indicators_for_target(lam, gamma, psi)

    ms = np.unique(np.round(np.logspace(0, 3, 60)).astype(int))
    rho = [rho_reduced(int(m), lam, gamma, psi) for m in ms]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(ms, rho, lw=1.6, color=series_color(0), label=rf"$\\lambda$ = {lam:.2f}")
    ax.axhline(ceiling, ls=":", lw=1.4, color=series_color(5))
    ax.axhline(0.90, ls="--", lw=1.0, color=INK)
    ax.set_xscale("log")
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("indicators per domain (log scale)")
    ax.set_ylabel(r"$\\rho_D$")
    ax.legend(fontsize=8, loc="lower right")
    reach = "unreachable at any number of indicators" if m_needed is None else f"{m_needed} per domain"
    ax.set_title(f"c = {c_now:.3f},  ceiling = {ceiling:.4f},  0.90 needs {reach}", fontsize=9)
    plt.show()

    print(f"second-order signal c        {c_now:.4f}   (0.90 needs c >= 4.2632)")
    print(f"ceiling                      {ceiling:.4f}")
    print(f"indicators per domain for 0.90: {reach}")


explore(
    ceiling_panel,
    controls=dict(
        g1=(0.10, 0.99, 0.01), g2=(0.10, 0.99, 0.01), g3=(0.10, 0.99, 0.01),
        p1=(0.05, 0.95, 0.01), p2=(0.05, 0.95, 0.01), p3=(0.05, 0.95, 0.01),
        lam=(0.30, 0.95, 0.05),
    ),
    defaults=dict(g1=0.80, g2=0.70, g3=0.62, p1=0.36, p2=0.51, p3=0.62, lam=0.75),
)
"""),
        md("""
**What the sliders show.** The horizontal ceiling moves only when $\\gamma$ or $\\Psi$
moves. The curve slides up and down with the loading, and stretches right forever, and
still never crosses the ceiling. When the ceiling is below 0.90 the answer to "how many
indicators do I need" is not a large number. There is no number.

**What they do not show.** Whether any real instrument has these $\\gamma$ and $\\Psi$. The
ceiling is computed from a hypothesised second-order structure. If the true loadings are
larger, the ceiling is higher, and the result still holds with the new $c$.
"""),
        md(r"""
## Identification

A model is locally identified at a parameter point when the Jacobian of the map from free
parameters to unique model-implied moments has full column rank there (Rothenberg, 1971).
With 12 indicators there are $12 \cdot 13 / 2 = 78$ unique moments and, under marker
scaling, 27 free parameters.

A rank check that never fails is not a check. So the run below includes two controls. The
first, two indicators per domain over three domains, is the case people expect to fail; it
does not, because the second-order factor supplies the missing restrictions. The second, a
second-order factor over only two domains, must fail: two loadings and two disturbances
cannot be recovered from a single between-domain covariance.
"""),
        code("""
banner()

from hpotential.core.measurement import identification_report

reports = [identification_report(lam, f"planned design, loading {lam}", GAMMA, PSI, per_domain=4)
           for lam in (0.55, 0.62, 0.75, 0.90)]
reports.append(identification_report(0.75, "control (a): 2 indicators per domain, 3 domains",
                                     GAMMA, PSI, per_domain=2))
reports.append(identification_report(0.75, "control (b): second-order factor over 2 domains",
                                     GAMMA[:2], PSI[:2], per_domain=4))

table = pd.DataFrame(reports)[
    ["label", "n_free", "n_moments", "rank", "identified", "smallest_singular_value", "condition_number"]
]
table.columns = ["design", "free", "moments", "rank", "identified", "smallest sv", "condition"]
table
"""),
        code("""
banner()

ctrl = reports[-1]
print("The negative control:")
print(f"  {ctrl['label']}")
print(f"  rank {ctrl['rank']} of {ctrl['n_free']} free parameters -> identified: {ctrl['identified']}")
print(f"  smallest singular value {ctrl['smallest_singular_value']:.3e}")
print(f"  condition number        {ctrl['condition_number']:.3e}")
print()
print("A smallest singular value at 1e-11 next to condition numbers around 11 for the")
print("planned designs is the difference between a rank deficiency and rounding.")
"""),
        md("""
## Sample size

Determinacy does not depend on $n$, but everything else does: whether the fit converges,
whether the solution is admissible, whether the estimates are unbiased, and whether a
nominal 95 per cent interval covers 95 per cent of the time. Coverage is counted here, not
assumed. A study that reports 0.95 without counting has reported the nominal level.

The run below is a simulation. Under a reduced run it uses a short grid and few replicates,
so its numbers are noisy and are not the published ones.
"""),
        code("""
banner()

import time

from hpotential.analyses import paper1_validation
from hpotential.config import Paper1Config

if FULL_RUN:
    cfg1 = Paper1Config()                                     # 150 reps, six sample sizes
else:
    cfg1 = Paper1Config(n_grid=(150, 400, 900), reps=10)      # noisy on purpose, and quick

start = time.time()
res1 = paper1_validation.run(cfg1)
print(f"{cfg1.reps} replicates per point, {len(cfg1.n_grid)} sample sizes, "
      f"{time.time() - start:.1f} s")

summary = res1["sample_size"]["summary"]
rows = [{"scenario": v["scenario"], "n": v["n"], "converged": v["convergence"],
         "admissible": v["admissible"], "worst coverage": v["min_coverage"],
         "max |rel bias|": v["max_abs_rel_bias"], "rho_D": v["determinacy"]}
        for v in summary.values()]
pd.DataFrame(rows).round(4)
"""),
        code("""
banner()

frame = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
for i, scenario in enumerate(sorted(frame["scenario"].unique())):
    part = frame[frame["scenario"] == scenario].sort_values("n")
    axes[0].plot(part["n"], part["admissible"], "o-", ms=4, lw=1.3, color=series_color(i),
                 label=scenario)
    axes[1].plot(part["n"], part["worst coverage"], "o-", ms=4, lw=1.3, color=series_color(i))
    axes[2].plot(part["n"], part["rho_D"], "o-", ms=4, lw=1.3, color=series_color(i))
axes[0].axhline(0.95, ls="--", lw=0.8, color=INK); axes[0].set_ylabel("admissible solutions")
axes[1].axhline(0.95, ls="--", lw=0.8, color=INK); axes[1].set_ylabel("worst 95% coverage")
axes[2].axhline(0.90, ls="--", lw=0.8, color=INK); axes[2].set_ylabel(r"$\\rho_D$")
axes[2].axhline(rho_inf, ls=":", lw=1.1, color=series_color(5))
for ax in axes:
    ax.set_xscale("log"); ax.set_xlabel("sample size $n$")
    ax.set_xticks(sorted(frame["n"].unique()))
    ax.set_xticklabels([str(n) for n in sorted(frame["n"].unique())], fontsize=7)
axes[0].legend(fontsize=8)
fig.tight_layout()
plt.show()

print("The third panel is the point: determinacy is flat in n and sits under the ceiling.")
print("More people buy better parameter estimates. They do not buy a better score.")
"""),
        md(NOT_ESTABLISH_TAIL + """
**The ceiling is algebra on an assumed population model.** It says nothing about whether
that model fits anything, and a determinacy computed from estimated parameters carries
their error. If the real $\\gamma$ is larger than 0.80, 0.70, 0.62, the ceiling is higher,
and the theorem still holds at the new $c$.

**The result is about the regression score.** Bartlett scores have a different, generally
lower, determinacy. Swapping scoring methods does not escape the bound.

**Identification is local, not global.** A full-rank Jacobian at one parameter point rules
out a flat direction there. It does not rule out a distant equivalent solution.

**The sample-size run is a simulation of the planned model.** No participant has been
recruited. Coverage near 0.95 says the estimator behaves when the model is true, which is
the easy case.

**A reduced run does not reproduce the published values.** The closed-form results are
unaffected by the flag, because they involve no simulation. The sample-size numbers are.
"""),
    ]
    return write(name, cells)


# ======================================================================= 02
def build_prognostic() -> pathlib.Path:
    name = "02_prognostic"
    cells = head(
        name,
        "Paper 2: attenuation, and what measuring instead of observing costs",
        """
Fit a survival model on a factor score rather than on the construct itself and the
coefficient shrinks toward the null. That shrinkage has a size, and the size has a
consequence for how many events a study needs. The consequence has a floor no instrument
can beat.
""",
        runtime=("**Runtime.** Reduced run: about 40 seconds. Full run: about ninety seconds."),
        full_note="Full run here takes about ninety seconds.",
    )

    cells += [
        md(r"""
## The claim

Let $\zeta$ be the standardised construct and $\widehat\zeta$ its regression score, with
determinacy $\rho_D$. Under a proportional-hazards model with hazard
$h(t) \exp(\beta\zeta)$ and measurement error that carries no information about survival,
the coefficient fitted on the standardised score is, to first order,

$$\beta_{\text{obs}} \approx \rho_D\,\beta_{\text{true}} .$$

This is classical regression dilution with the reliability replaced by the determinacy.
The step worth care is that $\rho_D$, not $\rho_D^2$, appears: the score is standardised,
so the shrinkage is the correlation.
"""),
        code("""
banner()

from hpotential.analyses.paper2_prognostic import population_model
from hpotential.config import Paper2Config
from hpotential.core.measurement import determinacy_ceiling
from hpotential.core.prognostic import (
    attenuated_beta, events_inflation, pipeline_replicate, required_events,
)

cfg = Paper2Config()
model = population_model(cfg)
rho_planned = model.determinacy()
rho_inf, c = determinacy_ceiling(np.asarray(cfg.gamma), np.asarray(cfg.psi))

print(f"planned design: {cfg.per_domain} indicators per domain, standardised loading {cfg.lam_std}")
print(f"rho_D of the planned design   {rho_planned:.4f}")
print(f"ceiling for this structure    {rho_inf:.4f}")
print()
print(f"beta_true                     {cfg.beta_true:+.4f}   (HR {np.exp(cfg.beta_true):.4f} per SD)")
print(f"predicted attenuated beta     {attenuated_beta(cfg.beta_true, rho_planned):+.4f}"
      f"   (HR {np.exp(attenuated_beta(cfg.beta_true, rho_planned)):.4f})")
"""),
        md("""
## Simulate the whole pipeline

A prediction about attenuation is worth having only if it survives the pipeline it
describes. One replicate does all of it: draw people, draw their indicators from the
measurement model, fit the factor model on those indicators, score it, simulate survival
from the true construct, then fit two Cox models, one on the true construct and one on the
score. The ratio of the two coefficients is the attenuation, and both come from the same
people, so the ratio is not polluted by differences in the survival data.
"""),
        code("""
banner()

import time

reps = cfg.reps if FULL_RUN else 20
n_grid = cfg.n_grid if FULL_RUN else (1500, 2500)

start = time.time()
records = []
for endpoint_index, (endpoint, fraction) in enumerate(cfg.event_fractions):
    for n in n_grid:
        for rep in range(reps):
            rng = np.random.default_rng([cfg.seed, endpoint_index, int(n), int(rep)])
            rec = pipeline_replicate(int(n), endpoint, fraction, model, cfg.beta_true, rng,
                                     per_domain=cfg.per_domain)
            if rec is not None:
                records.append(rec)
print(f"{len(records)} replicates in {time.time() - start:.1f} s")

frame = pd.DataFrame([{"endpoint": r.endpoint, "n": r.n, "events": r.events,
                       "beta_oracle": r.beta_oracle, "beta_observed": r.beta_observed,
                       "rho_hat": r.rho_hat, "significant": r.significant,
                       "significant_oracle": r.significant_oracle} for r in records])

grouped = frame.groupby(["endpoint", "n"]).agg(
    reps=("events", "size"), events=("events", "mean"),
    beta_oracle=("beta_oracle", "mean"), beta_observed=("beta_observed", "mean"),
    power=("significant", "mean"), power_oracle=("significant_oracle", "mean"),
).reset_index()
grouped["attenuation"] = grouped["beta_observed"] / grouped["beta_oracle"]
grouped.round(4)
"""),
        code("""
banner()

pooled = frame["beta_observed"].mean() / frame["beta_oracle"].mean()
print(f"pooled attenuation ratio over {len(frame)} replicates   {pooled:.4f}")
print(f"predicted by rho_D of the population model              {rho_planned:.4f}")
print(f"difference                                              {abs(pooled - rho_planned):.4f}")
print()
print("Pooling matters. A ratio from one design point carries several per cent of Monte")
print("Carlo error, which is larger than the thing being estimated, so the per-point")
print("ratios scatter around the prediction by more than the quantity itself.")
print(f"per-point range in this run: {grouped['attenuation'].min():.3f} to "
      f"{grouped['attenuation'].max():.3f}")
"""),
        code("""
banner()

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

ax = axes[0]
for i, endpoint in enumerate(grouped["endpoint"].unique()):
    part = grouped[grouped["endpoint"] == endpoint].sort_values("n")
    ax.plot(part["n"], part["attenuation"], "o-", ms=5, lw=1.4, color=series_color(i),
            label=endpoint)
ax.axhline(rho_planned, ls="--", lw=1.1, color=INK)
ax.text(grouped["n"].max(), rho_planned + 0.006, rf"predicted $\\rho_D$ = {rho_planned:.4f}",
        ha="right", fontsize=8)
ax.set_xlabel("sample size $n$"); ax.set_ylabel("attenuation ratio")
ax.set_title("Fitted shrinkage against the prediction", fontsize=10)
ax.legend(fontsize=8)

ax = axes[1]
width = 0.35
labels = [f"{e}\\nn={n}" for e, n in zip(grouped["endpoint"], grouped["n"])]
pos = np.arange(len(grouped))
ax.bar(pos - width / 2, grouped["power_oracle"], width, color=series_color(0), label="oracle")
ax.bar(pos + width / 2, grouped["power"], width, color=series_color(1), label="on the score")
ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("power"); ax.set_ylim(0, 1.05)
ax.set_title("The same study, two covariates", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
"""),
        md(r"""
## The power corollary

For a single standardised covariate the information about $\beta$ from $d$ events is about
$d$, so the noncentrality of the score test is about $\beta^2 d$. Matching noncentrality
between the oracle study and the study that uses the score requires

$$\beta_{\text{obs}}^2\, d_{\text{score}} = \beta_{\text{true}}^2\, d_{\text{oracle}},$$

and substituting $\beta_{\text{obs}} = \rho_D\beta_{\text{true}}$ gives

$$\frac{d_{\text{score}}}{d_{\text{oracle}}} = \frac{1}{\rho_D^{2}} > \frac{1+c}{c} .$$

The lower bound depends only on the second-order structure. No number of indicators and no
number of participants brings the ratio below it.
"""),
        code("""
banner()

target_power, alpha = cfg.power, cfg.alpha
rows = []
for label, rho in [("construct observed (oracle)", 1.0),
                   ("at the structural ceiling", rho_inf),
                   ("at the planned design", rho_planned)]:
    rows.append({"design": label, "rho_D": rho,
                 "events inflation": events_inflation(rho),
                 "events for 90% power": required_events(cfg.beta_true, target_power, alpha, rho)})
events = pd.DataFrame(rows)
events["events for 90% power"] = np.ceil(events["events for 90% power"])
events.round(4)
"""),
        code("""
banner()

oracle_d, ceiling_d, planned_d = np.ceil(
    [required_events(cfg.beta_true, target_power, alpha, r) for r in (1.0, rho_inf, rho_planned)]
)
print(f"detecting HR {np.exp(cfg.beta_true):.3f} per SD at 90% power, 5% two-sided:")
print(f"  {oracle_d:.0f} events if the construct is observed")
print(f"  {ceiling_d:.0f} events at the ceiling      (+{100 * (ceiling_d / oracle_d - 1):.1f}%)")
print(f"  {planned_d:.0f} events at the planned design (+{100 * (planned_d / oracle_d - 1):.1f}%)")
print()
print(f"The {ceiling_d:.0f} is a floor. Every instrument built on gamma = {tuple(cfg.gamma)}")
print(f"and Psi = {tuple(cfg.psi)} pays it, however many items it has, because")
print("rho_D approaches the ceiling from below and never passes it.")
print()
print("A 30 per cent increase in events is a design cost, not a barrier. What the")
print("corollary rules out is the belief that the cost can be written away in items.")
"""),
        md(r"""
## Interactive: move $\rho_D$

One slider. It sets the determinacy and everything else follows: the attenuated hazard
ratio, the inflation factor, and the events needed for 90 per cent power. The dotted line
marks the ceiling for this second-order structure. Everything to the right of it is
unreachable.
"""),
        code("""
banner()


def power_panel(rho_D=0.8361):
    beta_obs = attenuated_beta(cfg.beta_true, rho_D)
    d_needed = required_events(cfg.beta_true, target_power, alpha, rho_D)

    grid = np.linspace(0.50, 1.0, 300)
    d_curve = [required_events(cfg.beta_true, target_power, alpha, r) for r in grid]

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(grid, d_curve, lw=1.6, color=series_color(0))
    ax.axvline(rho_inf, ls=":", lw=1.4, color=series_color(5))
    ax.axvline(rho_D, ls="-", lw=1.0, color=series_color(1))
    ax.plot([rho_D], [d_needed], "o", ms=7, color=series_color(1))
    ax.fill_betweenx([0, max(d_curve)], rho_inf, 1.0, color=GRID, alpha=0.6)
    ax.text(rho_inf - 0.005, max(d_curve) * 0.9, "unreachable", ha="right", fontsize=8,
            color=series_color(5))
    ax.set_ylim(0, min(max(d_curve), 3000))
    ax.set_xlabel(r"determinacy $\\rho_D$")
    ax.set_ylabel("events for 90% power")
    ax.set_title(f"rho_D = {rho_D:.4f}: {np.ceil(d_needed):.0f} events", fontsize=10)
    plt.show()

    print(f"attenuated log HR   {beta_obs:+.4f}")
    print(f"attenuated HR       {np.exp(beta_obs):.4f}   against a true {np.exp(cfg.beta_true):.4f}")
    print(f"events inflation    {events_inflation(rho_D):.4f}")
    print(f"events for 90% power {np.ceil(d_needed):.0f}   against {oracle_d:.0f} for an oracle study")
    if rho_D > rho_inf:
        print()
        print(f"NOTE: {rho_D:.4f} is above the ceiling {rho_inf:.4f}. No instrument")
        print("built on this second-order structure can reach it.")


explore(power_panel, controls=dict(rho_D=(0.50, 0.99, 0.005)), defaults=dict(rho_D=0.8361))
"""),
        md(NOT_ESTABLISH_TAIL + """
**Attenuation here is a first-order result.** It assumes joint normality, a standardised
covariate, and measurement error that carries no information about survival given
covariates. Under a frailty drift the observed coefficient moves further toward the null
over follow-up, so the true inflation is if anything larger than $1/\\rho_D^2$. The bound
is conservative in the direction that matters.

**The events calculation is the standard normal approximation** for a single standardised
covariate, with information about $\\beta$ from $d$ events taken as $d$. Adjust for other
covariates and the constant changes; the direction and the bound do not.

**The survival data are simulated from the model.** No cohort has been followed, no event
has been observed, and the event fractions are design assumptions.

**"947 events is a floor" is a statement about this second-order structure.** A different
instrument with genuinely larger $\\gamma$ has a different, lower floor. The claim is that
you cannot get under the floor by adding items to this one.

**A reduced run does not reproduce the published attenuation ratio.** The events arithmetic
is closed form and is the same under either flag.
"""),
    ]
    return write(name, cells)


# ======================================================================= 03
def build_consensus() -> pathlib.Path:
    name = "03_consensus"
    cells = head(
        name,
        "Paper 3: what the Delphi rule does when nobody is watching",
        """
A modified Delphi rule keeps an item when the median is high, the spread is small and
enough raters call it relevant. Reasonable on its face. Simulate it against items whose
true endorsement you know, and one of the three criteria turns out to be doing nearly all
the work, in the wrong direction.
""",
        runtime=("**Runtime.** Reduced run: about 45 seconds. Full run: about two minutes."),
        full_note="Full run here takes about two minutes.",
    )

    cells += [
        md(r"""
## The rule and the rating model

On the integer scale $1,\dots,9$ an item is retained when

$$\text{median} \ge 7, \qquad \mathrm{IQR} \le \delta, \qquad \text{I-CVI} \ge 0.78,$$

where the content-validity index is the share of raters giving at least 7. Quartiles are
the linear-interpolation order statistics, `numpy.percentile` with `method="linear"`, type
7 in the usual taxonomy. That convention is part of the rule, and later on it turns out to
be load bearing.

Ratings are generated as $x = \operatorname{clip}(\operatorname{round}(N(\mu,\sigma)),1,9)$.
An item's true endorsement $\pi$ is the probability that a single rating is at least 7 on
that rounded scale, so $\mu$ is obtained by inverting the endorsement of the rounded
variable, not the continuous one. Inverting the continuous version would leave a request
for $\pi = 0.90$ realising 0.956, and every error rate computed from it would describe a
different item.
"""),
        code("""
banner()

from hpotential.core.consensus import (
    endorsement_probability, iqr_lemma_violations, mu_for_endorsement,
    operating_characteristics, retention_masks, simulate_ratings,
)

SIGMA = 1.2
mu_good = mu_for_endorsement(0.90, SIGMA)
mu_marginal = mu_for_endorsement(0.65, SIGMA)

print(f"a well-endorsed item, pi = 0.90 -> mu = {mu_good:.4f}")
print(f"a marginal item,      pi = 0.65 -> mu = {mu_marginal:.4f}")
print()
print("check the inverse round-trips on the rounded scale:")
print(f"  endorsement at mu = {mu_good:.4f}: {endorsement_probability(mu_good, SIGMA):.6f}")

# Now solve the same request on the continuous scale and see what it really delivers.
from scipy.stats import norm

mu_continuous = 7.0 - SIGMA * norm.ppf(0.10)          # P(N(mu, sigma) >= 7) = 0.90
realised = endorsement_probability(mu_continuous, SIGMA)
print()
print(f"the continuous solution      mu = {mu_continuous:.4f}")
print(f"what it realises once ratings are rounded: {realised:.4f}, not 0.90")
print()
print("Half a scale point of mu, and the item under test is a different item. Every")
print("error rate below would describe that other item instead.")
"""),
        md("""
## The retention curve

For a genuinely good item, $\\pi = 0.90$, how often does the rule drop it? Intuition says
a bigger panel is a safer panel. Run it.
"""),
        code("""
banner()

import time

n_sim = 20000 if FULL_RUN else 5000
panel_sizes = (12, 20, 30, 45, 60)

start = time.time()
rng = np.random.default_rng(20260815)
rows = []
for n_panel in panel_sizes:
    good = operating_characteristics(n_panel, 0.90, SIGMA, rng, n_sim, iqr_threshold=1.5)
    marginal = operating_characteristics(n_panel, 0.65, SIGMA, rng, n_sim, iqr_threshold=1.5)
    rows.append({"panel": n_panel,
                 "false drop (pi=0.90)": good["p_drop"],
                 "false keep (pi=0.65)": marginal["p_retain"],
                 "median bound": good["median_fails"],
                 "IQR bound": good["iqr_fails"],
                 "CVI bound": good["cvi_fails"]})
oc = pd.DataFrame(rows)
print(f"{n_sim} panels per cell, {time.time() - start:.1f} s")
oc.round(4)
"""),
        code("""
banner()

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

ax = axes[0]
ax.plot(oc["panel"], oc["false drop (pi=0.90)"], "o-", ms=5, lw=1.5, color=series_color(1),
        label=r"false drop, $\\pi$ = 0.90")
ax.plot(oc["panel"], oc["false keep (pi=0.65)"], "s-", ms=5, lw=1.5, color=series_color(0),
        label=r"false keep, $\\pi$ = 0.65")
ax.set_xlabel("panel size"); ax.set_ylabel("rate"); ax.set_ylim(0, 1)
ax.set_title("A bigger panel drops good items more often", fontsize=10)
ax.legend(fontsize=8)

ax = axes[1]
bottom = np.zeros(len(oc))
for i, col in enumerate(["median bound", "IQR bound", "CVI bound"]):
    ax.bar(oc["panel"].astype(str), oc[col], 0.6, bottom=bottom, color=series_color(i), label=col)
    bottom = bottom + oc[col].to_numpy()
ax.set_xlabel("panel size"); ax.set_ylabel("share of dropped panels failing each test")
ax.set_title("Which criterion did the dropping", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

print("The shares sum past one because several criteria can fail together. That is")
print("the point of reporting all three rather than the first one that fired.")
"""),
        md("""
**The diagnosis.** The median criterion essentially never bites. The content-validity
index bites less and less as the panel grows. The dispersion limit bites on nearly every
drop, and it gets worse with panel size, because the sample interquartile range converges
on a population value above 1.5. The rule is not becoming more reliable with a larger
panel. It is becoming more reliably wrong.
"""),
        md(r"""
## Is it the threshold or the criterion?

Hold everything else fixed and sweep the dispersion limit $\delta$ over four candidates,
while panel disagreement $\sigma$ grows. If the problem were the value 1.5, some other
value would fix it.
"""),
        code("""
banner()

sigmas = np.round(np.arange(0.5, 2.5, 0.1), 3) if FULL_RUN else np.round(np.arange(0.5, 2.5, 0.2), 3)
candidates = (1.0, 1.5, 2.0, 2.5)
diag_sim = 20000 if FULL_RUN else 4000

start = time.time()
rng = np.random.default_rng(20260815)
diagnostic = {}
for threshold in candidates:
    diagnostic[threshold] = [
        operating_characteristics(30, 0.90, float(s), rng, diag_sim, iqr_threshold=threshold)["p_drop"]
        for s in sigmas
    ]
print(f"{len(candidates) * len(sigmas)} cells in {time.time() - start:.1f} s")

fig, ax = plt.subplots(figsize=(6.6, 4.0))
for i, threshold in enumerate(candidates):
    ax.plot(sigmas, diagnostic[threshold], "o-", ms=4, lw=1.4, color=series_color(i),
            label=rf"$\\delta$ = {threshold}")
ax.set_xlabel(r"panel disagreement $\\sigma$")
ax.set_ylabel(r"false-drop rate for an item with $\\pi$ = 0.90")
ax.set_ylim(0, 1)
ax.set_title("Panel of 30. Two of these thresholds are alive; two are not.", fontsize=10)
ax.legend(fontsize=8)
plt.show()

pd.DataFrame(diagnostic, index=pd.Index(sigmas, name="sigma")).round(3)
"""),
        md(r"""
## The dispersion lemma, and why $\delta \ge 2$ is inert

The 2.0 and 2.5 curves are flat and low. That is not luck.

**Lemma.** On the integer scale $1,\dots,9$ with relevance cut 7 and linear-interpolation
quartiles, an item whose I-CVI is strictly above 0.75 has $Q_1 \ge 7$. The scale maximum is
9, so $Q_3 \le 9$, and therefore $\mathrm{IQR} \le 2$. A limit $\delta \ge 2$ can never
bind, and the rule collapses to the median and the endorsement.

*Proof.* $Q_{0.25}$ interpolates between the order statistics at position $1 + 0.25(n-1)$.
If strictly more than three quarters of ratings are at least 7, then fewer than a quarter
fall below 7, so both order statistics bracketing that position are at least 7, hence
$Q_{0.25} \ge 7$. With $\kappa = 0.78 > 0.75$ the hypothesis holds. $\square$

Both hypotheses are load bearing. At exactly $\kappa = 0.75$ the panel $(1,7,8,8)$ has
I-CVI 0.75, median 7.5 and IQR 2.5. And under a different quantile convention the lemma is
simply false: under the Weibull rule the panel $(1,7,9,9,9)$ has I-CVI 0.8 and IQR 5.

You do not have to take this on trust. Every panel of every size up to 10 can be
enumerated, because the statistics do not care about rating order, so sorted multisets
suffice.
"""),
        code("""
banner()

max_panel = 10 if FULL_RUN else 8
start = time.time()
linear = iqr_lemma_violations(max_panel=max_panel, cvi_min=0.78, iqr_bound=2.0, method="linear")
weibull = iqr_lemma_violations(max_panel=max_panel, cvi_min=0.78, iqr_bound=2.0, method="weibull")
print(f"exhaustive enumeration up to panel size {max_panel}, {time.time() - start:.1f} s")
print()
for label, out in [("linear interpolation (type 7)", linear), ("Weibull rule (type 6)", weibull)]:
    print(f"{label}")
    print(f"  panels enumerated        {out['n_panels_checked']}")
    print(f"  passing I-CVI >= 0.78    {out['n_passing_cvi']}")
    print(f"  IQR > 2 among those      {out['violations']}")
    print(f"  largest IQR among those  {out['max_iqr_among_passing']}   at {out['worst_panel']}")
    print()
print("Zero violations under the convention the rule names, and hundreds under one it")
print("does not. A protocol that adopts this lemma has to name its quantile rule, not")
print("only its thresholds.")
"""),
        code("""
banner()

# The worst case is reachable, not hypothetical. Check the boundary panel by hand.
worst = np.array([[7, 7, 9, 9]], dtype=float)
masks = retention_masks(worst, median_min=7.0, iqr_threshold=2.0, cvi_min=0.78)
print(f"panel (7, 7, 9, 9): median {masks['median'][0]}, IQR {masks['iqr'][0]}, "
      f"I-CVI {masks['cvi'][0]}")
print(f"retained at delta = 2.0: {bool(masks['retain'][0])}")

edge = np.array([[1, 7, 8, 8]], dtype=float)
m2 = retention_masks(edge, median_min=7.0, iqr_threshold=2.0, cvi_min=0.75)
print()
print(f"panel (1, 7, 8, 8) at the excluded boundary kappa = 0.75:")
print(f"  median {m2['median'][0]}, IQR {m2['iqr'][0]}, I-CVI {m2['cvi'][0]}")
print(f"  IQR exceeds 2: {m2['iqr'][0] > 2.0}   <- why the lemma needs kappa > 0.75 strictly")
"""),
        md("""
## Interactive: rediscover which criterion binds

Three sliders: panel size, panel disagreement, and the dispersion limit. The output is the
false-drop rate for an item that genuinely deserves to stay, and the share of drops each
criterion is responsible for.

Push the dispersion limit to 2.0 and watch the false-drop rate collapse and the IQR share
go to zero. Then push the panel size up at a limit of 1.5 and watch the false-drop rate
climb. Those two moves are the paper.
"""),
        code("""
banner()

widget_sim = 3000


def rule_panel(panel=30, sigma=1.2, iqr_limit=1.5):
    rng_w = np.random.default_rng(11)
    good = operating_characteristics(int(panel), 0.90, float(sigma), rng_w, widget_sim,
                                     iqr_threshold=float(iqr_limit))
    marginal = operating_characteristics(int(panel), 0.65, float(sigma), rng_w, widget_sim,
                                         iqr_threshold=float(iqr_limit))

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    names = ["median", "IQR", "I-CVI"]
    shares = [good["median_fails"], good["iqr_fails"], good["cvi_fails"]]
    ax.barh(names, shares, color=[series_color(i) for i in range(3)])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("share of wrongly dropped panels on which this criterion failed")
    ax.set_title(f"panel {int(panel)}, sigma {sigma:.1f}, IQR limit {iqr_limit:.1f}: "
                 f"false drop {good['p_drop']:.3f}", fontsize=10)
    plt.show()

    print(f"false-drop rate for a good item (pi = 0.90)   {good['p_drop']:.4f}")
    print(f"false-keep rate for a weak item (pi = 0.65)   {marginal['p_retain']:.4f}")
    print(f"binding shares: median {good['median_fails']:.3f}, "
          f"IQR {good['iqr_fails']:.3f}, I-CVI {good['cvi_fails']:.3f}")
    if iqr_limit >= 2.0:
        print()
        print("At a limit of 2.0 or more the dispersion criterion cannot bind at all,")
        print("by the lemma above. Its share is zero because it is inert, not because")
        print("it agrees with the others.")


explore(
    rule_panel,
    controls=dict(panel=(6, 60, 2), sigma=(0.4, 2.6, 0.1), iqr_limit=(0.5, 3.0, 0.5)),
    defaults=dict(panel=30, sigma=1.2, iqr_limit=1.5),
)
"""),
        md("""
## The stakeholder split

Panels are mixed. Suppose patients rate an item highly and clinicians do not. As the
clinician endorsement falls, how long does the item survive the pooled rule? And does the
answer depend on how many of each sit on the panel?
"""),
        code("""
banner()

split_sim = 20000 if FULL_RUN else 3000
clinician_pi = np.linspace(0.05, 0.98, 41 if FULL_RUN else 21)
patient_mu = 8.3

start = time.time()
rng = np.random.default_rng(20260815)
curves = {}
for n_patients, n_clinicians in ((6, 24), (12, 18), (15, 15)):
    probs = []
    for pi_clin in clinician_pi:
        mu_clin = mu_for_endorsement(float(pi_clin), SIGMA)
        patients = simulate_ratings(rng, patient_mu, SIGMA, (split_sim, n_patients))
        clinicians = simulate_ratings(rng, mu_clin, SIGMA, (split_sim, n_clinicians))
        masks = retention_masks(np.concatenate([patients, clinicians], axis=1),
                                median_min=7.0, iqr_threshold=1.5, cvi_min=0.78)
        probs.append(float(masks["retain"].mean()))
    curves[f"{n_patients} patients : {n_clinicians} clinicians"] = np.array(probs)
print(f"{time.time() - start:.1f} s")

fig, ax = plt.subplots(figsize=(6.6, 4.0))
for i, (label, probs) in enumerate(curves.items()):
    ax.plot(clinician_pi, probs, lw=1.6, color=series_color(i), label=label)
ax.axhline(0.80, ls="--", lw=0.9, color=INK)
ax.set_xlabel(r"clinician endorsement $\\pi$")
ax.set_ylabel("probability the item is retained")
ax.set_ylim(0, 1)
ax.set_title(r"An item patients rate at $\\mu$ = 8.3, as clinicians cool on it", fontsize=10)
ax.legend(fontsize=8)
plt.show()

for label, probs in curves.items():
    at_half = float(np.interp(0.50, clinician_pi, probs))
    need = float(np.interp(0.80, probs, clinician_pi)) if probs.max() >= 0.80 else float("nan")
    print(f"{label:<32} P(retained | clinician pi = 0.50) = {at_half:.3f}, "
          f"clinician pi needed for 0.80 = {need:.3f}")
"""),
        md("""
**What that shows.** Even a 15:15 split gives a patient-endorsed item roughly a one in
twelve chance of surviving when clinicians are split down the middle, and clinicians have
to reach about 0.97 endorsement before the item is safe. Weighting the panel toward
patients barely helps, because a mixed panel raises dispersion, and dispersion is the
criterion doing the dropping.
"""),
        md(NOT_ESTABLISH_TAIL + """
**These are properties of a decision rule, not of a panel.** No Delphi has been run. No
expert has rated anything. The error rates describe what the rule does to items whose true
endorsement is known by construction, which is exactly the thing you never know in
practice.

**The rating model is an assumption.** Rounded, clipped normal ratings with one common
$\\sigma$. Real panels cluster, argue and drift between rounds. A different rating model
gives different rates, though the mechanism, an order-statistic criterion tightening as
$n$ grows, does not depend on normality.

**The lemma is about a rule and a quantile convention together.** It is true under linear
interpolation and false under Weibull, Hazen and the median-unbiased rules. It is not a
statement about the thresholds alone.

**"Dispersion binds on 99 per cent of drops" is conditional on a drop happening.** It is
not the probability that dispersion is wrong.

**A reduced run does not reproduce the published rates.** The lemma enumeration is exact
either way; only its panel-size ceiling changes.
"""),
    ]
    return write(name, cells)


# ======================================================================= 04
def build_dynamics() -> pathlib.Path:
    name = "04_dynamics"
    cells = head(
        name,
        "Paper 4: verifying the solvers, and finding out what the data can identify",
        """
A continuous-time model of health state is only as trustworthy as the code that solves it.
Four checks, each of which can fail: the order of the deterministic solver, the strong
order of the stochastic one under two noise structures, conservation of probability mass in
the density solver, and whether an observation design can identify the diffusion at all.
""",
        runtime=("**Runtime.** Reduced run: about 50 seconds. Full run: about ninety seconds."),
        full_note="Full run here takes about ninety seconds.",
    )

    cells += [
        md(r"""
## The process

A mean-reverting state with either additive or multiplicative noise,

$$dX_t = (a - bX_t)\,dt + \sigma\,dW_t
\qquad\text{or}\qquad
dX_t = (a - bX_t)\,dt + \sigma X_t\,dW_t ,$$

with $a = 1.2$, $b = 0.6$, $\sigma = 0.45$, $X_0 = 3$ and $T = 6$. The deterministic part
has the closed-form solution

$$x(t) = \frac{a}{b} + \left(x_0 - \frac{a}{b}\right)e^{-bt},$$

so errors can be measured against the truth rather than against a finer run of the same
scheme.
"""),
        code("""
banner()

from hpotential.core.dynamical import (
    FokkerPlanck, LinearSDE, euler_maruyama, profile_sigma, rk4, simulate_state,
    stable_time_step, strong_error_study, weak_error_study,
)
from hpotential.utils.numerical import convergence_order, pairwise_orders

A, B, SIGMA, X0, T = 1.2, 0.6, 0.45, 3.0, 6.0
sde = LinearSDE(A, B, SIGMA)
print(f"mean-reversion level a / b = {sde.mean_reversion_level:.4f}")
print(f"exact x(T)                 = {sde.exact_solution(T, X0):.10f}")
"""),
        md(r"""
## Check 1: RK4 is fourth order

Halve the step and a fourth-order method should cut the error by a factor of 16. The
observed order is the slope of $\log(\text{error})$ against $\log(h)$.
"""),
        code("""
banner()

steps = np.array([0.4, 0.2, 0.1, 0.05, 0.025, 0.0125])
exact = sde.exact_solution(T, X0)
errors = np.array([abs(rk4(lambda t, x: sde.drift(x), X0, T, h) - exact) for h in steps])

print(f"{'h':>8} {'error':>14} {'ratio':>10} {'local order':>12}")
prev = None
for h, e in zip(steps, errors):
    ratio = "" if prev is None else f"{prev / e:10.2f}"
    order = "" if prev is None else f"{np.log2(prev / e):12.3f}"
    print(f"{h:8.4f} {e:14.4e} {ratio:>10} {order:>12}")
    prev = e
print()
print(f"fitted order over the whole sweep: {convergence_order(steps, errors):.3f}")
"""),
        md(r"""
## Check 2: strong order, and why the noise structure decides it

Strong error is $\sqrt{E|X^h_T - X_T|^2}$, pathwise. The reference and every coarse
solution are driven by the same Brownian increments, so what is measured is discretisation
error and not sampling noise.

Euler-Maruyama has strong order 1 when the diffusion does not depend on the state, because
the Milstein correction term vanishes. It drops to order $1/2$ when the diffusion is
$\sigma X_t$. Both cases are run because the difference is the whole point.
"""),
        code("""
banner()

import time

n_paths = 4000 if FULL_RUN else 1000
counts = [75, 150, 300, 600, 1200]

start = time.time()
studies = {}
for index, multiplicative in enumerate((False, True)):
    process = LinearSDE(A, B, SIGMA, multiplicative=multiplicative)
    label = "multiplicative" if multiplicative else "additive"
    strong = strong_error_study(process, X0, T, counts, n_paths, np.random.default_rng([2026, index]), 9600)
    weak = weak_error_study(process, X0, T, counts)
    studies[label] = {**strong, **weak}
    print(f"{label:<15} strong order {strong['strong_order']:.3f}   weak order {weak['weak_order']:.3f}")
print(f"{n_paths} paths per step size, {time.time() - start:.1f} s")
"""),
        code("""
banner()

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

ax = axes[0]
for i, (label, s) in enumerate(studies.items()):
    ax.loglog(s["steps"], s["strong_errors"], "o-", ms=5, lw=1.5, color=series_color(i),
              label=f"{label}, fitted {s['strong_order']:.2f}")
h = np.array(studies["additive"]["steps"])
ax.loglog(h, errors[0] * 0 + h * (studies["additive"]["strong_errors"][0] / h[0]), ls="--",
          lw=0.9, color=INK, label="slope 1")
ax.loglog(h, np.sqrt(h) * (studies["multiplicative"]["strong_errors"][0] / np.sqrt(h[0])), ls=":",
          lw=0.9, color=INK, label="slope 1/2")
ax.set_xlabel("step $h$"); ax.set_ylabel(r"strong error $\\sqrt{E|X^h_T - X_T|^2}$")
ax.set_title("Strong convergence", fontsize=10); ax.legend(fontsize=8)

ax = axes[1]
ax.loglog(steps, errors, "o-", ms=5, lw=1.5, color=series_color(2),
          label=f"RK4, fitted {convergence_order(steps, errors):.2f}")
ax.loglog(steps, errors[0] * (steps / steps[0]) ** 4, ls="--", lw=0.9, color=INK, label="slope 4")
ax.set_xlabel("step $h$"); ax.set_ylabel("error at $T$")
ax.set_title("RK4 on the deterministic part", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

print("Monte Carlo standard errors on the terminal mean, additive case:")
print("  " + "  ".join(f"{v:.2e}" for v in studies["additive"]["mc_se"]))
print("These are larger than the weak errors, which is why the weak order is computed")
print("in closed form from the deterministic recursion rather than estimated from paths.")
"""),
        md(r"""
## Check 3: mass conservation without renormalisation

The density $p(x,t)$ obeys the Fokker-Planck equation

$$\partial_t p = -\partial_x\!\left[(a - bx)p\right]
 + \tfrac{1}{2}\,\partial_x^2\!\left[\sigma^2 p\right],$$

which is a conservation law: written in flux form $\partial_t p = -\partial_x J$, with no
flux through the boundaries, total mass is constant for all time.

A finite-volume scheme that updates cells by the difference of face fluxes conserves mass
to machine precision by construction, because each interior flux is added to one cell and
subtracted from its neighbour. That is the check below. It matters that no renormalisation
is applied: dividing by the total at each step would produce a flat mass history from a
scheme that was losing mass, and hide the bug it was meant to reveal.
"""),
        code("""
banner()

grids = (100, 200, 400, 800) if FULL_RUN else (100, 200, 400)
t_end = 25.0

start = time.time()
fp = FokkerPlanck(sde, -2.0, 8.0)
rows, widths, l1 = [], [], []
for nx in grids:
    _, _, dx = fp.grid(nx)
    dt_cap = stable_time_step(sde, dx, -2.0, 8.0, safety=0.4)
    n_steps = int(np.ceil(t_end / dt_cap))
    sol = fp.solve(nx, n_steps, t_end, x0=X0)
    err = sol.l1_error(sde.stationary_pdf(sol.centres))
    widths.append(dx); l1.append(err)
    rows.append({"cells": nx, "dx": sol.dx, "dt": sol.dt, "steps": n_steps,
                 "mass drift": sol.mass_drift, "min density": sol.min_density,
                 "L1 vs stationary": err})
fp_table = pd.DataFrame(rows)
print(f"{time.time() - start:.1f} s")
print(f"L1 grid convergence order: {convergence_order(np.array(widths), np.array(l1)):.4f}")
fp_table
"""),
        code("""
banner()

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

ax = axes[0]
sol = fp.solve(grids[-1], int(np.ceil(t_end / stable_time_step(sde, fp.grid(grids[-1])[2],
                                                               -2.0, 8.0, safety=0.4))),
               t_end, x0=X0)
ax.plot(sol.mass - 1.0, lw=1.2, color=series_color(0))
ax.set_xlabel("time step"); ax.set_ylabel("total mass minus one")
ax.set_title(f"Mass drift over {len(sol.mass)} steps, no renormalisation", fontsize=10)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

ax = axes[1]
ax.loglog(widths, l1, "o-", ms=5, lw=1.5, color=series_color(1),
          label=f"fitted order {convergence_order(np.array(widths), np.array(l1)):.2f}")
ax.loglog(widths, np.array(l1)[0] * (np.array(widths) / widths[0]), ls="--", lw=0.9, color=INK,
          label="slope 1")
ax.set_xlabel(r"cell width $\\Delta x$"); ax.set_ylabel("$L_1$ error against the stationary density")
ax.set_title("Grid convergence", fontsize=10); ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

print(f"largest mass drift over every grid: {fp_table['mass drift'].abs().max():.3e}")
print(f"smallest density anywhere:          {fp_table['min density'].min():.3e}   (never negative)")
print()
print("Machine precision on mass, first order in space on accuracy. The two are")
print("separate claims: a scheme can conserve mass exactly and still be inaccurate.")
"""),
        md(r"""
## Check 4: what an observation design can identify

The last check is not about code. Suppose the state is observed with error at a handful of
visits. Can the diffusion $\sigma$ be recovered? Profile the exact Kalman likelihood over a
grid of $\sigma$, summing over subjects, and read the interval where the profile is within
1.92 log-likelihood units of its maximum.

Three designs: five visits with one indicator, five visits with three, and twenty-five
visits with three.
"""),
        code("""
banner()

n_subjects = 60 if FULL_RUN else 20
grid_n = 40 if FULL_RUN else 24
obs_sd = 0.35

start = time.time()
grid = np.linspace(0.12, 1.2, grid_n)
designs = (("sparse, 1 indicator", 5, 1), ("sparse, 3 indicators", 5, 3),
           ("dense, 3 indicators", 25, 3))
profiles, ident_rows = {}, []
for design_index, (label, n_visits, n_indicators) in enumerate(designs):
    times = np.linspace(0, T, n_visits)
    rng = np.random.default_rng([2026, 5, design_index])
    total = np.zeros_like(grid)
    for _ in range(n_subjects):
        y = simulate_state(sde, times, rng, obs_sd, n_indicators)
        total += profile_sigma(y, times, sde, grid)
    total -= total.max()
    profiles[label] = total
    inside = grid[total >= -1.92]
    ident_rows.append({"design": label, "visits": n_visits, "indicators": n_indicators,
                       "sigma hat": grid[int(np.argmax(total))],
                       "interval low": inside.min(), "interval high": inside.max(),
                       "width": inside.max() - inside.min()})
print(f"{n_subjects} subjects per design, {time.time() - start:.1f} s")
pd.DataFrame(ident_rows).round(4)
"""),
        code("""
banner()

fig, ax = plt.subplots(figsize=(6.6, 4.0))
for i, (label, profile) in enumerate(profiles.items()):
    ax.plot(grid, profile, lw=1.6, color=series_color(i), label=label)
ax.axhline(-1.92, ls="--", lw=0.9, color=INK)
ax.axvline(SIGMA, ls=":", lw=1.2, color=series_color(5))
ax.text(SIGMA + 0.01, ax.get_ylim()[0] * 0.55, rf"true $\\sigma$ = {SIGMA}", fontsize=8,
        color=series_color(5))
ax.set_ylim(-30, 2)
ax.set_xlabel(r"$\\sigma$"); ax.set_ylabel("profile log-likelihood, relative to its maximum")
ax.set_title("A flat profile is a design problem, not a solver problem", fontsize=10)
ax.legend(fontsize=8)
plt.show()

print("One indicator at five visits leaves a wide, flat interval: the data cannot")
print("separate process noise from measurement noise. Adding indicators at the same")
print("visits helps more than adding visits, because it pins the observation variance.")
"""),
        md(NOT_ESTABLISH_TAIL + """
**Verification is not validation.** Every check here says the code solves the equations it
claims to solve. None of them says the equations describe health.

**The observed orders are properties of this problem.** A different drift, a stiffer
regime, or a coarser grid outside the asymptotic range gives different numbers. Under a
reduced run the grids are deliberately coarser, so the fitted orders drift.

**Mass conservation is a statement about the scheme, not about accuracy.** The solver holds
mass to 1e-16 and still carries an $L_1$ error of a few per cent against the stationary
density at coarse grids. Those are two different claims and the notebook reports both.

**The identifiability result is about a design, not about a study.** Sixty synthetic
subjects with a known truth. It shows which designs could recover $\\sigma$ if the model
were right, which is the necessary condition, not the sufficient one.

**A reduced run does not reproduce the published numbers.** In particular the fitted orders
become unreliable, because the coarse sweeps are not in the asymptotic regime.
"""),
    ]
    return write(name, cells)


# ======================================================================= 05
def build_access() -> pathlib.Path:
    name = "05_access"
    cells = head(
        name,
        "Paper 5: costing the choice between access and efficacy",
        """
Given a fixed budget, do you spend it widening access to an existing treatment or
improving how well the treatment works? A tidy uncosted answer says compare the current
levels. Put price and feasibility back in and the answer changes on about half the draws.
""",
        runtime=(
            "**Runtime.** Reduced run: about 20 seconds. Full run: about 25 seconds. "
            "This one is cheap either way."
        ),
        full_note="Full run here adds only a few seconds.",
    )

    cells += [
        md(r"""
## Deriving $V$

Attainable capability is

$$H = \frac{(LQ)(STA)}{D\,(1 + C + \psi)} - \text{burden}.$$

Both $A$ and $T$ enter through the same product $STA$, so

$$\frac{\partial H}{\partial A} = \frac{(LQ)S\,T}{D(1+C+\psi)},
\qquad
\frac{\partial H}{\partial T} = \frac{(LQ)S\,A}{D(1+C+\psi)},
\qquad
\frac{\partial H/\partial A}{\partial H/\partial T} = \frac{T}{A}.$$

That ratio is a fact about the function, not an allocation rule. A decision maker does not
compare marginal effects; they compare gain per unit spend. An access programme buys
$\Delta_A$ at cost $c_A$, an efficacy programme buys $\Delta_T$ at cost $c_T$, so access is
the better buy exactly when

$$V = \underbrace{\frac{T}{A}}_{\text{structural}}
\cdot \underbrace{\frac{\Delta_A}{\Delta_T}}_{\text{feasibility}}
\cdot \underbrace{\frac{c_T}{c_A}}_{\text{price}} > 1 .$$

The common factor $(LQ)S/[D(1+C+\psi)]$ cancels, which is why $V$ needs none of $L$, $Q$,
$S$, $D$, $C$ or $\psi$. The uncosted rule $T > A$ is the special case where the last two
factors are both one, and there is no reason to expect that.
"""),
        code("""
banner()

from hpotential.core.access import AccessModel, attainable_capability, costed_statistic

high = AccessModel.high_resource()
low = AccessModel.low_resource()
pd.DataFrame({"high-resource": {k: f"{v[0]} to {v[1]}" for k, v in high.ranges.items()},
              "low-resource": {k: f"{v[0]} to {v[1]}" for k, v in low.ranges.items()}})
"""),
        md("""
### Check the cancellation

The closed form drops six inputs. If that cancellation is wrong, the whole rule is wrong.
The check below builds $\\partial H/\\partial A$ and $\\partial H/\\partial T$ from the
capability expression, forms the ratio of gain per unit spend directly, and compares. The
two sides share no code, so agreement is evidence about the algebra rather than about one
implementation of it.
"""),
        code("""
banner()

n_check = 20_000 if FULL_RUN else 5_000
rng = np.random.default_rng(20260815)

max_rel = low.closed_form_check(rng, n_check)
print(f"maximum relative discrepancy over {n_check} draws: {max_rel:.3e}")
print(f"machine epsilon for float64:                       {np.finfo(float).eps:.3e}")
print()
print("A few multiples of machine epsilon is what an exact identity looks like in")
print("floating point. Anything at 1e-6 would mean the cancellation is approximate.")
"""),
        code("""
banner()

# Do the comparison by hand once, so the identity is visible rather than asserted.
p = low.draw(np.random.default_rng(1), 5)
common = (p["L"] * p["Q"]) / (p["D"] * (1.0 + p["C"] + p["psi"]))
dh_da = common * p["S"] * p["T"]
dh_dt = common * p["S"] * p["A"]
direct = (dh_da * p["dA"] / p["cA"]) / (dh_dt * p["dT"] / p["cT"])
closed = costed_statistic(p["T"], p["A"], p["dA"], p["dT"], p["cA"], p["cT"])
pd.DataFrame({"direct from dH/dA and dH/dT": direct, "closed form V": closed,
              "relative gap": np.abs(closed - direct) / direct})
"""),
        md("""
## Probabilistic sensitivity analysis

Draw every input independently and uniformly over its range and ask two questions of each
draw: does the uncosted rule favour access, and does the costed rule? The interesting
number is how often they point different ways.
"""),
        code("""
banner()

import time

n_psa = 200_000 if FULL_RUN else 40_000
start = time.time()
rng = np.random.default_rng(20260815)
psa = {m.name: m.psa(rng, n_psa) for m in (high, low)}
print(f"{n_psa} draws per setting, {time.time() - start:.1f} s")
pd.DataFrame(psa).T[[
    "median_structural_term", "median_feasibility_term", "median_price_term", "median_V",
    "V_2_5", "V_97_5", "P_uncosted_rule_favours_access", "P_costed_rule_favours_access",
    "P_disagreement",
]].round(4)
"""),
        code("""
banner()

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
for i, model in enumerate((high, low)):
    q = model.draw(np.random.default_rng(100 + i), min(n_psa, 60_000))
    structural = q["T"] / q["A"]
    v = structural * (q["dA"] / q["dT"]) * (q["cT"] / q["cA"])
    ax = axes[i]
    ax.hist(np.log10(v), bins=70, color=series_color(i), edgecolor="none", label="costed $V$")
    ax.hist(np.log10(structural), bins=70, color=INK, histtype="step", lw=1.0,
            label="uncosted $T/A$")
    ax.axvline(0.0, ls="--", lw=1.1, color=INK)
    ax.set_xlabel(r"$\\log_{10}$ of the statistic"); ax.set_ylabel("draws")
    ax.set_title(f"{model.name}: rules disagree on "
                 f"{psa[model.name]['P_disagreement']:.1%} of draws", fontsize=10)
    ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

print("High-resource: the uncosted statistic sits mostly below one and the costed one")
print("straddles it, so the two rules split. Low-resource: both are far above one on")
print("every draw, so the shortcut costs nothing there. The disagreement is exactly")
print("zero, not approximately zero, because T/A exceeds one in every draw.")
"""),
        md("""
## Interactive: three factors, one decision

The three sliders are the three factors of $V$. The decision flips live. Start at the
uncosted position, feasibility and price both at 1.0, and see what the structural term
alone says. Then move the other two and watch how little the structural term settles.
"""),
        code("""
banner()


def v_panel(structural=0.91, feasibility=0.67, price=1.73):
    v = structural * feasibility * price
    uncosted = structural > 1.0
    costed = v > 1.0

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    names = ["structural $T/A$", r"feasibility $\\Delta_A/\\Delta_T$", "price $c_T/c_A$", "$V$"]
    values = [structural, feasibility, price, v]
    colours = [series_color(0), series_color(1), series_color(2),
               series_color(2) if costed else series_color(5)]
    ax.barh(names, values, color=colours)
    ax.axvline(1.0, ls="--", lw=1.2, color=INK)
    ax.set_xlim(0, max(3.5, v * 1.15))
    ax.set_xlabel("value (the dashed line is the decision boundary at one)")
    ax.set_title(f"V = {v:.3f}: spend on {'access' if costed else 'efficacy'}", fontsize=11)
    plt.show()

    print(f"V = {structural:.3f} x {feasibility:.3f} x {price:.3f} = {v:.4f}")
    print(f"uncosted rule (T > A):   spend on {'access' if uncosted else 'efficacy'}")
    print(f"costed rule   (V > 1):   spend on {'access' if costed else 'efficacy'}")
    if uncosted != costed:
        print()
        print("The two rules disagree here. The uncosted rule is throwing away the")
        print("factors that decided it.")


explore(
    v_panel,
    controls=dict(structural=(0.2, 3.0, 0.01), feasibility=(0.1, 3.0, 0.01), price=(0.2, 4.0, 0.01)),
    defaults=dict(structural=0.91, feasibility=0.67, price=1.73),
)
"""),
        md(NOT_ESTABLISH_TAIL + """
**Every input range is illustrative.** Nothing is estimated from a health system. The
48.6 per cent disagreement rate is a property of these ranges and of the choice to draw
inputs independently. Real inputs are correlated, and a correlated prior moves the number.

**$V > 1$ ranks two programmes.** It does not say either is worth funding, and it carries no
distributional weight: a programme that helps a few people a lot and one that helps many a
little are compared only through the aggregate.

**The rule is first order.** It compares marginal gains at the current $A$ and $T$. A large
$\\Delta$ takes you away from the point where the derivative was taken.

**The capability expression is a modelling choice.** The cancellation that makes $V$ so
simple depends on $A$ and $T$ entering as a product. Under a different functional form the
common factor does not cancel and $V$ is not this.

**A reduced run does not reproduce the published disagreement rate.** The closed-form check
is unaffected in substance, since it is a statement about floating-point agreement rather
than about a sample size.
"""),
    ]
    return write(name, cells)


# ======================================================================= 06
def build_hta() -> pathlib.Path:
    name = "06_hta"
    cells = head(
        name,
        "Paper 6: early assessment, value of information, and two outcome measures",
        """
A durable therapy against established prophylaxis, judged before the evidence exists. What
is the chance it is cost-effective, what would perfect information be worth, which single
input is worth learning first, and what happens to the decision if you swap the outcome
measure from QALYs to biomedical potential.
""",
        runtime=("**Runtime.** Reduced run: about 25 seconds. Full run: about 40 seconds."),
        full_note="Full run here adds about fifteen seconds.",
    )

    cells += [
        md(r"""
## The model in one paragraph

Six uncertain inputs: remaining years $L_0$, utility gain $q$, waning rate $w$,
implementation access $\alpha$, one-off price, and the annual cost of prophylaxis the
therapy displaces. Per treated person,

$$\Delta\text{QALY} = \alpha \sum_{t} q\,e^{-wt}\,\mathbb{1}\{t < L_0\}\,(1+r)^{-t},
\qquad
\Delta\text{cost} = \text{price} - \text{proph}\sum_{t} e^{-wt}\,\mathbb{1}\{t<L_0\}(1+r)^{-t},$$

and incremental net monetary benefit at threshold $\lambda$ is
$\text{INB} = \lambda\,\Delta\text{QALY} - \Delta\text{cost}$.

Durability drives both sides: the therapy displaces prophylaxis only while its effect
lasts. The utility gain and the durability share a latent quality factor, so a therapy that
helps more also tends to last longer. Sampling them independently would understate the
spread of net benefit and overstate the value of learning either one alone.
"""),
        code("""
banner()

from hpotential.core.hta import HTAModel, evpi, evppi

model = HTAModel()
print(f"threshold lambda      {model.threshold:,.0f} per QALY")
print(f"eligible population   {model.population} over {model.horizon_years} years")
print(f"discount rate         {model.discount}")
print(f"discounted population multiplier {model.population_multiplier():,.1f}")
print()
for name, spec in model.scenarios.items():
    print(f"{name:<12} waning {spec['waning']}   access {spec['access']}")
"""),
        code("""
banner()

import time

draws = 100_000 if FULL_RUN else 20_000
start = time.time()
rng = np.random.default_rng(20260815)
results, summaries = {}, {}
for scenario in ("optimistic", "central", "limited"):
    results[scenario] = model.run_scenario(rng, scenario, draws)
    summaries[scenario] = model.summarise(results[scenario], 120)
print(f"{draws} draws per scenario, {time.time() - start:.1f} s")

table = pd.DataFrame({
    s: {"mean incremental QALY": v["mean_incremental_qaly"],
        "mean incremental BMP": v["mean_incremental_bmp"],
        "mean incremental cost": v["mean_incremental_cost"],
        "mean INB": v["mean_INB"],
        "P(cost-effective, QALY)": v["P_cost_effective_QALY"],
        "P(cost-effective, BMP)": v["P_cost_effective_BMP"],
        "P(decisions disagree)": v["P_decision_disagreement"],
        "per-person EVPI": v["per_person_EVPI"],
        "population EVPI": v["population_EVPI"]}
    for s, v in summaries.items()})
table.round(3)
"""),
        code("""
banner()

fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharey=True)
for i, scenario in enumerate(("optimistic", "central", "limited")):
    res = results[scenario]
    ax = axes[i]
    take = slice(0, min(6000, draws))
    ax.scatter(res.incremental_qaly[take], res.incremental_cost[take] / 1000, s=3, alpha=0.2,
               color=series_color(i))
    q = np.linspace(0, max(res.incremental_qaly.max(), 0.1), 50)
    ax.plot(q, model.threshold * q / 1000, ls="--", lw=1.1, color=INK)
    ax.axhline(0, lw=0.7, color=GRID)
    ax.set_xlabel("incremental QALYs")
    ax.set_title(f"{scenario}: P(cost-effective) = "
                 f"{summaries[scenario]['P_cost_effective_QALY']:.3f}", fontsize=10)
axes[0].set_ylabel("incremental cost (thousands)")
fig.tight_layout()
plt.show()

print("The dashed line is the threshold. Points below it are cost-effective draws.")
print("Optimistic scenario draws sit low because a durable effect displaces years of")
print("prophylaxis, which is a cost saving, so incremental cost is often negative.")
"""),
        md(r"""
## What is perfect information worth?

Under uncertainty you take the option with the better expected net benefit and are
sometimes wrong. With perfect information you would be right every time. The gap is

$$\text{EVPI} = E\!\left[\max(\text{INB},0)\right] - \max\!\left(E[\text{INB}],0\right),$$

per person, and it is zero only when every draw has the same sign. Multiply by the
discounted population the decision covers to get what the whole decision is worth.

For one input at a time, expected value of partial perfect information uses the
single-parameter regression method of Strong, Oakley and Brennan: estimate
$g_i(\theta_i) = E[\text{INB} \mid \theta_i]$ by a smoothed conditional mean over
equal-count bins, then $\text{EVPPI}_i = E[\max(g_i,0)] - \max(E[\text{INB}],0)$.
"""),
        code("""
banner()

evppi_rows = {}
for scenario, summary in summaries.items():
    row = dict(summary["EVPPI"])
    row["(EVPI)"] = summary["per_person_EVPI"]
    evppi_rows[scenario] = row
evppi_table = pd.DataFrame(evppi_rows)
evppi_table.round(1)
"""),
        code("""
banner()

fig, ax = plt.subplots(figsize=(7.2, 4.0))
inputs = ["L0", "q_gain", "waning", "access", "price", "proph"]
width = 0.26
pos = np.arange(len(inputs))
for i, scenario in enumerate(("optimistic", "central", "limited")):
    shares = [max(summaries[scenario]["EVPPI"][k], 0.0) / summaries[scenario]["per_person_EVPI"]
              for k in inputs]
    ax.bar(pos + (i - 1) * width, shares, width, color=series_color(i), label=scenario)
ax.set_xticks(pos)
ax.set_xticklabels(["remaining years", "utility gain", "waning", "access", "price",
                    "prophylaxis cost"], fontsize=8)
ax.set_ylabel("EVPPI as a share of EVPI")
ax.set_title("Which single input is worth learning first", fontsize=10)
ax.legend(fontsize=8)
plt.show()

central = summaries["central"]
ranked = sorted(central["EVPPI"].items(), key=lambda kv: -kv[1])
print("central scenario, EVPPI per person, largest first:")
for key, value in ranked:
    share = max(value, 0.0) / central["per_person_EVPI"]
    print(f"  {key:<10} {value:10.1f}   {share:6.1%} of EVPI")
print()
print("Shares do not sum to one. Learning two inputs is worth more than the sum of")
print("learning each alone whenever they interact, and a small negative EVPPI is")
print("estimation error in the conditional mean, reported rather than clipped away.")
"""),
        md("""
## Two outcome measures are not one outcome measure

Biomedical potential is $L \\times Q$: undiscounted, on a bespoke quality scale, with no
waning term. Three differences from the QALY calculation, and all three push the same way.
"""),
        code("""
banner()

rows = []
for scenario, summary in summaries.items():
    ratio = summary["mean_incremental_bmp"] / summary["mean_incremental_qaly"]
    rows.append({"scenario": scenario,
                 "incremental QALY": summary["mean_incremental_qaly"],
                 "incremental BMP": summary["mean_incremental_bmp"],
                 "BMP / QALY": ratio,
                 "BMP is larger by": f"{100 * (ratio - 1):.0f}%",
                 "P(cost-effective, QALY)": summary["P_cost_effective_QALY"],
                 "P(cost-effective, BMP)": summary["P_cost_effective_BMP"],
                 "decisions disagree": summary["P_decision_disagreement"]})
pd.DataFrame(rows).round(3)
"""),
        md("""
The central scenario is the one to look at. The incremental benefit measured in biomedical
potential is about 1.88 times the incremental QALY gain, roughly 88 per cent larger, and
the probability of being judged cost-effective moves from 0.410 to 0.469. The two measures
disagree outright on about 6 per cent of draws. A single deterministic score would hide all
of this, which is why the paper does not use one.
"""),
        md(r"""
## Interactive: move the willingness to pay

The threshold $\lambda$ is a policy choice, not an estimate. Move it and both the
probability of being cost-effective and the value of perfect information move with it. The
EVPI curve peaks where the decision is closest to a coin toss, which is exactly where more
evidence is worth most.

No resampling happens here: the incremental QALYs and costs are fixed, and only
$\text{INB} = \lambda\,\Delta\text{QALY} - \Delta\text{cost}$ is recomputed.
"""),
        code("""
banner()

lam_grid = np.linspace(5_000, 120_000, 60)
curves = {s: {"p": [], "evpi": []} for s in results}
for scenario, res in results.items():
    for lam in lam_grid:
        inb = lam * res.incremental_qaly - res.incremental_cost
        curves[scenario]["p"].append(float(np.mean(inb > 0)))
        curves[scenario]["evpi"].append(evpi(inb))


def wtp_panel(wtp=30_000):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for i, scenario in enumerate(("optimistic", "central", "limited")):
        axes[0].plot(lam_grid / 1000, curves[scenario]["p"], lw=1.5, color=series_color(i),
                     label=scenario)
        axes[1].plot(lam_grid / 1000, np.array(curves[scenario]["evpi"]) / 1000, lw=1.5,
                     color=series_color(i), label=scenario)
    for ax in axes:
        ax.axvline(wtp / 1000, ls="-", lw=1.0, color=series_color(5))
    axes[0].set_ylabel("P(cost-effective)"); axes[0].set_ylim(0, 1)
    axes[1].set_ylabel("per-person EVPI (thousands)")
    for ax in axes:
        ax.set_xlabel("willingness to pay per QALY (thousands)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    plt.show()

    print(f"willingness to pay {wtp:,.0f} per QALY")
    for scenario, res in results.items():
        inb = wtp * res.incremental_qaly - res.incremental_cost
        print(f"  {scenario:<12} P(cost-effective) {np.mean(inb > 0):.3f}   "
              f"per-person EVPI {evpi(inb):9,.0f}   "
              f"population EVPI {evpi(inb) * model.population_multiplier():13,.0f}")


explore(wtp_panel, controls=dict(wtp=(5_000, 120_000, 2_500)), defaults=dict(wtp=30_000))
"""),
        md(NOT_ESTABLISH_TAIL + """
**Every input is synthetic.** There is no therapy, no trial, no price negotiation and no
patient population behind these numbers. The case is a haemophilia-shaped container for a
workflow.

**EVPI is an upper bound on the value of any study.** No real study delivers perfect
information, so a trial is worth less than the EVPI reported here, sometimes far less.

**EVPPI by single input understates joint value.** The shares do not add up, and learning
two correlated inputs together is worth more than learning each alone. A small negative
value is estimation error in the conditional mean and marks an input carrying essentially
no information value.

**The biomedical-potential comparison is a caution, not a proposal.** The point is that two
measures which both rise with health can order the same decisions differently. It is not a
recommendation to use either one.

**The scenarios are structural assumptions, not probabilities.** Nothing here says the
central scenario is more likely than the limited one.

**A reduced run does not reproduce the published values.** EVPPI is the worst affected,
since each conditional mean is then built from far fewer draws per bin.
"""),
    ]
    return write(name, cells)


# ======================================================================= 99
def build_reproduce() -> pathlib.Path:
    name = "99_reproduce_all"
    cells = head(
        name,
        "Reproduce every published number",
        """
One table. For each published value: what the papers say, what this package computes now,
the absolute difference, the tolerance, and PASS or FAIL. Closed-form results are held to
the digit. Simulation results are held to an interval wide enough for Monte Carlo error at
the published replicate counts and narrow enough to fail if a method changes.
""",
        runtime=(
            "**Runtime.** This notebook always runs at full size. Expect about five minutes "
            "here and up to ten on a free Colab CPU. Papers 1 and 2 are almost all of it."
        ),
        full_note="This notebook has no reduced mode.",
        full_run=True,
    )

    cells += [
        md("""
## Where the expected values come from

The expected values live in `tests/fixtures/*.json`, the files the published analysis
scripts wrote. Those files ship with the repository but not inside the installed package,
so this notebook looks for them in three places, in order:

1. a local checkout, when the notebook is run from a clone;
2. a shallow clone of the repository, which is what happens in Colab;
3. a small set of values embedded in this notebook, as a last resort.

The third path covers the case where the repository is private or the org name has not been
filled in yet. It is clearly flagged when it fires, because a fixture you carry with you is
weaker evidence than a fixture you fetched.
"""),
        code('''
import json
import pathlib
import subprocess

FIXTURE_FALLBACK = {
    "p1_determinacy": {"ceiling_rho_D": 0.8778, "c_gamma_Psi": 3.3586,
                       "c_required_for_target": 4.2632, "target": 0.9},
    "p2_pipeline": {"rho_D_population": 0.8361, "beta_true": -0.12,
                    "predicted_attenuated_beta": -0.1003},
    "p3_delphi_oc": {"operating_characteristics": {
        "12": {"false_drop_at_pi_0_90": 0.38075}, "20": {"false_drop_at_pi_0_90": 0.48805},
        "30": {"false_drop_at_pi_0_90": 0.6921}, "45": {"false_drop_at_pi_0_90": 0.76185},
        "60": {"false_drop_at_pi_0_90": 0.80965}}},
    "p4_numerics": {"rk4": {"observed_order": 4.052},
                    "euler_maruyama": {"additive_noise": {"strong_order": 1.03},
                                       "multiplicative_noise": {"strong_order": 0.53}},
                    "fokker_planck": {"grid_convergence_order_L1": 0.9668}},
    "p5_costed": {"psa": {"high-resource": {"P_disagreement": 0.486275}}},
    "p6_voi": {"scenarios": {"central": {"P_cost_effective_QALY": 0.40967,
                                         "per_person_EVPI": 34873.64}}},
}


def find_fixtures():
    """Return (directory, how it was found). Directory is None when embedded values are used."""
    for parent in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:
        candidate = parent / "tests" / "fixtures"
        if (candidate / "p1_determinacy.json").exists():
            return candidate, "local checkout"
    target = pathlib.Path("/content") / REPO if IN_COLAB else pathlib.Path.cwd() / f"_{REPO}"
    package = target / SUBDIR
    if not (package / "tests" / "fixtures" / "p1_determinacy.json").exists():
        url = f"https://github.com/{GITHUB_OWNER}/{REPO}.git"
        try:
            subprocess.run(["git", "clone", "--depth", "1", url, str(target)],
                           check=True, capture_output=True, timeout=180)
        except Exception as exc:
            print(f"could not clone {url}: {type(exc).__name__}")
    if (package / "tests" / "fixtures" / "p1_determinacy.json").exists():
        return package / "tests" / "fixtures", "shallow clone"
    return None, "embedded fallback"


FIXTURE_DIR, FIXTURE_SOURCE = find_fixtures()


def published(stem):
    if FIXTURE_DIR is None:
        return FIXTURE_FALLBACK[stem]
    return json.loads((FIXTURE_DIR / f"{stem}.json").read_text())


print(f"fixtures from: {FIXTURE_SOURCE}")
if FIXTURE_DIR is None:
    print("!" * 74)
    print("Using values embedded in this notebook, not the published JSON files.")
    print("Clone the repository, or run from a clone, for the real thing.")
    print("!" * 74)
else:
    print(f"directory:     {FIXTURE_DIR}")
'''),
        md("""
## The checking harness

One row per value. `tol` is absolute. Closed-form rows use 1e-6 or tighter; simulation rows
carry a tolerance chosen for the published replicate count and are marked so you can tell
them apart.
"""),
        code("""
CHECKS = []


def check(paper, what, expected, actual, tol, kind="closed form"):
    diff = abs(float(expected) - float(actual))
    CHECKS.append({"paper": paper, "quantity": what, "kind": kind,
                   "expected": float(expected), "actual": float(actual),
                   "abs diff": diff, "tol": float(tol),
                   "result": "PASS" if diff <= tol else "FAIL"})


def report():
    frame = pd.DataFrame(CHECKS)
    failures = int((frame["result"] == "FAIL").sum())
    print("=" * 96)
    print(f"{len(frame)} checks, {len(frame) - failures} PASS, {failures} FAIL")
    print("=" * 96)
    with pd.option_context("display.width", 200, "display.max_colwidth", 46,
                           "display.float_format", lambda v: f"{v:,.6g}"):
        print(frame.to_string(index=False))
    return frame


print("harness ready")
"""),
        md("## Paper 1: determinacy, identification, sample size"),
        code("""
import time

from hpotential.analyses import paper1_validation
from hpotential.core.measurement import (
    determinacy_ceiling, identification_report, required_second_order_signal,
    second_order_determinacy,
)

p1 = published("p1_determinacy")
rho_inf, c = determinacy_ceiling()
check(1, "ceiling rho_D_inf", p1["ceiling_rho_D"], round(rho_inf, 4), 1e-9)
check(1, "second-order signal c", p1["c_gamma_Psi"], round(c, 4), 1e-9)
check(1, "c required for rho_D = 0.90", p1["c_required_for_target"],
      round(required_second_order_signal(0.90), 4), 1e-9)
check(1, "rho_D at 400 indicators, lambda 0.85", 0.877594, second_order_determinacy(400, 0.85), 1e-5)

for lam in (0.55, 0.65, 0.75, 0.85):
    row = p1.get(f"lambda={lam}") if FIXTURE_DIR is not None else None
    if row is None:
        continue
    for m in (4, 8, 40):
        check(1, f"rho_D at m={m}, lambda={lam}", row[f"rho_D_at_m{m}"],
              round(second_order_determinacy(m, lam), 4), 1e-9)

print(f"{len(CHECKS)} checks so far")
"""),
        code("""
if FIXTURE_DIR is not None:
    ident = published("p1_identification")
    for record in ident["planned_designs"]:
        rep = identification_report(record["loading"], record["label"])
        check(1, f"Jacobian rank, loading {record['loading']}", record["rank"], rep["rank"], 0)
        check(1, f"smallest singular value, loading {record['loading']}",
              record["smallest_singular_value"], rep["smallest_singular_value"], 1e-8)
    control_b = ident["controls"][1]
    rep = identification_report(0.75, control_b["label"], np.array([0.80, 0.70]),
                                np.array([0.36, 0.51]), per_domain=4)
    check(1, "negative control rank (must be short)", control_b["rank"], rep["rank"], 0)
    print("identification checks added")
else:
    print("identification fixtures unavailable under the embedded fallback; skipped")
"""),
        code("""
start = time.time()
res1 = paper1_validation.run()          # published configuration, 150 replicates
print(f"paper 1 full run: {time.time() - start:.1f} s")

if FIXTURE_DIR is not None:
    samples = published("p1_samplesize")["summary"]
    for key, expected in samples.items():
        actual = res1["sample_size"]["summary"].get(key)
        if actual is None:
            continue
        check(1, f"determinacy, {key}", expected["determinacy"], actual["determinacy"], 0.01,
              "simulation")
        check(1, f"worst coverage, {key}", expected["min_coverage"], actual["min_coverage"], 0.06,
              "simulation")
print(f"{len(CHECKS)} checks so far")
"""),
        md("## Paper 2: attenuation and events"),
        code("""
from hpotential.analyses import paper2_prognostic
from hpotential.core.prognostic import events_inflation

p2 = published("p2_pipeline")
start = time.time()
res2 = paper2_prognostic.run()
print(f"paper 2 full run: {time.time() - start:.1f} s")

check(2, "rho_D of the planned design", p2["rho_D_population"],
      round(res2["rho_D_population"], 4), 1e-9)
check(2, "predicted attenuated beta", p2["predicted_attenuated_beta"],
      round(res2["predicted_attenuated_beta"], 4), 1e-9)
check(2, "events inflation at the ceiling", 1.2977, events_inflation(0.877819), 1e-4)
check(2, "events inflation at the planned design", 1.4305, events_inflation(0.8361), 1e-4)
check(2, "pooled attenuation ratio", 0.843, res2["attenuation_ratio_pooled"], 0.02, "simulation")

if FIXTURE_DIR is not None:
    for key, expected in p2.items():
        if not isinstance(expected, dict) or "attenuation_ratio" not in expected:
            continue
        actual = res2.get(key)
        if actual is None:
            continue
        check(2, f"attenuation ratio, {key}", expected["attenuation_ratio"],
              actual["attenuation_ratio"], 0.06, "simulation")
print(f"{len(CHECKS)} checks so far")
"""),
        md("## Paper 3: Delphi operating characteristics"),
        code("""
from hpotential.analyses import paper3_consensus
from hpotential.core.consensus import iqr_lemma_violations

p3 = published("p3_delphi_oc")
start = time.time()
res3 = paper3_consensus.run()
print(f"paper 3 full run: {time.time() - start:.1f} s")

for panel, expected in p3["operating_characteristics"].items():
    actual = res3["operating_characteristics"][panel]
    check(3, f"false-drop rate, panel {panel}", expected["false_drop_at_pi_0_90"],
          actual["false_drop_at_pi_0_90"], 0.02, "simulation")
    if "false_keep_at_pi_0_65" in expected:
        check(3, f"false-keep rate, panel {panel}", expected["false_keep_at_pi_0_65"],
              actual["false_keep_at_pi_0_65"], 0.02, "simulation")

lemma = iqr_lemma_violations(max_panel=10, cvi_min=0.78, iqr_bound=2.0)
check(3, "dispersion lemma violations (linear)", 0, lemma["violations"], 0)
check(3, "largest IQR among passing panels", 2.0, lemma["max_iqr_among_passing"], 1e-12)
print(f"{len(CHECKS)} checks so far")
"""),
        md("## Paper 4: solver verification"),
        code("""
from hpotential.analyses import paper4_dynamics

p4 = published("p4_numerics")
start = time.time()
res4 = paper4_dynamics.run()
print(f"paper 4 full run: {time.time() - start:.1f} s")

check(4, "RK4 observed order", p4["rk4"]["observed_order"], res4["rk4"]["observed_order"], 0.02)
check(4, "Euler-Maruyama strong order, additive",
      p4["euler_maruyama"]["additive_noise"]["strong_order"],
      res4["euler_maruyama"]["additive_noise"]["strong_order"], 0.05, "simulation")
check(4, "Euler-Maruyama strong order, multiplicative",
      p4["euler_maruyama"]["multiplicative_noise"]["strong_order"],
      res4["euler_maruyama"]["multiplicative_noise"]["strong_order"], 0.08, "simulation")
check(4, "Fokker-Planck L1 grid convergence order",
      p4["fokker_planck"]["grid_convergence_order_L1"],
      res4["fokker_planck"]["grid_convergence_order_L1"], 0.02)
worst_drift = max(abs(v["mass_drift"]) for k, v in res4["fokker_planck"].items()
                  if isinstance(v, dict))
check(4, "Fokker-Planck mass drift (must be at machine precision)", 0.0, worst_drift, 1e-14)
print(f"{len(CHECKS)} checks so far")
"""),
        md("## Paper 5: the costed rule"),
        code("""
from hpotential.analyses import paper5_access

p5 = published("p5_costed")
start = time.time()
res5 = paper5_access.run()
print(f"paper 5 full run: {time.time() - start:.1f} s")

check(5, "closed form vs direct evaluation", 0.0,
      res5["closed_form_check"]["max_relative_discrepancy"], 1e-12)
for setting, expected in p5["psa"].items():
    actual = res5["psa"][setting]
    check(5, f"rules disagree, {setting}", expected["P_disagreement"],
          actual["P_disagreement"], 0.01, "simulation")
    check(5, f"median V, {setting}", expected["median_V"], actual["median_V"], 0.30, "simulation")
print(f"{len(CHECKS)} checks so far")
"""),
        md("## Paper 6: early assessment and value of information"),
        code("""
from hpotential.analyses import paper6_hta

p6 = published("p6_voi")
start = time.time()
res6 = paper6_hta.run()
print(f"paper 6 full run: {time.time() - start:.1f} s")

for scenario, expected in p6["scenarios"].items():
    actual = res6["scenarios"][scenario]
    check(6, f"P(cost-effective, QALY), {scenario}", expected["P_cost_effective_QALY"],
          actual["P_cost_effective_QALY"], 0.01, "simulation")
    if "per_person_EVPI" in expected:
        check(6, f"per-person EVPI, {scenario}", expected["per_person_EVPI"],
              actual["per_person_EVPI"], max(300.0, 0.03 * expected["per_person_EVPI"]),
              "simulation")
    if "mean_incremental_bmp" in expected:
        check(6, f"BMP over QALY ratio, {scenario}",
              expected["mean_incremental_bmp"] / expected["mean_incremental_qaly"],
              actual["mean_incremental_bmp"] / actual["mean_incremental_qaly"], 0.05, "simulation")
print(f"{len(CHECKS)} checks so far")
"""),
        md("## The table"),
        code("""
frame = report()
"""),
        code("""
failures = frame[frame["result"] == "FAIL"]
if len(failures) == 0:
    print("Every published value reproduced within tolerance.")
else:
    print(f"{len(failures)} value(s) did not reproduce. A failure means one of two things:")
    print("the package changed what it computes, or a published number was wrong.")
    print("Both need a human. Widening a tolerance is not a fix.")
    print()
    print(failures.to_string(index=False))
"""),
        code("""
fig, ax = plt.subplots(figsize=(7.0, 4.0))
for i, kind in enumerate(sorted(frame["kind"].unique())):
    part = frame[frame["kind"] == kind]
    ratio = np.where(part["tol"] > 0, part["abs diff"] / part["tol"].replace(0, np.nan), np.nan)
    ax.scatter(range(len(part)), np.clip(ratio, 1e-12, None), s=22, color=series_color(i),
               label=kind, alpha=0.8)
ax.axhline(1.0, ls="--", lw=1.1, color=INK)
ax.set_yscale("log")
ax.set_xlabel("check")
ax.set_ylabel("absolute difference as a share of its tolerance")
ax.set_title("Everything under the dashed line passed", fontsize=10)
ax.legend(fontsize=8)
plt.show()
"""),
        md(NOT_ESTABLISH_TAIL + """
**A green table means the code is stable, not that the science is right.** Every check here
compares this package against numbers this package produced earlier. It is a regression
test, not an external validation.

**Simulation tolerances are judgement calls.** They are set wide enough for Monte Carlo
error at the published replicate counts and narrow enough that a change of method fails
them. A pass inside a wide tolerance is weaker evidence than a pass inside a tight one, and
the `kind` column tells you which you are looking at.

**The published values themselves rest on synthetic data.** Reproducing them exactly says
nothing about patients, panels or health systems.

**Reproducing a number is not agreeing with it.** If a value here is wrong in the papers,
this notebook will faithfully reproduce the wrong value and print PASS.
"""),
    ]
    return write(name, cells)


BUILDERS = [
    build_quickstart,
    build_measurement,
    build_prognostic,
    build_consensus,
    build_dynamics,
    build_access,
    build_hta,
    build_reproduce,
]


if __name__ == "__main__":
    for builder in BUILDERS:
        path = builder()
        nb = nbf.read(str(path), as_version=4)
        print(f"{path.name:<28} {len(nb.cells):>3} cells")
