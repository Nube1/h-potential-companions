# Notebooks

Eight Colab notebooks covering the H-potential methods series. Each one opens in Colab with
no setup: the first cell installs the package from GitHub, prints the resolved version, and
says whether it is running in Colab. Run from a clone instead and the same cell installs the
local checkout.

Every notebook carries a `FULL_RUN` flag near the top. With `FULL_RUN = False` the replicate
counts are cut so the notebook finishes quickly, and every results cell prints a banner
saying the numbers are a reduced run and do not reproduce the published values. With
`FULL_RUN = True` they reproduce them.

**All data in these notebooks are synthetic.** Papers 1 to 3 are protocols: no participant
has been recruited and no Delphi panel has met. Papers 4 to 6 are method illustrations whose
inputs show a workflow. No figure in any notebook is evidence about patients. Each notebook
says so in its first markdown cell and closes with a "What this does not establish" section.

## Where the notebooks install from

Each setup cell installs this package from `Nube1/h-potential-companions`, folder `methods-series`. The repository is private, so in Colab sign in to GitHub when asked, or clone the repository and run the notebooks from the clone. To change the source, edit `ORG`, `REPO` and `SUBDIR` in `scripts/build_notebooks.py` and run it again; it rewrites every notebook and badge.

## The notebooks

| notebook | what it covers | reduced run | full run |
| --- | --- | --- | --- |
| [00_quickstart](00_quickstart.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nube1/h-potential-companions/blob/main/methods-series/notebooks/00_quickstart.ipynb) | The six headline results in one pass, about ten lines each, closing with a table of what each paper claims and what it does not. | 5 s | ~4 min |
| [01_measurement](01_measurement.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nube1/h-potential-companions/blob/main/methods-series/notebooks/01_measurement.ipynb) | Paper 1. Derives the ceiling `sqrt(c/(1+c))`, checks the closed form against direct evaluation at 400 indicators per domain (0.877819 against 0.877594), plots determinacy against indicators for four loadings, runs the identification rank check with its two controls, and runs the sample-size simulation. Sliders for gamma and Psi drive the ceiling live. | 11 s | ~3 min |
| [02_prognostic](02_prognostic.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nube1/h-potential-companions/blob/main/methods-series/notebooks/02_prognostic.ipynb) | Paper 2. Simulates the whole pipeline and checks the attenuation lands near rho_D = 0.8361, then works the power corollary: 730 events with the construct observed, 947 at the ceiling, 1044 at the planned design. A slider for rho_D drives the attenuated hazard ratio and the events needed. | 9 s | ~90 s |
| [03_consensus](03_consensus.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nube1/h-potential-companions/blob/main/methods-series/notebooks/03_consensus.ipynb) | Paper 3. The retention curve, the binding-criterion diagnosis, the IQR threshold diagnostic, the stakeholder split, and the dispersion lemma checked by exhaustive enumeration of small panels. Sliders for panel size, disagreement and the IQR limit let you rediscover which criterion binds. | 5 s | ~2 min |
| [04_dynamics](04_dynamics.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nube1/h-potential-companions/blob/main/methods-series/notebooks/04_dynamics.ipynb) | Paper 4. RK4 order 4.05; Euler-Maruyama strong order 1.03 additive against 0.53 multiplicative on a log-log plot; Fokker-Planck mass conservation to machine precision with no renormalisation, plus first-order grid convergence; profile likelihood for three observation designs. | 7 s | ~90 s |
| [05_access](05_access.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nube1/h-potential-companions/blob/main/methods-series/notebooks/05_access.ipynb) | Paper 5. Derives V, checks the closed form against direct evaluation to machine precision, runs the probabilistic sensitivity analysis, and shows the costed and uncosted rules disagreeing on about half the high-resource draws. Sliders for the three factors of V flip the decision live. | 4 s | ~25 s |
| [06_hta](06_hta.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nube1/h-potential-companions/blob/main/methods-series/notebooks/06_hta.ipynb) | Paper 6. Three scenarios, EVPI, EVPPI by input, and the comparison between biomedical potential and QALYs that opens the 88 per cent gap in the central scenario. A willingness-to-pay slider drives the cost-effectiveness probability and the EVPI. | 4 s | ~40 s |
| [99_reproduce_all](99_reproduce_all.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nube1/h-potential-companions/blob/main/methods-series/notebooks/99_reproduce_all.ipynb) | Reads the published fixtures, recomputes every value, and prints expected, actual, absolute difference, tolerance and PASS or FAIL. 93 checks. No reduced mode. | n/a | 4 min 20 s here, allow 10 min on a free Colab CPU |

Times are wall clock on one CPU core with the package already installed. Add roughly a
minute in Colab for the pip install in the first cell.

## Reading order

Read 00 first if you want the shape of the series in a sitting. Read 01 first if you want
the argument, since papers 2 and 3 lean on the ceiling it derives. 99 is a regression check
rather than a teaching document, and it is the one to run after changing library code.

## Notes on the fixtures

`99_reproduce_all` needs `tests/fixtures/*.json`, which ship with the repository but not
inside the installed package. It looks for them in a local checkout, then in a shallow clone
of the repository, then falls back to a small set of values embedded in the notebook. The
fallback is flagged loudly when it fires, because a value carried in the notebook is weaker
evidence than one fetched from the repository.

## Widgets

The sliders use `ipywidgets`. Colab does not always provide a working widget stack, so every
interactive cell renders a static case first and attaches sliders only if the import
succeeds. Nothing breaks when widgets are unavailable; you get the static figure and a note
saying so.
