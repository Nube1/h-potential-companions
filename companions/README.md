# H-potential thesis companion research

Three distinct manuscripts and executable research code grounded in the supplied 283-page H_Potential_Thesis_V12_4 draft. Current status: **theory and synthetic computational evidence; not clinical validation**.

## Read the papers

| Paper | Contribution | Formats |
|---|---|---|
| 01 Capability realisation | Bounded CRF, assumptions, derivatives, log-gap decomposition and substitution sensitivity | [PDF](papers/01_capability_realisation.pdf), [source](papers/01_capability_realisation.md), [LaTeX](papers/01_capability_realisation.tex) |
| 02 Measurement uncertainty | Known-model Gaussian scoring, missing indicators, drift counterexample and future NHATS protocol | [PDF](papers/02_measurement_uncertainty.pdf), [source](papers/02_measurement_uncertainty.md), [LaTeX](papers/02_measurement_uncertainty.tex) |
| 03 Equity and policy | Weighted EDE, causal-response separation, exact binary allocation and policy-ranking reversal | [PDF](papers/03_equity_policy.pdf), [source](papers/03_equity_policy.md), [LaTeX](papers/03_equity_policy.tex) |

## Run

Python 3.10 or newer. A virtual environment is recommended.

```sh
python -m pip install -r requirements-dev.txt
python -m pytest                  # 41 tests, including randomised property tests
python analysis/reproduce.py      # regenerates results/
python analysis/build_papers.py   # writes papers/*.tex and papers/*.pdf
python analysis/verify_delivery.py
```

`build_papers.py` compiles the PDFs with LaTeX (pdflatex, Times) when `latexmk` is installed and falls back to ReportLab otherwise. Force either with `--engine latex` or `--engine reportlab`. The Markdown files are the source; edit those, not the `.tex`.

The CRF and policy code uses only the standard library. Measurement scoring uses NumPy.

Command-line scoring of compatible research profiles (`pip install -e .` also gives a `hpotential` command):

```sh
python -m hpotential score examples/profiles.csv results/example_scores.csv --alpha 0.5 --rho 0
python -m hpotential ede results/example_scores.csv --epsilon 1
```

Supply P in one common declared unit, S as social enablement, and R as structural freedom (1 minus a justified normalised burden). The command does not infer or clinically calibrate these inputs. Bad input gives a one-line error naming the CSV line, not a traceback.

## Proof figures

`python analysis/proof_figures.py` draws ten figures into `figures/`, one for each mathematical result of the three papers: level sets of the CES family, the ordering and limits in rho, the capacity ceiling, the balance penalty, marginal effects, the exact log-gap decomposition, determinacy and missingness, measurement drift, the EDE against inequality aversion, and the allocation reversal. Each figure carries a numeric check of the result it shows (for example, the finite-difference error of the analytic derivative, or the simulated against the formula bias under drift), and the build stops if a check fails. `figures/FIGURES.md` lists every figure with its statement, caption and check; `figures/figure_index.json` holds the same for the paper text. The papers do not yet cite these figures; each needs a sentence in the text before it goes into a manuscript.

## Reproducibility

`reproduce.py` writes floats to 12 significant digits. The result files and their SHA-256 hashes in `results/manifest.json` are then identical across NumPy versions and operating systems; this was checked against the original Windows run (NumPy 2.3.5) and Linux runs with NumPy 2.1 and 2.5. The tests regenerate every table and compare it with the committed copy, and check that every number in the three manuscript tables equals the generated value rounded to four decimals.

## Scope of working code

- Stable geometric/CES/bottleneck scoring, exact zero limits and invalid-input checks.
- Geometric derivatives, individual log-gap decomposition, weighted EDE and EDE derivatives.
- Exhaustive binary portfolio allocation for at most 20 candidate actions.
- Known-parameter continuous Gaussian posterior scores, covariance and determinacy.
- Frozen-anchor conversion with explicit extrapolation failure.
- Synthetic experiments with 4,000 generated records and six numerical tables.

The code does **not** fit a CFA, implement survey-weighted SEM, fit a survival model, identify causal effects, optimise medical dosing or validate NHATS/haemophilia outcomes. Those are named future stages, not hidden completed features.

## Relation to existing GitHub work

This folder is one of several projects in the repository; see the repository README. The separate haemophilia PK/PD notebook in `Nube1/Biomedical-Potential-Calculator-` is not changed, and its placeholder parameters are not treated as evidence for the thesis. GitHub Actions runs the tests on Python 3.10 to 3.13 and on Windows, then rebuilds the results and the LaTeX PDFs and fails if any committed result changes.

## Thesis and approval status

The supplied supervisor email says to follow up Faculty protocol approval with Claire and not to start the PhD before Senate approval. It permits Chapter One literature review and requests a traditional chapter without case studies. This package records that instruction; it does not assert approval, contact Claire, or alter the thesis. A concrete Chapter One revision map is in docs/thesis_alignment.md.

All examples are synthetic. No patient or NHATS individual data are included. The six original ZIP packages were identified but could not be downloaded for inspection; they are not represented as reviewed or silently merged. Source provenance and gaps are documented in docs/source_register.md and docs/gap_register.md.

## Submission status

These are complete research drafts for supervisory review. Empirical validation, full literature/novelty coverage, author-approved affiliations/co-authors, declarations and journal-specific formatting remain before submission. No journal acceptance, ethics approval or thesis readiness is asserted. The source and code are provided together so reviewers can assess exactly what the reported evidence supports.
