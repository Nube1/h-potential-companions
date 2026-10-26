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
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python analysis/reproduce.py
python analysis/build_papers.py
```

The CRF and policy module uses only the standard library. Measurement scoring uses NumPy. PDF generation uses ReportLab. Exact locally observed versions are recorded in results/environment.txt and results/manifest.json. Version ranges permit installation; exact bitwise reproduction across numerical-library versions is not promised.

Command-line scoring of compatible research profiles:

```sh
python -m hpotential score examples/profiles.csv results/example_scores.csv --alpha 0.5 --rho 0
python -m hpotential ede results/example_scores.csv --epsilon 1
```

Supply P in one common declared unit, S as social enablement, and R as structural freedom (1 minus a justified normalised burden). These inputs are not inferred or clinically calibrated by the command.

The generated CSV tables include a seed, version record and hashes. The manuscript tables are rounded from these results; see docs/claim_evidence.csv. LaTeX sources can be built with `pdflatex` in an environment with the declared packages. LaTeX compilation was not available locally; PDFs were generated directly with ReportLab and visually checked.

## Scope of working code

- Stable geometric/CES/bottleneck scoring, exact zero limits and invalid-input checks.
- Geometric derivatives, individual log-gap decomposition, weighted EDE and EDE derivatives.
- Exhaustive binary portfolio allocation for at most 20 candidate actions.
- Known-parameter continuous Gaussian posterior scores, covariance and determinacy.
- Frozen-anchor conversion with explicit extrapolation failure.
- Synthetic experiments with 4,000 generated records and six numerical tables.

The code does **not** fit a CFA, implement survey-weighted SEM, fit a survival model, identify causal effects, optimise medical dosing or validate NHATS/haemophilia outcomes. Those are named future stages, not hidden completed features.

## Relation to existing GitHub work

The connected Nube1 account contains `Biomedical-Potential-Calculator-`, whose README describes a separate haemophilia PK/PD notebook. This companion project does not overwrite that notebook or treat its placeholder parameters as evidence for the thesis. The intended integration is an additive `thesis_companions/` directory on a research branch, subject to repository write access.

GitHub branch creation was rejected by the session approval policy. This delivery is therefore a standalone local Git repository plus a portable Git bundle. It has not been pushed. No existing remote history has been rewritten. See docs/git_handoff.md for exact import commands.

## Thesis and approval status

The supplied supervisor email says to follow up Faculty protocol approval with Claire and not to start the PhD before Senate approval. It permits Chapter One literature review and requests a traditional chapter without case studies. This package records that instruction; it does not assert approval, contact Claire, or alter the thesis. A concrete Chapter One revision map is in docs/thesis_alignment.md.

All examples are synthetic. No patient or NHATS individual data are included. The six original ZIP packages were identified but could not be downloaded for inspection; they are not represented as reviewed or silently merged. Source provenance and gaps are documented in docs/source_register.md and docs/gap_register.md.

## Submission status

These are complete research drafts for supervisory review. Empirical validation, full literature/novelty coverage, author-approved affiliations/co-authors, declarations and journal-specific formatting remain before submission. No journal acceptance, ethics approval or thesis readiness is asserted. The source and code are provided together so reviewers can assess exactly what the reported evidence supports.
