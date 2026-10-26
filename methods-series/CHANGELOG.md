# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html); a
change that moves a published value is a major change.

## [1.1.0] - 2026-08-16

Adds a production-oriented, privacy-conscious real-data ingestion layer while
leaving the published computational analyses unchanged.

### Added

- Schema-governed CSV, TSV, Excel, and Parquet ingestion with explicit types,
  ranges, categories, roles, unique keys, and row-missingness limits.
- Structured validation reports that contain aggregate diagnostics rather than
  participant values.
- Reproducible analysis-bundle preparation with source SHA-256 fingerprints,
  identifier removal by default, manifests, and missingness summaries.
- The `hpotential-data validate` and `hpotential-data prepare` command-line
  workflows, a starter schema and cohort template, and a real-data operations
  guide.
- Automated tests for coercion, constraint failures, duplicate identifiers,
  source fingerprinting, identifier removal, and strict rejection.

### Changed

- The README now distinguishes bundled synthetic examples from user-supplied
  real-world cohorts and documents the supported workflow.
- The `full` optional dependency set includes Excel and Parquet readers.

## [1.0.0] - 2026-08-15

First release. Brings together the software supplement and the eight analysis
scripts that produced the published numbers, behind one tested interface.

### Added

- `core.measurement`: the second-order model, factor-score determinacy and its
  ceiling, local identification by Jacobian rank, and normal-theory maximum
  likelihood estimation with expected information.
- `core.prognostic`: the Cox partial likelihood with Breslow ties, the
  two-stage pipeline, the attenuation prediction, and the events cost implied by
  the determinacy ceiling.
- `core.consensus`: the Delphi retention rule, content validity, tie-corrected
  Kendall's W, operating characteristics of the rule, and an exhaustive check of
  the dispersion lemma.
- `core.dynamical`: the Ornstein-Uhlenbeck model, Runge-Kutta and
  Euler-Maruyama with convergence studies, a conservative Fokker-Planck solver,
  and profile-likelihood identifiability by exact Kalman filtering.
- `core.access`: attainable capability and the costed decision statistic.
- `core.hta`: the early-assessment model with EVPI and EVPPI.
- `core.itembank`, `core.ctt`, `core.irt`, `core.dif`: carried over from the
  software supplement unchanged.
- `analyses.paper1_validation` through `analyses.paper6_hta`, each exposing
  `run(cfg=None, outdir=None, quick=False)` and `figures(...)`.
- `config.Paper1Config` through `Paper6Config`, frozen dataclasses holding the
  exact published parameters.
- `utils.plotting`, `utils.numerical`, `utils.stats`.
- `scripts/run_all.py` and `scripts/regenerate_figures.py`.
- `tests/test_published_values.py`, which asserts that the package still
  reproduces every published value, reading the expected numbers from
  `tests/fixtures/`.

### Changed from the analysis scripts

- Replicate generators are derived from a seed sequence over
  `(seed, scenario, sample size, replicate)` instead of from `hash` of a
  scenario name. Python salts string hashes per process, so the original runs
  were not reproducible even on one machine. Simulation results now reproduce
  within Monte Carlo error rather than to the digit; closed-form results are
  unaffected.
- The endorsement of a simulated Delphi item is defined on the rounded scale and
  inverted in closed form, rather than defined on the continuous scale and
  inverted by Monte Carlo. The two differ: a request for 0.90 at sigma 1.2
  realises 0.9553 under the continuous definition.
- Paper 2 additionally reports `attenuation_ratio_pooled` over all replicates.
  A ratio from one design point carries several per cent of Monte Carlo error at
  120 replicates, which is more than the spread being interpreted.

### Notes

All data are synthetic. Papers 1 to 3 are protocols. See the integrity note in
the README, and `docs/DRAFT_REVIEW.md` for what changed from the co-author's
draft architecture.
