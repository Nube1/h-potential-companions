# H-Potential real-world validation pipeline

This repository turns the thesis protocol into an auditable Colab/Git workflow. It is built for the approved, de-identified NHATS analytic extract; no restricted participant data are bundled. The included synthetic data are only a software test and tutorial and **must never be reported as empirical evidence**.

## What is implemented

- strict schema, identifier, event/time, weight, missingness and SHA-256 checks;
- an outcome-blind measurement stage with a frozen measurement artifact;
- oriented Biomedical, Social and Systemic domain scores;
- additive, geometric CRF and predeclared CES sensitivity scores;
- baseline, comparator and H-Potential Cox models on the same split;
- fixed-horizon AUC, Brier score, calibration slope/intercept and decision-curve net benefit;
- sex/race subgroup calibration and error tables;
- immutable configuration/data hashes, deterministic seeds and run audit files;
- synthetic smoke data, unit tests, notebook execution checks and GitHub Actions.

The implementation is deliberately honest about its boundary. It provides a robust development/internal hold-out workflow, but a hold-out from the same cohort is not external validation. The study configuration must be frozen and committed before outcomes are opened. Confirmatory publication analysis should additionally use the full-pipeline paired bootstrap and design-consistent multiple-imputation/variance estimation described in `docs/PROTOCOL_AND_GAPS.md`.

## Colab

Open `notebooks/H_Potential_Real_World_Pipeline.ipynb` in Colab. Run the synthetic demonstration first. For real data, upload the approved extract or mount a controlled Google Drive folder, set `DATA_PATH`, and keep `USE_SYNTHETIC = False`. The notebook refuses real-data mode if the expected file is absent.

## Local/Git use

```bash
python -m pip install -e ".[dev]"
hpotential generate-synthetic --config configs/study.yaml
hpotential run --config configs/study.yaml --data data/sample/synthetic.csv
pytest
```

Never commit raw or row-level derived participant data. `.gitignore` blocks them. Only disclosure-reviewed aggregate outputs belong in Git.

## Before a real run

1. Map the approved source variables to canonical columns and document special missing codes.
2. Freeze cohort entry, index date, endpoints, horizons, exclusions, indicator directions, alpha, rho, bounds, comparators, subgroups, thresholds and rejection rules.
3. Commit that configuration **before** outcome access and record the commit SHA.
4. Obtain governance/ethics approval and an approved de-identified extract. Do not scrape or bypass NHATS access.
5. Run the contract check, inspect missingness, then run under the frozen configuration.
6. Treat haemophilia as a translation protocol until a separate cohort, Delphi, scale identification and external validation exist.

## Outputs

`data_manifest.json`, `measurement_model_frozen.json`, `model_performance.csv`, `equity_performance.csv`, `participant_scores_RESTRICTED.csv`, and `run_audit.json`.

