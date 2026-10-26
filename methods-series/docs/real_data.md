# Real-data workflow

The package keeps published simulations separate from participant data. Raw data should remain in an approved secure environment and must never be committed to Git.

## 1. Create a study schema

Copy `data/templates/cohort_schema.json` and edit the columns, categories, ranges, unique key, and missingness threshold for the registered protocol. Every variable needs a role and plain-language description. Do not change a schema after database lock without versioning it and documenting the reason.

## 2. Validate without exporting rows

```bash
hpotential-data validate secure/cohort.xlsx \
  --schema study/cohort_schema.json \
  --sheet baseline \
  --report validation_report.json
```

The report contains counts and missingness only, not participant values. Exit status `0` means valid; `2` means the schema failed.

## 3. Create a de-identified analysis snapshot

```bash
hpotential-data prepare secure/cohort.xlsx \
  --schema study/cohort_schema.json \
  --sheet baseline \
  --out derived/cohort_2026_08_16
```

Identifier-role columns are removed by default. The output directory is immutable by convention and contains:

- `analysis_data.csv`: coerced analysis data without identifiers;
- `manifest.json`: SHA-256 fingerprint, schema version, software versions, and validation report;
- `missingness.csv`: variable-level missingness.

The command refuses to overwrite an existing snapshot. Use a new dated/versioned directory after an authorised data update.

## 4. Analyse under a frozen plan

Read the de-identified snapshot, not the raw workbook, in paper-specific scripts. Record exclusions, transformations, imputation, survey weights, clustering, and analysis-set counts. Fit measurement models before outcome models where required by the protocol.

## Privacy and governance

- Use pseudonymous study IDs; keep the linkage key outside the repository.
- Never commit raw, row-level, free-text, geographic microdata, or dates not required by the analysis.
- Small-cell suppression and access controls remain study responsibilities.
- A valid schema is a technical check, not ethics approval or consent.
- `--include-identifiers` exists for controlled internal pipelines and should not be used for shared analysis bundles.
