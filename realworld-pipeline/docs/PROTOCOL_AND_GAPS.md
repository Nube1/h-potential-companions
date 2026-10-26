# Prescribed analysis and implementation boundary

## Trace to the thesis

The implementation follows the editable thesis sources, especially the formalism, measurement/results, prognostic validation, haemophilia translation, Appendices A–F, `THESIS_GRILL_V2.md`, `REVISION_MEMO.md`, and earlier verification code. The empirical order is:

1. approve the de-identified NHATS extract and map variables;
2. freeze estimand, cohort/index date, endpoints and five-year horizon;
3. freeze indicator direction, normalisation bounds, alpha and rho without outcomes;
4. estimate the three-domain measurement model without future outcomes;
5. preserve the frozen transformation and produce candidate scores;
6. compare identical people, outcomes and horizons across baseline, conventional comparators, additive, factor-score, geometric CRF and CES models;
7. evaluate association separately from discrimination, calibration, Brier error and net benefit;
8. repeat the complete pipeline in paired resamples and examine equity strata;
9. narrow or reject claims under the declared criteria; and
10. perform external validation before clinical or policy use.

## Non-negotiable scientific controls

- **Measurement/outcome firewall:** indicator selection, alpha, rho and score bounds may not be tuned to mortality or hospitalisation.
- **Paired comparison:** candidate models use identical splits/resamples and endpoints.
- **Survey design:** preserve NHATS weights, PSU and strata. Final design-based inference should be independently checked in R with survey/replicate weights.
- **Missingness:** special missing codes must be converted before ingestion. Packaged median imputation is an executable baseline, not final multiple imputation.
- **Competing risks:** first unplanned admission may require death as competing event. Pre-specify cause-specific versus cumulative-incidence interpretation.
- **Censoring:** packaged fixed-horizon evaluation uses known status. Publication analysis must implement IPCW Uno C/AUC/Brier with positivity diagnostics.
- **Uncertainty:** repeat imputation, measurement fitting, bounds, scoring and prognosis inside every paired bootstrap replicate.
- **Equity:** inspect invariance, calibration and Brier error by predeclared groups; suppress small cells.
- **No causal overclaim:** gradients are model-implied rankings, not intervention effects.

## Thesis rejection logic

Abandon the scalar for `(P,S,R)` if HTMT > 0.85, factor correlation > 0.90, or determinacy < 0.80. Narrow claims when optimism-corrected delta C is non-positive or error is worse than parsimonious comparators. Refute the geometric specification when its calibration or net benefit is worse than additive. Restrict claims to the developmental population if external transport fails.

## Data-access reality

The source archive contains no approved NHATS analysis extract and therefore no real empirical result. This package accepts an approved local CSV/Parquet and records its checksum; it does not redistribute restricted data. A real-world run remains blocked until the governed extract and `TO_MAP` crosswalk are supplied.

## Haemophilia extension

Do not reuse NHATS loadings as transportable. Preserve construct meaning, conduct the proposed modified Delphi, map disease-specific indicators, re-identify the scale, test the loading pattern and invariance, recalibrate, then validate in an independent South African cohort.

