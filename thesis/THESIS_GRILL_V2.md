# Examiner-style audit of the H-Potential thesis

## Bottom line

The concept is promising, but Draft 1 is a sophisticated protocol-shaped shell,
not yet a defensible completed thesis. Its strongest move is the separation of a
latent measurement model from later prognostic testing. Against the authoritative
revised 2026 proposal, the broad design is aligned; the decisive weakness is that
the proposed analyses and translational work have not yet been evidenced or made
reproducible in the archive.

## Critical issues before submission

1. **Protocol version control.** The revised proposal supplied separately specifies
   the NHATS/SEM/survival design and is well aligned with the thesis. However, the
   ZIP also contains a superseded `Protocol.tex` describing a different synthetic
   global simulation. Remove it from the submission archive or label it prominently
   as obsolete; otherwise an examiner could reasonably infer protocol confusion.
2. **No empirical results.** Dozens of tables, figures, derivations, the abstract,
   discussion, and conclusion contain placeholders. Until real output replaces
   them, claims of validation, prediction, or superiority are unavailable.
3. **No reproducible analysis.** Draft 1 named an R workflow and a Vol-OS engine
   but included neither. V2 removes the unsupported Vol-OS claims and supplies an
   input-validation scaffold. The full scripts, package lockfile, data provenance,
   and generated outputs still have to be added.
4. **Construct/outcome leakage risk.** Several biomedical indicators (function,
   multimorbidity) may overlap strongly with frailty comparators and predict the
   outcomes almost by definition. Pre-specify overlap audits, comparator fairness,
   and sensitivity models excluding near-outcome indicators.
5. **Survey design.** NHATS is a complex survey. The thesis must state how weights,
   strata, clusters, replenishment cohorts, and repeated observations enter both
   SEM and survival analyses. Ignoring design can bias uncertainty and population
   interpretation.
6. **Factor-score uncertainty.** A two-stage model does not eliminate uncertainty
   or optimism. Factor scores are estimated regressors. Refit the full measurement
   plus prognostic pipeline inside every bootstrap replicate, and add a joint or
   plausible-values sensitivity analysis.
7. **Higher-order identification.** Three first-order factors give a just-identified
   second-order structure under common constraints, limiting the informativeness
   of global second-order fit. Compare plausible bifactor and correlated-factor
   alternatives and justify why a single score is preferable to a profile.
8. **Causal language and policy use.** Prognostic association cannot support claims
   that changing access or social support will change outcomes. Policy-gradient
   optimisation requires a causal model, intervention definitions, costs,
   constraints, and distributive objectives; keep it future work unless implemented.
9. **Equity validation is missing.** A metric designed to expose inequity needs
   measurement invariance, calibration, error, and net-benefit analyses by sex,
   race/ethnicity, socioeconomic position, and relevant intersections—not merely
   those variables inside the score.
10. **Hemophilia is currently translational framing, not validation.** The disease
    chapter provides a roadmap. The revised proposal also specifies a two- or
    three-round modified Delphi with Wits clinicians. Supply completed consensus
    methods/results and governance evidence, or state clearly that this component
    remains planned and narrow Aim 4 claims accordingly.
11. **Administrative and layout cleanup.** The compiled title page still says
    November 2025 while the revised proposal is dated 2026. Resolve the submission
    date, class name mismatch, duplicated PDF page anchors, tight appendix headings,
    and remaining overfull boxes before deposit.

## Conceptual upgrade

Treat H-Potential as a **state-and-capability profile**, not automatically as one
scalar. Retain the three domain scores, the higher-order score only if empirically
supported, and an explicit uncertainty interval. This avoids hiding compensatory
trade-offs: severe structural burden should not disappear merely because a strong
biomedical domain offsets it numerically.

The next-level research programme has three linked layers:

- **Measurement:** invariant latent domains with uncertainty and subgroup fairness.
- **Prediction:** landmark or dynamic prediction with external validation and
  decision-curve analysis.
- **Intervention:** a separate causal model estimating which feasible action changes
  which domain, for whom, at what cost, under an explicit equity objective.

## Required completion sequence

1. Freeze the estimand, cohort entry, index date, horizons, endpoints, and the
   authoritative revised-proposal version.
2. Freeze the NHATS wave/variable dictionary, coding directions, missing codes,
   survey design variables, and data-flow diagram.
3. Run data-quality and overlap audits before modelling.
4. Fit CFA alternatives; assess reliability, local fit, invariance, and score
   determinacy. Avoid post-hoc correlated errors without substantive justification.
5. Lock the measurement model before accessing outcome results.
6. Fit and diagnose endpoint-specific survival models; use competing-risk methods
   where death precludes hospitalisation, and report absolute risk at five years.
7. Bootstrap the entire pipeline, pool imputation correctly, and publish calibration
   plots with uncertainty.
8. Benchmark fairly against pre-specified comparators and report paired uncertainty
   for performance differences.
9. Run subgroup fairness and sensitivity analyses, including removal of overlapping
   indicators and alternative latent structures.
10. Populate results once, regenerate tables/figures from code, rewrite the abstract,
    discussion, and conclusions from those outputs, then conduct an independent
    reproducibility check.

## Submission gate

Do not submit while any bracketed placeholder remains, while the analysis cannot
run from a clean environment, or while claims in the abstract exceed the evidence
available in the archived outputs.
