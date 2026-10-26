# Head-to-head comparison scaffold. Requires 00_preflight.R to pass and the
# frozen variable mapping to be completed before outcome access.
source("analysis/00_preflight.R")

required_model_columns <- c(
  "h_sem", "biomedical_potential", "social_enablement", "structural_burden",
  "mortality_time", "mortality_event", "hospital_time", "hospital_event"
)
missing_model_columns <- setdiff(required_model_columns, names(dat))
if (length(missing_model_columns)) {
  stop("Comparison columns missing: ", paste(missing_model_columns, collapse = ", "))
}

alpha <- 0.5  # Replace only by a pre-outcome measurement-derived or normative value.
source("analysis/01_capability_realisation.R")
scores <- hp_candidates(
  dat$biomedical_potential, dat$social_enablement, dat$structural_burden,
  alpha = alpha, rho = -1,
  beta = c(0, 1, 1, 1, 0)
)
dat$h_additive <- scores$additive
dat$h_interaction <- scores$interaction
dat$h_multiplicative <- scores$multiplicative
dat$h_ces <- scores$ces
dat$h_relational <- scores$relational

candidate_scores <- c(
  "h_additive", "h_interaction", "h_multiplicative", "h_ces",
  "h_relational", "h_sem"
)
models_mortality <- lapply(candidate_scores, function(score) {
  survival::coxph(
    stats::as.formula(paste("survival::Surv(mortality_time, mortality_event) ~", score,
                            "+ age + sex")), data = dat, x = TRUE
  )
})
names(models_mortality) <- candidate_scores

# The final implementation must add: multiple imputation; survey design;
# competing-risk evaluation for hospitalisation; paired full-pipeline bootstrap;
# calibration, Brier score, Uno C, time-dependent AUC, decision curves; subgroup
# error/calibration; and generated tables/figures. Do not report this scaffold's
# apparent in-sample fits as validation results.
saveRDS(models_mortality, "outputs/crf_candidate_models_unvalidated.rds")
