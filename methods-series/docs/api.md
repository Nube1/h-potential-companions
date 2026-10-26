# API reference

Every public function carries a numpy-style docstring stating its assumptions and
what it does not do. This page is the map; the docstrings are the detail.

```python
import hpotential
hpotential.__version__
help(hpotential.core.measurement.rho_determinacy)
```

## Conventions

- Every stochastic function takes an explicit `rng` (a
  `numpy.random.Generator`) or a `seed`. Nothing in the library touches the
  global numpy random state, so two analyses in one process cannot disturb each
  other and the same seed always gives the same output.
- Configurations are frozen dataclasses. `MeasurementModel` is frozen and its
  arrays are read-only, so a model can be shared between analyses safely.
- Functions raise `ValueError` on bad input rather than returning a filled-in
  value. A replicate that fails returns `None` and is dropped from the average,
  not replaced.
- The package ships `py.typed`; type checkers will use the annotations.

---

## `hpotential.core.measurement`

The second-order model, its determinacy, its identification, and its estimation.

| name | what it does |
| --- | --- |
| `MeasurementModel` | frozen model: loadings, second-order loadings, disturbances, residuals |
| `MeasurementModel.balanced` | equal standardised loadings in every domain |
| `MeasurementModel.marker_scaled` | first indicator of each domain loads at one |
| `rho_determinacy` | `rho_D` from a full parameterisation |
| `second_order_determinacy` | `rho_D` of a balanced design |
| `determinacy_ceiling` | `(sqrt(c/(1+c)), c)` with `c = gamma' Psi^-1 gamma` |
| `required_second_order_signal` | `c` needed for a target `rho_D` |
| `determinacy_curve` | a grid of designs as a dataframe |
| `identification_report` | Jacobian rank, smallest singular value, condition number |
| `moment_jacobian` | derivative of the unique moments in the free parameters |
| `fit_cfa` | normal-theory ML fit, returning `CFAFit` |
| `expected_information` | Fisher information from central differences |
| `sample_size_replicate` | one replicate: simulate, fit, coverage, determinacy |
| `simulate_indicators` | draw `zeta`, the domains and the indicators |
| `regression_scores` | Thurstone factor scores |

Use `marker_scaled` whenever the estimated `gamma` is compared with the true
one. A balanced population model fitted under marker scaling returns
`lam * gamma`, and bias computed against `gamma` then measures the scale
mismatch rather than the estimator.

## `hpotential.core.prognostic`

| name | what it does |
| --- | --- |
| `cox_fit` | Cox partial likelihood, Newton-Raphson, Breslow ties |
| `CoxFit` | coefficient, standard error, iterations, convergence, hazard ratio |
| `simulate_survival` | exponential times with administrative censoring |
| `attenuated_beta` | `rho_D * beta_true`, the regression-dilution prediction |
| `events_inflation` | `1 / rho_D^2`, events needed relative to an oracle study |
| `required_events` | events for a given power, level and determinacy |
| `pipeline_replicate` | indicators, factor model, score, Cox model, once |

## `hpotential.core.consensus`

| name | what it does |
| --- | --- |
| `DelphiRule`, `DelphiConsensusEngine` | the three-part retention rule on real ratings |
| `scale_cvi` | scale-level content validity, both definitions |
| `kendalls_w` | concordance with the tie correction |
| `stability` | round-to-round agreement |
| `simulate_panel` | synthetic ratings as a dataframe |
| `simulate_ratings` | many panels at once, as an array |
| `endorsement_probability` | `pi` on the rounded scale, in closed form |
| `mu_for_endorsement` | the exact inverse of the above |
| `retention_masks` | apply the rule to many panels, with per-criterion masks |
| `operating_characteristics` | false-drop and false-keep rates, and what binds |
| `iqr_lemma_violations` | exhaustive check of the dispersion lemma |

The engine's default dispersion threshold is 2.0, not the 1.5 usually quoted.
The module docstring gives the error rates that decided it.

## `hpotential.core.dynamical`

| name | what it does |
| --- | --- |
| `LinearSDE` | Ornstein-Uhlenbeck process, additive or multiplicative noise |
| `rk4` | classical fourth-order Runge-Kutta |
| `euler_maruyama` | stochastic paths, optionally on a shared Brownian path |
| `strong_error_study`, `weak_error_study` | convergence orders |
| `FokkerPlanck` | conservative finite-volume solver, no renormalisation |
| `FokkerPlanckSolution` | density, mass history, minimum density, `l1_error` |
| `stable_time_step` | diffusive and advective stability limit |
| `kalman_loglik` | exact filter for a latent OU state observed with error |
| `simulate_state` | one subject's noisy observations |
| `profile_sigma` | profile likelihood over the process diffusion |

## `hpotential.core.access`

| name | what it does |
| --- | --- |
| `attainable_capability` | `H = (L Q)(S T A) / (D (1 + C + psi)) - burden` |
| `costed_statistic` | `V = (T/A)(dA/dT)(cT/cA)` |
| `AccessModel` | input ranges for one setting, with `psa` and `closed_form_check` |

## `hpotential.core.hta`

| name | what it does |
| --- | --- |
| `HTAModel` | the synthetic early-assessment model |
| `HTAModel.sample`, `.evaluate`, `.run_scenario`, `.summarise` | the workflow |
| `evpi` | per-person expected value of perfect information |
| `evppi` | single-parameter EVPPI by the regression method |
| `smooth_conditional_mean` | equal-count binned conditional mean |

## `hpotential.core.itembank`, `.ctt`, `.irt`, `.dif`

Carried over unchanged from the software supplement: item metadata and its
schema, Cronbach's alpha and McDonald's omega, KMO and Bartlett, exploratory
factor analysis with promax rotation, Samejima's graded response model fitted by
marginal maximum likelihood over Gauss-Hermite quadrature, and ordinal logistic
differential item functioning with three nested models.

## `hpotential.analyses`

Six modules, one per paper, all the same shape:

```python
run(cfg=None, outdir=None, quick=False) -> dict
figures(results, outdir, cfg=None) -> dict[str, Path]
FIGURES: tuple[str, ...]
```

`hpotential.analyses.PAPERS` maps a paper number to its module, which is what
the scripts loop over.

## `hpotential.config`

`Paper1Config` through `Paper6Config`, frozen dataclasses holding the exact
published parameters. Every analysis takes one and defaults to the published
one.

```python
from hpotential.config import Paper1Config
from hpotential.analyses import paper1_validation

published = Paper1Config()
variant = Paper1Config(gamma=(0.9, 0.85, 0.8), reps=50)
paper1_validation.run(variant)
```

## `hpotential.utils`

| module | contents |
| --- | --- |
| `plotting` | `apply_style`, `save_figure`, `print_safe`, `series_color`, `SERIES`, `LINESTYLES`, `HATCHES`, `FILLS`, `INK`, `GRID` |
| `numerical` | `convergence_order`, `pairwise_orders`, `richardson`, `safe_cholesky` |
| `stats` | `credible_interval`, `binned_conditional_mean`, `bootstrap_ci`, `relative_bias`, `summarise` |
