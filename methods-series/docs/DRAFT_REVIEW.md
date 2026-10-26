# What changed from the draft architecture, and why

The draft architecture supplied by a co-author set the module layout used here:
`core/` split by method area, `analyses/` with one module per paper, `config/`
holding the parameters, `utils/` for shared helpers. That layout is kept.

The function bodies were mostly placeholders. Six of them would have produced
numbers that look like results but are not, and three of those would have
reported a result no calculation supports. This note lists each one, what the
package does instead, and where a number is available, the number that shows the
difference matters. It is a record for the co-author, not a complaint: the draft
was explicitly a skeleton.

Every claim below was run before it was written.

---

## 1. Fit indices and coverage drawn from a random number generator

**Draft.** Fit statistics generated as `0.95 + np.random.normal(0, 0.02)`, and
interval coverage returned as the constant `0.95`.

**Why it matters.** These are the two quantities a sample-size study exists to
estimate. A coverage of 0.95 written into the code passes any check a reader
could apply, including a check that coverage is near 0.95, so the study cannot
fail and therefore says nothing. The same holds for a fit index drawn from a
normal centred on a value the reader is hoping to see.

**Now.** `core/measurement.py` fits the model by maximum likelihood
(`fit_cfa`), builds the expected (Fisher) information from central differences
of the model-implied covariance (`expected_information`), and forms a Wald
interval per replicate. `sample_size_replicate` returns a 0 or 1 per parameter
for whether that interval covered the truth, and
`analyses/paper1_validation.py` averages them. A replicate that does not
converge contributes a zero to the convergence rate and nothing else; it is not
replaced by a filled-in value.

`tests/test_core_measurement.py::test_coverage_is_counted_rather_than_assumed`
checks that both outcomes occur, so the reported number is not a constant. The
realised worst-case coverage over the twelve published design points runs from
0.92 to 0.977.

---

## 2. "Cox fit" computed as a difference of group means

**Draft.** `beta = mean(z[event == 1]) - mean(z[event == 0])`.

**Why it matters.** That statistic is not the Cox coefficient. It ignores
follow-up time and the risk sets entirely, and it is biased away from the null.

**The number.** On the package's own simulation, 20000 people, true coefficient
`-0.12`, 22 per cent events, seed 20260815:

| estimator | value |
| --- | --- |
| partial likelihood, Breslow ties | **-0.1244** |
| difference of group means | **-0.1400** |

The difference of means is 17 per cent further from the truth, in the direction
that overstates the effect. Since the whole point of paper 2 is that measurement
error *attenuates* the coefficient toward the null, an estimator with a bias
away from the null would work against the result being demonstrated.

**Now.** `core/prognostic.py::cox_fit` maximises the Cox partial likelihood by
Newton-Raphson with Breslow's handling of ties, using reverse cumulative sums for
the risk-set totals so each iteration is linear in the sample size. It returns
the coefficient, the standard error from the inverse observed information, the
iteration count, and whether it converged. With no event or a non-positive
information it returns `nan` rather than a number that looks like an estimate.

`tests/test_core_prognostic.py::test_cox_fit_beats_a_difference_of_group_means`
asserts both values above.

---

## 3. A Fokker-Planck solver that renormalises the density every step

**Draft.** After each time step, `p /= p.sum() * dx`, followed by a check that
total mass is conserved.

**Why it matters.** Renormalising forces the mass to one, so the mass check then
reports machine epsilon whatever the flux computation did. A sign error in the
advective flux, a wrong boundary condition, or a step past the stability limit
would all leave the check passing. The check is not weak, it is vacuous.

**Now.** `core/dynamical.py::FokkerPlanck` uses a conservative finite-volume
scheme: upwind advective flux and central diffusive flux at cell faces, both
boundary fluxes set to zero, and the update written as the difference of face
fluxes. Mass conservation is then a property of the discretisation, and the
density is never rescaled. `stable_time_step` supplies a step that honours both
the diffusive and the advective limit of the explicit scheme; the advective limit
binds at coarse grids, where the drift is large at the domain edges.

**The numbers.** Over 25 time units with no renormalisation:

| cells | mass drift | minimum density | L1 error vs stationary |
| --- | --- | --- | --- |
| 100 | 3.3e-16 | 4.2e-44 | 0.072559 |
| 200 | 8.9e-16 | 1.5e-44 | 0.037667 |
| 400 | 2.2e-16 | 9.2e-45 | 0.019239 |
| 800 | 6.7e-16 | 7.1e-45 | 0.009725 |

Grid convergence order in L1: 0.967, consistent with the first-order upwind
flux. The drift is at the level of floating-point rounding because the scheme
conserves mass, not because anything rescaled it.

---

## 4. A rating generator whose realised endorsement is not the requested one

**Draft.** Endorsement `pi` defined on the continuous scale, as
`P(N(mu, sigma) >= 7)`, then ratings simulated as
`clip(round(N(mu, sigma)), 1, 9)` and the endorsement evaluated after rounding.

**Why it matters.** Rounding sends a draw to 7 or above whenever it is at least
6.5, so the endorsement of the reported ratings is not the endorsement of the
latent draws. Every error rate is then computed for a different item from the one
requested, and the false-drop rate of a "0.90 item" is really the false-drop rate
of a much stronger item.

**The number.** At `sigma = 1.2`, a request for `pi = 0.90` defined on the
continuous scale gives `mu = 7 + 1.2 * Phi^-1(0.90) = 8.5379`. The endorsement
realised after rounding is `1 - Phi((6.5 - 8.5379) / 1.2) = ` **0.9553**, not
0.90. The rule is then being scored on an item far stronger than the one
described.

**Now.** `core/consensus.py::endorsement_probability` defines endorsement on the
scale the rule actually sees:

```
pi = 1 - Phi((cut - 0.5 - mu) / sigma)
```

and `mu_for_endorsement` inverts it in closed form,
`mu = cut - 0.5 + sigma * Phi^-1(pi)`. For `pi = 0.90` at `sigma = 1.2` this
gives `mu = 8.0379`, against the 8.0363 the published Monte Carlo inversion found
from 400000 draws per grid point. The closed form agrees and carries no
simulation error.

---

## 5. Domain reliability summing the whole residual diagonal for every domain

**Draft.** Each domain's reliability computed from `sum(theta)` over all
indicators, not over the indicators in that domain.

**Why it matters.** Every domain then gets the same denominator, so the three
reported reliabilities differ only through their numerators. A domain measured
with four sharp indicators and one measured with four noisy ones come out closer
together than they are, and the ordering between them can invert. The number is
also wrong in level, not merely in comparison: the residual variance charged to
each domain is three times what belongs to it.

**Now.** The package computes determinacy from the full model-implied covariance
rather than from a per-domain shortcut. `rho_determinacy` forms
`Sigma = Lambda Phi Lambda' + diag(theta)` and evaluates
`rho_D^2 = (Lambda gamma)' Sigma^-1 (Lambda gamma)`, so each indicator's residual
enters exactly once and in the right place. Where a per-domain quantity is
wanted, `core/ctt.py` supplies alpha and omega computed on that domain's items
alone.

`tests/test_core_measurement.py::test_general_form_agrees_with_the_balanced_helper`
and the determinacy regression tests hold the result to the published four
decimal places at every point of the published table.

---

## 6. `stats` used at class level but imported only under `__main__`

**Draft.** The dynamical module referenced `scipy.stats` in a class body while
importing it inside an `if __name__ == "__main__":` block.

**Why it matters.** The module raises `NameError` on import for every caller
except the one that runs it as a script. Any test, notebook or downstream module
would fail, and the failure would appear at import time rather than where the
name is used.

**Now.** `core/dynamical.py` imports `numpy`, `scipy.stats` and
`scipy.optimize.minimize_scalar` at module level, and no module in the package
has an `if __name__ == "__main__":` block. The entry points are
`scripts/run_all.py` and `scripts/regenerate_figures.py`, which import the
library rather than being part of it.

---

## Two smaller changes worth recording

**Replicate seeding.** The published scripts derived a replicate's generator
partly from `hash(scenario_name)`. Python salts string hashes per process, so two
runs of the same script on the same machine draw different numbers. The package
derives every generator from a seed sequence over
`(seed, scenario index, sample size, replicate)`, which is reproducible across
processes and machines. The consequence is that the simulation results here do
not match the published run digit for digit; they match within Monte Carlo error,
and the regression tests state intervals rather than pretending to an equality
that does not hold.

**The attenuation headline.** The published run reported the attenuation ratio at
each of eight design points, from 0.805 to 0.859 around a predicted 0.8361. At
120 replicates the ratio at a single design point carries several per cent of
Monte Carlo error, which is larger than the spread being interpreted. The package
also reports `attenuation_ratio_pooled`, the ratio of mean coefficients over all
960 replicates, which is the stable statistic to compare with `rho_D`. It comes
out at 0.843.

---

## What was kept from the draft

The module layout, the names `MeasurementModel`, `AccessModel`, `HTAModel`,
`LinearSDE` and `FokkerPlanck`, the `run(cfg=None, outdir=None)` shape shared by
every analysis, the split between `core` and `analyses`, and the idea of a frozen
config per paper. Those decisions are good and the repository follows them.
