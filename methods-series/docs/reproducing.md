# Reproducing every published number

Every number in the six manuscripts traces to one function in this package and
one entry in `tests/fixtures/`. This page says which, and how to run it.

## Setup

```bash
git clone https://github.com/Nube1/h-potential-companions.git
cd h-potential-companions/methods-series
pip install -e ".[full,dev]"
```

Python 3.10 or later. The `full` extra brings in `statsmodels` and
`factor_analyzer`, which the differential-item-functioning and factor-analysis
paths use.

## The short version

```bash
make test-all     # fast suite, then the published-value regressions
make figures      # every figure at published settings
```

`make test-all` is the check that matters. It runs
`tests/test_published_values.py`, which reads the expected values from
`tests/fixtures/*.json` (the files the published analysis scripts wrote) and
asserts that the package still produces them.

## Running one paper

```bash
python scripts/run_all.py --papers 1 --outdir out
python scripts/regenerate_figures.py --paper 1 --all --outdir figs
```

Or from python:

```python
from hpotential.analyses import paper1_validation
from hpotential.config import Paper1Config

results = paper1_validation.run(Paper1Config(), outdir="out")
paper1_validation.figures(results, "figs")
```

Change any field of the config to vary the design. The defaults are the
published values.

## Quick mode

```bash
python scripts/run_all.py --quick --outdir /tmp/hpq
```

Quick mode cuts replicate counts, grids and Monte Carlo draws so the whole
series finishes in under three minutes. **It does not reproduce the published
numbers.** The driver says so on the way in and on the way out, and every result
dictionary carries `quick_mode: true`. Use it to check that a change runs, not
to check what it computes.

## What each result costs to run

Timings on two cores.

| paper | analysis | full | quick |
| --- | --- | --- | --- |
| 1 | `paper1_validation` | about 3 minutes | 4 seconds |
| 2 | `paper2_prognostic` | about 65 seconds | 1 second |
| 3 | `paper3_consensus` | about 10 seconds | 2 seconds |
| 4 | `paper4_dynamics` | about 12 seconds | 2 seconds |
| 5 | `paper5_access` | under 1 second | under 1 second |
| 6 | `paper6_hta` | about 1 second | under 1 second |

## Result by result

### Paper 1: the determinacy ceiling

| result | function | published value |
| --- | --- | --- |
| ceiling `rho_D^inf` | `core.measurement.determinacy_ceiling` | 0.877819 |
| second-order signal `c` | same | 3.3586 |
| numerical check at 400 indicators per domain | `core.measurement.second_order_determinacy(400, 0.85)` | 0.877594 |
| signal needed for `rho_D = 0.90` | `core.measurement.required_second_order_signal(0.90)` | 4.2632 |

Closed form throughout, so these reproduce to the digit. The numerical check is
strictly below the ceiling, which is the point: the ceiling is a supremum
approached from below.

```python
from hpotential import determinacy_ceiling, second_order_determinacy
determinacy_ceiling()                  # (0.8778191930600775, 3.358562091503268)
second_order_determinacy(400, 0.85)    # 0.8775944308679847
```

### Paper 1: identification

| result | function | published value |
| --- | --- | --- |
| planned design, four loadings | `core.measurement.identification_report` | rank 27 of 27 |
| control (a), two indicators per domain | same, `per_domain=2` | rank 15 of 15, identified |
| control (b), two domains | same, two-element `gamma` | rank 17 of 18, **not** identified |

The smallest singular values and condition numbers reproduce to six significant
figures. Control (b) has to fail; a rank test that cannot detect an
unidentified model is not a test.

### Paper 1: sample size

`paper1_validation.run()` at 150 replicates over six sample sizes and two
scenarios. Reproduces within Monte Carlo error, not to the digit, for the reason
in the note on seeding below. The published run reported worst-case coverage
between 0.92 and 0.977 and maximum absolute relative bias below 0.014.

### Paper 2: attenuation and the events cost

| result | function | published value |
| --- | --- | --- |
| `rho_D` of the planned design | `paper2_prognostic.population_model(...).determinacy()` | 0.8361 |
| predicted attenuated coefficient | `core.prognostic.attenuated_beta` | -0.1003 |
| events inflation at the ceiling | `core.prognostic.events_inflation(0.877819)` | 1.2977 |
| events inflation at the planned design | `core.prognostic.events_inflation(0.8361)` | 1.4305 |
| attenuation ratio, simulated | `paper2_prognostic.run()` | 0.805 to 0.859 per point |

The first four are closed form. The last is 120 replicates at each of eight
design points; the package also reports `attenuation_ratio_pooled` over all 960,
which is the stable statistic and comes out at 0.843.

### Paper 3: the Delphi rule

| result | function | published value |
| --- | --- | --- |
| false drop, `pi = 0.90`, `n = 30`, IQR <= 1.5 | `core.consensus.operating_characteristics` | 0.692 |
| the same at IQR <= 2.0 | same, `iqr_threshold=2.0` | 0.023 |
| false drop by panel size, 12 to 60 | same | 0.381 to 0.810 |
| dispersion lemma, panels up to 10 | `core.consensus.iqr_lemma_violations` | 0 violations, max IQR 2 |

The lemma is exhaustive, not simulated: every multiset of ratings from 1 to 9
for panel sizes 1 to 10 is visited. The largest interquartile range among panels
meeting `I-CVI >= 0.78` is exactly 2, attained at `(7, 7, 9, 9)`. Under the
Weibull quantile rule the same enumeration finds hundreds of violations, so the
lemma is about a rule and a convention together.

### Paper 4: solvers

| result | function | published value |
| --- | --- | --- |
| RK4 observed order | `paper4_dynamics.run()` | 4.052 |
| Euler-Maruyama strong order, additive | same | 1.030 |
| Euler-Maruyama strong order, multiplicative | same | 0.530 |
| Fokker-Planck mass drift | same | below 9e-16 |
| Fokker-Planck L1 grid order | same | 0.967 |

The L1 errors reproduce to six significant figures because the Fokker-Planck run
is deterministic. The strong orders come out at 1.03 and 0.52 here against the
published 1.03 and 0.53; the difference is the reference Brownian path, which
depends on the generator.

### Paper 5: the costed rule

| result | function | published value |
| --- | --- | --- |
| closed form vs direct evaluation | `core.access.AccessModel.closed_form_check` | 6.4e-16 |
| disagreement rate, high-resource | `core.access.AccessModel.psa` | 0.486 |
| disagreement rate, low-resource | same | exactly 0 |

Both reproduce bit for bit: the analysis draws from a single seeded generator in
a fixed order.

The low-resource zero is not a rounded small number. Over the stated ranges
`T / A` exceeds one in every draw and so does `V`, so the two rules cannot
disagree there.

### Paper 6: the early assessment

| result | function | published value |
| --- | --- | --- |
| P(cost-effective on QALYs), central | `paper6_hta.run()` | 0.410 |
| mean incremental BMP / mean incremental QALY, central | same | 1.882 |
| per-person EVPI, central | same | 34874 |
| decision disagreement, central | same | 0.059 |

These reproduce bit for bit for the same reason as paper 5.

## A note on seeding

The published scripts derived a replicate's generator partly from
`hash(scenario_name)`. Python salts string hashes per process, so those runs are
not reproducible even on the same machine. This package derives every generator
from a seed sequence over `(seed, scenario index, sample size, replicate)`.

The consequence: results that pass through that derivation, the paper 1
sample-size study and the paper 2 pipeline, reproduce within Monte Carlo error
rather than to the digit. The regression tests state intervals for those, and
the intervals are wide enough to hold the sampling error at the published
replicate counts and narrow enough to fail if a method changes.

Everything closed form, and everything drawn from a single generator in a fixed
order, reproduces exactly.

## If a regression test fails

Two possibilities, and both need a person:

1. The package changed what it computes. Find the change; it was probably not
   intended.
2. A published number was wrong. Then the fixture is wrong too, and the
   manuscript needs a correction before the fixture is edited.

Do not widen a tolerance to make a test pass.
