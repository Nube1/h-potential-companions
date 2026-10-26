# hpotential

[![CI](https://github.com/Nube1/h-potential-companions/actions/workflows/ci.yml/badge.svg)](https://github.com/Nube1/h-potential-companions/actions/workflows/ci.yml)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Reproducible code for the H-potential health-capability methods series: a
second-order measurement model and the structural ceiling on its factor-score
determinacy, a two-stage prognostic pipeline and the attenuation that ceiling
forces on it, a modified Delphi consensus rule and the error rates it carries, a
continuous-time state model with verified solvers, a costed rule for choosing
between access and efficacy, and an early health technology assessment with
value of information. Every number the package produces is computed from a
stated model. Nothing is asserted from memory and nothing is filled in.

## Install

```bash
pip install -e ".[full,dev]"
```

Python 3.10 or later.

## Quickstart

```python
from hpotential import determinacy_ceiling, events_inflation

rho_inf, c = determinacy_ceiling()          # (0.877819, 3.358562)
events_inflation(rho_inf)                   # 1.2977: the events an oracle study saves
```

## Real cohort data

The repository now accepts CSV, TSV, Excel, and Parquet data through a schema-first ingestion layer. Raw participant data remain outside Git. The workflow validates types, ranges, categories, uniqueness, and missingness; removes identifier-role columns by default; fingerprints the source file; and writes an auditable analysis snapshot.

```bash
hpotential-data validate secure/cohort.xlsx \
  --schema data/templates/cohort_schema.json --sheet baseline

hpotential-data prepare secure/cohort.xlsx \
  --schema data/templates/cohort_schema.json --sheet baseline \
  --out derived/cohort_v1
```

See [`docs/real_data.md`](docs/real_data.md) and the editable schema/template in [`data/templates`](data/templates). A passing validation report is not evidence of scientific validity, consent, or regulatory compliance; those remain study-specific requirements.

Adding indicators cannot pass `rho_inf`, and reaching 0.90 would need
`c >= 4.2632` against the planned design's 3.3586. The consequence for a
prognostic study is the third line: at least 30 per cent more events than if the
construct were observed rather than measured.

## What each paper computes

| paper | analysis module | headline computed result |
| --- | --- | --- |
| 1. Theory and psychometrics | `analyses.paper1_validation` | determinacy ceiling `rho_D^inf = 0.877819` at `c = 3.3586`; 0.877594 at 400 indicators per domain, still below it; Jacobian rank 27 of 27 |
| 2. Prognostic pipeline | `analyses.paper2_prognostic` | attenuation ratio 0.843 pooled against a predicted `rho_D = 0.8361`; events inflation 1.4305 |
| 3. Disease-specific substitution | `analyses.paper3_consensus` | false-drop rate 0.692 for a well-endorsed item at IQR <= 1.5, and 0.023 at IQR <= 2.0; dispersion binds on 99 per cent of drops |
| 4. Continuous-time model | `analyses.paper4_dynamics` | RK4 order 4.05; Euler-Maruyama strong order 1.03 additive and 0.52 multiplicative; Fokker-Planck mass drift below 1e-15 with no renormalisation |
| 5. Access elasticity | `analyses.paper5_access` | closed form matches direct evaluation to 6.4e-16; costed and uncosted rules disagree on 48.6 per cent of high-resource draws and 0 per cent of low-resource ones |
| 6. Early assessment | `analyses.paper6_hta` | P(cost-effective on QALYs) 0.410 in the central scenario; incremental BMP is 1.88 times the incremental QALY gain |

## Notebooks

| paper | open in Colab |
| --- | --- |
| 1 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nube1/h-potential-companions/blob/main/methods-series/notebooks/01_measurement.ipynb) |
| 2 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nube1/h-potential-companions/blob/main/methods-series/notebooks/02_prognostic.ipynb) |
| 3 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nube1/h-potential-companions/blob/main/methods-series/notebooks/03_consensus.ipynb) |
| 4 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nube1/h-potential-companions/blob/main/methods-series/notebooks/04_dynamics.ipynb) |
| 5 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nube1/h-potential-companions/blob/main/methods-series/notebooks/05_access.ipynb) |
| 6 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nube1/h-potential-companions/blob/main/methods-series/notebooks/06_hta.ipynb) |

## Reproducing the published numbers

```bash
make test        # fast suite
make test-all    # and the regressions against every published value
make figures     # regenerate every figure
```

`tests/test_published_values.py` reads the expected values from
`tests/fixtures/*.json`, the files the published analysis scripts wrote, and
asserts that the package still produces them. Closed-form results are held to
the digit; simulation results are held to an interval wide enough for Monte
Carlo error at the published replicate counts and narrow enough to fail if a
method changes.

To run one paper:

```bash
python scripts/run_all.py --papers 1,2,3 --outdir out
python scripts/regenerate_figures.py --paper 1 --all --outdir figs
```

Add `--quick` to finish the whole series in under three minutes. Quick mode
cuts every replicate count and **does not reproduce the published numbers**; the
driver says so and every result carries `quick_mode: true`.

`docs/reproducing.md` maps each published number to the function that computes
it. `docs/api.md` is the interface. `docs/DRAFT_REVIEW.md` records what changed
from the co-author's draft architecture and why.

## Integrity note

**All bundled data in this repository are synthetic.** The software can ingest real data, but no participant data are distributed here. No participant has been
recruited, no Delphi panel has been run, and no cohort has been analysed.
Papers 1 to 3 are protocols: they specify what will be done, not what was found.
Papers 4 to 6 are methodological illustrations whose inputs are chosen to
demonstrate a workflow and are not estimates from any health system or any real
product.

Every result here is a property of a stated model or a stated decision rule. The
Delphi error rates describe the rule, not a panel. The health technology
assessment describes a workflow, not a therapy. Nothing in this repository is
evidence about any patient group.

## Determinism

Every stochastic function takes an explicit `seed` or `rng`. No library code
touches the global numpy random state, so the same seed gives bit-identical
output and two analyses in one process cannot disturb each other.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: `make lint` and `make test-all`
must pass, public functions need numpy-style docstrings stating their
assumptions, and a change that moves a published value needs a human decision
recorded in the pull request, not a widened tolerance.

## Citation

See [CITATION.cff](CITATION.cff), or:

```bibtex
@software{hpotential,
  title   = {hpotential: reproducible code for the H-potential health-capability
             methods series},
  author  = {{H-potential methods series authors}},
  year    = {2026},
  version = {1.1.0},
  license = {MIT},
  url     = {https://github.com/Nube1/h-potential-companions}
}
```

## License

MIT. See [LICENSE](LICENSE).
