# H-potential

[![CI](https://github.com/Nube1/h-potential-companions/actions/workflows/ci.yml/badge.svg)](https://github.com/Nube1/h-potential-companions/actions/workflows/ci.yml)

Research code, manuscripts and working record for the H-potential PhD thesis by Ronald Rethabile Malamule Mahomane (University of the Witwatersrand).

H-potential is a bounded measure of health capability. It combines biomedical potential, social modifiers and systemic burdens, and asks how much of a person's health capacity their circumstances let them use. The thesis develops it on the National Health and Aging Trends Study (NHATS), with South Africa and haemophilia care as later applications.

**Status.** Everything here is theory, proofs and synthetic computation. No result in this repository is a clinical validation, and no participant data are stored here.

## What is in the repository

| Folder | What it is | Version | Checked by CI |
|---|---|---|---|
| [companions/](companions/) | Three companion papers (capability realisation, measurement uncertainty, equity and policy) with the code that produces every number in them | 0.2.0 | Tests on Python 3.10 to 3.13 and Windows; results and LaTeX papers rebuilt and compared with the committed copies |
| [methods-series/](methods-series/) | `hpotential` package behind the six-paper methods series: measurement, prognosis, Delphi consensus, dynamics, access and health technology assessment, with Colab notebooks | 1.1.0 | Tests on Python 3.10 to 3.12; published-value regressions |
| [realworld-pipeline/](realworld-pipeline/) | Validation pipeline for the approved, de-identified NHATS extract: frozen measurement stage, Cox models, calibration and subgroup checks | 1.0.0 | Tests and an end-to-end run on synthetic data |
| [archive/](archive/) | Every earlier H-potential file, all versions: thesis drafts, proposal and protocol submissions, decks, poster, the five-paper series and earlier code | | Not run |

Each project is self-contained, with its own README, dependencies and tests. All three install a Python package called `hpotential`, so give each its own virtual environment.

## Quick start

Python 3.10 or newer. One command drives everything; each project gets its own environment in `.venvs/` because all three install a package called `hpotential`.

```sh
python tools/hp.py setup     # create the environments
python tools/hp.py test      # every test suite
python tools/hp.py build     # results, LaTeX papers and every figure
python tools/hp.py dist      # bundle papers and figures into dist/
python tools/hp.py all       # data guard, tests, build and bundle
```

`make setup`, `make test` and so on do the same where `make` is available.

## NHATS data

NHATS has no public data API: the files come from nhats.org after you register and accept its data use terms. Download the Round 1 sample-person file, the follow-up sample-person files and the latest tracker file (Stata, SAS, CSV or Parquet; the zip files as downloaded are fine) into one folder. Then plug them in any of three ways:

```sh
python tools/hp.py nhats --data-dir PATH/TO/NHATS          # one command
```

```python
from hpotential.api import analyse                           # from Python
summary = analyse("PATH/TO/NHATS", "outputs/nhats")
```

```sh
hpotential serve --port 8765                                 # local HTTP API
curl -X POST localhost:8765/analyse -d '{"source": "PATH/TO/NHATS", "run": "round1"}'
```

The HTTP API listens on this computer only, so the data never leave it. It answers `/check`, `/analyse`, `/runs`, `/runs/<run>` and `/runs/<run>/figures/<file>`, and returns aggregate numbers only.

Every route first checks each variable named in `realworld-pipeline/configs/nhats_layout.yaml` and stops if any is missing or carries an undeclared code. Those names follow NHATS naming and were written without the codebook to hand, so expect to correct a few on the first run. Participant-level files carry RESTRICTED in their names; the guard below keeps them out of git.

## Figures

| Set | Where | What it shows |
|---|---|---|
| Mathematical proofs | `companions/figures/` | Ten figures, one per result in the companion papers, each with a numeric check the build enforces |
| Methods series | `methods-series/figures/` | Determinacy, attenuation, Delphi, solver verification, costed access and value of information |
| Biostatistics | `realworld-pipeline/outputs/*/figures/` | Cohort flow, missingness, distributions, Kaplan-Meier and cumulative incidence, calibration, decision curves, discrimination, subgroups |
| Medical | same | Domain profiles by number of conditions, H by age and sex, condition contrasts |
| Economics and policy | same | Concentration curves by income, inequality aversion, the exact split of the income gap |

Every figure separates its series by shade and by line style or hatch, so it reads in black and white. Each folder has `FIGURES.md` and `figure_index.json` with captions and the numbers behind each figure. The demo figures in a fresh checkout come from made-up NHATS-shaped files and are not evidence about anyone.

## Working with git

- **Pre-commit checks.** `pip install pre-commit && pre-commit install` runs the linters and `tools/guard_data.py` on every commit. The guard refuses statistical data files, anything marked RESTRICTED, files under `data/raw/` or `data/processed/`, CSVs carrying the NHATS person ID, and large files outside Git LFS. CI runs the same guard on every push.
- **Releases.** Push a tag such as `v0.3.0` and GitHub Actions rebuilds everything and attaches the bundle to a GitHub release.
- **Large files.** The archive is stored in Git LFS, so a clone downloads small pointer files until you run `git lfs pull`.

## The archive

[archive/README.md](archive/README.md) indexes every file by kind, newest first, marks the latest version of each document and flags byte-for-byte copies. The latest versions of the main documents are:

| Document | Latest file |
|---|---|
| Thesis | `archive/thesis/pdf/H_Potential_Thesis_V12_5.pdf` and its source `archive/thesis/source/H_Potential_Thesis_V12_source_2.zip` |
| Wits PhD proposal | `archive/protocol/proposal/Hpotential_proposal_Wits_v14_revised_3 (2).pdf` |
| Protocol assessment deck | `archive/presentations/protocol-assessment/H_potential_Protocol_Assessment_8Oct2026_9 (1) (1).pptx` |
| Papers 1 to 5 | `archive/papers/paper-1/` to `paper-5/`, latest marked in the index |

The archive is a record. Edit the maintained projects, not the archived copies. Continuous integration skips the archive: it is large, and some of its paths are too long for Windows checkouts.

## Data

No participant data are stored here. The real-world pipeline runs on an approved, de-identified NHATS extract that stays outside the repository; its tests and CI use synthetic data only. Files the pipeline writes with `RESTRICTED` in the name hold participant-level output and must not be committed.

## Licence and citation

The projects carry their own licences (MIT for methods-series and realworld-pipeline) and citation files. The manuscripts in `companions/papers/` and the documents in `archive/` are unpublished drafts and are not licensed for reuse.
