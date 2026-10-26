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

Python 3.10 or newer. From the repository root:

```sh
cd companions
python -m pip install -r requirements-dev.txt
python -m pytest
python analysis/reproduce.py
```

The other projects follow the same pattern: `cd` into the folder, `pip install -e ".[dev]"` (or `".[full,dev]"` for methods-series), then `pytest`. Each folder's README gives the details.

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
