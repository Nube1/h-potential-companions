# H-Potential thesis — V3 Chapter 2 and model-family revision

This revision makes the draft auditable and compilation-ready without inventing
empirical findings. It aligns the thesis explicitly to the revised 2026 NHATS
proposal, flags the conflicting superseded protocol bundled in Draft 1, removes
claims about absent software, and adds a reproducible-analysis scaffold.

## Build

```sh
latexmk -pdf -interaction=nonstopmode thesis.tex
```

## Analysis status

The thesis is **not submission-ready** until the NHATS extract, a frozen variable
dictionary, and final analysis outputs are provided. Run:

```sh
python analysis/00_preflight.py
```

The script stops safely if required inputs are absent. The whole analysis is in
Python; there is no R step. Results reach the thesis only through
`hpotential thesis-export`, which writes `generated/` and leaves every
placeholder in place that the run did not fill.

```sh
HPOTENTIAL_DATA=path/to/extract.csv python analysis/03_empirical_comparison.py
```

Chapter 2 now traces the argument from the WHO conceptual relation through the
H-Potential construct, candidate additive, interaction, multiplicative and CES
forms, reflective-versus-formative measurement, and prespecified falsification.
The same candidate family is implemented in `hpotential.candidates`,
stress-tested in Python, and compared empirically by the pipeline.

The mathematical novelty extension is in Appendix E. Its illustrative,
non-empirical experiments can be reproduced with:

```sh
python3 analysis/02_crf_simulation.py
```

This regenerates the three figures and CSV tables under
`outputs/crf_simulation/`. The empirical head-to-head comparison is run by
`analysis/03_empirical_comparison.py`; it requires the final NHATS analytic
extract, and the CRF parameters are frozen before any outcome is used.

See `THESIS_GRILL_V2.md` for the examiner-style audit and completion roadmap.
