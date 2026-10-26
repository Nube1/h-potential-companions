# Validation record

Version 0.2.0, 26 October 2026.

- Test suite: 41 tests pass on Python 3.10 and 3.13 (pytest). Line coverage of the `hpotential` package is 95%.
- Property tests (Hypothesis) check, on random inputs, the capacity ceiling, essential context domains for rho <= 0, unit equivariance, the CES mean lying between min and max and rising with rho, the exact log-gap identity, and the EDE lying between the minimum and the mean.
- Regression tests regenerate every result table into a temporary folder and compare it with `results/`. They also check each number in the three manuscript tables against the generated value.
- Results: 4,000 synthetic observations, fixed seed 20261025. Written to 12 significant digits, the CSV hashes are identical under NumPy 2.1.3 and 2.5.3 on Linux, and all values agree with the original Windows run (NumPy 2.3.5) after the same rounding.
- PDFs: built with pdflatex from the generated LaTeX, with no overfull lines over 5pt. Pages were rendered and checked by eye. The ReportLab fallback also builds.
- `analysis/verify_delivery.py` checks the result hashes, the quoted drift bias and coverage, and that each PDF has an Abstract and References.

Fixes found while refining:

- The allocator rejected feasible portfolios when floating-point round-off pushed a domain a hair above 1 (for example 0.1 + 0.34 + 0.56). It now allows 1e-12 of slack and clips only that round-off. A test covers the case.
- Paper 2, Table 1, social drift domain 3: coverage is 3827/4000 = 0.95675 and is now quoted as 0.9568, matching the half-up rounding used elsewhere in the table.

No patient-level validation, SEM fitting, survival-model execution, journal acceptance or institutional approval is represented by these checks.
