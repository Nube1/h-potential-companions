# Contributing

Thank you for looking at this. The rules below exist because the repository
backs six manuscripts: a change here can change a published number.

## Before you open a pull request

```bash
pip install -e ".[full,dev]"
pre-commit install
make lint
make test-all
```

`make test-all` runs the fast suite and then the regressions against every
published value. Both must pass.

## The one rule that matters

**Every number the package produces must be computed.**

No placeholder that looks like a result. Concretely, and these are all things
that have appeared in drafts of this code:

- Do not generate a fit index or a coverage rate from a random number generator.
- Do not hardcode a rate that the code is supposed to estimate.
- Do not substitute a cheaper statistic for the one named. A difference of group
  means is not a Cox coefficient.
- Do not normalise away the quantity a check is testing. A density solver that
  rescales to unit mass every step makes its own mass-conservation check
  vacuous.
- Do not let a generative model realise something other than what was asked for.
  If a function takes an endorsement of 0.90, the simulated ratings must have an
  endorsement of 0.90 on the scale the rule sees.

`docs/DRAFT_REVIEW.md` has the full list with the numbers showing why each one
matters.

## Style

- Type hints on every public function.
- Numpy-style docstrings, with a `Notes` section for assumptions and a
  `Limitations` section where the method has a real one. Say what a function
  does **not** do.
- `ruff` and `black` at 100 columns. `make format` fixes most things.
- Every stochastic function takes an explicit `rng` or `seed`. No
  `np.random.seed` in library code, ever.
- Raise `ValueError` on bad input. Do not return a filled-in value.

## Tests

- A unit test for each new public function, covering the normal case, an edge
  case, and the error it raises.
- A determinism test if the function draws random numbers: same seed, same
  output.
- If a test bounds a simulated quantity, the bound has to be tight enough to
  fail when the method changes. A test that passes whatever the code does is
  worse than no test, because it reads as evidence.

## Changing a published value

If a change moves a value in `tests/fixtures/`, say so in the pull request and
say why. Do not widen a tolerance to make a test pass. Two things can be true:
the code was wrong, or the published number was wrong. The second needs a
correction to the manuscript before the fixture is touched.

## Reporting a problem

Use the issue templates. If a published value is affected, name it.
