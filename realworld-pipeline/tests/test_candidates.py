"""The boundary checks of the R script, now in Python."""

import numpy as np
import pytest

from hpotential.candidates import crf, crf_gap, hp_candidates


def a(x):
    return np.array([x], float)


def test_axiomatic_boundary_cases():
    assert crf(a(10), a(1), a(0))[0] == 10
    assert crf(a(10), a(0), a(0))[0] == 0
    assert crf(a(10), a(1), a(1))[0] == 0
    assert crf(a(10), a(0.8), a(0.2))[0] <= 10
    assert crf(a(10), a(0.8), a(0.2), rho=-1)[0] <= 10
    assert crf(a(10), a(0), a(0.2), rho=-1)[0] == 0


def test_multiplicative_candidate_is_the_crf():
    c = hp_candidates(a(10), a(0.8), a(0.2))
    assert c["multiplicative"][0] == pytest.approx(crf(a(10), a(0.8), a(0.2))[0])


def test_gap_decomposition_is_exact():
    x, y = (0.30, 0.70, 0.15), (0.30, 0.70, 0.80)
    parts = crf_gap(x, y)
    total = np.log(crf(a(x[0]), a(x[1]), a(x[2]))[0] / crf(a(y[0]), a(y[1]), a(y[2]))[0])
    assert sum(parts.values()) == pytest.approx(total)
    assert parts["biomedical"] == 0 and parts["social"] == 0
    assert parts["structural"] == pytest.approx(0.723, abs=5e-4)


def test_relational_is_bounded():
    rng = np.random.default_rng(0)
    p, s, b = rng.uniform(0, 1, (3, 500))
    rel = hp_candidates(p, s, b)["relational"]
    assert rel.between(0, 1).all()


def test_rejects_out_of_range():
    with pytest.raises(ValueError):
        crf(a(1), a(1.2), a(0))
    with pytest.raises(ValueError):
        crf(a(1), a(0.5), a(0.5), rho=0.5)
