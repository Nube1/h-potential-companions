"""Appendix A: the axioms every admissible aggregator must satisfy."""
import math
from itertools import product
import pytest
from hpv import crf

GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
RHOS = [-8.0, -2.0, -0.5, 0.0, 0.5, 1.0]   # includes rho > 0:
#        the sensitivity range App. E pre-specifies, and where a
#        short-circuit bug lived until the grid was widened to it
PROFILES = list(product(GRID, repeat=3))


@pytest.mark.parametrize("rho", RHOS)
def test_axiom_normalisation(rho):
    """Normalisation: H(1,1,1)=1, and H(0,s,r)=0 for every rho because the
    capacity factor multiplies the whole expression."""
    assert crf.H(1, 1, 1, rho) == pytest.approx(1.0)
    for s, r in product(GRID, repeat=2):
        assert crf.H(0, s, r, rho) == pytest.approx(0.0)


@pytest.mark.parametrize("rho", RHOS)
def test_contextual_mean_is_continuous_at_a_zero_enabler(rho):
    """A zero enabler must be the limit of small enablers, not a special case.

    This is the test that would have caught the short-circuit bug: for
    0 < rho <= 1 the code returned the arithmetic mean at s = 0 regardless of
    rho, giving 0.5 where 0.25 is correct, and a jump at s = 0.001.
    """
    for r in (0.25, 0.5, 1.0):
        at_zero = crf.contextual_mean(0.0, r, rho)
        # the value at zero must be the limit of a sequence approaching it
        gaps = [abs(crf.contextual_mean(eps, r, rho) - at_zero)
                for eps in (1e-4, 1e-6, 1e-8, 1e-10)]
        assert gaps == sorted(gaps, reverse=True)      # monotonically closing
        assert gaps[-1] < 1e-4                          # and reaching it
    # and the value itself, at the case the bug got wrong
    assert crf.contextual_mean(0.0, 1.0, 0.5) == pytest.approx(0.25)
    assert crf.contextual_mean(0.0, 1.0, 1.0) == pytest.approx(0.5)
    assert crf.contextual_mean(0.0, 1.0, -2.0) == pytest.approx(0.0)


def test_essential_enablement_holds_only_for_non_positive_rho():
    """The non-compensation axiom bites only on the admissible range.

    For rho <= 0 a failed enabler drives realised capability to zero. For
    rho > 0 it does not, which is exactly why Chapter 4 restricts the family
    to rho <= 0 and why rho > 0 members are reported as excluded comparators
    rather than as candidates.
    """
    for rho in (-8.0, -2.0, -0.5, 0.0):
        assert crf.H(1.0, 0.0, 1.0, rho) == pytest.approx(0.0)
    for rho in (0.5, 1.0):
        assert crf.H(1.0, 0.0, 1.0, rho) > 0.0


@pytest.mark.parametrize("rho", RHOS)
def test_axiom_capacity_boundedness(rho):
    """Axiom A.3(iv): 0 <= H(p,s,r) <= p. Context cannot manufacture capacity."""
    for p, s, r in PROFILES:
        h = crf.H(p, s, r, rho)
        assert -1e-12 <= h <= p + 1e-12


@pytest.mark.parametrize("rho", RHOS)
def test_axiom_weak_monotonicity(rho):
    """Axiom A.3(ii): improving any domain never lowers the score."""
    step = 0.05
    for p, s, r in PROFILES:
        h = crf.H(p, s, r, rho)
        for dp, ds, dr in ((step, 0, 0), (0, step, 0), (0, 0, step)):
            h2 = crf.H(min(1, p + dp), min(1, s + ds), min(1, r + dr), rho)
            assert h2 + 1e-12 >= h


@pytest.mark.parametrize("rho", RHOS)
def test_axiom_continuity(rho):
    """Axiom A.3(i): no jumps. A small input change gives a small output change."""
    eps = 1e-6
    for p, s, r in product([0.2, 0.5, 0.8], repeat=3):
        base = crf.H(p, s, r, rho)
        moved = crf.H(p + eps, s + eps, r + eps, rho)
        assert abs(moved - base) < 1e-3


def test_proposition_dominance_preservation():
    """Proposition A.1: weak monotonicity preserves dominance."""
    for a in PROFILES:
        for b in PROFILES:
            if all(ai >= bi for ai, bi in zip(a, b)):
                assert crf.H(*a) + 1e-12 >= crf.H(*b)


def test_theorem_ordinal_not_cardinal():
    """Theorem A.2: any strictly increasing g preserves the ordering but
    changes differences, so a score has no intrinsic cardinal unit."""
    g = lambda h: h ** 3
    pairs = [(a, b) for a in PROFILES[:40] for b in PROFILES[:40]]
    order_preserved = all(
        (crf.H(*a) >= crf.H(*b)) == (g(crf.H(*a)) >= g(crf.H(*b)))
        for a, b in pairs)
    assert order_preserved
    a, b, c = (0.9, 0.9, 0.9), (0.6, 0.6, 0.6), (0.3, 0.3, 0.3)
    d1 = crf.H(*a) - crf.H(*b)
    d2 = crf.H(*b) - crf.H(*c)
    assert not math.isclose(d1 / d2, (g(crf.H(*a)) - g(crf.H(*b)))
                            / (g(crf.H(*b)) - g(crf.H(*c))), rel_tol=1e-3)


def test_proposition_additive_compensation():
    """Proposition A.3: additivity enforces an exact trade. Equal weights let
    a gain in biology cancel a loss in structural freedom one for one."""
    w = (1 / 3, 1 / 3, 1 / 3)
    before = crf.H_additive(0.30, 0.70, 0.20, w)
    after = crf.H_additive(0.30 + 0.15, 0.70, 0.20 - 0.15, w)
    assert after == pytest.approx(before)


def test_worked_example_additive_perverse():
    """Worked Example A.2: the additive index scores a man higher after his
    inhibitor clears and his specialist funding is withdrawn."""
    w = (1 / 3, 1 / 3, 1 / 3)
    rationed = crf.H_additive(0.30, 0.70, 0.20, w)
    cleared_but_stranded = crf.H_additive(0.60, 0.70, 0.04, w)
    assert rationed == pytest.approx(0.40)
    assert cleared_but_stranded == pytest.approx(0.4467, abs=1e-4)
    assert cleared_but_stranded > rationed
    # the CRF gives the opposite sign, which is the point of the example
    assert crf.H(0.60, 0.70, 0.04) == pytest.approx(0.1004, abs=1e-4)
    assert crf.H(0.30, 0.70, 0.20) == pytest.approx(0.1122, abs=1e-4)
    assert crf.H(0.60, 0.70, 0.04) < crf.H(0.30, 0.70, 0.20)


def test_proposition_ranking_reversal():
    """Proposition A.7: additive and bottleneck rules can rank two profiles
    in opposite orders, so the aggregation rule is not a technical detail."""
    x, y = (1.0, 1.0, 0.2), (1.0, 0.55, 0.55)
    arith = lambda z: z[0] * 0.5 * (z[1] + z[2])
    bottle = lambda z: z[0] * min(z[1], z[2])
    assert arith(x) > arith(y)
    assert bottle(x) < bottle(y)


def test_proposition_relational_denies_ceiling():
    """Proposition A.8: the relational aggregator is bounded, monotone and
    normalised, and gives a positive score at zero biomedical capacity."""
    for p, s, r in PROFILES:
        v = crf.H_relational(p, s, r)
        assert -1e-12 <= v <= 1 + 1e-12
    assert crf.H_relational(1, 1, 1) == pytest.approx(1.0)
    assert crf.H_relational(0.0, 0.8, 0.9) > 0.0      # ceiling axiom violated
    assert crf.H(0.0, 0.8, 0.9) == pytest.approx(0.0)  # CRF respects it


def test_definition_deprivation_count():
    """Definition A.13 and Worked Example A.4: counting, not smoothing. At a
    threshold of 0.5 Profile A is deprived in two domains of three, Profile B rationed
    in two of three, and Profile B funded in one of three."""
    assert crf.deprivation_count((0.55, 0.35, 0.40)) == pytest.approx(2 / 3)
    assert crf.deprivation_count((0.30, 0.70, 0.20)) == pytest.approx(2 / 3)
    assert crf.deprivation_count((0.30, 0.70, 0.85)) == pytest.approx(1 / 3)
    assert crf.deprivation_count((0.9, 0.9, 0.9)) == pytest.approx(0.0)
    # counting discards magnitude: 0.49 and 0.05 score identically
    assert (crf.deprivation_count((0.49, 0.9, 0.9))
            == crf.deprivation_count((0.05, 0.9, 0.9)))


def test_worked_example_four_distinct_objects_for_profile_a():
    """Worked Example A.7: the aggregate is a rule, not a measurement.
    Profile A's profile is fixed, and the number attached to it moves with the
    substitution parameter alone. Nothing about her changes between rows."""
    profile = (0.55, 0.35, 0.40)
    geometric = crf.H(*profile, 0.0)
    harmonic = crf.H(*profile, -1.0)
    bottleneck = crf.H(*profile, -800.0)
    assert geometric == pytest.approx(0.2058, abs=1e-4)
    assert harmonic == pytest.approx(0.2053, abs=1e-4)
    assert bottleneck == pytest.approx(0.1925, abs=1e-4)
    assert geometric > harmonic > bottleneck
    # the additive rule on the same profile is a different object again
    assert crf.H_additive(*profile) == pytest.approx(0.4333, abs=1e-4)
    # and the profile itself is unchanged by any of this
    assert profile == (0.55, 0.35, 0.40)
