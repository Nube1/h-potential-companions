"""Appendix E: characterisation, complementarity, decomposition, equity."""
import math
from itertools import product
import pytest
from hpv import crf

MRS_A = (0.55, 0.35, 0.40)
MR_B_FUNDED = (0.30, 0.70, 0.85)
MR_B_RATIONED = (0.30, 0.70, 0.20)


def test_running_cases_reproduce_the_thesis_numbers():
    """Every chapter quotes these three values. They must agree."""
    assert crf.H(*MRS_A) == pytest.approx(0.206, abs=5e-4)
    assert crf.H(*MR_B_FUNDED) == pytest.approx(0.231, abs=5e-4)
    assert crf.H(*MR_B_RATIONED) == pytest.approx(0.112, abs=5e-4)


def test_theorem_characterisation_constant_elasticities():
    """Theorem E.3: constant contextual elasticities pin down PS^a R^(1-a).

    Checked by confirming the elasticities really are constant across the
    domain, which is the assumption the theorem trades on.
    """
    alpha = 0.4
    seen = []
    for p, s, r in product([0.2, 0.5, 0.9], repeat=3):
        ep, es, er = crf.elasticities(p, s, r, alpha)
        seen.append((ep, es, er))
    for ep, es, er in seen:
        assert ep == pytest.approx(1.0)
        assert es == pytest.approx(alpha)
        assert er == pytest.approx(1 - alpha)


def test_proposition_balance_penalty_am_gm():
    """Proposition E.3: PS^a R^(1-a) <= PM with equality iff S = R.

    The gap is the non-compensation penalty an additive index throws away.
    """
    alpha = 0.5
    for s, r in product([0.05, 0.2, 0.45, 0.7, 0.95], repeat=2):
        M = alpha * s + (1 - alpha) * r
        geo = s ** alpha * r ** (1 - alpha)
        assert geo <= M + 1e-12
        if math.isclose(s, r):
            assert geo == pytest.approx(M)
        else:
            assert geo < M


def test_worked_example_penalty_measured():
    """Worked Example 4.2: four contexts all averaging 0.45. The penalty is
    zero when balanced and rises to 79 per cent when one enabler fails."""
    expected = {(0.45, 0.45): 0.0000,
                (0.60, 0.30): 0.0257,
                (0.70, 0.20): 0.0758,
                (0.89, 0.01): 0.3557}
    for (s_, r_), pen in expected.items():
        M = 0.5 * s_ + 0.5 * r_
        assert M == pytest.approx(0.45)
        got = M - crf.contextual_mean(s_, r_)
        assert got == pytest.approx(pen, abs=1e-4)
    # zero exactly when balanced, and monotone in the imbalance
    pens = [0.45 - crf.contextual_mean(a, b) for a, b in expected]
    assert pens[0] == pytest.approx(0.0, abs=1e-12)
    assert pens == sorted(pens)
    assert pens[-1] / 0.45 == pytest.approx(0.790, abs=2e-3)


def test_proposition_complementarity_cross_partials_positive():
    """Proposition E.5: both cross-partials strictly positive, which is the
    intersectionality claim written as mathematics. An additive index has
    cross-partials of zero and cannot express it."""
    for p, s, r in product([0.1, 0.4, 0.8], repeat=3):
        hps, hpr = crf.cross_partials(p, s, r)
        assert hps > 0
        assert hpr > 0


def test_theorem_gap_decomposition_is_exact():
    """Theorem E.6: the three parts sum to the total with no residual."""
    for a in ([0.55, 0.35, 0.40], [0.30, 0.70, 0.85], [0.8, 0.2, 0.6]):
        for b in ([0.30, 0.70, 0.20], [0.45, 0.45, 0.45], [0.1, 0.9, 0.3]):
            bio, soc, stc, total = crf.gap_decomposition(a, b)
            assert bio + soc + stc == pytest.approx(total, abs=1e-12)


def test_worked_example_gap_is_entirely_structural():
    """Worked Example 4.3: Profile B funded against Profile B rationed. Nothing about
    his body or his relationships differs, so the whole gap is structural."""
    bio, soc, stc, total = crf.gap_decomposition(MR_B_FUNDED, MR_B_RATIONED)
    assert bio == pytest.approx(0.0, abs=1e-12)
    assert soc == pytest.approx(0.0, abs=1e-12)
    assert stc == pytest.approx(0.723, abs=1e-3)
    assert total == pytest.approx(0.723, abs=1e-3)
    assert stc / total == pytest.approx(1.0)


def test_worked_example_policy_gradient_points_at_the_bottleneck():
    """Worked Example 4.4: for Profile A the marginal return is higher on the
    domain she is worse in. The gradient finds the bottleneck without being
    told that bottlenecks matter."""
    gp, gs, gr = crf.gradients(*MRS_A)
    assert gs == pytest.approx(0.294, abs=1e-3)
    assert gr == pytest.approx(0.257, abs=1e-3)
    assert gs > gr                       # S = 0.35 is worse than R = 0.40


def test_substitution_ordering_across_rho():
    """The power-mean ordering: lower rho punishes imbalance harder."""
    s, r = 0.70, 0.20
    vals = [crf.contextual_mean(s, r, rho) for rho in (1.0, 0.0, -2.0, -800.0)]
    assert vals == sorted(vals, reverse=True)
    assert vals[0] == pytest.approx(0.45)      # arithmetic
    assert vals[1] == pytest.approx(0.3742, abs=1e-4)  # geometric
    assert vals[-1] == pytest.approx(0.20)     # bottleneck


def test_worked_example_rho_reverses_the_ranking():
    """Worked Example 4.1: Profile A against Profile B rationed. The additive rule
    ranks him first, the bottleneck rule ranks her first, the geometric rule
    calls them equal. The data cannot choose between the three."""
    a, b = (0.35, 0.40), (0.70, 0.20)
    assert crf.contextual_mean(*a, 1.0) < crf.contextual_mean(*b, 1.0)
    assert crf.contextual_mean(*a, 0.0) == pytest.approx(
        crf.contextual_mean(*b, 0.0), abs=1e-9)
    assert crf.contextual_mean(*a, -800.0) > crf.contextual_mean(*b, -800.0)


def test_definition_equity_ede_and_limits():
    """Definition E.9: at eps=0 the EDE is the mean; at eps=1 the geometric
    mean; as eps grows it approaches the minimum."""
    v = [0.4, 0.8]
    assert crf.equity_ede(v, 0.0) == pytest.approx(0.60)
    assert crf.equity_ede(v, 1.0) == pytest.approx(math.sqrt(0.32))
    assert crf.equity_ede(v, 2.0) == pytest.approx(2 / (1 / 0.4 + 1 / 0.8))
    assert crf.equity_ede(v, 60.0) == pytest.approx(0.4, abs=1e-2)


def test_worked_example_atkinson_transfer_values():
    """Worked Example A.7: the equally distributed equivalent before and after
    moving 0.05 of health from the better-off group to the worse-off."""
    before, after = [0.4, 0.8], [0.45, 0.75]
    assert crf.equity_ede(before, 0.0) == pytest.approx(0.6000, abs=1e-4)
    assert crf.equity_ede(before, 1.0) == pytest.approx(0.5657, abs=1e-4)
    assert crf.equity_ede(before, 2.0) == pytest.approx(0.5333, abs=1e-4)
    assert crf.equity_ede(after, 0.0) == pytest.approx(0.6000, abs=1e-4)
    assert crf.equity_ede(after, 1.0) == pytest.approx(0.5809, abs=1e-4)
    assert crf.equity_ede(after, 2.0) == pytest.approx(0.5625, abs=1e-4)


def test_worked_example_transfer_only_registers_above_eps_zero():
    """Worked Example 4.5: moving 0.05 of health from the better-off group to
    the worse-off leaves the plain mean unchanged and raises every
    inequality-averse measure. That difference is the price of equality,
    stated in health."""
    before, after = [0.4, 0.8], [0.45, 0.75]
    assert crf.equity_ede(before, 0.0) == pytest.approx(crf.equity_ede(after, 0.0))
    for eps in (1.0, 2.0):
        assert crf.equity_ede(after, eps) > crf.equity_ede(before, eps)


def test_proposition_social_weight():
    """Proposition E.10: the marginal social weight is (EDE/H_i)^eps, so a
    decision-maker at eps=2 is committed to counting Profile A almost four times."""
    assert crf.social_weight(0.206, 0.40, 0.0) == pytest.approx(1.0)
    assert crf.social_weight(0.206, 0.40, 1.0) == pytest.approx(1.942, abs=1e-3)
    assert crf.social_weight(0.206, 0.40, 2.0) == pytest.approx(3.77, abs=1e-2)


# ---------------------------------------------------------------------------
# Chapter 2: results promoted from the literature synthesis.
# ---------------------------------------------------------------------------

def test_theorem_lit_choice_map_is_not_injective():
    """Theorem 2.1: observed functioning does not identify the capability set.

    Three people record the same functioning, 'did not attend', from three
    different opportunity sets. Any map from the record back to a set is
    wrong for at least two of them.
    """
    sets = {
        "no fare": frozenset({"home"}),
        "breathless": frozenset({"home", "attend"}),
        "informed refusal": frozenset({"home", "attend", "attend and shop"}),
    }
    chosen = {name: "home" for name in sets}          # chi_i(C_i)
    assert len(set(chosen.values())) == 1             # same F for all three
    assert len(set(sets.values())) == 3               # different C for all three

    # Any inverse map g takes the single observed value to a single set,
    # so it can match at most one of the three.
    for guess in sets.values():
        correct = sum(1 for c in sets.values() if c == guess)
        assert correct == 1
        assert len(sets) - correct == 2


def test_proposition_lit_heuristic_admits_negative_values():
    """Proposition 2.2: h = (LQ)S - B has range [-1, 1], so it cannot be a
    capability scale. Profile A's own profile puts her below zero."""
    h = lambda lq, s, b: lq * s - b
    assert h(0.55, 0.35, 0.60) == pytest.approx(-0.4075)
    assert h(0.0, 0.0, 1.0) == pytest.approx(-1.0)     # infimum, attained
    assert h(1.0, 1.0, 0.0) == pytest.approx(1.0)      # supremum, attained
    # the CRF on the same profile is non-negative and inside [0, p]
    assert 0.0 <= crf.H(0.55, 0.35, 0.40) <= 0.55
    # sign flips at B = (LQ)S
    assert h(0.55, 0.35, 0.1925) == pytest.approx(0.0, abs=1e-12)
    assert h(0.55, 0.35, 0.20) < 0.0


def test_proposition_lit_heuristic_negative_on_most_of_the_cube():
    """Proposition 2.2, second part: with L, Q, S, B independent uniform on
    [0,1], P(h < 0) = 1 - E[LQS] = 7/8."""
    import random
    rng = random.Random(20260902)
    n = 200_000
    hits = sum(1 for _ in range(n)
               if rng.random() > rng.random() * rng.random() * rng.random())
    assert hits / n == pytest.approx(0.875, abs=0.005)


def test_worked_example_lit_ranking_reversal():
    """Worked example: x = (1, 1, 0.2) and y = (1, 0.55, 0.55) are ranked
    oppositely by the additive rule and by the geometric and bottleneck
    rules. Cross-referenced to Appendix A."""
    x, y = (1.0, 1.0, 0.2), (1.0, 0.55, 0.55)
    arith = lambda z: z[0] * 0.5 * (z[1] + z[2])
    bottle = lambda z: z[0] * min(z[1], z[2])
    assert arith(x) == pytest.approx(0.60)
    assert arith(y) == pytest.approx(0.55)
    assert crf.H(*x) == pytest.approx(0.4472, abs=1e-4)
    assert crf.H(*y) == pytest.approx(0.55, abs=1e-4)
    assert bottle(x) == pytest.approx(0.20)
    assert bottle(y) == pytest.approx(0.55)
    assert arith(x) > arith(y)
    assert crf.H(*x) < crf.H(*y)
    assert bottle(x) < bottle(y)


def test_worked_example_lit_qaly_versus_crf():
    """Worked example: a QALY calculation and the CRF rank two interventions
    for Profile A in opposite orders, because they answer different questions."""
    base = crf.H(0.55, 0.35, 0.40)
    knee = crf.H(0.70, 0.35, 0.40)          # raises P
    transport = crf.H(0.55, 0.35, 0.70)     # raises R
    assert base == pytest.approx(0.2058, abs=1e-4)
    assert knee - base == pytest.approx(0.0561, abs=1e-4)
    assert transport - base == pytest.approx(0.0664, abs=1e-4)
    assert transport - base > knee - base
    # a utility gain of 0.05 held for eight years against no recorded
    # utility change at all
    qaly_knee, qaly_transport = 0.05 * 8, 0.0
    assert qaly_knee == pytest.approx(0.40)
    assert qaly_knee > qaly_transport
