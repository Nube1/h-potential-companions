"""Worked examples that appear in the chapters rather than the appendices.

One test per numbered Worked Example. Each asserts the figures the chapter
prints, so that editing a parameter in the text without editing the code, or
the reverse, fails the build.
"""
import math
import pytest
from hpv import crf
from hpv import haemophilia as hem
from hpv import prognostic as prog


# --------------------------------------------------------------------------
# Chapter 1
# --------------------------------------------------------------------------
def test_worked_example_intro_mrs_a_geometric_versus_additive():
    """Worked Example 1.1: 3.14, 4.71 and 1.78 good years under the CRF
    against 3.15, 5.25 and 3.99 under the additive rule. At S = 0.05 the
    additive rule overstates Mrs A by a factor of 2.24."""
    P = 8.4
    expected = {
        (0.35, 0.40): (3.14, 3.15),
        (0.35, 0.90): (4.71, 5.25),
        (0.05, 0.90): (1.78, 3.99),
    }
    for (s, r), (crf_years, add_years) in expected.items():
        geo = crf.contextual_mean(s, r, rho=0.0)
        add = crf.contextual_mean(s, r, rho=1.0)
        assert P * geo == pytest.approx(crf_years, abs=5e-3)
        assert P * add == pytest.approx(add_years, abs=5e-3)

    # the two rules agree when the enablers are balanced and diverge otherwise
    balanced = crf.contextual_mean(0.35, 0.40, 1.0) / crf.contextual_mean(0.35, 0.40, 0.0)
    collapsed = crf.contextual_mean(0.05, 0.90, 1.0) / crf.contextual_mean(0.05, 0.90, 0.0)
    assert balanced == pytest.approx(1.00, abs=5e-3)
    assert collapsed == pytest.approx(2.24, abs=5e-3)

    # and the same profile on the unit scale
    assert crf.H(0.55, 0.35, 0.40) == pytest.approx(0.2058, abs=5e-4)


def test_worked_example_intro_mr_b_funded_versus_rationed():
    """Worked Example 1.2: H = 0.231 funded and 0.112 rationed, a ratio of
    2.06, with the whole of the log gap on the structural coordinate."""
    funded, rationed = (0.30, 0.70, 0.85), (0.30, 0.70, 0.20)
    hf, hr = crf.H(*funded), crf.H(*rationed)
    assert hf == pytest.approx(0.231, abs=5e-4)
    assert hr == pytest.approx(0.112, abs=5e-4)
    assert hf / hr == pytest.approx(2.06, abs=5e-3)

    # in the good-years units used earlier in the chapter
    assert 4.0 * crf.contextual_mean(0.70, 0.85) == pytest.approx(3.09, abs=5e-3)
    assert 4.0 * crf.contextual_mean(0.70, 0.20) == pytest.approx(1.50, abs=5e-3)

    bio, soc, struct, total = crf.gap_decomposition(funded, rationed)
    assert bio == pytest.approx(0.0, abs=1e-12)
    assert soc == pytest.approx(0.0, abs=1e-12)
    assert struct == pytest.approx(0.723, abs=5e-4)
    assert total == pytest.approx(0.723, abs=5e-4)


# --------------------------------------------------------------------------
# Chapter 7
# --------------------------------------------------------------------------
def test_worked_example_hem_mr_b_funded_and_rationed():
    """Worked Example 7.1: scored from haemophilia indicators, Mr B reaches
    77 per cent of his biomedical potential funded and 37 per cent rationed."""
    P, S = 0.30, 0.70
    mult_f = crf.contextual_mean(S, 0.85)
    mult_r = crf.contextual_mean(S, 0.20)
    assert mult_f == pytest.approx(0.771, abs=5e-4)
    assert mult_r == pytest.approx(0.374, abs=5e-4)
    assert crf.H(P, S, 0.85) == pytest.approx(0.231, abs=5e-4)
    assert crf.H(P, S, 0.20) == pytest.approx(0.112, abs=5e-4)
    # the multiplier is the fraction of biomedical potential he reaches
    assert crf.H(P, S, 0.85) / P == pytest.approx(0.77, abs=5e-3)
    assert crf.H(P, S, 0.20) / P == pytest.approx(0.37, abs=5e-3)
    assert crf.H(P, S, 0.85) / crf.H(P, S, 0.20) == pytest.approx(2.06, abs=5e-3)


def test_worked_example_hem_threshold_moves_with_dose():
    """Worked Example 7.2: tolerable titres of 3.37, 5.00, 7.42 and 13.23 BU
    as the delivered input rises from 45 to 90, and no finite threshold at
    either end of the range."""
    REG = dict(alpha=4.0, K_I=5.0, beta=2.0, lam=1.0)
    expected = {45.0: 3.37, 60.0: 5.00, 75.0: 7.42, 90.0: 13.23}
    for dose, thr in expected.items():
        got = hem.inhibitor_threshold(numerator=dose, S_prot=20.0, **REG)
        assert got == pytest.approx(thr, abs=5e-3)

    # convex in the dose: the last increment buys more than the first
    t = {d: hem.inhibitor_threshold(numerator=d, S_prot=20.0, **REG)
         for d in (60.0, 75.0, 90.0)}
    assert t[75.0] - t[60.0] == pytest.approx(2.4, abs=0.05)
    assert t[90.0] - t[75.0] == pytest.approx(5.8, abs=0.05)
    assert t[90.0] - t[75.0] > t[75.0] - t[60.0]

    # a regimen that always protects, and one that never does
    with pytest.raises(ValueError):
        hem.inhibitor_threshold(numerator=110.0, S_prot=20.0, **REG)
    with pytest.raises(ValueError):
        hem.inhibitor_threshold(numerator=20.0, S_prot=20.0, **REG)


def test_worked_example_hem_intercept_shift_moves_cut_point():
    """Worked Example 7.3: an intercept 0.072 above the developmental value at
    a loading of 0.80 moves a cut-point from 0.28 to 0.19, and Mr B at 0.23
    crosses it."""
    lam, delta = 0.80, 0.072
    shift = delta / lam
    assert shift == pytest.approx(0.09, abs=5e-4)
    cut_dev = 0.28
    cut_hem = cut_dev - shift
    assert cut_hem == pytest.approx(0.19, abs=5e-4)

    mr_b = 0.23
    assert mr_b < cut_dev          # flagged against the developmental cut-point
    assert mr_b > cut_hem          # not flagged against the recalibrated one

    # the direction of the misclassification follows the sign of delta
    assert (cut_dev - shift) < cut_dev
    assert (cut_dev - (-delta) / lam) > cut_dev


# --------------------------------------------------------------------------
# Chapters 8 and 9
# --------------------------------------------------------------------------
def test_worked_example_disc_incremental_value_two_cases():
    """Worked Example 8.1: statistical distinguishability and clinical worth
    are separate findings. Case one gives 0.8 points and one person in 122;
    case two gives 9.4 points and one in 11."""
    base = 0.20

    r0 = prog.hazard_ratio_to_risk(base, 1.35)
    r1 = prog.hazard_ratio_to_risk(base, 1.40)
    assert r0 == pytest.approx(0.260, abs=1e-3)
    assert r1 == pytest.approx(0.268, abs=1e-3)
    assert r1 - r0 == pytest.approx(0.008, abs=1e-3)
    assert prog.number_needed(r0, r1) == pytest.approx(122, abs=3)

    r0 = prog.hazard_ratio_to_risk(base, 1.30)
    r1 = prog.hazard_ratio_to_risk(base, 1.90)
    assert r0 == pytest.approx(0.252, abs=1e-3)
    assert r1 == pytest.approx(0.346, abs=1e-3)
    assert r1 - r0 == pytest.approx(0.094, abs=1e-3)
    assert prog.number_needed(r0, r1) == pytest.approx(11, abs=1)


def test_worked_example_conc_null_result_at_low_reliability():
    """Worked Example 9.1: at reliability 0.55 a true HR of 0.80 is reported
    as 0.886 with an interval spanning 1, so the null is uninformative about
    an effect of that size. Reliability above about 0.67 would rule it out."""
    beta_true, se = -0.22, 0.075

    b_perfect = prog.attenuated_coefficient(beta_true, 1.00)
    lo, hi = math.exp(b_perfect - 1.96 * se), math.exp(b_perfect + 1.96 * se)
    assert math.exp(b_perfect) == pytest.approx(0.803, abs=1e-3)
    assert (lo, hi) == (pytest.approx(0.693, abs=1e-3),
                        pytest.approx(0.930, abs=1e-3))
    assert hi < 1.0                                   # excludes the null

    b_real = prog.attenuated_coefficient(beta_true, 0.55)
    lo, hi = math.exp(b_real - 1.96 * se), math.exp(b_real + 1.96 * se)
    assert math.exp(b_real) == pytest.approx(0.886, abs=1e-3)
    assert (lo, hi) == (pytest.approx(0.765, abs=1e-3),
                        pytest.approx(1.026, abs=1e-3))
    assert hi > 1.0                                   # includes it

    # the reliability at which the interval just clears 1
    needed = 1.96 * se / abs(beta_true)
    assert needed == pytest.approx(0.668, abs=1e-3)
    b_edge = prog.attenuated_coefficient(beta_true, 0.90)
    assert math.exp(b_edge + 1.96 * se) < 1.0
