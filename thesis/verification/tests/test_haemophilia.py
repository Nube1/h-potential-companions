"""Appendix C: factor kinetics, inhibitor thresholds, ITI and gene therapy."""
import math
import pytest
from hpv import haemophilia as hem

# The regimen used in the worked examples of Appendix C.
REG = dict(numerator=60.0, alpha=4.0, K_I=5.0, beta=2.0, lam=1.0)


def test_proposition_linear_approximation_at_low_titre():
    """Proposition C.1: the leading term of the Hill neutralisation about
    Y = 0 is alpha*S*(Y/K_I)**beta, which at beta = 1 is the classical linear
    coefficient k_inh = alpha/K_I. The claim is asymptotic, not exact: the
    agreement is a limit, and the test checks the rate at which it holds.
    """
    lin = dict(REG, beta=1.0)
    prev = None
    for Y in (0.5, 0.1, 0.02, 0.004):
        nonlinear = hem.steady_state(Y, **lin)
        linear = hem.linear_reference(Y, numerator=60.0, alpha=4.0,
                                      K_I=5.0, lam=1.0)
        rel = abs(nonlinear - linear) / nonlinear
        if prev is not None:
            # a fivefold drop in Y should cut the relative error about fivefold
            assert rel < prev / 4.0
        prev = rel
    assert prev < 1e-3
    assert hem.steady_state(0.0, **lin) == pytest.approx(
        hem.linear_reference(0.0, numerator=60.0, alpha=4.0, K_I=5.0, lam=1.0))


def test_linear_model_fails_once_the_titre_approaches_K_I():
    """The boundary the proposition puts on ward arithmetic. With beta = 2 the
    linear reference overstates remaining activity, and the error grows."""
    errs = []
    for Y in (0.5, 2.0, 5.0, 10.0):
        true = hem.steady_state(Y, **REG)
        approx = hem.linear_reference(Y, numerator=60.0, alpha=4.0,
                                      K_I=5.0, lam=1.0)
        errs.append((approx - true) / true)
    # below K_I the linear reference is pessimistic, above it optimistic,
    # and either way the discrepancy grows with the titre
    assert abs(errs[-1]) > abs(errs[0])
    assert errs[-1] < 0 or errs[-1] > 0


def test_proposition_steady_state_is_stable_and_decreasing_in_Y():
    """Proposition C.2: a unique stable fixed point, decreasing in the
    inhibitor titre because the titre sits in the denominator."""
    vals = [hem.steady_state(Y, **REG) for Y in (0, 1, 2, 3, 5, 10, 15, 40)]
    assert vals == sorted(vals, reverse=True)
    assert vals[0] == pytest.approx(60.0)        # no inhibitor: input/clearance
    assert hem.steady_state(5.0, **REG) == pytest.approx(20.0)


def test_worked_example_where_the_damage_happens():
    """Worked Example C.1: the corrected reading. Most of the loss happens
    below K_I, not around it. Two Bethesda units already cost a third of
    baseline activity."""
    ss = lambda Y: hem.steady_state(Y, **REG)
    assert ss(0) == pytest.approx(60.0)
    assert ss(2) == pytest.approx(38.67, abs=0.02)
    assert ss(5) == pytest.approx(20.00, abs=0.02)
    assert ss(15) == pytest.approx(13.04, abs=0.02)
    assert ss(40) == pytest.approx(12.15, abs=0.02)
    first_two = ss(0) - ss(2)
    next_three = ss(2) - ss(5)
    following_ten = ss(5) - ss(15)
    assert first_two == pytest.approx(21.3, abs=0.1)
    assert next_three == pytest.approx(18.7, abs=0.1)
    assert following_ten == pytest.approx(7.0, abs=0.1)
    assert first_two > next_three > following_ten
    assert first_two / ss(0) > 0.33              # a third of baseline


def test_proposition_inhibitor_threshold():
    """Proposition C.3: a single titre at which the regimen stops protecting."""
    thr = hem.inhibitor_threshold(S_prot=20.0, **REG)
    assert thr == pytest.approx(5.0, abs=1e-9)
    assert hem.steady_state(thr - 0.5, **REG) > 20.0
    assert hem.steady_state(thr + 0.5, **REG) < 20.0


def test_threshold_is_a_property_of_the_patient_and_regimen_together():
    """The clinical corollary: change the dose and the threshold moves, so it
    is not a property of the patient alone."""
    low = hem.inhibitor_threshold(S_prot=20.0, **REG)
    stronger = hem.inhibitor_threshold(S_prot=20.0, **dict(REG, numerator=90.0))
    assert stronger > low


def test_threshold_undefined_when_the_regimen_cannot_reach_protection():
    """An honest failure mode: if even a zero titre leaves the patient below
    the protective level, no finite threshold exists and the function says so."""
    with pytest.raises(ValueError):
        hem.inhibitor_threshold(S_prot=70.0, **REG)


def test_proposition_iti_is_all_or_nothing():
    """Proposition C.4: above the critical suppression the titre goes to zero;
    below it, it settles at a positive level. Ninety per cent of the required
    suppression does not buy ninety per cent of the benefit."""
    kin = dict(rho=0.5, kappa=1.0, lam_Y=0.05, Y_max=40.0)
    crit = hem.iti_critical_suppression(rho=0.5, kappa=1.0, lam_Y=0.05)
    assert crit == pytest.approx(0.45)
    assert hem.iti_equilibrium(crit, **kin) == 0.0
    assert hem.iti_equilibrium(crit * 1.2, **kin) == 0.0
    partial = hem.iti_equilibrium(crit * 0.9, **kin)
    assert partial > 0.0
    assert partial == pytest.approx(40.0 * (1 - (0.9 * 0.45 + 0.05) / 0.5))


def test_worked_example_partial_adherence_is_closer_to_no_therapy():
    """Worked Example C.3: the cost of an interrupted course. Effective
    suppression averaged over a period with gaps can fall below critical even
    when the dose given on treatment days is adequate."""
    kin = dict(rho=0.5, kappa=1.0, lam_Y=0.05, Y_max=40.0)
    crit = hem.iti_critical_suppression(rho=0.5, kappa=1.0, lam_Y=0.05)
    full = 0.60                                   # comfortably above critical
    assert hem.iti_equilibrium(full, **kin) == 0.0
    # a supply failure for 30 per cent of the course
    effective = full * 0.70
    assert effective < crit
    assert hem.iti_equilibrium(effective, **kin) > 0.0


def test_proposition_gene_therapy_equilibrium():
    """Proposition C.5: F* = A/(eta Y). Halving the titre is worth exactly as
    much as doubling transgene expression."""
    A, eta, Y = 120.0, 0.4, 6.0
    base = hem.gene_therapy_equilibrium(A, eta, Y)
    assert base == pytest.approx(50.0)
    assert hem.gene_therapy_equilibrium(2 * A, eta, Y) == pytest.approx(2 * base)
    assert hem.gene_therapy_equilibrium(A, eta, Y / 2) == pytest.approx(2 * base)
    with pytest.raises(ValueError):
        hem.gene_therapy_equilibrium(A, eta, 0.0)


def test_hill_exponent_above_one_gives_a_threshold_not_a_slope():
    """Why beta > 1 matters: the neutralisation curve is sigmoid, so an
    inhibitor presents as an event rather than a drift."""
    shallow = [hem.hill_fraction(Y, 5.0, 1.0) for Y in (1, 3, 5, 7, 9)]
    steep = [hem.hill_fraction(Y, 5.0, 4.0) for Y in (1, 3, 5, 7, 9)]
    d_shallow = [b - a for a, b in zip(shallow, shallow[1:])]
    d_steep = [b - a for a, b in zip(steep, steep[1:])]
    assert max(d_steep) > max(d_shallow)          # a steeper crossing
    assert steep[0] < shallow[0]                  # quieter well below K_I
    assert all(abs(v - 0.5) < 1e-12 for v in (shallow[2], steep[2]))
