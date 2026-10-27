"""Appendix A.11 and Appendix D: net benefit, budget impact, DALYs."""
import math
import pytest
from hpv import economics as econ


def test_definition_net_health_benefit():
    """Definition A.15: NHB = dQ - dC/lambda, measured in health."""
    assert econ.net_health_benefit(1.8, 540_000, 30_000) == pytest.approx(-16.2)
    with pytest.raises(ValueError):
        econ.net_health_benefit(1.0, 100.0, 0.0)


def test_worked_example_treatment_change_for_profile_b():
    """Worked Example A.6: the drug works and is still net negative, because
    the money buys more health elsewhere. The sign is the whole content."""
    dQ, dC, lam = 1.8, 540_000.0, 30_000.0
    displaced = econ.displaced_health(dC, lam)
    assert displaced == pytest.approx(18.0)
    assert dQ > 0                                   # it does produce health
    assert econ.net_health_benefit(dQ, dC, lam) < 0  # and still fails the test
    # the decision flips on lambda, which is a local fact, not a clinical one
    assert econ.net_health_benefit(dQ, dC, 400_000.0) > 0


def test_proposition_need_does_not_imply_cost_effectiveness():
    """Proposition A.9: for any baseline score there is an intervention with
    zero effect and positive cost, so a low score proves nothing about value."""
    assert econ.net_health_benefit(0.0, 1.0, 30_000.0) < 0


def test_inmb_and_nhb_agree_on_the_sign():
    """The two forms are the same decision rescaled by lambda."""
    for dQ, dC, lam in ((1.8, 540_000, 30_000), (0.5, 5_000, 30_000),
                        (0.02, 100, 20_000)):
        a = econ.net_health_benefit(dQ, dC, lam)
        b = econ.inmb(dQ, dC, lam)
        assert math.copysign(1, a) == math.copysign(1, b)
        assert b == pytest.approx(a * lam)


def test_theorem_policy_rankings_do_not_transport():
    """Theorem E.8: equal capability shift and equal health gain in two
    jurisdictions still permit opposite decisions, because lambda, cost and
    the budget ceiling are local."""
    dQ = 1.8
    j = econ.inmb(dQ, dC=40_000.0, lam=30_000.0)
    k = econ.inmb(dQ, dC=40_000.0, lam=15_000.0)
    assert j > 0 and k < 0


def test_budget_impact_is_not_cost_effectiveness():
    """A positive INMB does not imply affordability."""
    assert econ.inmb(1.8, 40_000.0, 30_000.0) > 0
    bi = econ.budget_impact(N=[1200, 1250, 1300], dc=[9_000, 9_000, 9_000])
    assert bi == pytest.approx(33_750_000.0)
    assert bi > 20_000_000.0                       # over a plausible ceiling


def test_discounting():
    assert econ.discounted([100, 100, 100], 0.035) == pytest.approx(
        100 + 100 / 1.035 + 100 / 1.035 ** 2)


def test_icer_is_undefined_at_zero_effect():
    """Why the net-benefit form is reported alongside: the ratio misbehaves
    exactly where the denominator is small."""
    with pytest.raises(ZeroDivisionError):
        econ.icer(1000.0, 0.0)
    big = econ.icer(1000.0, 1e-6)
    assert big > 1e8


def test_algorithm_one_daly_identity():
    """Appendix D Algorithm 1: DALY = YLD + YLL, by construction."""
    yld, yll, total = econ.daly(L0=30.0, w=0.28, E_L=22.0)
    assert yld == pytest.approx(6.16)
    assert yll == pytest.approx(8.0)
    assert total == pytest.approx(yld + yll)


def test_worked_example_profile_b_daly_split():
    """Worked Example D.1: for Profile B the burden is disability, not death.
    A mortality-only measure would record him as almost unaffected."""
    yld, yll, total = econ.daly(L0=30.0, w=0.28, E_L=28.0)
    assert yll == pytest.approx(2.0)               # treatment closed the gap
    assert yld == pytest.approx(7.84)              # the joints did not
    assert yld > 3 * yll
    bp = econ.biomedical_potential(L0=30.0, Q=0.72, dalys=total)
    assert 0.0 <= bp <= 1.0
    assert bp == pytest.approx(0.72 * (1 - 9.84 / 30.0), abs=1e-6)
