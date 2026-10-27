"""Chapters 3 and 6: hazards, competing risks, calibration, net benefit."""
import math
import pytest
from hpv import prognostic as prog


def test_definition_hazard_ratio_to_absolute_risk():
    """Definition 3.4 and its reading. A hazard ratio is a rate multiplier;
    a patient lives in absolute risk, and the conversion needs a baseline."""
    assert prog.hazard_ratio_to_risk(0.20, 1.0) == pytest.approx(0.20)
    assert prog.hazard_ratio_to_risk(0.20, 1.4) == pytest.approx(0.268, abs=1e-3)
    with pytest.raises(ValueError):
        prog.hazard_ratio_to_risk(0.0, 1.4)


def test_worked_example_profile_a_absolute_risk():
    """Worked Example 8.1 and 6.1: HR 1.4 per SD, Profile A one SD below the mean, a
    20 per cent five-year baseline. About 27 per cent, seven extra deaths per
    hundred, one per fourteen women like her."""
    base, hr = 0.20, 1.4
    hers = prog.hazard_ratio_to_risk(base, hr)
    assert hers == pytest.approx(0.268, abs=1e-3)
    assert hers - base == pytest.approx(0.068, abs=1e-3)
    assert prog.number_needed(hers, base) == pytest.approx(14.6, abs=0.2)


def test_the_same_hazard_ratio_means_less_at_a_low_baseline():
    """Why the absolute scale has to be reported. The identical ratio buys a
    much smaller absolute difference in a low-risk group."""
    hr = 1.4
    high = prog.hazard_ratio_to_risk(0.20, hr) - 0.20
    low = prog.hazard_ratio_to_risk(0.05, hr) - 0.05
    assert high > 3 * low
    assert prog.number_needed(prog.hazard_ratio_to_risk(0.05, hr), 0.05) > 45


def test_proposition_competing_risk_biases_kaplan_meier_upward():
    """Proposition 6.1: treating death as censoring overstates the cumulative
    incidence of hospitalisation, and overstates it most in the sickest."""
    t = 5.0
    for death_rate in (0.02, 0.10, 0.30):
        true = prog.cumulative_incidence_competing(0.12, death_rate, t)
        naive = prog.km_complement(0.12, t)
        assert naive > true
    gap_well = (prog.km_complement(0.12, t)
                - prog.cumulative_incidence_competing(0.12, 0.02, t))
    gap_sick = (prog.km_complement(0.12, t)
                - prog.cumulative_incidence_competing(0.12, 0.30, t))
    assert gap_sick > gap_well                 # worst where it matters most


def test_worked_example_competing_risk_numbers():
    """Worked Example 3.1: constant rates, five years. The naive figure is
    0.45; the true cumulative incidence is 0.41 in a well group and 0.29 in a
    sick one, from the same event rate."""
    t = 5.0
    naive = prog.km_complement(0.12, t)
    well = prog.cumulative_incidence_competing(0.12, 0.02, t)
    sick = prog.cumulative_incidence_competing(0.12, 0.30, t)
    assert naive == pytest.approx(0.451, abs=1e-3)
    assert well == pytest.approx(0.431, abs=1e-3)
    assert sick == pytest.approx(0.251, abs=1e-3)
    # the overstatement is 2 points in the well group and 20 in the sick one
    assert naive - well == pytest.approx(0.020, abs=2e-3)
    assert naive - sick == pytest.approx(0.201, abs=2e-3)


def test_definition_net_benefit_and_the_threshold_as_an_exchange_rate():
    """Definition 6.5: the threshold probability is the rate at which a
    clinician trades a missed case against an unnecessary intervention."""
    nb = prog.net_benefit(tp=60, fp=90, n=1000, threshold=0.10)
    assert nb == pytest.approx(0.06 - 0.09 * (0.1 / 0.9), abs=1e-9)
    assert nb == pytest.approx(0.05, abs=1e-3)
    # a higher threshold means the clinician minds false positives more
    assert prog.net_benefit(60, 90, 1000, 0.30) < nb


def test_worked_example_net_benefit_against_treat_all():
    """Worked Example 6.3 and 3.1: a model earns its place only by beating both
    treat-all and treat-none at the threshold in question."""
    prevalence, thr = 0.10, 0.10
    all_ = prog.net_benefit_treat_all(prevalence, thr)
    model = prog.net_benefit(tp=70, fp=120, n=1000, threshold=thr)
    assert all_ == pytest.approx(0.10 - 0.90 * (1 / 9), abs=1e-9)
    assert all_ == pytest.approx(0.0, abs=1e-9)
    assert model > all_
    assert model > 0.0                        # and beats treat-none


def test_proposition_attenuation_by_measurement_error():
    """Proposition 3.2: beta_naive ~= rho_H * beta. An estimated factor score
    biases the coefficient towards no effect, so a null result is weaker
    evidence against the construct than a positive result is for it."""
    beta = -0.40
    assert prog.attenuated_coefficient(beta, 1.0) == pytest.approx(beta)
    att = prog.attenuated_coefficient(beta, 0.80)
    assert att == pytest.approx(-0.32)
    assert abs(att) < abs(beta)               # towards the null, never past it
    assert math.exp(beta) == pytest.approx(0.670, abs=1e-3)
    assert math.exp(att) == pytest.approx(0.726, abs=1e-3)
    with pytest.raises(ValueError):
        prog.attenuated_coefficient(beta, 0.0)


def test_worked_example_attenuation_with_the_numbers():
    """Worked Example 3.2: a true HR of 0.67 is reported as 0.73 when the
    factor score has reliability 0.80."""
    beta = -0.40
    expected = {1.0: 0.670, 0.9: 0.698, 0.8: 0.726, 0.6: 0.787}
    for rho, hr in expected.items():
        b = prog.attenuated_coefficient(beta, rho)
        assert math.exp(b) == pytest.approx(hr, abs=1e-3)


def test_attenuation_is_monotone_in_reliability():
    beta = -0.40
    vals = [abs(prog.attenuated_coefficient(beta, r))
            for r in (0.5, 0.7, 0.9, 1.0)]
    assert vals == sorted(vals)


# ---------------------------------------------------------------------------
# Chapter 6 additions: absolute risk for Profile A, the independence of
# discrimination and calibration, net benefit at a stated threshold, and the
# optimism gap.
# ---------------------------------------------------------------------------
import numpy as np


def _auc(pred, outcome):
    """Concordance for a binary outcome observed at a fixed horizon."""
    pos = [p for p, y in zip(pred, outcome) if y == 1]
    neg = [p for p, y in zip(pred, outcome) if y == 0]
    total = sum((1.0 if a > b else 0.5 if a == b else 0.0)
                for a in pos for b in neg)
    return total / (len(pos) * len(neg))


def test_worked_example_profile_a_absolute_risk_from_a_hazard_ratio_of_072():
    """Worked Example 6.1: a hazard ratio of 0.72 per SD, Profile A one SD below
    the mean, a 20 per cent five-year baseline. About 27 per cent, a gap of
    seven points, one death prevented per fifteen women lifted."""
    multiplier = 1.0 / 0.72
    hers = prog.hazard_ratio_to_risk(0.20, multiplier)
    assert multiplier == pytest.approx(1.389, abs=1e-3)
    assert hers == pytest.approx(0.266, abs=1e-3)
    assert hers - 0.20 == pytest.approx(0.066, abs=1e-3)
    assert prog.number_needed(hers, 0.20) == pytest.approx(15.0, abs=0.1)
    # the same multiplier at a 5 per cent baseline buys far less
    low = prog.hazard_ratio_to_risk(0.05, multiplier)
    assert low == pytest.approx(0.069, abs=1e-3)
    assert prog.number_needed(low, 0.05) == pytest.approx(53.3, abs=0.2)


PREDICTED = [0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55, 0.70]
OBSERVED = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1]


def test_proposition_discrimination_and_calibration_are_independent():
    """Proposition 6.1: a strictly increasing transformation of the predicted
    risks leaves concordance untouched and wrecks calibration; a constant
    prediction is calibrated in the large and ranks nobody."""
    c_before = _auc(PREDICTED, OBSERVED)
    warped = [math.sqrt(p) for p in PREDICTED]
    assert _auc(warped, OBSERVED) == pytest.approx(c_before, abs=1e-12)
    citl_before = sum(OBSERVED) / 10 - sum(PREDICTED) / 10
    citl_after = sum(OBSERVED) / 10 - sum(warped) / 10
    assert abs(citl_after) > 5 * abs(citl_before)
    # the reverse failure
    flat = [sum(OBSERVED) / 10] * 10
    assert _auc(flat, OBSERVED) == pytest.approx(0.50, abs=1e-12)
    assert sum(OBSERVED) / 10 - sum(flat) / 10 == pytest.approx(0.0, abs=1e-12)


def test_worked_example_monotone_transformation_keeps_c_and_breaks_calibration():
    """Worked Example 6.2: concordance 0.952 before and after, while mean
    predicted risk moves from 0.28 to 0.49 against an observed 0.30."""
    assert _auc(PREDICTED, OBSERVED) == pytest.approx(0.952, abs=1e-3)
    warped = [math.sqrt(p) for p in PREDICTED]
    assert _auc(warped, OBSERVED) == pytest.approx(0.952, abs=1e-3)
    assert sum(PREDICTED) / 10 == pytest.approx(0.280, abs=1e-9)
    assert sum(warped) / 10 == pytest.approx(0.495, abs=1e-3)
    assert sum(OBSERVED) / 10 == pytest.approx(0.300, abs=1e-9)


def test_worked_example_net_benefit_at_a_twenty_per_cent_threshold():
    """Worked Example 6.3 and 3.1: at a threshold above the event rate, treating
    everyone is worse than treating nobody, and the model beats both."""
    thr, n = 0.20, 1000
    model = prog.net_benefit(tp=75, fp=125, n=n, threshold=thr)
    treat_all = prog.net_benefit_treat_all(0.15, thr)
    assert model == pytest.approx(0.04375, abs=1e-9)
    assert treat_all == pytest.approx(-0.0625, abs=1e-9)
    assert treat_all < 0.0 < model            # treat-none scores zero
    # treat-all turns negative exactly when the threshold exceeds prevalence
    assert prog.net_benefit_treat_all(0.15, 0.15) == pytest.approx(0.0,
                                                                   abs=1e-12)


def _fit(X, y):
    design = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(design, y, rcond=None)[0]


def _predict(coef, X):
    return np.column_stack([np.ones(len(X)), X]) @ coef


def _auc_np(pred, y):
    pos, neg = pred[y == 1], pred[y == 0]
    return float((np.greater.outer(pos, neg)
                  + 0.5 * np.equal.outer(pos, neg)).mean())


def test_worked_example_bootstrap_optimism_gap():
    """Worked Example 6.4: 200 people, 20 candidate predictors of which two
    carry signal. Apparent concordance about 0.76, optimism about 0.10, so
    corrected performance about 0.66."""
    rng = np.random.default_rng(6)
    n, p = 200, 20
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[0], beta[1] = 0.8, 0.5
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(X @ beta - 0.8)))).astype(int)
    coef = _fit(X, y)
    apparent = _auc_np(_predict(coef, X), y)
    gaps = []
    for _ in range(200):
        idx = rng.integers(0, n, n)
        Xb, yb = X[idx], y[idx]
        if yb.sum() in (0, len(yb)):
            continue
        cb = _fit(Xb, yb)
        gaps.append(_auc_np(_predict(cb, Xb), yb) - _auc_np(_predict(cb, X), y))
    optimism = float(np.mean(gaps))
    assert apparent == pytest.approx(0.76, abs=0.02)
    assert optimism == pytest.approx(0.10, abs=0.03)
    assert apparent - optimism == pytest.approx(0.66, abs=0.03)
    assert optimism > 0.0                     # apparent always flatters


def test_hazard_ratio_scales_with_distance_from_the_mean():
    """The SD multiplier must actually be applied.

    Every other test calls hazard_ratio_to_risk at the default of one standard
    deviation, where hr**1.0 == hr, so dropping the exponent entirely is
    invisible to them. A mutation that removed it survived the suite. This
    test exercises sd_from_mean != 1, which is the case a real cohort will
    contain.
    """
    base, hr = 0.20, 1.4
    r1 = prog.hazard_ratio_to_risk(base, hr, 1.0)
    r2 = prog.hazard_ratio_to_risk(base, hr, 2.0)
    rh = prog.hazard_ratio_to_risk(base, hr, 0.5)
    r0 = prog.hazard_ratio_to_risk(base, hr, 0.0)

    assert r0 == pytest.approx(base)          # at the mean, the baseline
    assert rh < r1 < r2                        # monotone in distance
    assert r1 == pytest.approx(0.268, abs=1e-3)
    assert r2 == pytest.approx(0.354, abs=1e-3)
    assert rh == pytest.approx(0.232, abs=1e-3)

    # two SDs applies the ratio twice on the cumulative-hazard scale
    H0 = -math.log(1 - base)
    assert r2 == pytest.approx(1 - math.exp(-H0 * hr ** 2))

    # and it works in the protective direction too
    prot = prog.hazard_ratio_to_risk(base, 0.72, 2.0)
    assert prot < prog.hazard_ratio_to_risk(base, 0.72, 1.0) < base
