"""Regression tests: does the package still reproduce the published numbers?

Expected values are read from ``tests/fixtures/*.json``, which are the files the
published analysis scripts wrote. A test here failing means one of two things:
the package has changed what it computes, or a published number was wrong. Both
need a human.

Two kinds of assertion appear below.

*Closed-form results* are asserted to the digit, because they involve no
simulation: the determinacy ceiling, the required second-order signal, the
Jacobian ranks, the events inflation, and the closed form of the costed
statistic.

*Simulation results* are asserted against an interval wide enough to hold Monte
Carlo error at the published replicate counts and narrow enough to fail if the
method changes. Where the published run and this package differ in how a
replicate's generator is derived, the interval is the honest statement and a
point equality would be a false one.

The slow ones carry ``@pytest.mark.slow``. ``make test`` skips them;
``make test-all`` runs them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hpotential.analyses import (
    paper1_validation,
    paper2_prognostic,
    paper3_consensus,
    paper4_dynamics,
    paper5_access,
    paper6_hta,
)
from hpotential.core.access import AccessModel
from hpotential.core.consensus import (
    iqr_lemma_violations,
    mu_for_endorsement,
    operating_characteristics,
    retention_masks,
    simulate_ratings,
)
from hpotential.core.measurement import (
    determinacy_ceiling,
    identification_report,
    required_second_order_signal,
    second_order_determinacy,
)
from hpotential.core.prognostic import events_inflation

#: The JSON written by the published analysis scripts.
FIXTURE_DIR = Path(__file__).parent / "fixtures"


def load_published(name: str) -> dict:
    """Read one published result file by stem, for example ``p1_determinacy``."""
    path = FIXTURE_DIR / f"{name}.json"
    if not path.exists():  # pragma: no cover - a missing fixture is a setup error
        raise FileNotFoundError(f"missing published fixture: {path}")
    return json.loads(path.read_text())


# ==========================================================================
# Paper 1: determinacy ceiling and identification (closed form, exact)
# ==========================================================================
def test_determinacy_ceiling_matches_the_published_value():
    published = load_published("p1_determinacy")
    rho_inf, c = determinacy_ceiling()
    assert rho_inf == pytest.approx(0.877819, abs=1e-5)
    # The published table stores c rounded to four decimals, 3.3586. The exact
    # value is 3.3585621, which differs from the stored one by 3.8e-5, so the
    # digit-level check is the rounding and the tight check is against the full
    # precision value.
    assert c == pytest.approx(3.358562, abs=1e-5)
    assert round(rho_inf, 4) == published["ceiling_rho_D"]
    assert round(c, 4) == published["c_gamma_Psi"] == 3.3586


def test_numerical_determinacy_at_400_indicators_sits_below_the_ceiling():
    """The ceiling is a supremum, approached from below and never reached."""
    rho_inf, _ = determinacy_ceiling()
    numeric = second_order_determinacy(400, 0.85)
    assert numeric == pytest.approx(0.877594, abs=1e-4)
    assert numeric < rho_inf
    assert rho_inf - numeric < 1e-3


def test_required_signal_for_the_individual_use_benchmark():
    published = load_published("p1_determinacy")
    required = required_second_order_signal(0.90)
    assert required == pytest.approx(4.2632, abs=1e-4)
    assert round(required, 4) == published["c_required_for_target"]
    _, c = determinacy_ceiling()
    assert required > c  # the planned design cannot reach 0.90


def test_published_determinacy_table_reproduces_to_four_decimals():
    published = load_published("p1_determinacy")
    for lam in (0.55, 0.65, 0.75, 0.85):
        row = published[f"lambda={lam}"]
        assert round(second_order_determinacy(4, lam), 4) == row["rho_D_at_m4"]
        assert round(second_order_determinacy(8, lam), 4) == row["rho_D_at_m8"]
        assert round(second_order_determinacy(40, lam), 4) == row["rho_D_at_m40"]
        assert row["indicators_per_domain_for_target"] is None


def test_identification_ranks_match_the_published_report():
    published = load_published("p1_identification")
    for record in published["planned_designs"]:
        report = identification_report(record["loading"], record["label"])
        assert report["rank"] == record["rank"] == 27
        assert report["n_free"] == record["n_free"] == 27
        assert report["n_moments"] == record["n_moments"] == 78
        assert report["identified"] is True
        assert report["smallest_singular_value"] == pytest.approx(
            record["smallest_singular_value"], rel=1e-6
        )
        assert report["condition_number"] == pytest.approx(record["condition_number"], rel=1e-6)


def test_two_domain_negative_control_still_fails_the_rank_test():
    published = load_published("p1_identification")
    record = next(r for r in published["controls"] if "2 domains" in r["label"])
    report = identification_report(
        0.75, record["label"], np.array([0.80, 0.70]), np.array([0.36, 0.51]), 4
    )
    assert report["rank"] == record["rank"] == 17
    assert report["n_free"] == record["n_free"] == 18
    assert report["identified"] is False


def test_two_indicators_per_domain_control_is_identified():
    published = load_published("p1_identification")
    record = next(r for r in published["controls"] if "2 indicators" in r["label"])
    report = identification_report(0.75, record["label"], per_domain=2)
    assert report["rank"] == record["rank"] == 15
    assert report["identified"] is True


# ==========================================================================
# Paper 2: attenuation and the events cost
# ==========================================================================
def test_predicted_determinacy_of_the_planned_design():
    published = load_published("p2_pipeline")
    rho = paper2_prognostic.population_model(paper2_prognostic.Paper2Config()).determinacy()
    assert rho == pytest.approx(0.8361, abs=1e-4)
    assert round(rho, 4) == published["rho_D_population"]


def test_events_inflation_at_the_ceiling_and_at_the_planned_design():
    """Corollary: matching noncentrality costs a factor of ``1 / rho_D^2``."""
    rho_inf, _ = determinacy_ceiling()
    rho_planned = paper2_prognostic.population_model(paper2_prognostic.Paper2Config()).determinacy()
    assert events_inflation(rho_inf) == pytest.approx(1.2977, abs=1e-3)
    assert events_inflation(rho_planned) == pytest.approx(1.4305, abs=1e-3)
    assert events_inflation(rho_planned) > events_inflation(rho_inf)


@pytest.mark.slow
def test_pipeline_attenuation_matches_the_predicted_determinacy():
    """The whole pipeline, 120 replicates at eight design points.

    The pooled ratio is the statistic to compare with ``rho_D``. A ratio from a
    single design point carries several per cent of Monte Carlo error, so the
    per-point values scatter more widely than the quantity being estimated; the
    published run reported 0.805 to 0.859 across the same eight points.
    """
    results = paper2_prognostic.run()
    pooled = results["attenuation_ratio_pooled"]
    assert 0.79 <= pooled <= 0.87
    assert results["predicted_attenuated_beta"] == pytest.approx(-0.1003, abs=1e-4)
    assert results["rho_D_population"] == pytest.approx(0.8361, abs=1e-4)
    assert 0.75 < results["attenuation_ratio_min"]
    assert results["attenuation_ratio_max"] < 0.92
    assert results["n_replicates_pooled"] >= 900

    published = load_published("p2_pipeline")
    for key, record in results.items():
        if not isinstance(record, dict) or "power_observed" not in record:
            continue
        # Measurement error costs power at every design point, as published.
        assert record["power_observed"] <= record["power_oracle"] + 1e-9
        assert record["mean_events"] == pytest.approx(published[key]["mean_events"], rel=0.02)


# ==========================================================================
# Paper 3: the Delphi rule
# ==========================================================================
@pytest.mark.slow
def test_delphi_false_drop_rate_at_the_specified_threshold():
    """At IQR <= 1.5 a well-endorsed item is dropped most of the time."""
    published = load_published("p3_delphi_oc")
    rng = np.random.default_rng(20260815)
    strict = operating_characteristics(30, 0.90, 1.2, rng, 20000, iqr_threshold=1.5)
    assert 0.60 <= strict["p_drop"] <= 0.75
    assert strict["p_drop"] == pytest.approx(
        published["operating_characteristics"]["30"]["false_drop_at_pi_0_90"], abs=0.02
    )
    # It is dispersion that binds, not endorsement and not the median.
    assert strict["iqr_fails"] > 0.95
    assert strict["median_fails"] < 0.01


@pytest.mark.slow
def test_delphi_false_drop_rate_at_a_threshold_of_two():
    """At IQR <= 2.0 the dispersion criterion never binds on its own.

    The lemma bounds the interquartile range only for panels that meet the
    content-validity threshold, so a panel can still show an interquartile range
    above 2 while failing on endorsement. What the lemma forbids is a panel that
    passes I-CVI and is then dropped on dispersion, and that is what is checked
    here: every dropped panel fails I-CVI, and no panel passing I-CVI has an
    interquartile range above 2.
    """
    rng = np.random.default_rng(20260815)
    relaxed = operating_characteristics(30, 0.90, 1.2, rng, 20000, iqr_threshold=2.0)
    assert 0.00 <= relaxed["p_drop"] <= 0.06
    assert relaxed["cvi_fails"] == 1.0

    rng = np.random.default_rng(20260815)
    ratings = simulate_ratings(rng, mu_for_endorsement(0.90, 1.2), 1.2, (20000, 30))
    masks = retention_masks(ratings, iqr_threshold=2.0)
    assert int((masks["ok_cvi"] & ~masks["ok_iqr"]).sum()) == 0
    assert masks["iqr"][masks["ok_cvi"]].max() <= 2.0


@pytest.mark.slow
def test_delphi_error_rates_rise_with_panel_size_as_published():
    published = load_published("p3_delphi_oc")["operating_characteristics"]
    rng = np.random.default_rng(20260815)
    rates = {}
    for n_panel in (12, 20, 30, 45, 60):
        rates[n_panel] = operating_characteristics(
            n_panel, 0.90, 1.2, rng, 20000, iqr_threshold=1.5
        )["p_drop"]
    assert list(rates.values()) == sorted(rates.values())
    for n_panel, rate in rates.items():
        assert rate == pytest.approx(published[str(n_panel)]["false_drop_at_pi_0_90"], abs=0.02)


def test_dispersion_lemma_holds_exhaustively_up_to_ten_panellists():
    """I-CVI >= 0.78 forces IQR <= 2 under the linear-interpolation rule.

    Every multiset of ratings from 1 to 9 for panel sizes 1 to 10 is visited.
    The largest interquartile range among panels that pass the content-validity
    threshold is exactly 2, attained at (7, 7, 9, 9), and there are no
    violations.
    """
    out = iqr_lemma_violations(max_panel=10, cvi_min=0.78, iqr_bound=2.0, method="linear")
    assert out["violations"] == 0
    assert out["max_iqr_among_passing"] == 2.0
    assert out["worst_panel"] == (7, 7, 9, 9)
    assert out["n_passing_cvi"] > 1000


def test_dispersion_lemma_is_false_under_another_quantile_convention():
    """The lemma is about a rule and a convention together, not thresholds alone."""
    weibull = iqr_lemma_violations(max_panel=10, cvi_min=0.78, iqr_bound=2.0, method="weibull")
    assert weibull["violations"] > 0
    assert weibull["max_iqr_among_passing"] > 2.0


# ==========================================================================
# Paper 4: solvers and the Fokker-Planck scheme
# ==========================================================================
@pytest.mark.slow
def test_rk4_and_euler_maruyama_orders_match_the_published_run():
    published = load_published("p4_numerics")
    results = paper4_dynamics.run()

    assert 3.9 <= results["rk4"]["observed_order"] <= 4.2
    assert results["rk4"]["observed_order"] == pytest.approx(
        published["rk4"]["observed_order"], abs=0.01
    )

    additive = results["euler_maruyama"]["additive_noise"]
    multiplicative = results["euler_maruyama"]["multiplicative_noise"]
    assert 0.95 <= additive["strong_order"] <= 1.10
    assert 0.45 <= multiplicative["strong_order"] <= 0.60
    assert 0.95 <= additive["weak_order"] <= 1.05


@pytest.mark.slow
def test_fokker_planck_conserves_mass_and_converges_at_first_order():
    """Mass drift below 1e-14 without any renormalisation of the density."""
    published = load_published("p4_numerics")
    results = paper4_dynamics.run()
    fp = results["fokker_planck"]
    for nx in ("100", "200", "400", "800"):
        assert fp[nx]["mass_drift"] < 1e-14
        assert fp[nx]["min_density"] >= 0.0
        assert fp[nx]["L1_error_vs_stationary"] == pytest.approx(
            published["fokker_planck"][nx]["L1_error_vs_stationary"], rel=1e-6
        )
    assert 0.90 <= fp["grid_convergence_order_L1"] <= 1.05
    assert fp["grid_convergence_order_L1"] == pytest.approx(
        published["fokker_planck"]["grid_convergence_order_L1"], abs=1e-3
    )


# ==========================================================================
# Paper 5: the costed statistic
# ==========================================================================
def test_costed_statistic_closed_form_agrees_with_direct_evaluation():
    published = load_published("p5_costed")
    discrepancy = AccessModel.low_resource().closed_form_check(
        np.random.default_rng(20260815), 20_000
    )
    assert discrepancy < 1e-12
    assert discrepancy == pytest.approx(
        published["closed_form_check"]["max_relative_discrepancy"], rel=1e-6
    )


@pytest.mark.slow
def test_costed_and_uncosted_rules_disagree_as_published():
    published = load_published("p5_costed")["psa"]
    results = paper5_access.run()["psa"]

    high = results["high-resource"]
    assert 0.45 <= high["P_disagreement"] <= 0.52
    assert high["P_disagreement"] == pytest.approx(
        published["high-resource"]["P_disagreement"], abs=0.01
    )
    assert high["median_V"] == pytest.approx(published["high-resource"]["median_V"], rel=0.02)

    low = results["low-resource"]
    assert low["P_disagreement"] == 0.0
    assert low["P_costed_rule_favours_access"] == 1.0


# ==========================================================================
# Paper 6: the early assessment
# ==========================================================================
@pytest.mark.slow
def test_central_scenario_decision_probabilities_match_the_published_run():
    published = load_published("p6_voi")["scenarios"]["central"]
    central = paper6_hta.run()["scenarios"]["central"]

    assert 0.38 <= central["P_cost_effective_QALY"] <= 0.44
    assert central["P_cost_effective_QALY"] == pytest.approx(
        published["P_cost_effective_QALY"], abs=0.01
    )

    ratio = central["mean_incremental_bmp"] / central["mean_incremental_qaly"]
    assert 1.85 <= ratio <= 1.92
    assert ratio == pytest.approx(central["bmp_to_qaly_ratio"], rel=1e-12)
    assert central["mean_incremental_qaly"] == pytest.approx(
        published["mean_incremental_qaly"], rel=0.02
    )
    assert central["mean_incremental_bmp"] == pytest.approx(
        published["mean_incremental_bmp"], rel=0.02
    )


@pytest.mark.slow
def test_value_of_information_is_positive_and_bounded_by_evpi():
    published = load_published("p6_voi")["scenarios"]
    results = paper6_hta.run()["scenarios"]
    for name, record in results.items():
        assert record["per_person_EVPI"] > 0
        assert record["per_person_EVPI"] == pytest.approx(
            published[name]["per_person_EVPI"], rel=0.05
        )
        for value in record["EVPPI"].values():
            assert value <= record["per_person_EVPI"] + 1e-6


# ==========================================================================
# Paper 1 simulation study, and determinism
# ==========================================================================
@pytest.mark.slow
def test_sample_size_study_reports_realised_coverage_near_the_nominal_level():
    """Coverage is counted from Wald intervals, not written in.

    The published run reported worst-case coverage between 0.92 and 0.977 across
    twelve design points at 150 replicates. A binomial standard error at that
    count is about 0.018, so a band from 0.88 to 1.0 is the honest one.
    """
    results = paper1_validation.run()
    summary = results["sample_size"]["summary"]
    assert len(summary) == 12
    for record in summary.values():
        assert record["convergence"] >= 0.95
        assert 0.88 <= record["min_coverage"] <= 1.0
        assert record["max_abs_rel_bias"] < 0.05
        assert 0.80 < record["determinacy"] < 0.89
    # Admissibility improves with sample size in the stress scenario.
    assert summary["stress|1400"]["admissible"] >= summary["stress|150"]["admissible"]


def test_the_same_seed_gives_the_same_numbers():
    """Determinism, checked rather than asserted in a docstring."""
    a = paper5_access.run(quick=True)
    b = paper5_access.run(quick=True)
    assert a["psa"] == b["psa"]

    c = paper6_hta.run(quick=True)
    d = paper6_hta.run(quick=True)
    assert c["scenarios"]["central"]["mean_INB"] == d["scenarios"]["central"]["mean_INB"]

    e = paper3_consensus.run(quick=True)
    f = paper3_consensus.run(quick=True)
    assert e["operating_characteristics"] == f["operating_characteristics"]


def test_quick_mode_says_it_is_quick():
    """Nothing in quick mode should be mistaken for a published number."""
    for module in (paper3_consensus, paper5_access, paper6_hta):
        assert module.run(quick=True)["quick_mode"] is True
