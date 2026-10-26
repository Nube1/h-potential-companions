"""Tests for ordinal logistic differential item functioning."""

from __future__ import annotations

import numpy as np
import pytest

from hpotential.core import dif, irt

N_ITEMS = 6
A_TRUE = [1.3] * N_ITEMS
B_TRUE = [-1.2, -0.1, 1.0]
DIF_ITEM = 2


def planted_data(shift: float = 1.0, n: int = 1500, seed: int = 20260815):
    """Responses with uniform DIF planted on one item.

    The focal group needs a higher trait to reach the same category on item
    ``DIF_ITEM``: its thresholds are shifted up by ``shift``. Every other item
    is identical across groups, and both groups share one trait distribution,
    so any difference the scan finds on those items is noise.
    """
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, n)
    theta = rng.normal(size=n)
    columns = []
    for j in range(N_ITEMS):
        col = np.zeros(n, dtype=int)
        for level in (0, 1):
            mask = group == level
            thresholds = np.asarray(B_TRUE) + (shift if j == DIF_ITEM else 0.0) * level
            responses, _ = irt.simulate_grm(
                int(mask.sum()),
                [A_TRUE[j]],
                [thresholds],
                seed=seed + 10 * j + level,
                theta=theta[mask],
            )
            col[mask] = responses[:, 0]
        columns.append(col)
    return np.column_stack(columns), group


# ------------------------------------------------------------ the base fit
def test_proportional_odds_fit_recovers_a_known_slope():
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.normal(size=n)
    beta, gammas = 1.2, np.array([-1.0, 0.5, 1.8])
    cum = 1.0 / (1.0 + np.exp(-(gammas[None, :] - beta * x[:, None])))
    probs = np.hstack([cum, np.ones((n, 1))]) - np.hstack([np.zeros((n, 1)), cum])
    draws = rng.random(n)[:, None]
    y = (draws > np.cumsum(probs, axis=1)).sum(axis=1)
    fit = dif.fit_proportional_odds(y, x[:, None], names=["x"])
    assert fit.converged
    assert fit.coefficient("x") == pytest.approx(beta, abs=0.1)
    assert np.allclose(fit.thresholds, gammas, atol=0.12)
    assert np.all(np.diff(fit.thresholds) > 0)


def test_statsmodels_agrees_with_the_scipy_fitter():
    pytest.importorskip("statsmodels")
    responses, group = planted_data(n=600)
    match = dif.rest_score(responses, DIF_ITEM)
    design = np.column_stack([match, group])
    ours = dif.fit_proportional_odds(responses[:, DIF_ITEM], design, names=["m", "g"])
    theirs = dif.fit_ordinal_statsmodels(responses[:, DIF_ITEM], design)
    assert ours.loglik == pytest.approx(theirs["loglik"], abs=1e-3)
    assert np.allclose(ours.beta, theirs["params"][:2], atol=1e-3)


# ------------------------------------------------------------- the DIF scan
@pytest.fixture(scope="module")
def scan():
    responses, group = planted_data()
    return dif.dif_scan(responses, group), responses, group


def test_planted_uniform_dif_is_found(scan):
    table, _, _ = scan
    row = table.loc[f"item{DIF_ITEM + 1}"]
    assert row["p_uniform"] < 1e-6
    assert row["delta_r2"] >= dif.NEGLIGIBLE_MAX
    assert row["classification"] == "uniform"
    assert bool(row["flagged"])


def test_clean_items_are_not_flagged(scan):
    table, _, _ = scan
    clean = table.drop(index=f"item{DIF_ITEM + 1}")
    assert not clean["flagged"].any()
    assert (clean["delta_r2"] < dif.NEGLIGIBLE_MAX).all()


def test_the_sign_convention_reads_as_documented(scan):
    """Thresholds shifted up for the focal group means it answers lower.

    Under ``logit P(Y <= c) = gamma_c - x'beta`` a positive coefficient pushes
    responses towards higher categories, so the group coefficient must be
    negative here.
    """
    table, _, _ = scan
    assert table.loc[f"item{DIF_ITEM + 1}", "beta_group"] < -0.5


def test_no_non_uniform_dif_is_claimed_when_none_was_planted(scan):
    table, _, _ = scan
    assert table.loc[f"item{DIF_ITEM + 1}", "p_nonuniform"] > 0.01


def test_mcfadden_delta_r2_is_positive_and_uses_the_null_model(scan):
    """The draft divided by loglik(M1), giving a negative effect size."""
    table, responses, group = scan
    result = dif.ordinal_logistic_dif(
        responses[:, DIF_ITEM],
        group,
        dif.rest_score(responses, DIF_ITEM),
        item_name="studied",
    )
    ll = result.loglik
    assert ll["M0"] < ll["M1"] < ll["M2"] <= ll["M3"]
    expected = (ll["M3"] - ll["M1"]) / (-ll["M0"])
    assert result.delta_r2 == pytest.approx(expected)
    assert result.delta_r2 > 0
    # what the draft computed, kept here to show the sign it produced
    draft = (ll["M3"] - ll["M1"]) / ll["M1"]
    assert draft < 0
    assert table.loc[f"item{DIF_ITEM + 1}", "delta_r2"] == pytest.approx(expected, abs=1e-6)


def test_delta_r2_matches_the_difference_of_two_mcfadden_r2(scan):
    _, responses, group = scan
    result = dif.ordinal_logistic_dif(
        responses[:, DIF_ITEM], group, dif.rest_score(responses, DIF_ITEM)
    )
    ll = result.loglik
    r2_m1 = 1 - ll["M1"] / ll["M0"]
    r2_m3 = 1 - ll["M3"] / ll["M0"]
    assert result.delta_r2 == pytest.approx(r2_m3 - r2_m1)


def test_significance_alone_does_not_flag_an_item():
    """A tiny but real difference in a large sample is significant, not DIF."""
    responses, group = planted_data(shift=0.18, n=6000, seed=99)
    result = dif.ordinal_logistic_dif(
        responses[:, DIF_ITEM],
        group,
        dif.rest_score(responses, DIF_ITEM),
        item_name="small effect",
    )
    assert result.p_uniform < 0.05
    assert result.delta_r2 < dif.NEGLIGIBLE_MAX
    assert result.magnitude == "negligible"
    assert not result.flagged
    assert result.classification == "none"


# ---------------------------------------------------- matching on rest score
def test_rest_score_leaves_out_the_studied_item():
    values = np.arange(12).reshape(4, 3)
    assert np.array_equal(dif.rest_score(values, 1), values[:, [0, 2]].sum(axis=1))


def _short_scale(shift: float = 1.0, n: int = 2000, seed: int = 4242):
    """Three items, DIF planted on the first, where one item is a third of the total."""
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, n)
    theta = rng.normal(size=n)
    columns = []
    for j in range(3):
        col = np.zeros(n, dtype=int)
        for level in (0, 1):
            mask = group == level
            thresholds = np.asarray(B_TRUE) + (shift if j == 0 else 0.0) * level
            responses, _ = irt.simulate_grm(
                int(mask.sum()),
                [1.3],
                [thresholds],
                seed=seed + 10 * j + level,
                theta=theta[mask],
            )
            col[mask] = responses[:, 0]
        columns.append(col)
    return np.column_stack(columns), group


def test_matching_on_the_total_costs_power_on_a_short_scale():
    """The studied item is a third of a three-item total, so including it
    matches partly on the very difference under test."""
    responses, group = _short_scale()
    rest = dif.ordinal_logistic_dif(
        responses[:, 0], group, dif.rest_score(responses, 0), matching="rest"
    )
    total = dif.ordinal_logistic_dif(
        responses[:, 0], group, responses.sum(axis=1), matching="total"
    )
    assert total.chi2_uniform < rest.chi2_uniform
    assert rest.matching == "rest"
    assert total.matching == "total"


def test_matching_on_the_total_spreads_dif_onto_clean_items(scan):
    """The contaminated total carries one item's DIF to the rest of the scale."""
    _, responses, group = scan
    item = f"item{DIF_ITEM + 1}"
    rest = dif.dif_scan(responses, group, matching="rest").drop(index=item)
    total = dif.dif_scan(responses, group, matching="total").drop(index=item)
    assert total["beta_group"].abs().mean() > rest["beta_group"].abs().mean()
    assert total["chi2_uniform"].mean() > rest["chi2_uniform"].mean()


# ------------------------------------------------------------------ guards
def test_group_must_have_two_levels():
    responses, _ = planted_data(n=200)
    with pytest.raises(ValueError, match="two levels"):
        dif.ordinal_logistic_dif(
            responses[:, 0], np.zeros(responses.shape[0]), dif.rest_score(responses, 0)
        )


def test_lengths_must_agree():
    with pytest.raises(ValueError, match="same length"):
        dif.ordinal_logistic_dif(np.zeros(10), np.zeros(9), np.zeros(10))


def test_magnitude_labels():
    assert dif._magnitude(0.01) == "negligible"
    assert dif._magnitude(0.05) == "moderate"
    assert dif._magnitude(0.20) == "large"
