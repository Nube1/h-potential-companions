"""Capability-realisation function (CRF), population aggregation and allocation.

Conventions
-----------
* ``P`` is nonnegative biomedical capacity in one declared unit. ``H`` is
  returned in the same unit.
* ``S`` (social enablement) and ``R`` (structural freedom, 1 minus normalised
  burden) are dimensionless and lie in ``[0, 1]``.
* ``alpha`` in ``(0, 1)`` is the weight on ``S``; ``1 - alpha`` is the weight
  on ``R``.

Nothing here is clinically calibrated. The functions implement the equations
of the companion papers and nothing more.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypedDict

__all__ = [
    "Action",
    "Allocation",
    "H",
    "allocate",
    "contextual_mean",
    "ede",
    "ede_gradient",
    "gap_decomposition",
    "gradients",
]

Profile = tuple[float, float, float]
"""One person's ``(P, S, R)``."""

# Switch to a second-order expansion when the CES exponent term is this small,
# so that the rho -> 0 (geometric) limit does not suffer cancellation.
_SERIES_THRESHOLD = 1e-6
# Largest number of candidate actions the exhaustive allocator will enumerate.
MAX_ACTIONS = 20
# Slack for floating-point round-off when testing budgets and domain bounds.
# Without it, e.g. 0.1 + 0.2 + 0.7 = 1.0000000000000002 would be rejected.
_BOUND_TOL = 1e-12
# Two objective values closer than this are treated as a tie.
_TIE_TOL = 1e-14


def _check_finite(x: float, name: str) -> None:
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x!r}")


def _check_profile(p: float, s: float, r: float, alpha: float) -> None:
    for x, name in ((p, "P"), (s, "S"), (r, "R"), (alpha, "alpha")):
        _check_finite(x, name)
    if p < 0:
        raise ValueError(f"P must be >= 0, got {p!r}")
    if not 0 <= s <= 1:
        raise ValueError(f"S must lie in [0, 1], got {s!r}")
    if not 0 <= r <= 1:
        raise ValueError(f"R must lie in [0, 1], got {r!r}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")


def contextual_mean(s: float, r: float, rho: float = 0.0, alpha: float = 0.5) -> float:
    """Weighted CES (power) mean of ``S`` and ``R``.

    ``rho = 0`` is the geometric mean, ``rho = -1`` harmonic, ``rho = 1``
    arithmetic (an explicitly compensatory comparator), and ``rho = -inf``
    the exact bottleneck ``min(S, R)``.

    The mean is evaluated in the log domain. Exact zeros are handled by their
    limits; no hidden epsilon floor is added.
    """
    _check_profile(1.0, s, r, alpha)
    if math.isnan(rho) or rho == math.inf:
        raise ValueError(f"rho must be finite or -inf, got {rho!r}")
    if rho == -math.inf:
        return min(s, r)
    if s == 0 or r == 0:
        if rho <= 0:
            return 0.0
        return (alpha * s**rho + (1 - alpha) * r**rho) ** (1 / rho)

    ls, lr = math.log(s), math.log(r)
    mu = alpha * ls + (1 - alpha) * lr
    if abs(rho) * abs(ls - lr) < _SERIES_THRESHOLD:
        # Cumulant expansion: log M_rho = mu + rho * Var / 2 + O(rho^2).
        variance = alpha * (1 - alpha) * (ls - lr) ** 2
        return math.exp(mu + rho * variance / 2)
    a, b = rho * ls, rho * lr
    m = max(a, b)  # log-sum-exp shift
    log_sum = m + math.log(alpha * math.exp(a - m) + (1 - alpha) * math.exp(b - m))
    return math.exp(log_sum / rho)


def H(p: float, s: float, r: float, rho: float = 0.0, alpha: float = 0.5) -> float:  # noqa: N802
    """Realised capability ``H = P * M_rho(S, R)`` in the unit of ``P``.

    With the default ``rho = 0`` this is paper 1 equation (1),
    ``H = P * S**alpha * R**(1 - alpha)``.
    """
    _check_profile(p, s, r, alpha)
    return p * contextual_mean(s, r, rho, alpha)


def gradients(p: float, s: float, r: float, alpha: float = 0.5) -> Profile:
    """Partial derivatives ``(dH/dP, dH/dS, dH/dR)`` of the geometric CRF.

    Defined only on the strictly positive interior, where the geometric form
    is differentiable.
    """
    _check_profile(p, s, r, alpha)
    if min(p, s, r) <= 0:
        raise ValueError("Interior derivatives require P, S, R > 0")
    h = H(p, s, r, alpha=alpha)
    return h / p, alpha * h / s, (1 - alpha) * h / r


def gap_decomposition(
    a: Profile, b: Profile, alpha: float = 0.5
) -> tuple[float, float, float, float]:
    """Exact split of ``log(H(a) / H(b))`` for two individual profiles.

    Returns ``(biomedical, social, structural, total)``; the first three sum
    to ``total``. Applies to two positive profiles, not to group mean scores.
    """
    _check_profile(*a, alpha)
    _check_profile(*b, alpha)
    if min(*a, *b) <= 0:
        raise ValueError("Log decomposition needs strictly positive profiles")
    parts = (
        math.log(a[0]) - math.log(b[0]),
        alpha * (math.log(a[1]) - math.log(b[1])),
        (1 - alpha) * (math.log(a[2]) - math.log(b[2])),
    )
    return (*parts, math.fsum(parts))


def _normalised_weights(n: int, weights: Iterable[float] | None) -> list[float]:
    w = [1.0] * n if weights is None else [float(x) for x in weights]
    if len(w) != n:
        raise ValueError(f"Expected {n} weights, got {len(w)}")
    if any(not math.isfinite(x) or x < 0 for x in w):
        raise ValueError("Weights must be finite and nonnegative")
    total = math.fsum(w)
    if total <= 0:
        raise ValueError("Weights must have a positive sum")
    return [x / total for x in w]


def ede(
    values: Iterable[float], epsilon: float = 0.0, weights: Iterable[float] | None = None
) -> float:
    """Weighted Atkinson equally distributed equivalent (EDE).

    ``epsilon`` is inequality aversion: 0 gives the weighted mean, 1 the
    weighted geometric mean, and larger values move towards the minimum.
    Zero scores are handled by their exact limits: for ``epsilon >= 1`` any
    zero with positive weight makes the EDE zero.
    """
    v = [float(x) for x in values]
    if not v:
        raise ValueError("Population must be nonempty")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError(f"epsilon must be finite and >= 0, got {epsilon!r}")
    if any(not math.isfinite(x) or x < 0 for x in v):
        raise ValueError("Scores must be finite and nonnegative")
    q = _normalised_weights(len(v), weights)
    pairs = [(x, w) for x, w in zip(v, q) if w > 0]

    if all(x == 0 for x, _ in pairs):
        return 0.0
    if epsilon >= 1 and any(x == 0 for x, _ in pairs):
        return 0.0
    if epsilon == 1:
        return math.exp(math.fsum(w * math.log(x) for x, w in pairs))

    power = 1 - epsilon
    if abs(power) < _SERIES_THRESHOLD and all(x > 0 for x, _ in pairs):
        # Expansion around the geometric mean as epsilon -> 1.
        mu = math.fsum(w * math.log(x) for x, w in pairs)
        var = math.fsum(w * (math.log(x) - mu) ** 2 for x, w in pairs)
        return math.exp(mu + power * var / 2)
    # Zero scores contribute nothing to sum(w * x**power) when power > 0.
    terms = [(power * math.log(x), w) for x, w in pairs if x > 0]
    m = max(t for t, _ in terms)
    log_sum = m + math.log(math.fsum(w * math.exp(t - m) for t, w in terms))
    return math.exp(log_sum / power)


def ede_gradient(
    values: Sequence[float], epsilon: float = 0.0, weights: Iterable[float] | None = None
) -> list[float]:
    """Gradient of :func:`ede` with respect to each individual score.

    ``dEDE/dh_i = q_i * (EDE / h_i) ** epsilon`` with normalised weights
    ``q_i``. Requires strictly positive scores.
    """
    v = [float(x) for x in values]
    if any(x <= 0 for x in v):
        raise ValueError("Population gradient requires strictly positive scores")
    w = None if weights is None else list(weights)
    total = ede(v, epsilon, w)
    q = _normalised_weights(len(v), w)
    return [qi * (total / x) ** epsilon for x, qi in zip(v, q)]


@dataclass(frozen=True)
class Action:
    """A binary policy option with a cost and an assumed response per person.

    ``changes[i]`` is the additive ``(dP, dS, dR)`` for population member
    ``i``. Responses are specified inputs, not estimated causal effects.
    """

    name: str
    cost: float
    changes: tuple[Profile, ...]


class Allocation(TypedDict):
    selected: list[str]
    cost: float
    ede: float
    scores: list[float]


def _apply(profiles: Sequence[Profile], chosen: Sequence[Action]) -> list[Profile] | None:
    """Post-policy profiles, or ``None`` if a domain leaves its bounds."""
    post = []
    for i, base in enumerate(profiles):
        p, s, r = (base[k] + math.fsum(a.changes[i][k] for a in chosen) for k in range(3))
        if (
            p < -_BOUND_TOL
            or not (-_BOUND_TOL <= s <= 1 + _BOUND_TOL)
            or not (-_BOUND_TOL <= r <= 1 + _BOUND_TOL)
        ):
            return None
        # Remove round-off only; real violations were rejected above.
        post.append((max(p, 0.0), min(max(s, 0.0), 1.0), min(max(r, 0.0), 1.0)))
    return post


def allocate(
    profiles: Sequence[Profile],
    actions: Sequence[Action],
    budget: float,
    epsilon: float = 0.0,
    alpha: float = 0.5,
    weights: Iterable[float] | None = None,
) -> Allocation:
    """Exact best portfolio of binary actions under a budget.

    Enumerates every subset of ``actions`` (so at most ``MAX_ACTIONS``),
    applies the additive responses, discards portfolios that push any domain
    outside its bounds, and maximises the EDE of the geometric CRF scores.
    Ties go to the cheaper portfolio, then to the earlier one in enumeration
    order. Responses are not clipped to fit: an infeasible portfolio is
    excluded.
    """
    if not math.isfinite(budget) or budget < 0:
        raise ValueError(f"budget must be finite and >= 0, got {budget!r}")
    if len(actions) > MAX_ACTIONS:
        raise ValueError(f"At most {MAX_ACTIONS} actions can be enumerated, got {len(actions)}")
    if len({a.name for a in actions}) != len(actions):
        raise ValueError("Action names must be unique")
    for prof in profiles:
        if len(prof) != 3:
            raise ValueError("Each profile must be (P, S, R)")
        _check_profile(*prof, alpha)
    w = None if weights is None else list(weights)
    if w is not None:
        _normalised_weights(len(profiles), w)
    for a in actions:
        if not math.isfinite(a.cost) or a.cost < 0:
            raise ValueError(f"Action {a.name!r} has invalid cost {a.cost!r}")
        if len(a.changes) != len(profiles):
            raise ValueError(f"Action {a.name!r} needs one response per population member")
        if any(len(d) != 3 or not all(math.isfinite(x) for x in d) for d in a.changes):
            raise ValueError(f"Action {a.name!r} has an invalid (dP, dS, dR) response")

    best: Allocation | None = None
    for mask in range(1 << len(actions)):
        chosen = [a for j, a in enumerate(actions) if mask & (1 << j)]
        cost = math.fsum(a.cost for a in chosen)
        if cost > budget + _BOUND_TOL * max(1.0, budget):
            continue
        post = _apply(profiles, chosen)
        if post is None:
            continue
        scores = [H(*prof, alpha=alpha) for prof in post]
        objective = ede(scores, epsilon, w)
        if (
            best is None
            or objective > best["ede"] + _TIE_TOL
            or (abs(objective - best["ede"]) <= _TIE_TOL and cost < best["cost"])
        ):
            best = Allocation(
                selected=[a.name for a in chosen], cost=cost, ede=objective, scores=scores
            )
    # The empty portfolio is always feasible because base profiles are valid.
    assert best is not None
    return best
