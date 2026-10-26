"""Weighted survey statistics used by the figures."""

from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_km(time, event, weight=None) -> pd.DataFrame:
    """Weighted Kaplan-Meier survival with unweighted numbers at risk."""
    t = np.asarray(time, float)
    e = np.asarray(event, float)
    w = np.ones_like(t) if weight is None else np.asarray(weight, float)
    ok = ~(np.isnan(t) | np.isnan(e) | np.isnan(w))
    t, e, w = t[ok], e[ok], w[ok]
    order = np.argsort(t, kind="stable")
    t, e, w = t[order], e[order], w[order]
    times = np.unique(t[e == 1])
    surv, s = [], 1.0
    rows = []
    for u in times:
        at_risk = t >= u
        n_w = w[at_risk].sum()
        d_w = w[(t == u) & (e == 1)].sum()
        s *= 1 - d_w / n_w if n_w > 0 else 1
        surv.append(s)
        rows.append((u, s, int(at_risk.sum())))
    frame = pd.DataFrame(rows, columns=["time", "survival", "n_at_risk"])
    return pd.concat(
        [pd.DataFrame({"time": [0.0], "survival": [1.0], "n_at_risk": [len(t)]}), frame]
    )


def survival_at(km: pd.DataFrame, horizon: float) -> float:
    s = km.loc[km["time"] <= horizon, "survival"]
    return float(s.iloc[-1]) if len(s) else 1.0


def n_at_risk(time, ticks) -> list[int]:
    t = np.asarray(time, float)
    return [int((t >= x).sum()) for x in ticks]


def weighted_mean_ci(x, w, z: float = 1.959963984540054) -> tuple[float, float, float]:
    """Weighted mean with a normal interval from the effective sample size.

    A design-based variance needs strata and PSUs; this interval ignores
    clustering and is labelled as approximate wherever it is shown.
    """
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = ~(np.isnan(x) | np.isnan(w))
    x, w = x[ok], w[ok]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = np.average(x, weights=w)
    n_eff = w.sum() ** 2 / (w**2).sum()
    sd = np.sqrt(np.average((x - m) ** 2, weights=w))
    half = z * sd / np.sqrt(max(n_eff, 1))
    return float(m), float(m - half), float(m + half)


def atkinson_ede(x, w, epsilon: float) -> float:
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = ~(np.isnan(x) | np.isnan(w)) & (x > 0)
    x, w = x[ok], w[ok] / w[ok].sum()
    if abs(epsilon - 1) < 1e-9:
        return float(np.exp(np.sum(w * np.log(x))))
    return float(np.sum(w * x ** (1 - epsilon)) ** (1 / (1 - epsilon)))


def concentration_curve(h, rank_by, w) -> tuple[np.ndarray, np.ndarray, float]:
    """Cumulative population share against cumulative share of ``h``.

    Ranked from poorest to richest by ``rank_by``. Returns the curve and the
    concentration index (negative = concentrated among the poorer).
    """
    df = pd.DataFrame({"h": h, "y": rank_by, "w": w}).dropna()
    df = df.sort_values("y", kind="stable")
    w = df["w"].to_numpy() / df["w"].sum()
    hw = df["h"].to_numpy() * w
    pop = np.concatenate([[0], np.cumsum(w)])
    share = np.concatenate([[0], np.cumsum(hw) / hw.sum()])
    rank = np.cumsum(w) - w / 2
    mu = hw.sum()
    ci = 2 * np.sum(w * (df["h"].to_numpy() - mu) * (rank - 0.5)) / mu
    return pop, share, float(ci)


def bootstrap(stat, n: int, reps: int, seed: int) -> np.ndarray:
    """Statistic over ``reps`` resamples of ``n`` rows (index arrays)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        try:
            out.append(stat(idx))
        except ValueError:
            continue
    return np.asarray(out, float)


def weighted_cif(time, event, competing, weight=None) -> pd.DataFrame:
    """Weighted Aalen-Johansen cumulative incidence of ``event`` with a competing event.

    Returns columns time and cif. A person with the competing event leaves the
    risk set and can no longer have the event of interest, unlike censoring.
    """
    t = np.asarray(time, float)
    e = np.asarray(event, float)
    c = np.asarray(competing, float)
    w = np.ones_like(t) if weight is None else np.asarray(weight, float)
    ok = ~(np.isnan(t) | np.isnan(e) | np.isnan(c) | np.isnan(w))
    t, e, c, w = t[ok], e[ok], c[ok], w[ok]
    times = np.unique(t[(e == 1) | (c == 1)])
    surv, cif, rows = 1.0, 0.0, [(0.0, 0.0)]
    for u in times:
        n_w = w[t >= u].sum()
        if n_w <= 0:
            continue
        d1 = w[(t == u) & (e == 1)].sum()
        d2 = w[(t == u) & (c == 1)].sum()
        cif += surv * d1 / n_w
        surv *= 1 - (d1 + d2) / n_w
        rows.append((u, cif))
    return pd.DataFrame(rows, columns=["time", "cif"])


def cif_at(frame: pd.DataFrame, horizon: float) -> float:
    s = frame.loc[frame["time"] <= horizon, "cif"]
    return float(s.iloc[-1]) if len(s) else 0.0
