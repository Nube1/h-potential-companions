from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


def horizon_status(time, event, horizon, competing=None):
    """Whose status at ``horizon`` is known, and who had the event by then.

    With a competing event (death before a first hospital stay), a person who
    had the competing event first is a known non-case.
    """
    time, event = np.asarray(time, float), np.asarray(event)
    y = ((event == 1) & (time <= horizon)).astype(int)
    known = (time >= horizon) | (y == 1)
    if competing is not None:
        known |= (np.asarray(competing) == 1) & (time <= horizon)
    return known, y


def _cox(frame, time_col, event_col, predictors, weight_col):
    cols = [time_col, event_col, *predictors] + ([weight_col] if weight_col else [])
    data = frame[cols].dropna().copy()
    cph = CoxPHFitter(penalizer=0.01)
    kwargs = {"duration_col": time_col, "event_col": event_col}
    if weight_col:
        kwargs.update(weights_col=weight_col, robust=True)
    return cph.fit(data, **kwargs)


def fit_predict_cox(
    train, test, time_col, event_col, predictors, horizon, weight_col=None, competing_col=None
):
    """Predicted risk of the event by ``horizon`` for the test rows.

    Without a competing event this is 1 - S(horizon) from one Cox model, with S
    the product-limit form of the Breslow estimate. With
    one, two cause-specific Cox models are fitted and the absolute risk is the
    cumulative incidence, sum over event times of S(t-) dH_1(t), where S is
    survival from both causes. Treating death as ordinary censoring would
    overstate the risk of hospitalisation.
    """
    cph = _cox(train, time_col, event_col, predictors, weight_col)
    if not competing_col:
        # Product-limit survival from the Breslow hazard steps, as Kaplan-Meier
        # is to Nelson-Aalen. With annual rounds exp(-H) would understate risk.
        grid = cph.baseline_cumulative_hazard_.index.to_numpy()
        grid = grid[grid <= horizon]
        if len(grid) == 0:
            return np.zeros(len(test)), cph
        h = cph.predict_cumulative_hazard(test[predictors], times=grid).to_numpy()
        d_h = np.clip(np.diff(np.vstack([np.zeros((1, h.shape[1])), h]), axis=0), 0, 1)
        return 1 - np.prod(1 - d_h, axis=0), cph
    other = _cox(train, time_col, competing_col, predictors, weight_col)
    grid = np.unique(
        np.concatenate(
            [cph.baseline_cumulative_hazard_.index, other.baseline_cumulative_hazard_.index]
        )
    )
    grid = grid[grid <= horizon]
    if len(grid) == 0:
        return np.zeros(len(test)), cph
    h1 = cph.predict_cumulative_hazard(test[predictors], times=grid).to_numpy()
    h2 = other.predict_cumulative_hazard(test[predictors], times=grid).to_numpy()
    zero = np.zeros((1, h1.shape[1]))
    d_h1 = np.clip(np.diff(np.vstack([zero, h1]), axis=0), 0, None)
    d_h2 = np.clip(np.diff(np.vstack([zero, h2]), axis=0), 0, None)
    # Product-limit (Aalen-Johansen) steps: with annual rounds the jumps are
    # large, and exp(-dH) would understate each step. The step probability is
    # capped at 1 and split between causes, so risk stays within [0, 1].
    total = d_h1 + d_h2
    step = np.minimum(total, 1.0)
    share = np.divide(d_h1, total, out=np.zeros_like(total), where=total > 0)
    surv_before = np.cumprod(np.vstack([np.ones((1, h1.shape[1])), 1 - step[:-1]]), axis=0)
    return (surv_before * step * share).sum(axis=0), cph


def ipcw_weights(time, event, horizon, competing=None) -> np.ndarray:
    """Inverse probability of censoring weights for status at ``horizon``.

    People whose status is known get 1 / G, with G the Kaplan-Meier survivor
    function of censoring just before their event (or at the horizon if still
    event-free); people censored before the horizon get 0. Without weights,
    dropping the censored inflates the observed risk.
    """
    t = np.asarray(time, float)
    e = np.asarray(event) == 1
    c = np.zeros_like(e) if competing is None else np.asarray(competing) == 1
    censored = ~(e | c) & np.isfinite(t)
    grid = np.unique(t[censored])
    at_risk = np.array([(t >= u).sum() for u in grid], float)
    d = np.array([(censored & (t == u)).sum() for u in grid], float)
    surv = np.cumprod(1 - d / at_risk)

    def g_before(x):
        if not len(grid):  # no censoring at all
            return np.ones_like(x, dtype=float)
        i = np.searchsorted(grid, x, side="left") - 1
        return np.where(i >= 0, surv[np.clip(i, 0, None)], 1.0)

    known, _ = horizon_status(t, event, horizon, competing)
    g = np.where(t >= horizon, g_before(np.full_like(t, horizon)), g_before(t))
    return np.where(known & (g > 0), 1 / np.where(g > 0, g, 1), 0.0)


def _logit(p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _citl(y, lp, w) -> float:
    """Calibration-in-the-large: the intercept with the slope fixed at one."""
    a = 0.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(a + lp)))
        step = (w * (y - mu)).sum() / max((w * mu * (1 - mu)).sum(), 1e-12)
        a += step
        if abs(step) < 1e-10:
            break
    return float(a)


def metrics(time, event, risk, horizon, thresholds=(0.05, 0.10, 0.20), competing=None):
    """Hold-out performance at ``horizon``, weighted for censoring (IPCW).

    AUC, Brier score and calibration use inverse probability of censoring
    weights, so people censored before the horizon are represented by those
    still followed. Calibration-in-the-large holds the slope at one; the slope
    comes from an unpenalised logistic recalibration.
    """
    known, y = horizon_status(time, event, horizon, competing)
    w_all = ipcw_weights(time, event, horizon, competing)
    y, p, w = y[known], np.asarray(risk, float)[known], w_all[known]
    if len(np.unique(y)) < 2:
        return {"n_known": len(y), "auc": np.nan, "brier": np.nan}
    lp = _logit(p)
    cal = LogisticRegression(C=np.inf).fit(lp.reshape(-1, 1), y, sample_weight=w)
    result = {
        "n_known": len(y),
        "events": int(y.sum()),
        "auc": roc_auc_score(y, p, sample_weight=w),
        "brier": brier_score_loss(y, p, sample_weight=w),
        "calibration_intercept": _citl(y, lp, w),
        "calibration_slope": float(cal.coef_[0, 0]),
        "observed_ipcw": float((w * y).sum() / w.sum()),
        "expected": float(np.mean(np.asarray(risk, float))),
    }
    total = w.sum()
    for t in thresholds:
        pred = p >= t
        result[f"net_benefit_{t:g}"] = (w * (pred & (y == 1))).sum() / total - (
            w * (pred & (y == 0))
        ).sum() / total * t / (1 - t)
    return result


def equity_metrics(frame, group_cols, time_col, event_col, risk_col, horizon, competing_col=None):
    rows = []
    for col in group_cols:
        for level, g in frame.groupby(col, dropna=False):
            comp = g[competing_col] if competing_col else None
            row = metrics(g[time_col], g[event_col], g[risk_col], horizon, competing=comp)
            row.update(group=col, level=str(level))
            rows.append(row)
    return pd.DataFrame(rows)
