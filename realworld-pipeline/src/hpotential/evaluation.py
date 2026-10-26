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

    Without a competing event this is 1 - S(horizon) from one Cox model. With
    one, two cause-specific Cox models are fitted and the absolute risk is the
    cumulative incidence, sum over event times of S(t-) dH_1(t), where S is
    survival from both causes. Treating death as ordinary censoring would
    overstate the risk of hospitalisation.
    """
    cph = _cox(train, time_col, event_col, predictors, weight_col)
    if not competing_col:
        surv = cph.predict_survival_function(test[predictors], times=[horizon]).iloc[0].to_numpy()
        return 1 - surv, cph
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
    surv_before = np.exp(-np.vstack([zero, np.cumsum(d_h1 + d_h2, axis=0)[:-1]]))
    total = d_h1 + d_h2
    # Split each step's drop in overall survival between the causes by their
    # hazard increments, so the risk never exceeds 1 - S(horizon).
    share = np.divide(d_h1, total, out=np.zeros_like(total), where=total > 0)
    return (surv_before * (1 - np.exp(-total)) * share).sum(axis=0), cph


def metrics(time, event, risk, horizon, thresholds=(0.05, 0.10, 0.20), competing=None):
    known, y = horizon_status(time, event, horizon, competing)
    y, p = y[known], np.asarray(risk)[known]
    if len(np.unique(y)) < 2:
        return {"n_known": len(y), "auc": np.nan, "brier": np.nan}
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    cal = LogisticRegression().fit(lp.reshape(-1, 1), y)
    result = {
        "n_known": len(y),
        "events": int(y.sum()),
        "auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "calibration_intercept": float(cal.intercept_[0]),
        "calibration_slope": float(cal.coef_[0, 0]),
    }
    n = len(y)
    for t in thresholds:
        pred = p >= t
        result[f"net_benefit_{t:g}"] = (pred & (y == 1)).sum() / n - (
            pred & (y == 0)
        ).sum() / n * t / (1 - t)
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
