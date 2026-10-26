from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


def horizon_status(time, event, horizon):
    time, event = np.asarray(time), np.asarray(event)
    known = (time >= horizon) | ((event == 1) & (time <= horizon))
    y = ((event == 1) & (time <= horizon)).astype(int)
    return known, y


def fit_predict_cox(train, test, time_col, event_col, predictors, horizon, weight_col=None):
    cols = [time_col, event_col, *predictors] + ([weight_col] if weight_col else [])
    tr = train[cols].dropna().copy()
    cph = CoxPHFitter(penalizer=0.01)
    kwargs = {"duration_col": time_col, "event_col": event_col}
    if weight_col:
        kwargs.update(weights_col=weight_col, robust=True)
    cph.fit(tr, **kwargs)
    surv = cph.predict_survival_function(test[predictors], times=[horizon]).iloc[0].to_numpy()
    return 1 - surv, cph


def metrics(time, event, risk, horizon, thresholds=(0.05, 0.10, 0.20)):
    known, y = horizon_status(time, event, horizon)
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


def equity_metrics(frame, group_cols, time_col, event_col, risk_col, horizon):
    rows = []
    for col in group_cols:
        for level, g in frame.groupby(col, dropna=False):
            row = metrics(g[time_col], g[event_col], g[risk_col], horizon)
            row.update(group=col, level=str(level))
            rows.append(row)
    return pd.DataFrame(rows)
