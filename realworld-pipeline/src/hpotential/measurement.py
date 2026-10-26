from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

EPS = 1e-6


def _orient(df, spec):
    x = df[spec["indicators"]].apply(pd.to_numeric, errors="coerce").copy()
    if spec.get("direction") == "higher_burden":
        x = -x
    return x


def fit_measurement(train: pd.DataFrame, cfg):
    model, scores = {"domains": {}}, {}
    for name, spec in cfg.indicators.items():
        x = _orient(train, spec)
        imp = SimpleImputer(strategy="median").fit(x)
        scaler = StandardScaler().fit(imp.transform(x))
        fa = FactorAnalysis(n_components=1, random_state=cfg.seed).fit(
            scaler.transform(imp.transform(x))
        )
        load = fa.components_[0]
        if load.mean() < 0:
            load = -load
        model["domains"][name] = {
            "columns": list(x.columns),
            "median": imp.statistics_.tolist(),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "loadings": load.tolist(),
        }
        scores[name] = scaler.transform(imp.transform(x)) @ load
    s = pd.DataFrame(scores, index=train.index)
    model["bounds"] = {c: [float(s[c].quantile(0.01)), float(s[c].quantile(0.99))] for c in s}
    corr = s.corr()
    model["max_abs_domain_correlation"] = float(np.max(np.abs(corr.values[np.triu_indices(3, 1)])))
    return model


def transform_measurement(df: pd.DataFrame, model, cfg, floor: float | None = None):
    """Map each domain onto the unit interval with the frozen bounds.

    ``floor=None`` (or 0) is the preregistered rule: scores are clipped to
    [EPS, 1 - EPS], so anyone below the 1st training percentile sits at about
    zero. A positive ``floor`` maps the same bounds onto [floor, 1] instead, so
    no one is pushed to zero; it is reported as a sensitivity analysis.
    """
    out = {}
    for name, m in model["domains"].items():
        spec = cfg.indicators[name]
        x = _orient(df, spec).to_numpy(float)
        med = np.array(m["median"])
        x = np.where(np.isnan(x), med, x)
        z = (x - np.array(m["mean"])) / np.array(m["scale"])
        raw = z @ np.array(m["loadings"])
        lo, hi = model["bounds"][name]
        unit = (raw - lo) / (hi - lo)
        if floor:
            out[name] = floor + (1 - floor) * np.clip(unit, 0.0, 1.0)
        else:
            out[name] = np.clip(unit, EPS, 1 - EPS)
    return pd.DataFrame(out, index=df.index)


def candidate_scores(domains, alpha=0.5, rhos=(-1.0, 0.5)):
    p, s, r = (domains[c].to_numpy() for c in ("biomedical", "social", "systemic"))
    out = pd.DataFrame(index=domains.index)
    out["additive"] = (p + s + r) / 3
    out["geometric_crf"] = p * np.power(s, alpha) * np.power(r, 1 - alpha)
    for rho in rhos:
        if abs(rho) < 1e-10:
            out["ces_rho_0"] = np.cbrt(p * s * r)
        else:
            out[f"ces_rho_{rho:g}"] = np.power((p**rho + s**rho + r**rho) / 3, 1 / rho)
    return out


def save_frozen_model(model, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, sort_keys=True)
