"""Measurement: multiple imputation, the joint three-domain CFA, and domain scores.

The measurement model is fitted on the development split only and never sees
an outcome:

1. Missing indicators are filled by multiple imputation by chained equations
   (``m`` imputations, posterior draws), using the indicators, age and sex.
   Outcomes are left out of this imputation model so that the measurement
   stage stays blind to them.
2. The correlated three-domain CFA of Chapter 3 is fitted by maximum
   likelihood to each imputed development set, with the survey weights.
3. Each person's domain scores and second-order factor score are the regression
   factor scores averaged over the imputations.
4. For the realisation function each domain is mapped onto the unit interval
   with bounds frozen at the 1st and 99th development percentiles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from .cfa import CFAResult, fit_cfa, rubin

EPS = 1e-6
DOMAINS = ("biomedical", "social", "systemic")


def _orient(df, spec):
    x = df[spec["indicators"]].apply(pd.to_numeric, errors="coerce").copy()
    if spec.get("direction") == "higher_burden":
        x = -x
    return x


def _indicator_frame(df, cfg) -> pd.DataFrame:
    """The oriented indicators, after the optional not-asked floor.

    ``measurement.not_asked_floor`` ({flag, indicators, value}) is a
    sensitivity analysis: where ``flag`` is 1 (a proxy said the person could
    not be tested) the listed indicators are set to ``value`` instead of being
    imputed. The primary analysis leaves them to the imputation model.
    """
    rule = cfg.raw["measurement"].get("not_asked_floor")
    if rule and rule["flag"] in df:
        df = df.copy()
        hit = pd.to_numeric(df[rule["flag"]], errors="coerce").eq(1)
        for col in rule["indicators"]:
            df.loc[hit, col] = float(rule["value"])
    return pd.concat([_orient(df, cfg.indicators[d]) for d in DOMAINS], axis=1)


def _aux_frame(df, cfg=None) -> pd.DataFrame:
    """Variables that inform the imputation but are not indicators.

    Age and sex always. ``measurement.auxiliary`` in the configuration adds
    baseline flags that bear on why an indicator is missing: whether a proxy
    answered, whether the proxy said the person could not do the cognitive
    tests, and reported dementia. With these in the model, a value missing
    because the person could not be asked is imputed from people in the same
    position rather than from everyone. Outcomes are never auxiliary.
    """
    aux = pd.DataFrame(index=df.index)
    if "age" in df:
        aux["age"] = pd.to_numeric(df["age"], errors="coerce")
    if "sex" in df:
        aux["female"] = (df["sex"].astype(str).str.lower() == "female").astype(float)
    names = cfg.raw["measurement"].get("auxiliary", []) if cfg is not None else []
    for name in names:
        if name in df:
            aux[name] = pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    return aux


@dataclass
class MeasurementModel:
    columns: list
    pattern: dict
    imputers: list = field(default_factory=list)
    centre: list = field(default_factory=list)
    scale: list = field(default_factory=list)
    fits: list = field(default_factory=list)
    bounds: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- use
    def imputed(self, df, cfg):
        """The m standardised imputed indicator matrices for these rows."""
        X = _indicator_frame(df, cfg)
        Z = pd.concat([X, _aux_frame(df, cfg)], axis=1).to_numpy(float)
        p = X.shape[1]
        out = []
        for imp, c, s in zip(self.imputers, self.centre, self.scale, strict=True):
            Zi = imp.transform(Z)[:, :p]
            out.append((Zi - c) / s)
        return out

    def raw_scores(self, df, cfg) -> pd.DataFrame:
        """Domain and general factor scores, averaged over imputations."""
        dom, gen = [], []
        for Xi, fit in zip(self.imputed(df, cfg), self.fits, strict=True):
            dom.append(fit.scores(Xi))
            gen.append(fit.general_scores(Xi))
        out = pd.DataFrame(np.mean(dom, axis=0), columns=list(DOMAINS), index=df.index)
        ok = [g for g in gen if np.all(np.isfinite(g))]
        # Only imputations whose second-order model is admissible contribute.
        out["factor_score"] = np.mean(ok, axis=0) if ok else np.nan
        return out

    # ------------------------------------------------------------ results
    def pooled(self) -> dict:
        """Estimates pooled over imputations by Rubin's rules, with averaged fit."""
        fits: list[CFAResult] = self.fits
        load = [f.standardised_loadings() for f in fits]
        est = np.array([[e for e, _ in row] for row in load])
        var = np.array([[s**2 for _, s in row] for row in load])
        pl = rubin(est, var)
        so = [f.second_order() for f in fits]
        pso = rubin(
            np.array([[e for e, _ in r] for r in so]), np.array([[s**2 for _, s in r] for r in so])
        )
        fit_keys = ("chi2", "df", "cfi", "tli", "rmsea", "srmr")
        fit_mean = {k: float(np.mean([f.fit_indices()[k] for f in fits])) for k in fit_keys}
        rel = {
            d: {
                k: float(np.mean([f.reliability()[d][k] for f in fits]))
                for k in ("omega", "alpha", "ave")
            }
            for d in DOMAINS
        }
        corr = np.mean([f.factor_correlations() for f in fits], axis=0)
        htmt = np.mean([f.htmt() for f in fits], axis=0)
        det = {}
        for k in (*DOMAINS, "general"):
            vals = [f.determinacy()[k] for f in fits]
            det[k] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float("nan")
        admissible = [f.second_order_admissible() for f in fits]
        return {
            "imputations": len(fits),
            "n": int(fits[0].n.sum()),
            "indicators": self.columns,
            "fit": fit_mean,
            "loadings": {
                c: {
                    "estimate": float(pl["estimate"][i]),
                    "se": float(pl["se"][i]),
                    "df": float(pl["df"][i]),
                }
                for i, c in enumerate(self.columns)
            },
            "second_order_admissible": float(np.mean(admissible)),
            "second_order": {
                d: {"estimate": float(pso["estimate"][i]), "se": float(pso["se"][i])}
                for i, d in enumerate(DOMAINS)
            },
            "factor_correlations": corr.tolist(),
            "htmt": htmt.tolist(),
            "reliability": rel,
            "determinacy": det,
            "bounds": self.bounds,
        }


def fit_measurement(train: pd.DataFrame, cfg, se: bool = True) -> MeasurementModel:
    """Fit imputation, CFA and frozen bounds on the development rows."""
    mcfg = cfg.raw["measurement"]
    m = int(mcfg.get("imputations", 10))
    X = _indicator_frame(train, cfg)
    pattern, start = {}, 0
    for d in DOMAINS:
        k = len(cfg.indicators[d]["indicators"])
        pattern[d] = list(range(start, start + k))
        start += k
    model = MeasurementModel(columns=list(X.columns), pattern=pattern)
    Z = pd.concat([X, _aux_frame(train, cfg)], axis=1).to_numpy(float)
    w = None
    wcol = cfg.raw["data"].get("weight")
    if wcol and wcol in train:
        w = pd.to_numeric(train[wcol], errors="coerce").fillna(0).to_numpy()
    p = X.shape[1]
    for i in range(m):
        imp = IterativeImputer(sample_posterior=True, max_iter=10, random_state=cfg.seed + i)
        Zi = imp.fit_transform(Z)[:, :p]
        c, s = Zi.mean(axis=0), Zi.std(axis=0)
        s[s == 0] = 1.0
        model.imputers.append(imp)
        model.centre.append(c)
        model.scale.append(s)
        model.fits.append(fit_cfa((Zi - c) / s, pattern, w, se=se))
    raw = model.raw_scores(train, cfg)
    model.bounds = {
        d: [float(raw[d].quantile(0.01)), float(raw[d].quantile(0.99))] for d in DOMAINS
    }
    return model


def transform_measurement(df: pd.DataFrame, model: MeasurementModel, cfg, floor=None):
    """Map each domain onto the unit interval with the frozen bounds.

    ``floor=None`` (or 0) is the preregistered rule: scores are clipped to
    [EPS, 1 - EPS], so anyone below the 1st development percentile sits at
    about zero. A positive ``floor`` maps the same bounds onto [floor, 1]
    instead, so no one is pushed to zero; it is reported as a sensitivity
    analysis.
    """
    raw = model.raw_scores(df, cfg)
    out = {}
    for d in DOMAINS:
        lo, hi = model.bounds[d]
        unit = (raw[d].to_numpy() - lo) / (hi - lo)
        if floor:
            out[d] = floor + (1 - floor) * np.clip(unit, 0.0, 1.0)
        else:
            out[d] = np.clip(unit, EPS, 1 - EPS)
    return pd.DataFrame(out, index=df.index)


def factor_scores(df: pd.DataFrame, model: MeasurementModel, cfg) -> pd.Series:
    """The second-order general factor score: the outcome-independent comparator."""
    return model.raw_scores(df, cfg)["factor_score"]


def candidate_scores(domains, alpha=0.5, rhos=(-1.0, 0.5)):
    p, s, r = (domains[c].to_numpy() for c in DOMAINS)
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
    summary = model.pooled() if isinstance(model, MeasurementModel) else model
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
