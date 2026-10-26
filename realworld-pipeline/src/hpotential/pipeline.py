"""Development and internal hold-out validation run."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import load_and_validate, write_manifest
from .evaluation import equity_metrics, fit_predict_cox, metrics
from .measurement import (
    candidate_scores,
    fit_measurement,
    save_frozen_model,
    transform_measurement,
)


def _encoded_covariates(frame: pd.DataFrame, cfg) -> pd.DataFrame:
    cols = cfg.raw["data"]["covariates"]
    categorical = [c for c in cols if not pd.api.types.is_numeric_dtype(frame[c])]
    return pd.get_dummies(frame[cols], columns=categorical, drop_first=True, dtype=float)


def _predictor_sets(cov_cols: list[str], cfg) -> dict[str, list[str]]:
    base = [f"cov_{c}" for c in cov_cols]
    sets = {
        "baseline": base,
        "additive": ["additive", *base],
        "geometric_crf": ["geometric_crf", *base],
    }
    if cfg.raw["measurement"].get("floor_sensitivity"):
        sets["geometric_crf_floor"] = ["geometric_crf_floor", *base]
    for c in cfg.raw["validation"].get("comparators", []):
        sets[c] = [c, *base]
    return sets


def run(data_path, cfg, output_dir="outputs") -> pd.DataFrame:
    """Fit measurement on the training split only, then compare models.

    Writes aggregate outputs plus two RESTRICTED participant-level files
    (scores and hold-out predictions) that must never leave the machine.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, manifest = load_and_validate(data_path, cfg)
    manifest["config_sha256"] = cfg.digest
    write_manifest(manifest, out / "data_manifest.json")

    val = cfg.raw["validation"]
    tr_idx, te_idx = train_test_split(
        np.arange(len(df)), test_size=val["test_fraction"], random_state=cfg.seed
    )
    model = fit_measurement(df.iloc[tr_idx], cfg)  # outcome-blind: training rows, indicators only
    save_frozen_model(model, out / "measurement_model_frozen.json")
    alpha = cfg.raw["measurement"]["alpha"]
    domains = transform_measurement(df, model, cfg).reset_index(drop=True)
    scores = candidate_scores(domains, alpha, cfg.raw["measurement"]["rho_sensitivity"])
    # Sensitivity: the same frozen bounds mapped onto [floor, 1].
    floor = float(cfg.raw["measurement"].get("floor_sensitivity", 0) or 0)
    if floor:
        floored = transform_measurement(df, model, cfg, floor).reset_index(drop=True)
        scores["geometric_crf_floor"] = candidate_scores(floored, alpha, [])["geometric_crf"]
        domains = pd.concat([domains, floored.add_suffix("_floor")], axis=1)
    analysis = pd.concat([df.reset_index(drop=True), domains, scores], axis=1)
    cov = _encoded_covariates(analysis, cfg)
    analysis = pd.concat([analysis, cov.add_prefix("cov_")], axis=1)
    predictors = _predictor_sets(list(cov.columns), cfg)

    weight = cfg.raw["data"]["weight"]
    train, test = analysis.iloc[tr_idx], analysis.iloc[te_idx]
    results, equity = [], []
    predictions = pd.DataFrame({cfg.raw["data"]["id"]: test[cfg.raw["data"]["id"]].to_numpy()})
    for endpoint, spec in cfg.raw["outcomes"]["endpoints"].items():
        competing = spec.get("competing")
        for horizon in cfg.horizons:
            for name, cols in predictors.items():
                risk, _ = fit_predict_cox(
                    train, test, spec["time"], spec["event"], cols, horizon, weight, competing
                )
                predictions[f"risk__{endpoint}__{name}__{horizon:g}"] = risk
                m = metrics(
                    test[spec["time"]],
                    test[spec["event"]],
                    risk,
                    horizon,
                    val["decision_thresholds"],
                    test[competing] if competing else None,
                )
                m.update(endpoint=endpoint, horizon_days=horizon, model=name)
                results.append(m)
                frame = test.assign(risk=risk)
                eq = equity_metrics(
                    frame,
                    val["equity_groups"],
                    spec["time"],
                    spec["event"],
                    "risk",
                    horizon,
                    competing,
                )
                eq["endpoint"], eq["horizon_days"], eq["model"] = endpoint, horizon, name
                equity.append(eq)

    pd.DataFrame(results).to_csv(out / "model_performance.csv", index=False)
    pd.concat(equity, ignore_index=True).to_csv(out / "equity_performance.csv", index=False)
    split = np.where(np.isin(np.arange(len(analysis)), te_idx), "test", "train")
    analysis.assign(split=split)[
        [cfg.raw["data"]["id"], "split", *domains.columns, *scores.columns]
    ].to_csv(out / "participant_scores_RESTRICTED.csv", index=False)
    predictions.to_csv(out / "predictions_RESTRICTED.csv", index=False)
    audit = {
        "status": "completed",
        "claim_boundary": "Development split only; not external validation.",
        "seed": cfg.seed,
        "python": sys.version,
        "platform": platform.platform(),
        "measurement_outcome_firewall": True,
        "data_path": str(Path(data_path).resolve()),
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
    }
    (out / "run_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return pd.DataFrame(results)
