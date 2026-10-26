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
from .measurement import candidate_scores, fit_measurement, save_frozen_model, transform_measurement


def _encoded_covariates(frame, cfg):
    cols=cfg.raw["data"]["covariates"]
    return pd.get_dummies(frame[cols], columns=[c for c in cols if frame[c].dtype == object], drop_first=True, dtype=float)

def run(data_path, cfg, output_dir="outputs"):
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    df, manifest=load_and_validate(data_path,cfg); manifest["config_sha256"]=cfg.digest
    write_manifest(manifest,out/"data_manifest.json")
    ids=np.arange(len(df)); tr_idx,te_idx=train_test_split(ids,test_size=cfg.raw["validation"]["test_fraction"],random_state=cfg.seed)
    train=df.iloc[tr_idx].copy()
    model=fit_measurement(train,cfg); save_frozen_model(model,out/"measurement_model_frozen.json")
    all_domains=transform_measurement(df,model,cfg); scores=candidate_scores(all_domains,cfg.raw["measurement"]["alpha"],cfg.raw["measurement"]["rho_sensitivity"])
    analysis=pd.concat([df.reset_index(drop=True),all_domains.reset_index(drop=True),scores.reset_index(drop=True)],axis=1)
    cov=_encoded_covariates(analysis,cfg); analysis=pd.concat([analysis,cov.add_prefix("cov_")],axis=1)
    predictors={"baseline":["cov_"+c for c in cov.columns], "additive":["additive",*["cov_"+c for c in cov.columns]],
                "geometric_crf":["geometric_crf",*["cov_"+c for c in cov.columns]]}
    for c in cfg.raw["validation"].get("comparators",[]): predictors[c]=[c,*predictors["baseline"]]
    result_rows=[]; equity_rows=[]
    for endpoint,spec in cfg.raw["outcomes"]["endpoints"].items():
      for horizon in cfg.horizons:
       for name,preds in predictors.items():
        risk,_=fit_predict_cox(analysis.iloc[tr_idx],analysis.iloc[te_idx],spec["time"],spec["event"],preds,horizon,cfg.raw["data"]["weight"])
        m=metrics(analysis.iloc[te_idx][spec["time"]],analysis.iloc[te_idx][spec["event"]],risk,horizon,cfg.raw["validation"]["decision_thresholds"])
        m.update(endpoint=endpoint,horizon_days=horizon,model=name); result_rows.append(m)
        eframe=analysis.iloc[te_idx].copy(); eframe["risk"]=risk
        eq=equity_metrics(eframe,cfg.raw["validation"]["equity_groups"],spec["time"],spec["event"],"risk",horizon)
        eq["endpoint"],eq["horizon_days"],eq["model"]=endpoint,horizon,name; equity_rows.append(eq)
    pd.DataFrame(result_rows).to_csv(out/"model_performance.csv",index=False)
    pd.concat(equity_rows,ignore_index=True).to_csv(out/"equity_performance.csv",index=False)
    analysis[[cfg.raw["data"]["id"],*all_domains.columns,*scores.columns]].to_csv(out/"participant_scores_RESTRICTED.csv",index=False)
    audit={"status":"completed","claim_boundary":"Development split only; not external validation.","seed":cfg.seed,
           "python":sys.version,"platform":platform.platform(),"measurement_outcome_firewall":True}
    (out/"run_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    return pd.DataFrame(result_rows)
