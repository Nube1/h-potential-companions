"""Everything a figure needs from one pipeline run, loaded once."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class RunContext:
    cfg: object
    out: Path
    data: pd.DataFrame  # analytic rows joined to scores; split column marks hold-out
    predictions: pd.DataFrame  # hold-out rows, one risk column per endpoint/model/horizon
    performance: pd.DataFrame
    equity: pd.DataFrame
    manifest: dict
    flow: dict | None

    @property
    def id(self) -> str:
        return self.cfg.raw["data"]["id"]

    @property
    def weight(self) -> str:
        return self.cfg.raw["data"]["weight"]

    @property
    def endpoints(self) -> dict:
        return self.cfg.raw["outcomes"]["endpoints"]

    @property
    def horizon(self) -> float:
        return self.cfg.horizons[0]

    @property
    def alpha(self) -> float:
        return float(self.cfg.raw["measurement"]["alpha"])

    def test(self) -> pd.DataFrame:
        return self.data[self.data["split"] == "test"].merge(self.predictions, on=self.id)

    def risk_column(self, endpoint: str, model: str) -> str:
        return f"risk__{endpoint}__{model}__{self.horizon:g}"


def load_run(output_dir: str | Path, cfg, data_path: str | Path | None = None) -> RunContext:
    out = Path(output_dir)
    audit = json.loads((out / "run_audit.json").read_text(encoding="utf-8"))
    data_path = Path(data_path or audit["data_path"])
    raw = pd.read_csv(data_path) if data_path.suffix == ".csv" else pd.read_parquet(data_path)
    scores = pd.read_csv(out / "participant_scores_RESTRICTED.csv")
    id_col = cfg.raw["data"]["id"]
    raw[id_col] = raw[id_col].astype(str)
    scores[id_col] = scores[id_col].astype(str)
    predictions = pd.read_csv(out / "predictions_RESTRICTED.csv")
    predictions[id_col] = predictions[id_col].astype(str)
    flow_path = out / "nhats_flow.json"
    return RunContext(
        cfg=cfg,
        out=out,
        data=raw.merge(scores, on=id_col, how="inner"),
        predictions=predictions,
        performance=pd.read_csv(out / "model_performance.csv"),
        equity=pd.read_csv(out / "equity_performance.csv"),
        manifest=json.loads((out / "data_manifest.json").read_text(encoding="utf-8")),
        flow=json.loads(flow_path.read_text(encoding="utf-8")) if flow_path.exists() else None,
    )
