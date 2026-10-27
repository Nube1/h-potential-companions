from pathlib import Path

import pandas as pd
import pytest

from hpotential.config import load_config
from hpotential.data import load_and_validate, synthetic_dataset
from hpotential.measurement import candidate_scores, fit_measurement, transform_measurement
from hpotential.pipeline import run

ROOT = Path(__file__).parents[1]


@pytest.fixture
def cfg():
    return load_config(ROOT / "configs/study.yaml")


def test_contract_rejects_missing(cfg, tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"participant_id": ["x"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_and_validate(p, cfg)


def test_duplicate_ids_refused(cfg, tmp_path):
    d = synthetic_dataset(cfg, 40)
    d.loc[1, "participant_id"] = d.loc[0, "participant_id"]
    p = tmp_path / "bad.csv"
    d.to_csv(p, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_and_validate(p, cfg)


def test_measurement_is_bounded_and_outcome_blind(cfg):
    d = synthetic_dataset(cfg, 100)
    m = fit_measurement(d.iloc[:70], cfg)
    s = transform_measurement(d, m, cfg)
    c = candidate_scores(s, 0.5, [-1, 0.5])
    assert set(s) == {"biomedical", "social", "systemic"}
    assert ((s > 0) & (s < 1)).all().all()
    assert ((c["geometric_crf"] > 0) & (c["geometric_crf"] < 1)).all()
    # The imputation model sees indicators, age and sex only, never an outcome.
    assert not any("mortality" in c or "hospital" in c for c in m.columns)
    assert m.imputers[0].n_features_in_ == len(m.columns) + 2
    pooled = m.pooled()
    assert pooled["imputations"] == cfg.raw["measurement"]["imputations"]
    assert 0 < pooled["fit"]["cfi"] <= 1


def test_noncompensation():
    d = pd.DataFrame({"biomedical": [0.9, 0.9], "social": [0.8, 0.8], "systemic": [0.8, 1e-6]})
    c = candidate_scores(d, 0.5, [])
    assert c.geometric_crf.iloc[1] < 0.01 * c.geometric_crf.iloc[0]
    assert c.additive.iloc[1] > 0.5 * c.additive.iloc[0]


def test_end_to_end(cfg, tmp_path):
    p = tmp_path / "synthetic.csv"
    synthetic_dataset(cfg, 220).to_csv(p, index=False)
    out = tmp_path / "out"
    result = run(p, cfg, out)
    assert not result.empty
    for name in (
        "data_manifest.json",
        "measurement_model_frozen.json",
        "model_performance.csv",
        "equity_performance.csv",
        "run_audit.json",
    ):
        assert (out / name).exists()
