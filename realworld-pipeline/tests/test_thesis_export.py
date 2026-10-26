"""The thesis export refuses demo runs unless asked, and stamps them when it does."""

import json
from pathlib import Path

import numpy as np
import pytest

from hpotential.cli import main
from hpotential.thesis_export import export, harrell_c, uno_c

ROOT = Path(__file__).resolve().parents[1]


def test_uno_c_matches_harrell_without_censoring():
    rng = np.random.default_rng(1)
    t = rng.exponential(1, 300)
    risk = -t + rng.normal(0, 0.5, 300)
    e = np.ones(300, int)
    assert abs(uno_c(t, e, risk, np.inf) - harrell_c(t, e, risk)) < 1e-9


def test_export_from_demo(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    run = tmp_path / "run"
    assert (
        main(
            [
                "demo",
                "--workdir",
                str(tmp_path / "w"),
                "--output",
                str(run),
                "--n",
                "700",
                "--reps",
                "10",
            ]
        )
        == 0
    )
    thesis = tmp_path / "thesis"
    thesis.mkdir()
    with pytest.raises(ValueError, match="made-up"):
        export(run, thesis, reps=5)
    report = export(run, thesis, reps=5, allow_synthetic=True, internal=False)
    gen = thesis / "generated"
    assert report["synthetic"]
    values = (gen / "values.tex").read_text()
    assert r"\hpsynthetictrue" in values and r"\hpsetrows{hr}" in values
    for key in ("assoc", "discrimination", "calibration"):
        assert (gen / "text" / f"{key}.tex").read_text().count(r"\hpstamp") == 1
    assert (gen / "figures" / "km_overall.tex").exists()
    assert json.loads((gen / "EXPORT.json").read_text())["not_filled"]
