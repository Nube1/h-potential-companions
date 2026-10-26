"""Every proof figure draws and its numeric check passes."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("matplotlib")
_spec = importlib.util.spec_from_file_location("proof_figures", ROOT / "analysis/proof_figures.py")
proof_figures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proof_figures)


def test_all_proof_figures_draw_and_check(tmp_path):
    proof_figures.main(["--out", str(tmp_path)])
    index = json.loads((tmp_path / "figure_index.json").read_text())
    assert len(index) == len(proof_figures.ALL)
    for entry in index:
        assert (tmp_path / f"{entry['name']}.pdf").exists()
        assert entry["statement"] and entry["caption"]
