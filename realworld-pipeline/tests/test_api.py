import json
import threading
import time
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from hpotential import api
from hpotential.nhats import Layout, write_synthetic_nhats
from hpotential.nhats.ingest import find_file

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def nhats_zip(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    files = write_synthetic_nhats(d, Layout.load(ROOT / "configs/nhats_layout.yaml"), n=700, seed=3)
    z = d / "NHATS_download.zip"
    with zipfile.ZipFile(z, "w") as archive:
        for f in files:
            archive.write(f, f"NHATS/{f.stem}_V2{f.suffix}")  # versioned names, as released
    return z


def test_find_file_prefers_highest_version(tmp_path):
    for name in (
        "NHATS_Round_1_SP_File.dta",
        "NHATS_Round_1_SP_File_V3.dta",
        "NHATS_Round_1_SP_File_V2.csv",
    ):
        (tmp_path / name).write_bytes(b"")
    assert find_file(tmp_path, "nhats_round_1_sp_file").name == "NHATS_Round_1_SP_File_V3.dta"


def test_zip_is_unpacked_safely(tmp_path):
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as archive:
        archive.writestr("../escape.txt", "x")
        archive.writestr("ok/NHATS_Round_1_SP_File.csv", "spid\n1\n")
    folder = api.prepare_source(z, tmp_path / "work")
    assert (folder / "NHATS_Round_1_SP_File.csv").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_analyse_from_zip(nhats_zip, tmp_path):
    summary = api.analyse(nhats_zip, tmp_path / "out", reps=10)
    assert summary["flow"]["analytic_rows"] > 0
    assert {e["name"] for e in summary["figures"]} >= {
        "bio_04_km_mortality",
        "pol_02_inequality_aversion",
    }
    text = json.dumps(summary)
    assert "S1000" not in text and "participant_id" not in text  # aggregates only


def test_http_api(nhats_zip, tmp_path):
    api._Handler.runs_dir = tmp_path / "runs"
    server = ThreadingHTTPServer(("127.0.0.1", 0), api._Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    def post(path, body):
        req = urllib.request.Request(
            base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())

    try:
        assert json.loads(urllib.request.urlopen(base + "/health").read())["status"] == "ok"
        assert post("/check", {"source": str(nhats_zip)})["ok"] is True
        result = post("/analyse", {"source": str(nhats_zip), "run": "t1", "reps": 10})
        assert result["figure_urls"]
        png = urllib.request.urlopen(base + result["figure_urls"][0]).read()
        assert png[:4] == b"\x89PNG"
        assert "t1" in json.loads(urllib.request.urlopen(base + "/runs").read())["runs"]
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/runs/t1/figures/..%2Fsummary.json")
    finally:
        server.shutdown()
        time.sleep(0.1)
