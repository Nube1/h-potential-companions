"""Plug NHATS data in and get the analysis back, from Python or over local HTTP.

NHATS has no public data API: the files come from nhats.org after you
register and accept the data use terms. This module is the plug on our side.
Point it at the downloaded files (a folder, or the zip files as downloaded)
and it checks them, builds the analytic extract, fits the models and draws
the figures.

Python::

    from hpotential.api import analyse
    summary = analyse("C:/data/nhats", "outputs/nhats")

HTTP (binds to this computer only, so the data never leave it)::

    hpotential serve --port 8765
    POST /check     {"source": "C:/data/nhats"}
    POST /analyse   {"source": "C:/data/nhats", "run": "nhats_2026_10"}
    GET  /runs                      list finished runs
    GET  /runs/<run>                summary JSON
    GET  /runs/<run>/figures/<file> a figure (PNG or PDF)
    GET  /health
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import load_config

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/study.yaml"
DEFAULT_LAYOUT = PACKAGE_ROOT / "configs/nhats_layout.yaml"
DEFAULT_RUNS = PACKAGE_ROOT / "outputs/api"
RUN_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
FIGURE_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.(png|pdf|json|md)$")


def prepare_source(source: str | Path, work_dir: str | Path | None = None) -> Path:
    """A folder holding the NHATS files, unpacking any zip files found.

    ``source`` may be a folder (zips inside it are unpacked beside a copy),
    a single zip file, or a folder of already-unpacked files.
    """
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"No such file or folder: {source}")
    zips = [source] if source.suffix.lower() == ".zip" else sorted(source.glob("*.zip"))
    if not zips:
        return source
    digest = hashlib.sha256("|".join(f"{z}:{z.stat().st_size}" for z in zips).encode()).hexdigest()[
        :12
    ]
    target = Path(work_dir or PACKAGE_ROOT / "data/raw/unpacked") / digest
    if not target.exists():
        target.mkdir(parents=True)
        for z in zips:
            with zipfile.ZipFile(z) as archive:
                for member in archive.infolist():
                    name = Path(member.filename)
                    if member.is_dir() or name.is_absolute() or ".." in name.parts:
                        continue  # refuse path traversal
                    dest = target / name.name
                    with archive.open(member) as src, dest.open("wb") as out:
                        shutil.copyfileobj(src, out)
    if source.is_dir():
        for f in source.iterdir():
            if f.is_file() and f.suffix.lower() != ".zip" and not (target / f.name).exists():
                shutil.copy2(f, target / f.name)
    return target


def check_source(source: str | Path, layout_path: str | Path = DEFAULT_LAYOUT) -> dict:
    from .nhats import Layout, check

    folder = prepare_source(source)
    report = check(folder, Layout.load(layout_path)).to_dict()
    report["folder"] = str(folder)
    return report


def analyse(
    source: str | Path,
    output_dir: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    layout_path: str | Path = DEFAULT_LAYOUT,
    figures: bool = True,
    reps: int | None = None,
) -> dict:
    """NHATS files in, results and figures out. Returns a summary dict.

    The summary holds only aggregate numbers. Participant-level files are
    written to ``output_dir`` with RESTRICTED in their names.
    """
    from .nhats import Layout, build
    from .pipeline import run

    folder = prepare_source(source)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame, flow = build(folder, Layout.load(layout_path))
    extract = out / "nhats_analytic_RESTRICTED.csv"
    frame.to_csv(extract, index=False)
    (out / "nhats_flow.json").write_text(json.dumps(flow, indent=2), encoding="utf-8")
    cfg = load_config(config_path)
    performance = run(extract, cfg, out)
    summary = {
        "flow": flow,
        "performance": json.loads(performance.to_json(orient="records")),
        "figures": [],
        "claim_boundary": "Development and internal hold-out validation only; not external validation.",
    }
    if figures:
        from .figures import draw_all

        index = draw_all(out, cfg, data_path=extract, reps=reps)
        summary["figures"] = [
            {k: e[k] for k in ("name", "group", "caption")} for e in json.loads(index.read_text())
        ]
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    return summary


# ----------------------------------------------------------------------- HTTP


class _Handler(BaseHTTPRequestHandler):
    runs_dir: Path = DEFAULT_RUNS
    lock = threading.Lock()

    def log_message(self, fmt, *args):  # quieter console
        print(f"[api] {self.address_string()} {fmt % args}")

    def _send(self, code: int, body, content_type: str = "application/json") -> None:
        data = (
            body if isinstance(body, bytes) else json.dumps(body, indent=2, default=float).encode()
        )
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):  # noqa: N802
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        if parts == ["health"]:
            return self._send(200, {"status": "ok"})
        if parts == ["runs"]:
            runs = sorted(p.name for p in self.runs_dir.glob("*") if (p / "summary.json").exists())
            return self._send(200, {"runs": runs})
        if len(parts) == 2 and parts[0] == "runs" and RUN_NAME.match(parts[1]):
            summary = self.runs_dir / parts[1] / "summary.json"
            if summary.exists():
                return self._send(200, json.loads(summary.read_text()))
        if len(parts) == 4 and parts[0] == "runs" and parts[2] == "figures":
            if RUN_NAME.match(parts[1]) and FIGURE_NAME.match(parts[3]):
                f = self.runs_dir / parts[1] / "figures" / parts[3]
                if f.exists():
                    kind = {
                        "png": "image/png",
                        "pdf": "application/pdf",
                        "json": "application/json",
                        "md": "text/markdown",
                    }
                    return self._send(200, f.read_bytes(), kind[f.suffix[1:]])
        return self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        try:
            body = self._json()
            if self.path == "/check":
                return self._send(200, check_source(body["source"]))
            if self.path == "/analyse":
                run_name = body.get("run") or "latest"
                if not RUN_NAME.match(run_name):
                    return self._send(
                        400, {"error": "run must be letters, digits, '.', '_' or '-'"}
                    )
                with self.lock:  # one analysis at a time
                    summary = analyse(
                        body["source"], self.runs_dir / run_name, reps=body.get("reps")
                    )
                summary["figure_urls"] = [
                    f"/runs/{run_name}/figures/{e['name']}.png" for e in summary["figures"]
                ]
                return self._send(200, summary)
        except KeyError as exc:
            return self._send(400, {"error": f"missing field {exc}"})
        except (FileNotFoundError, ValueError) as exc:
            return self._send(422, {"error": str(exc)})
        return self._send(404, {"error": "not found"})


def serve(port: int = 8765, runs_dir: str | Path = DEFAULT_RUNS) -> None:
    """Serve the API on 127.0.0.1 only."""
    _Handler.runs_dir = Path(runs_dir)
    _Handler.runs_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    print(f"H-potential data API on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ["analyse", "check_source", "prepare_source", "serve"]
