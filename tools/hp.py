#!/usr/bin/env python3
"""One command for the whole repository.

    python tools/hp.py setup            # one virtual environment per project, in .venvs/
    python tools/hp.py test             # every test suite
    python tools/hp.py build            # results, papers and every figure
    python tools/hp.py nhats --data-dir PATH   # real NHATS files -> extract -> run -> figures
    python tools/hp.py dist             # collect papers and figures into dist/ for a release
    python tools/hp.py all              # test, build, dist

Each project installs a package called hpotential, so each gets its own
environment. With --here, commands use the current interpreter instead (CI
installs one project per job).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENVS = ROOT / ".venvs"
PROJECTS = {
    "companions": ["-r", "requirements-dev.txt"],
    "methods-series": ["-e", ".[full,dev]"],
    "realworld-pipeline": ["-e", ".[dev]"],
}


def python_for(project: str, here: bool) -> str:
    if here:
        return sys.executable
    sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    exe = VENVS / project / sub
    if not exe.exists():
        sys.exit(f"No environment for {project}. Run: python tools/hp.py setup")
    return str(exe)


def sh(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    print(f"\n[{cwd.relative_to(ROOT) if cwd != ROOT else '.'}] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, env={**os.environ, **(env or {})})


def setup(args) -> None:
    for project in args.projects or PROJECTS:
        venv = VENVS / project
        if not venv.exists():
            sh([sys.executable, "-m", "venv", str(venv)], ROOT)
        py = python_for(project, False)
        sh([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], ROOT / project)
        sh([py, "-m", "pip", "install", "-q", *PROJECTS[project]], ROOT / project)


def test(args) -> None:
    for project in args.projects or PROJECTS:
        sh([python_for(project, args.here), "-m", "pytest", "-q"], ROOT / project)


def build(args) -> None:
    projects = args.projects or list(PROJECTS)
    env = {"MPLBACKEND": "Agg"}
    if "companions" in projects:
        py, cwd = python_for("companions", args.here), ROOT / "companions"
        sh([py, "analysis/reproduce.py", "--quiet"], cwd, env)
        sh([py, "analysis/proof_figures.py"], cwd, env)
        sh([py, "analysis/build_papers.py"], cwd, env)
        sh([py, "analysis/verify_delivery.py"], cwd, env)
    if "methods-series" in projects:
        py, cwd = python_for("methods-series", args.here), ROOT / "methods-series"
        sh([py, "scripts/regenerate_figures.py", "--all", "--outdir", "figures"], cwd, env)
    if "realworld-pipeline" in projects:
        py, cwd = python_for("realworld-pipeline", args.here), ROOT / "realworld-pipeline"
        sh(
            [py, "-m", "hpotential.cli", "demo", "--output", "outputs/demo", "--reps", str(args.reps)],
            cwd,
            env,
        )


def nhats(args) -> None:
    py, cwd = python_for("realworld-pipeline", args.here), ROOT / "realworld-pipeline"
    data_dir = str(Path(args.data_dir).resolve())
    extract = "data/processed/nhats_analytic.csv"
    sh([py, "-m", "hpotential.cli", "nhats-check", "--data-dir", data_dir], cwd)
    sh([py, "-m", "hpotential.cli", "nhats-build", "--data-dir", data_dir, "--output", extract], cwd)
    sh([py, "-m", "hpotential.cli", "run", "--data", extract, "--output", "outputs/nhats", "--figures"], cwd)
    print(
        "\nFigures and aggregate tables: realworld-pipeline/outputs/nhats/. "
        "Files marked RESTRICTED hold participant-level rows and must stay on this machine."
    )


def version() -> str:
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"


def dist(args) -> Path:
    tag = args.tag or version()
    out = ROOT / "dist" / f"h-potential-{tag}"
    if out.exists():
        shutil.rmtree(out)
    picks = {
        "companions/papers": ["*.pdf"],
        "companions/figures": ["*.pdf", "*.png", "FIGURES.md", "figure_index.json"],
        "companions/results": ["*.csv", "manifest.json"],
        "methods-series/figures": ["*.pdf", "*.png"],
        "realworld-pipeline/outputs/demo/figures": ["*.pdf", "*.png", "FIGURES.md", "figure_index.json"],
        "realworld-pipeline/outputs/demo": [
            "model_performance.csv",
            "equity_performance.csv",
            "nhats_flow.json",
        ],
    }
    copied = []
    for folder, patterns in picks.items():
        src = ROOT / folder
        for pattern in patterns:
            for f in sorted(src.glob(pattern)):
                if "RESTRICTED" in f.name:
                    continue
                dest = out / folder / f.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                copied.append(str(dest.relative_to(out)))
    (out / "MANIFEST.json").write_text(
        json.dumps({"version": tag, "files": copied}, indent=2), encoding="utf-8"
    )
    (out / "README.txt").write_text(
        "H-potential release bundle: companion papers, proof figures, methods-series figures and the\n"
        "real-world pipeline demo figures. The demo runs on made-up NHATS-shaped files and is not\n"
        "evidence about anyone.\n",
        encoding="utf-8",
    )
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out.parent))
    print(f"{len(copied)} files -> {archive.relative_to(ROOT)}")
    return archive


def guard(args) -> None:
    sh([sys.executable, "tools/guard_data.py"], ROOT)


def everything(args) -> None:
    guard(args)
    test(args)
    build(args)
    dist(args)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--here", action="store_true", help="use the current interpreter")
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn in [("setup", setup), ("test", test), ("build", build), ("all", everything)]:
        s = sub.add_parser(name)
        s.add_argument("projects", nargs="*", help=f"any of {', '.join(PROJECTS)} (default all)")
        s.add_argument("--reps", type=int, default=200, help="bootstrap replicates for the demo")
        s.add_argument("--tag", default=None)
        s.set_defaults(fn=fn)
    s = sub.add_parser("nhats")
    s.add_argument("--data-dir", required=True)
    s.set_defaults(fn=nhats)
    s = sub.add_parser("dist")
    s.add_argument("--tag", default=None)
    s.set_defaults(fn=dist)
    s = sub.add_parser("guard")
    s.set_defaults(fn=guard)
    args = p.parse_args(argv)
    unknown = set(getattr(args, "projects", None) or []) - set(PROJECTS)
    if unknown:
        p.error(f"unknown project(s): {', '.join(sorted(unknown))}")
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
