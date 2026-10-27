"""Command line: NHATS ingestion, the validation run and the figures.

hpotential nhats-check  --data-dir /path/to/nhats
hpotential nhats-build  --data-dir /path/to/nhats --output data/processed/nhats_analytic.csv
hpotential run          --data data/processed/nhats_analytic.csv --output outputs --figures
hpotential figures      --output outputs
hpotential demo         # the same chain on made-up NHATS-shaped files
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .config import load_config

DEFAULT_CONFIG = "configs/study.yaml"
DEFAULT_LAYOUT = "configs/nhats_layout.yaml"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hpotential", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate-synthetic", help="flat synthetic analytic file (tests only)")
    g.add_argument("--config", default=DEFAULT_CONFIG)
    g.add_argument("--output", default="data/sample/synthetic.csv")
    g.add_argument("--n", type=int, default=750)

    for name, help_text in [
        ("nhats-check", "list absent NHATS files, variables and codes"),
        ("nhats-build", "build the canonical analytic extract from NHATS files"),
    ]:
        s = sub.add_parser(name, help=help_text)
        s.add_argument("--data-dir", required=True, help="folder holding the NHATS files")
        s.add_argument("--layout", default=DEFAULT_LAYOUT)
        if name == "nhats-build":
            s.add_argument("--output", default="data/processed/nhats_analytic.csv")

    s = sub.add_parser("nhats-synthetic", help="write made-up files in the NHATS layout")
    s.add_argument("--out-dir", default="data/raw/synthetic_nhats")
    s.add_argument("--layout", default=DEFAULT_LAYOUT)
    s.add_argument("--n", type=int, default=2000)

    r = sub.add_parser("run", help="measurement, models and hold-out evaluation")
    r.add_argument("--config", default=DEFAULT_CONFIG)
    r.add_argument("--data", required=True)
    r.add_argument("--output", default="outputs")
    r.add_argument("--figures", action="store_true", help="draw the figures afterwards")

    f = sub.add_parser("figures", help="draw every figure from a finished run")
    f.add_argument("--config", default=DEFAULT_CONFIG)
    f.add_argument("--output", default="outputs")
    f.add_argument("--figure-dir", default=None)
    f.add_argument("--reps", type=int, default=None, help="bootstrap replicates (config default)")

    a = sub.add_parser("analyse", help="NHATS folder or zip in, results and figures out")
    a.add_argument("--source", required=True, help="folder of NHATS files, or the downloaded zip")
    a.add_argument("--output", default="outputs/nhats")
    a.add_argument("--reps", type=int, default=None)

    v = sub.add_parser("serve", help="local HTTP data API on 127.0.0.1")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--runs-dir", default="outputs/api")

    d = sub.add_parser("demo", help="synthetic NHATS files -> extract -> run -> figures")
    d.add_argument("--config", default=DEFAULT_CONFIG)
    d.add_argument("--layout", default=DEFAULT_LAYOUT)
    d.add_argument("--workdir", default="data/raw/demo")
    d.add_argument("--output", default="outputs/demo")
    d.add_argument("--n", type=int, default=2000)
    d.add_argument("--reps", type=int, default=200)

    x = sub.add_parser("thesis-export", help="fill the thesis results chapters from a run")
    x.add_argument("--run", default="outputs/nhats", help="output folder of a finished run")
    x.add_argument("--thesis", default="../thesis", help="thesis folder (holds thesis.tex)")
    x.add_argument("--config", default=DEFAULT_CONFIG)
    x.add_argument("--reps", type=int, default=200, help="bootstrap replicates")
    x.add_argument("--no-internal", action="store_true", help="skip the bootstrap refits")
    x.add_argument("--jobs", type=int, default=0, help="processes for bootstrap refits (0 = all)")
    x.add_argument(
        "--allow-synthetic", action="store_true", help="test the wiring on a demo run (stamped)"
    )
    return p


def _build(data_dir, layout_path, output) -> Path:
    from .nhats import Layout, build

    frame, flow = build(data_dir, Layout.load(layout_path))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    (output.parent / "nhats_flow.json").write_text(json.dumps(flow, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in flow.items() if k != "indicator_missing_fraction"}))
    return output


def _run(cfg, data, output, figures, reps=None) -> None:
    from .pipeline import run

    result = run(data, cfg, output)
    flow = Path(data).parent / "nhats_flow.json"
    if flow.exists():
        shutil.copyfile(flow, Path(output) / "nhats_flow.json")
    print(
        result[["endpoint", "model", "auc", "brier", "calibration_slope"]]
        .round(4)
        .to_string(index=False)
    )
    if figures:
        from .figures import draw_all

        print(draw_all(output, cfg, reps=reps))


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    if a.command == "generate-synthetic":
        from .data import synthetic_dataset

        cfg = load_config(a.config)
        dest = Path(a.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        synthetic_dataset(cfg, a.n, cfg.seed).to_csv(dest, index=False)
        print(dest)
    elif a.command == "nhats-check":
        from .nhats import Layout, check

        report = check(a.data_dir, Layout.load(a.layout))
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.ok else 1
    elif a.command == "nhats-build":
        _build(a.data_dir, a.layout, a.output)
    elif a.command == "nhats-synthetic":
        from .nhats import Layout, write_synthetic_nhats

        for path in write_synthetic_nhats(a.out_dir, Layout.load(a.layout), a.n):
            print(path)
    elif a.command == "thesis-export":
        from .thesis_export import export

        report = export(
            a.run,
            a.thesis,
            a.config,
            a.reps,
            a.allow_synthetic,
            internal=not a.no_internal,
            jobs=a.jobs,
        )
        print(json.dumps(report, indent=2))
    elif a.command == "run":
        _run(load_config(a.config), a.data, a.output, a.figures)
    elif a.command == "figures":
        from .figures import draw_all

        print(draw_all(a.output, load_config(a.config), a.figure_dir, reps=a.reps))
    elif a.command == "analyse":
        from .api import analyse

        summary = analyse(a.source, a.output, reps=a.reps)
        print(
            json.dumps({"flow": summary["flow"], "figures": len(summary["figures"])}, default=float)
        )
    elif a.command == "serve":
        from .api import serve

        serve(a.port, a.runs_dir)
    elif a.command == "demo":
        from .nhats import Layout, write_synthetic_nhats

        raw = Path(a.workdir) / "nhats"
        write_synthetic_nhats(raw, Layout.load(a.layout), a.n)
        extract = _build(raw, a.layout, Path(a.workdir) / "nhats_analytic.csv")
        _run(load_config(a.config), extract, a.output, True, a.reps)
        # Mark the output so nothing downstream mistakes it for evidence.
        (Path(a.output) / "SYNTHETIC").write_text(
            "Made-up NHATS-shaped data from `hpotential demo`. Not results.\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
