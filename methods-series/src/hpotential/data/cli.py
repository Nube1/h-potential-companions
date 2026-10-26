"""Command line entry point for real-data validation and bundle creation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_dataset, write_analysis_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hpotential-data")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate data without writing participant rows")
    validate.add_argument("data")
    validate.add_argument("--schema", required=True)
    validate.add_argument("--sheet")
    validate.add_argument("--report", type=Path)
    prepare = sub.add_parser("prepare", help="validate and write a de-identified analysis bundle")
    prepare.add_argument("data")
    prepare.add_argument("--schema", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--sheet")
    prepare.add_argument("--format", choices=("csv", "parquet"), default="csv")
    prepare.add_argument("--include-identifiers", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = load_dataset(args.data, args.schema, sheet_name=args.sheet, strict=False)
    report = bundle.report.to_dict()
    rendered = json.dumps(report, indent=2)
    if getattr(args, "report", None):
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered)
    if not bundle.report.valid:
        return 2
    if args.command == "prepare":
        files = write_analysis_bundle(
            bundle,
            args.out,
            include_identifiers=args.include_identifiers,
            output_format=args.format,
        )
        print(json.dumps({k: str(v) for k, v in files.items()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
