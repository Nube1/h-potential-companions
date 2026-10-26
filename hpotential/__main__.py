"""Command-line scoring and population aggregation.

Examples::

    python -m hpotential score examples/profiles.csv scores.csv --alpha 0.5 --rho 0
    python -m hpotential ede scores.csv --epsilon 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .crf import H, ede

SCORE_COLUMNS = ("id", "P", "S", "R")


class InputError(Exception):
    """Bad input file content; reported to the user without a traceback."""


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            return list(reader.fieldnames or []), rows
    except OSError as exc:
        raise InputError(f"Cannot read {path}: {exc.strerror}") from exc


def _number(row: dict[str, str], column: str, line: int) -> float:
    raw = (row.get(column) or "").strip()
    try:
        value = float(raw)
    except ValueError:
        raise InputError(f"Line {line}: {column}={raw!r} is not a number") from None
    if not math.isfinite(value):
        raise InputError(f"Line {line}: {column} must be finite")
    return value


def _score(args: argparse.Namespace) -> dict[str, object]:
    if args.input.resolve() == args.output.resolve():
        raise InputError("Output path must differ from input path")
    columns, rows = _read_rows(args.input)
    if sorted(columns) != sorted(SCORE_COLUMNS):
        raise InputError(f"Expected exactly the columns {','.join(SCORE_COLUMNS)}, got {columns}")
    if not rows:
        raise InputError("Input has no data rows")

    result = []
    for line, row in enumerate(rows, start=2):
        p, s, r = (_number(row, c, line) for c in ("P", "S", "R"))
        try:
            h = H(p, s, r, rho=args.rho, alpha=args.alpha)
        except ValueError as exc:
            raise InputError(f"Line {line} (id={row['id']}): {exc}") from None
        result.append({**row, "H": h, "alpha": args.alpha, "rho": args.rho})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
    return {"rows": len(result), "output": str(args.output), "units": "same as supplied P"}


def _ede(args: argparse.Namespace) -> dict[str, object]:
    columns, rows = _read_rows(args.input)
    if "H" not in columns:
        raise InputError("Input needs an H column (run the score command first)")
    if not rows:
        raise InputError("Input has no data rows")
    values = [_number(row, "H", line) for line, row in enumerate(rows, start=2)]
    weights = (
        [_number(row, "weight", line) for line, row in enumerate(rows, start=2)]
        if "weight" in columns
        else None
    )
    try:
        value = ede(values, args.epsilon, weights)
    except ValueError as exc:
        raise InputError(str(exc)) from None
    return {"epsilon": args.epsilon, "ede": value, "n": len(rows), "weighted": weights is not None}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hpotential",
        description="H-potential research scoring. Not clinically calibrated.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Score a CSV with exactly the columns id,P,S,R")
    score.add_argument("input", type=Path)
    score.add_argument("output", type=Path)
    score.add_argument("--alpha", type=float, default=0.5, help="weight on S, in (0, 1)")
    score.add_argument("--rho", type=float, default=0.0, help="CES exponent; 0 is geometric")
    score.set_defaults(handler=_score)

    pop = sub.add_parser("ede", help="EDE of a scored CSV with an H column and optional weight")
    pop.add_argument("input", type=Path)
    pop.add_argument("--epsilon", type=float, default=0.0, help="inequality aversion, >= 0")
    pop.set_defaults(handler=_ede)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = args.handler(args)
    except InputError as exc:
        print(f"hpotential {args.command}: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
