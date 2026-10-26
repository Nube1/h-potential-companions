#!/usr/bin/env python3
"""Run some or all of the six paper analyses and write their JSON results.

Examples
--------
Run everything at published settings, which takes a few minutes::

    python scripts/run_all.py --outdir out

Run papers 1 to 3 in quick mode, which takes under three minutes in total and
does NOT reproduce the published numbers::

    python scripts/run_all.py --papers 1,2,3 --quick --outdir /tmp/hpq
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from hpotential.analyses import PAPERS

QUICK_WARNING = (
    "QUICK MODE: replicate counts, grids and Monte Carlo draws are cut. "
    "The numbers below do NOT reproduce the published results. "
    "Run without --quick for those."
)


def parse_papers(text: str) -> list[int]:
    """Turn ``"1,3,6"`` into ``[1, 3, 6]``, rejecting unknown paper numbers."""
    wanted = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            number = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"not a paper number: {part!r}") from exc
        if number not in PAPERS:
            raise argparse.ArgumentTypeError(f"no such paper: {number}")
        wanted.append(number)
    if not wanted:
        raise argparse.ArgumentTypeError("no papers selected")
    return wanted


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--papers",
        type=parse_papers,
        default=list(PAPERS),
        help="comma-separated paper numbers, for example 1,2,3. Default: all six.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("out"),
        help="directory for the JSON results. Default: out",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="cut every replicate count so the run finishes in minutes. "
        "Quick mode does not reproduce the published values.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        print("=" * 78)
        print(QUICK_WARNING)
        print("=" * 78)
    else:
        print("Full mode: published settings. This takes a few minutes.")

    timings: dict[str, float] = {}
    failures: list[str] = []
    for number in args.papers:
        module = PAPERS[number]
        label = module.__name__.rsplit(".", 1)[-1]
        print(f"\npaper {number}  {label}")
        start = time.perf_counter()
        try:
            module.run(outdir=args.outdir, quick=args.quick)
        except Exception as exc:  # noqa: BLE001  (a driver reports, it does not raise)
            elapsed = time.perf_counter() - start
            print(f"  failed after {elapsed:.1f} s: {exc}")
            failures.append(label)
            continue
        elapsed = time.perf_counter() - start
        timings[label] = elapsed
        print(f"  done in {elapsed:.1f} s -> {args.outdir / f'paper{number}.json'}")

    total = sum(timings.values())
    print(f"\ntotal {total:.1f} s over {len(timings)} analyses")
    if args.quick:
        print(QUICK_WARNING)
    (args.outdir / "run_all_timings.json").write_text(
        json.dumps({"quick": args.quick, "seconds": timings}, indent=1)
    )
    if failures:
        print("failed: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
