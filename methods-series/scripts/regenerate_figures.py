#!/usr/bin/env python3
"""Regenerate the figures for one paper or for all six.

Each figure is drawn from the analysis it belongs to, so a figure and the JSON
beside it always describe the same run.

Examples
--------
Every figure of paper 1::

    python scripts/regenerate_figures.py --paper 1 --all --outdir figs

One named figure::

    python scripts/regenerate_figures.py --paper 4 --figure verification

Every figure of every paper, quickly, for a layout check::

    python scripts/regenerate_figures.py --all --quick --outdir /tmp/figs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from hpotential.analyses import PAPERS


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--paper", type=int, choices=sorted(PAPERS), help="paper number. Omit for all six."
    )
    parser.add_argument("--figure", type=str, help="single figure name to draw")
    parser.add_argument(
        "--all", action="store_true", help="draw every figure of the selected papers"
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("figs"), help="directory for the PNG files"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run the underlying analyses in quick mode. The figures then show "
        "values that do NOT reproduce the published results.",
    )
    parser.add_argument("--list", action="store_true", help="list the available figures and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    numbers = [args.paper] if args.paper else sorted(PAPERS)

    if args.list:
        for number in numbers:
            print(f"paper {number}: {', '.join(PAPERS[number].FIGURES)}")
        return 0

    if not args.all and args.figure is None:
        print("choose --all or --figure NAME (or --list to see the names)")
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.quick:
        print(
            "QUICK MODE: the analyses behind these figures are cut down. "
            "The figures do NOT show the published values."
        )

    written = 0
    for number in numbers:
        module = PAPERS[number]
        available = set(module.FIGURES)
        wanted = available if args.all else {args.figure} & available
        if not wanted:
            if args.figure is not None and args.paper is not None:
                print(
                    f"paper {number} has no figure named {args.figure!r}; "
                    f"available: {', '.join(sorted(available))}"
                )
                return 2
            continue
        print(f"\npaper {number}: running the analysis")
        start = time.perf_counter()
        results = module.run(quick=args.quick)
        print(f"  analysis done in {time.perf_counter() - start:.1f} s")
        paths = module.figures(results, args.outdir)
        for name, path in paths.items():
            if name in wanted:
                print(f"  wrote {path}")
                written += 1

    print(f"\n{written} figure(s) written to {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
