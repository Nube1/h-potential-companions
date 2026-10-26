"""One module per paper, each exposing ``run`` and ``figures``.

Every module has the same shape::

    run(cfg=None, outdir=None, quick=False) -> dict
    figures(results, outdir, cfg=None) -> dict[str, Path]

so a driver can loop over them without knowing what any one of them does.
"""

from __future__ import annotations

from . import (
    paper1_validation,
    paper2_prognostic,
    paper3_consensus,
    paper4_dynamics,
    paper5_access,
    paper6_hta,
)

#: Analysis module for each paper number.
PAPERS = {
    1: paper1_validation,
    2: paper2_prognostic,
    3: paper3_consensus,
    4: paper4_dynamics,
    5: paper5_access,
    6: paper6_hta,
}

__all__ = [
    "PAPERS",
    "paper1_validation",
    "paper2_prognostic",
    "paper3_consensus",
    "paper4_dynamics",
    "paper5_access",
    "paper6_hta",
]
