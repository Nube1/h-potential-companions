"""Publication figures from a pipeline run: biostatistics, medical, economics and policy."""

from __future__ import annotations

from pathlib import Path

from . import biostat, medical, policy, style
from .context import load_run

__all__ = ["draw_all"]


def draw_all(output_dir, cfg, figure_dir=None, data_path=None, reps=None) -> Path:
    """Draw every figure for one run and write figure_index.json and FIGURES.md."""
    style.apply()
    ctx = load_run(output_dir, cfg, data_path)
    index = style.FigureIndex(Path(figure_dir or Path(output_dir) / "figures"))
    for draw in biostat.ALL + medical.ALL + policy.ALL:
        if draw is biostat.discrimination and reps is not None:
            draw(ctx, index, reps=reps)
        else:
            draw(ctx, index)
    return index.write()
