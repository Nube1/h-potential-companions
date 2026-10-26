"""Shared figure style for the H-potential series.

One style is applied everywhere so that a figure from paper 1 and a figure from
paper 6 can sit on the same page without looking like they came from different
projects. The palette is ordered, muted and distinguishable in greyscale;
series colours are chosen by position, never at random, so a figure regenerated
tomorrow keeps the colours it had today.

The Agg backend is selected on import, because every figure in this package is
written to a file and none is shown on a screen.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

__all__ = ["GRID", "INK", "SERIES", "apply_style", "save_figure", "series_color"]

#: Near-black used for text, axes and rules.
INK: str = "#1a1a1a"

#: Grid line colour.
GRID: str = "#d9d9d9"

#: Ordered series palette. Index by position, not by name.
SERIES: list[str] = ["#2b5d8a", "#c05621", "#3f7d4e", "#7a4b8c", "#8a6d3b", "#8a2b3b"]


def apply_style() -> None:
    """Set the series defaults on the global matplotlib parameters.

    Notes
    -----
    This changes global state, which is what a style is. Call it once at the
    top of a figure script. It does not touch colours of individual artists;
    pass :func:`series_color` for those.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.size": 9,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.frameon": False,
        }
    )


def series_color(index: int) -> str:
    """Palette colour at ``index``, wrapping round.

    Parameters
    ----------
    index : int
        Position in the series.

    Returns
    -------
    str
        A hex colour.

    Examples
    --------
    >>> series_color(0)
    '#2b5d8a'
    >>> series_color(6) == series_color(0)
    True
    """
    return SERIES[int(index) % len(SERIES)]


def save_figure(fig: plt.Figure, path: str | Path, close: bool = True) -> Path:
    """Write a figure, creating the directory if it is missing.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure.
    path : str or pathlib.Path
        Destination, including the extension.
    close : bool, default True
        Close the figure afterwards, which matters when a script writes many.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    if close:
        plt.close(fig)
    return out
