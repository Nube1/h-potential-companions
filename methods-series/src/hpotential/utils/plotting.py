"""Shared figure style for the H-potential series.

One style is applied everywhere so that a figure from paper 1 and a figure from
paper 6 can sit on the same page without looking like they came from different
projects. The palette is ordered, muted and distinguishable in greyscale;
series colours are chosen by position, never at random, so a figure regenerated
tomorrow keeps the colours it had today.

Every series is told apart twice, by shade and by line style or hatch, so a
figure still reads once printed in black and white. ``save_figure`` applies
the line styles and hatches by series position, so analysis code only has to
pick colours with :func:`series_color`.

The Agg backend is selected on import, because every figure in this package is
written to a file and none is shown on a screen.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

__all__ = [
    "GRID",
    "HATCHES",
    "INK",
    "LINESTYLES",
    "SERIES",
    "apply_style",
    "print_safe",
    "save_figure",
    "series_color",
]

#: Near-black used for text, axes and rules.
INK: str = "#1a1a1a"

#: Grid line colour.
GRID: str = "#d9d9d9"

#: Ordered series palette: restrained greys and muted blues. Index by position.
SERIES: list[str] = ["#2f4b7c", "#1a1a1a", "#7a7a7a", "#4f6f8f", "#a6a6a6", "#23395b"]

#: Line style paired with each palette position.
LINESTYLES: list = ["-", "--", ":", "-.", (0, (5, 1, 1, 1)), (0, (1, 1))]

#: Light fill paired with each palette position, so hatching stays visible.
FILLS: list[str] = ["#c9d3e3", "#ffffff", "#dcdcdc", "#b9c7d6", "#f0f0f0", "#9fb0c8"]

#: Hatch paired with each palette position.
HATCHES: list[str] = ["", "///", "...", "xxx", "\\\\\\", "ooo"]


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
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "hatch.linewidth": 0.5,
            "pdf.fonttype": 42,
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
    '#2f4b7c'
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
    print_safe(fig)
    fig.savefig(out, metadata={"Software": "hpotential"} if out.suffix == ".png" else None)
    if out.suffix == ".png":
        fig.savefig(out.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    if close:
        plt.close(fig)
    return out


def _series_index(color) -> int | None:
    from matplotlib.colors import to_hex

    try:
        return SERIES.index(to_hex(color))
    except (ValueError, TypeError):
        return None


def print_safe(fig: plt.Figure) -> None:
    """Give each palette series its line style and hatch, and tidy log axes.

    Solid lines drawn in a palette colour take the line style of that colour's
    position; bars and filled areas take its hatch. Lines already given a
    style, and colours outside the palette (reference lines), are left alone.
    Minor tick labels on log axes are removed because they collide.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to adjust in place.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import NullFormatter

    for ax in fig.axes:
        artists = list(ax.get_lines())
        legend = ax.get_legend()
        if legend is not None:
            artists += [h for h in legend.legend_handles if isinstance(h, Line2D)]
        for line in artists:
            i = _series_index(line.get_color())
            if i is not None and line.get_linestyle() == "-":
                line.set_linestyle(LINESTYLES[i])
        patches = list(ax.patches)
        if legend is not None:
            patches += [h for h in legend.legend_handles if isinstance(h, Patch)]
        for patch in patches:
            i = _series_index(patch.get_facecolor())
            if i is None and not patch.get_fill():
                i = _series_index(patch.get_edgecolor())
                if i is not None:
                    patch.set_linestyle(LINESTYLES[i])
                continue
            if i is not None and not patch.get_hatch():
                if patch.get_fill():
                    patch.set_facecolor(FILLS[i])
                patch.set_hatch(HATCHES[i] or None)
                patch.set_edgecolor(INK)
                patch.set_linewidth(0.5)
        for axis in (ax.xaxis, ax.yaxis):
            if axis.get_scale() == "log":
                axis.set_minor_formatter(NullFormatter())
