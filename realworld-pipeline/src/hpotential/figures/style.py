"""House style for publication figures.

Serif type, restrained greys and one muted blue. Every series differs twice,
by shade and by line style or hatch, so a figure still reads in black and
white.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Twelve series. Shades repeat after six, but never with the same line style
# or marker, so every pair of series still differs twice.
SHADES = ["#1a1a1a", "#2f4b7c", "#6b6b6b", "#9a9a9a", "#4f6f8f", "#c4c4c4"] * 2
LINES = [
    "-",
    "--",
    ":",
    "-.",
    (0, (5, 1, 1, 1)),
    (0, (1, 1)),
    (0, (3, 1, 1, 1, 1, 1)),
    (0, (8, 2)),
    (0, (2, 2)),
    (0, (4, 1, 2, 1)),
    (0, (6, 1)),
    (0, (1, 2)),
]
HATCHES = ["", "///", "...", "xxx", "\\\\\\", "ooo", "---", "+++", "|||", "**", "OO", ".."]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "p"]
MIN_CELL = 11  # suppress any group or count below this


def apply() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.3,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "hatch.linewidth": 0.5,
        }
    )


def series(i: int) -> dict:
    """Line keyword arguments for series ``i``."""
    return {"color": SHADES[i % len(SHADES)], "linestyle": LINES[i % len(LINES)]}


def bar(i: int) -> dict:
    """Bar keyword arguments for series ``i``: shade and hatch."""
    return {
        "facecolor": SHADES[(i + 2) % len(SHADES)] if i else "white",
        "edgecolor": "#1a1a1a",
        "hatch": HATCHES[i % len(HATCHES)],
        "linewidth": 0.6,
    }


class FigureIndex:
    """Collects each figure's caption and the numbers behind it."""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []

    def save(self, fig, name: str, group: str, caption: str, numbers: dict | None = None) -> None:
        for ext in ("pdf", "png"):
            fig.savefig(self.out_dir / f"{name}.{ext}", metadata=_metadata(ext))
        plt.close(fig)
        self.entries.append(
            {"name": name, "group": group, "caption": caption, "numbers": numbers or {}}
        )

    def write(self) -> Path:
        path = self.out_dir / "figure_index.json"
        path.write_text(json.dumps(self.entries, indent=2, default=float), encoding="utf-8")
        md = ["# Figures", ""]
        for group in dict.fromkeys(e["group"] for e in self.entries):
            md += [f"## {group}", ""]
            for e in (x for x in self.entries if x["group"] == group):
                md += [
                    f"**{e['name']}.** {e['caption']}",
                    "",
                    f"![{e['name']}]({e['name']}.png)",
                    "",
                ]
        (self.out_dir / "FIGURES.md").write_text("\n".join(md), encoding="utf-8")
        return path


def _metadata(ext: str) -> dict:
    # Fixed metadata keeps the files byte-identical between runs.
    if ext == "pdf":
        return {"CreationDate": None, "ModDate": None, "Producer": "hpotential"}
    if ext == "png":
        return {"Software": "hpotential"}
    return {}
