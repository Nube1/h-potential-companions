"""Build the manuscripts from their Markdown sources.

Each ``papers/0*.md`` produces a ``.tex`` file (always) and a ``.pdf``:

* ``--engine latex`` (default when ``latexmk`` is installed) compiles the
  generated LaTeX with pdflatex in Times.
* ``--engine reportlab`` draws the PDF directly, for machines without TeX.

The Markdown subset is small on purpose: ``#``/``##``/``###`` headings,
paragraphs, pipe tables, fenced equation blocks (kept as fixed-width ASCII,
never rasterised) and ``[n]`` reference entries under ``## References``.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "papers"
MAX_EQUATION_WIDTH = 89
FOOTER = "H-potential | Companion research draft | Synthetic evidence only"


# --------------------------------------------------------------------------
# Parsing


@dataclass
class Block:
    kind: str  # heading2, heading3, paragraph, equation, table, reference, figure
    text: str = ""
    rows: list[list[str]] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)


@dataclass
class Paper:
    title: str
    author: str
    note: str
    blocks: list[Block]


def parse(path: Path) -> Paper:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError(f"{path.name}: first line must be '# Title'")
    title = lines[0][2:].strip()
    blocks: list[Block] = []
    in_references = False
    i = 1
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("```"):
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            if i == len(lines):
                raise ValueError(f"{path.name}: unclosed code fence")
            too_wide = [x for x in body if len(x) > MAX_EQUATION_WIDTH]
            if too_wide:
                raise ValueError(
                    f"{path.name}: equation line over {MAX_EQUATION_WIDTH} chars: {too_wide[0]!r}"
                )
            blocks.append(Block("equation", lines=body))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                row = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(set(x) <= set("-: ") for x in row):
                    rows.append(row)
                i += 1
            if len({len(r) for r in rows}) != 1:
                raise ValueError(f"{path.name}: table rows have unequal cell counts")
            blocks.append(Block("table", rows=rows))
            continue
        m = re.match(r"!\[(?P<cap>.*)\]\((?P<src>[^)]+)\)\s*$", line)
        if m:
            src = m.group("src")
            if not (path.parent / f"{src}.pdf").exists():
                raise ValueError(f"{path.name}: figure file {src}.pdf not found")
            cap = re.sub(r"^Figure \d+\.\s*", "", m.group("cap"))
            blocks.append(Block("figure", cap, lines=[src]))
            i += 1
            continue
        if line.startswith("### "):
            blocks.append(Block("heading3", line[4:].strip()))
            i += 1
            continue
        if line.startswith("## "):
            heading = line[3:].strip()
            in_references = heading.lower() == "references"
            blocks.append(Block("heading2", heading))
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "```")):
            paragraph.append(lines[i])
            i += 1
        text = " ".join(x.strip() for x in paragraph)
        kind = "reference" if in_references and re.match(r"\[\d+\]", text) else "paragraph"
        blocks.append(Block(kind, text))

    # The first two paragraphs are the author line and the status note.
    if len(blocks) < 2 or blocks[0].kind != "paragraph" or blocks[1].kind != "paragraph":
        raise ValueError(f"{path.name}: expected author and status lines after the title")
    return Paper(title, blocks[0].text, blocks[1].text, blocks[2:])


# --------------------------------------------------------------------------
# LaTeX

_TEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
}
_URL = re.compile(r"https?://\S+")


def tex_escape(text: str) -> str:
    return "".join(_TEX_SPECIAL.get(c, c) for c in text)


def tex_text(text: str) -> str:
    """Escape prose; wrap URLs in ``\\url`` so they can break across lines."""
    out, last = [], 0
    for m in _URL.finditer(text):
        url = m.group(0).rstrip(".,;)")
        out.append(tex_escape(text[last : m.start()]))
        out.append(r"\url{" + url.replace("%", r"\%").replace("#", r"\#") + "}")
        last = m.start() + len(url)
    out.append(tex_escape(text[last:]))
    return "".join(out)


_NUMERIC = re.compile(r"^-?\d+(\.\d+)?%?$")

TEX_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\pdftrailerid{}
\usepackage[margin=25mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{amsmath,amssymb,booktabs,array,microtype,fancyhdr,graphicx}
\usepackage[labelsep=none,skip=6pt,font=small]{caption}
\captionsetup[figure]{labelsep=period,justification=raggedright,singlelinecheck=false}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\urlstyle{same}
\makeatletter
\g@addto@macro{\UrlBreaks}{\do\a\do\b\do\c\do\d\do\e\do\f\do\g\do\h\do\i\do\j\do\k\do\l\do\m\do\n\do\o\do\p\do\q\do\r\do\s\do\t\do\u\do\v\do\w\do\x\do\y\do\z}
\makeatother
\setlength{\parindent}{0pt}
\setlength{\emergencystretch}{3em}
\setlength{\parskip}{0.6\baselineskip}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0.4pt}
\fancyfoot[L]{\footnotesize FOOTER}
\fancyfoot[R]{\footnotesize\thepage}
\fancypagestyle{plain}{\fancyhf{}%
  \fancyfoot[L]{\footnotesize FOOTER}\fancyfoot[R]{\footnotesize\thepage}}
"""


def _figure_width(png: Path) -> str:
    """Wide multi-panel figures fill the line; single panels take 70%."""
    try:
        from PIL import Image

        with Image.open(png) as im:
            ratio = im.width / im.height
    except (ImportError, OSError):
        return "0.9"
    return "1.0" if ratio > 2.2 else "0.7"


def to_tex(paper: Paper, paper_dir: Path = PAPERS) -> str:
    out = [TEX_PREAMBLE.replace("FOOTER", tex_escape(FOOTER))]
    out.append(r"\title{" + tex_escape(paper.title) + "}")
    out.append(r"\author{" + tex_escape(paper.author) + "}")
    out.append(r"\date{}")
    out.append(r"\begin{document}")
    out.append(r"\maketitle")
    out.append(r"\begin{center}\small\itshape " + tex_text(paper.note) + r"\end{center}" + "\n")
    in_refs = False
    for block in paper.blocks:
        if in_refs and block.kind != "reference":
            out.append(r"\end{list}")
            in_refs = False
        if block.kind == "heading2":
            out.append(r"\section*{" + tex_escape(block.text) + "}")
        elif block.kind == "heading3":
            out.append(r"\subsection*{" + tex_escape(block.text) + "}")
        elif block.kind == "paragraph":
            out.append(tex_text(block.text) + "\n")
        elif block.kind == "equation":
            out.append(r"{\footnotesize\begin{verbatim}")
            out.extend(block.lines)
            out.append(r"\end{verbatim}}")
        elif block.kind == "table":
            header, *body = block.rows
            numeric = [all(_NUMERIC.match(r[j]) for r in body) for j in range(len(header))]
            spec = "".join("r" if n else "l" for n in numeric)
            out.append(r"\begin{table}[htbp]\centering\small")
            out.append(r"\caption{}")
            out.append(r"\begin{tabular}{" + spec + "}")
            out.append(r"\toprule")
            out.append(" & ".join(tex_escape(c) for c in header) + r" \\")
            out.append(r"\midrule")
            out.extend(" & ".join(tex_escape(c) for c in row) + r" \\" for row in body)
            out.append(r"\bottomrule")
            out.append(r"\end{tabular}")
            out.append(r"\end{table}" + "\n")
        elif block.kind == "figure":
            out.append(r"\begin{figure}[htbp]\centering")
            width = _figure_width(paper_dir / f"{block.lines[0]}.png")
            out.append(
                r"\includegraphics[width=" + width + r"\linewidth]{" + block.lines[0] + ".pdf}"
            )
            out.append(r"\caption{" + tex_text(block.text) + "}")
            out.append(r"\end{figure}" + "\n")
        elif block.kind == "reference":
            if not in_refs:
                out.append(
                    r"\begin{list}{}{\setlength{\leftmargin}{2em}\setlength{\itemindent}{-2em}"
                    r"\setlength{\itemsep}{0.3\baselineskip}}\small"
                )
                in_refs = True
            out.append(r"\item " + tex_text(block.text))
    if in_refs:
        out.append(r"\end{list}")
    out.append(r"\end{document}")
    return "\n".join(out) + "\n"


def compile_latex(tex_path: Path) -> Path:
    """Compile with latexmk in a scratch folder; copy only the PDF back."""
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={tmp}",
            str(tex_path),
        ]
        # A fixed timestamp makes rebuilds byte-identical when nothing changed.
        env = {
            **os.environ,
            "SOURCE_DATE_EPOCH": os.environ.get("SOURCE_DATE_EPOCH", "1792972800"),
            "FORCE_SOURCE_DATE": "1",
        }
        # Run beside the .tex so relative figure paths resolve.
        cmd[-1] = tex_path.name
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=tex_path.parent)
        log = Path(tmp) / tex_path.with_suffix(".log").name
        if proc.returncode != 0:
            tail = log.read_text(errors="replace")[-3000:] if log.exists() else proc.stdout[-3000:]
            raise RuntimeError(f"LaTeX failed for {tex_path.name}:\n{tail}")
        overfull = re.findall(r"Overfull \\hbox \((\d+\.\d+)pt", log.read_text(errors="replace"))
        if any(float(x) > 5 for x in overfull):
            print(f"warning: {tex_path.name} has overfull lines: {overfull}", file=sys.stderr)
        pdf = tex_path.with_suffix(".pdf")
        shutil.copyfile(Path(tmp) / pdf.name, pdf)
    return pdf


# --------------------------------------------------------------------------
# ReportLab fallback


def build_reportlab(paper: Paper, pdf_path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import (
        Paragraph,
        Preformatted,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    body = ParagraphStyle("Body", fontName="Times-Roman", fontSize=11, leading=15, spaceAfter=8)
    small = ParagraphStyle("Small", fontName="Times-Italic", fontSize=9, leading=12, spaceAfter=10)
    title = ParagraphStyle(
        "PaperTitle", fontName="Times-Bold", fontSize=17, leading=21, spaceAfter=10
    )
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Times-Bold", keepWithNext=True)
    h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName="Times-Bold", keepWithNext=True)
    eq = ParagraphStyle(
        "Eq", fontName="Courier", fontSize=8, leading=11, spaceBefore=5, spaceAfter=10
    )
    cell = ParagraphStyle("Cell", fontName="Times-Roman", fontSize=9, leading=11)
    ref = ParagraphStyle(
        "Ref", parent=body, fontSize=9.5, leading=12, leftIndent=18, firstLineIndent=-18
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#999999"))
        canvas.line(50, 45, A4[0] - 50, 45)
        canvas.setFont("Times-Roman", 8)
        canvas.drawString(50, 32, FOOTER)
        canvas.drawRightString(A4[0] - 50, 32, str(doc.page))
        canvas.restoreState()

    story = [
        Paragraph(escape(paper.title), title),
        Paragraph(escape(paper.author), body),
        Paragraph(escape(paper.note), small),
    ]
    n_table = 0
    n_figure = 0
    for block in paper.blocks:
        if block.kind == "heading2":
            story.append(Paragraph(escape(block.text), h2))
        elif block.kind == "heading3":
            story.append(Paragraph(escape(block.text), h3))
        elif block.kind == "paragraph":
            story.append(Paragraph(escape(block.text), body))
        elif block.kind == "reference":
            story.append(Paragraph(escape(block.text), ref))
        elif block.kind == "equation":
            story.append(Preformatted("\n".join(block.lines), eq))
        elif block.kind == "figure":
            from reportlab.platypus import Image

            n_figure += 1
            img = Image(str(pdf_path.parent / f"{block.lines[0]}.png"))
            scale = (A4[0] - 100) / img.imageWidth
            img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
            story.extend(
                [img, Paragraph(f"Figure {n_figure}. " + escape(block.text), cell), Spacer(1, 8)]
            )
        elif block.kind == "table":
            n_table += 1
            story.append(Paragraph(f"Table {n_table}", cell))
            cells = [[Paragraph(escape(c), cell) for c in row] for row in block.rows]
            width = (A4[0] - 100) / len(block.rows[0])
            table = Table(
                cells, colWidths=[width] * len(block.rows[0]), repeatRows=1, hAlign="LEFT"
            )
            table.setStyle(
                TableStyle(
                    [
                        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.extend([table, Spacer(1, 10)])
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=46,
        bottomMargin=60,
        title=paper.title,
        author=paper.author,
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build manuscript .tex and .pdf files")
    parser.add_argument("--engine", choices=("auto", "latex", "reportlab"), default="auto")
    parser.add_argument("--tex-only", action="store_true", help="write .tex files, skip PDFs")
    args = parser.parse_args(argv)
    engine = args.engine
    if engine == "auto":
        engine = "latex" if shutil.which("latexmk") else "reportlab"
        if engine == "reportlab":
            print("latexmk not found; drawing PDFs with ReportLab", file=sys.stderr)

    for md in sorted(PAPERS.glob("0*.md")):
        paper = parse(md)
        tex = md.with_suffix(".tex")
        tex.write_text(to_tex(paper), encoding="utf-8")
        if args.tex_only:
            print(tex.name)
            continue
        if engine == "latex":
            compile_latex(tex)
        else:
            build_reportlab(paper, md.with_suffix(".pdf"))
        print(md.with_suffix(".pdf").name, f"({engine})")


if __name__ == "__main__":
    main()
