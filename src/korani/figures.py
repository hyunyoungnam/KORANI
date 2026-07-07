"""Visual evidence for the Result Analyst (stage F).

Three best-effort producers — each returns None instead of raising, because
missing pictures degrade the analysis to text, they don't sink it:
- downsampled CSV text of a simulated curve (for text-only models),
- a PNG plot of the simulated curve (matplotlib, Agg),
- a PNG render of the paper page that contains a cited figure (PyMuPDF —
  whole page, no fragile figure-region cropping).
"""

from __future__ import annotations

import base64
import csv
import re
from pathlib import Path
from typing import List, Optional


def read_curve_csv(path: str, max_points: int = 40) -> Optional[str]:
    """Header plus at most ``max_points`` evenly spaced rows, as CSV text."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.reader(f))
    except OSError:
        return None
    if len(rows) < 2:
        return None
    header, data = rows[0], rows[1:]
    if len(data) > max_points:
        step = len(data) / float(max_points)
        data = [data[int(i * step)] for i in range(max_points)]
    lines = [",".join(header)] + [",".join(r) for r in data]
    return "\n".join(lines)


def render_curve_png(csv_path: str, png_path: str, title: str) -> Optional[str]:
    """Plot the first two CSV columns; returns the PNG path or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.reader(f))
        header, data = rows[0], rows[1:]
        xs = [float(r[0]) for r in data]
        ys = [float(r[1]) for r in data]
        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=110)
        ax.plot(xs, ys)
        ax.set_xlabel(header[0] if header else "x")
        ax.set_ylabel(header[1] if len(header) > 1 else "y")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)
        return str(png_path)
    except Exception:
        return None


def find_figure_page_png(
    pdf_path: str, location: str, out_path: str, dpi: int = 110
) -> Optional[str]:
    """Render the first PDF page whose text cites the figure/table in
    ``location`` (e.g. "Figure 5" → a page containing 'Figure 5' / 'Fig. 5')."""
    match = re.search(r"(Fig(?:ure)?\.?|Table)\s*(\d+)", location, re.IGNORECASE)
    if not match:
        return None
    kind = "Table" if match.group(1).lower().startswith("t") else "Fig"
    number = match.group(2)
    if kind == "Table":
        pattern = re.compile(r"\bTable\s*%s\b" % number, re.IGNORECASE)
    else:
        pattern = re.compile(r"\bFig(?:ure)?\.?\s*%s\b" % number, re.IGNORECASE)
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                if pattern.search(page.get_text()):
                    pix = page.get_pixmap(dpi=dpi)
                    pix.save(out_path)
                    return str(out_path)
    except Exception:
        return None
    return None


def png_data_url(path: str) -> Optional[str]:
    try:
        payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except OSError:
        return None
    return "data:image/png;base64," + payload
