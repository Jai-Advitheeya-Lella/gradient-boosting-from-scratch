"""Shared matplotlib style: one colour per library, recessive grid, thin marks."""

from __future__ import annotations

import matplotlib as mpl

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

# categorical slots assigned by entity, never by rank
COLORS = {
    "ours": "#2a78d6",
    "ours-leafwise": "#2a78d6",
    "xgboost": "#eb6834",
    "lightgbm": "#1baf7a",
    "sklearn": "#6250d6",
}
LINESTYLES = {"ours": "-", "ours-leafwise": (0, (4, 2)), "xgboost": "-", "lightgbm": "-", "sklearn": "-"}
MARKERS = {"ours": "o", "ours-leafwise": "o", "xgboost": "s", "lightgbm": "^", "sklearn": "D"}


def apply() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.8,
            "font.family": "DejaVu Sans",
            "figure.dpi": 100,
            "savefig.dpi": 150,
        }
    )
