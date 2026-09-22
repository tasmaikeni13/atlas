"""Scientific visualization styling and color palettes for publication-grade figures."""

import matplotlib as mpl
import matplotlib.pyplot as plt

def set_publication_style():
    """Sets publication-grade matplotlib styling."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

COLOR_MAP = "magma"
TRAJECTORY_COLOR = "#00F5D4"
ANCHOR_COLOR = "#FFD166"
CERT_COLOR = "#EF476F"
MIN_COLOR = "#06D6A0"
