"""Figure 1: omics-anchored selection framework. Usage: python fig1_schematic.py FIG_DIR"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

F = Path(sys.argv[1]); F.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, INK, INK2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
FILL_B, FILL_O, FILL_G = "#e6f0fb", "#fdebe3", "#f2f1ed"
plt.rcParams.update({"font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(7.6, 3.6))
ax.set_xlim(0, 102); ax.set_ylim(0, 48); ax.axis("off")


def box(x, y, w, h, title, body, fill, edge, gap=6.2, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fill, ec=edge, lw=lw))
    ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top", fontsize=7.4, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - gap, body, ha="center", va="top", fontsize=6.7, color=INK2, linespacing=1.35)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, color=INK2, lw=1.2,
                                 shrinkA=0, shrinkB=0))


box(1, 29, 17, 16, "Expression", "RNA-seq\n(the only input\nat deployment)", FILL_B, BLUE)
box(1, 5, 17, 16, "Copy number +\nmethylation", "training cohort\nonly", FILL_O, ORANGE, gap=9.4)
box(22, 29, 18, 16, "Gene screen", "univariate Cox z\ntop 300 genes\n(training fold)", FILL_G, INK2)
box(22, 5, 18, 16, "Anchoring score", "max |Pearson r| of a\ngene's expression\nwith its own GISTIC\nCN or methylation", FILL_O, ORANGE)
box(44, 13, 15, 24, "Prior", "π_g = z̃_g + ã_g\n\ngenes whose\nexpression tracks\ntheir own genome\nare favoured", FILL_O, ORANGE, lw=1.8)
box(63, 13, 19, 24, "Anchored\nadaptive LASSO", "penalty weight\nw_g ∝ exp(−π_g / 2)\nCox-LASSO,\nλ by inner 3-fold\nCV", FILL_B, BLUE, gap=8.6, lw=1.8)
box(86, 13, 14, 24, "Signature", "few genes\n+ Cox model\n\nexpression\nonly; tested\non two micro-\narray platforms", FILL_G, INK2)

arrow(18.6, 37, 21.4, 37); arrow(18.6, 13, 21.4, 13)
arrow(40.6, 37, 43.4, 30); arrow(40.6, 13, 43.4, 20)
arrow(59.6, 25, 62.4, 25)
arrow(82.6, 25, 85.4, 25)
fig.savefig(F / "fig1_overview.png", dpi=300, bbox_inches="tight", facecolor="white")
