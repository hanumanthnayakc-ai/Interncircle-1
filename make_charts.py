"""
Matplotlib + Seaborn charts from two real datasets bundled with scikit-learn:
  * Breast Cancer Wisconsin (569 tumours, 30 measurements, benign / malignant)
  * Wine recognition        (178 wines, 13 chemical measurements, 3 cultivars)

Usage:  python make_charts.py [output_dir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.datasets import load_breast_cancer, load_wine

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "charts")
OUT.mkdir(parents=True, exist_ok=True)

# ---- palette (validated categorical slots 1-3, light surface) -------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("blue_red", [BLUE, NEUTRAL, "#e34948"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASE, "axes.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def header(fig, title, subtitle, left=0.06):
    fig.text(left, 0.97, title, fontsize=15, fontweight="bold", va="top", color=INK)
    fig.text(left, 0.925, subtitle, fontsize=10.5, va="top", color=INK2)


def legend_row(fig, items, y=0.885, left=0.06):
    """items: [(label, colour)] - a swatch plus a name in secondary ink."""
    handles = [Patch(facecolor=c, edgecolor="none", label=l) for l, c in items]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(left, y),
               ncol=len(items), fontsize=10.5, labelcolor=INK2,
               handlelength=1.0, handleheight=1.0, columnspacing=1.6)


def direct_label(ax, x, y, text, colour):
    """Colour swatch carries identity; the words stay in ink."""
    ax.plot([x], [y], marker="s", ms=8, color=colour, mec=SURFACE, mew=1.5,
            ls="none", clip_on=False, zorder=5)
    ax.annotate(text, (x, y), xytext=(9, 0), textcoords="offset points",
                va="center", fontsize=10.5, color=INK, zorder=5)


# ---- data ------------------------------------------------------------------
bc = load_breast_cancer(as_frame=True)
cancer = bc.frame.copy()
cancer["diagnosis"] = cancer["target"].map({0: "Malignant", 1: "Benign"})
CANCER_PAL = {"Malignant": ORANGE, "Benign": BLUE}

wine_b = load_wine(as_frame=True)
wine = wine_b.frame.copy()
wine["cultivar"] = wine["target"].map({0: "Cultivar 0", 1: "Cultivar 1", 2: "Cultivar 2"})
WINE_PAL = {"Cultivar 0": BLUE, "Cultivar 1": ORANGE, "Cultivar 2": AQUA}


# ---- 1. histogram: tumour size by diagnosis --------------------------------
fig, ax = plt.subplots(figsize=(9, 5.6))
fig.subplots_adjust(top=0.78, bottom=0.12, left=0.08, right=0.97)
bins = np.arange(6, 29.5, 0.75)
sns.histplot(data=cancer, x="mean radius", hue="diagnosis", hue_order=["Benign", "Malignant"],
             palette=CANCER_PAL, bins=bins, multiple="layer", alpha=0.8,
             edgecolor=SURFACE, linewidth=1.5, ax=ax, legend=False)
for name in ["Benign", "Malignant"]:
    v = cancer.loc[cancer.diagnosis == name, "mean radius"]
    ax.axvline(v.median(), color=INK2, lw=1, ymax=0.9)
    ax.text(v.median(), ax.get_ylim()[1] * 0.93, f" median {v.median():.1f}",
            color=INK2, fontsize=9.5, va="bottom")
ax.set_xlabel("Mean tumour radius (mm)")
ax.set_ylabel("Number of patients")
ax.grid(axis="x", visible=False)
n_b, n_m = (cancer.diagnosis == "Benign").sum(), (cancer.diagnosis == "Malignant").sum()
header(fig, "Malignant tumours skew larger",
       f"Distribution of mean radius, {n_b} benign vs {n_m} malignant (Breast Cancer Wisconsin)")
legend_row(fig, [("Benign", BLUE), ("Malignant", ORANGE)])
fig.savefig(OUT / "01_histogram_tumour_radius.png", dpi=160)
plt.close(fig)

# ---- 2. box plots: four features, small multiples (one scale each) ----------
feats = [("mean radius", "Radius (mm)"), ("mean texture", "Texture"),
         ("mean smoothness", "Smoothness"), ("mean concavity", "Concavity")]
fig, axes = plt.subplots(1, 4, figsize=(12, 5.2))
fig.subplots_adjust(top=0.72, bottom=0.1, left=0.06, right=0.98, wspace=0.45)
for ax, (col, label) in zip(axes, feats):
    sns.boxplot(data=cancer, x="diagnosis", y=col, order=["Benign", "Malignant"],
                hue="diagnosis", palette=CANCER_PAL, legend=False, width=0.55, ax=ax,
                linewidth=1.2, linecolor=INK2, saturation=1,
                medianprops={"color": INK, "linewidth": 2},
                whiskerprops={"color": INK2}, capprops={"color": INK2},
                flierprops={"marker": "o", "markersize": 4, "markerfacecolor": INK2,
                            "markeredgecolor": SURFACE, "markeredgewidth": 0.8,
                            "alpha": 0.7})
    ax.set_title(label, fontsize=11.5, color=INK, loc="left", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", colors=INK2)
    ax.grid(axis="x", visible=False)
header(fig, "Size and concavity separate the diagnoses; texture overlaps more",
       "Median (bold line), interquartile range (box) and 1.5×IQR whiskers; each panel has its own scale")
legend_row(fig, [("Benign", BLUE), ("Malignant", ORANGE)])
fig.savefig(OUT / "02_boxplots_tumour_features.png", dpi=160)
plt.close(fig)


# ---- 3 & 5. correlation heatmaps -------------------------------------------
def corr_heatmap(df, cols, title, subtitle, fname, size, rename=None, annot_min=0.7):
    c = df[cols].corr()
    if rename:
        c = c.rename(index=rename, columns=rename)
    mask = np.triu(np.ones_like(c, dtype=bool), k=1)
    strong = (np.abs(c.values) >= annot_min) & ~np.eye(len(c), dtype=bool)
    labels = np.where(strong, np.round(c.values, 2).astype(str), "")
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(top=0.84, bottom=0.2, left=0.2, right=0.9)
    sns.heatmap(c, mask=mask, cmap=DIVERGING, vmin=-1, vmax=1, center=0, square=True,
                annot=labels, fmt="", annot_kws={"fontsize": 8.5, "color": INK},
                linewidths=2, linecolor=SURFACE,
                cbar_kws={"shrink": 0.7, "label": "Pearson correlation", "ticks": [-1, -0.5, 0, 0.5, 1]},
                ax=ax)
    ax.grid(False)
    ax.tick_params(length=0, labelsize=9.5, colors=INK2)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    cb = ax.collections[0].colorbar
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9, colors=MUTED)
    cb.set_label("Pearson correlation", color=INK2, fontsize=10)
    header(fig, title, subtitle, left=0.04)
    fig.savefig(OUT / fname, dpi=160)
    plt.close(fig)


mean_cols = [c for c in cancer.columns if c.startswith("mean ")]
corr_heatmap(cancer, mean_cols,
             "Tumour size measures are almost interchangeable",
             "Correlation of the ten mean measurements; values shown where |r| ≥ 0.70",
             "03_heatmap_cancer_correlation.png", (8.6, 7.6),
             rename=lambda s: s.replace("mean ", "").title())

wine_cols = list(wine_b.feature_names)
corr_heatmap(wine, wine_cols,
             "Wine chemistry: phenols travel together",
             "Correlation of 13 chemical measurements across 178 wines; values shown where |r| ≥ 0.70",
             "05_heatmap_wine_correlation.png", (9.4, 8.2),
             rename=lambda s: ("OD280/OD315 ratio" if s.startswith("od280")
                               else s.replace("_", " ").title()))

# ---- 4. histogram small multiples: alcohol by cultivar ----------------------
fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.2), sharex=True, sharey=True)
fig.subplots_adjust(top=0.8, bottom=0.09, left=0.09, right=0.96, hspace=0.18)
bins = np.arange(11, 15.01, 0.25)
for ax, (name, colour) in zip(axes, WINE_PAL.items()):
    v = wine.loc[wine.cultivar == name, "alcohol"]
    sns.histplot(v, bins=bins, color=colour, alpha=1, edgecolor=SURFACE, linewidth=1.5,
                 ax=ax)
    ax.axvline(v.median(), color=INK2, lw=1, ymax=0.75)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 27)
    ax.set_yticks([0, 5, 10, 15, 20])
    direct_label(ax, 11.08, 24.5, f"{name}  (n={len(v)}, median {v.median():.1f}%)", colour)
axes[-1].set_xlabel("Alcohol (% by volume)")
fig.text(0.03, 0.45, "Number of wines", rotation=90, va="center", color=INK2, fontsize=11)
header(fig, "Cultivar 0 is the strongest wine; cultivar 1 the lightest",
       "Alcohol content by cultivar, shared axes (Wine recognition dataset)")
legend_row(fig, list(WINE_PAL.items()))
fig.savefig(OUT / "04_histogram_wine_alcohol.png", dpi=160)
plt.close(fig)

print("Wrote:", *sorted(p.name for p in OUT.glob("*.png")), sep="\n  ")
