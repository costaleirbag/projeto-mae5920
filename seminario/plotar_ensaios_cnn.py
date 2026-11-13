"""Gera o gráfico de comprimento da TextCNN a partir das métricas publicadas."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEAL, RUST, INK, RULE, PAPER = "#176c69", "#a34d34", "#142637", "#d9e3e2", "#faf9f4"


def main():
    summary = json.loads((ROOT / "results/cnn_length/summary.json").read_text())
    names = ("max_256", "max_400", "max_768", "max_integral")
    scopes = ("256", "400", "768", "integral")
    labels = ("256", "400", "768", "Integral")
    rows = [[run for run in summary["runs"] if run["recipe"] == name] for name in names]
    x = np.arange(len(rows))
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.bottom": False, "axes.spines.left": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.35), facecolor=PAPER,
                             gridspec_kw={"width_ratios": (1.35, 1)})
    fig.subplots_adjust(left=.065, bottom=.15, right=.98, top=.84, wspace=.28)
    accuracy, coverage = axes
    for key, label, color in (("test", "Teste inteiro", TEAL), ("test_long", "Resenhas acima de 400", RUST)):
        values = np.array([[100 * run[key]["accuracy"] for run in group] for group in rows])
        accuracy.fill_between(x, values.min(1), values.max(1), color=color, alpha=.12, linewidth=0)
        accuracy.plot(x, values.mean(1), color=color, label=label, linewidth=2.1, marker="o", markersize=5)
    rates = [summary["coverage"][key]["test"] for key in scopes]
    percentages = np.asarray([100 * row["reviews_cut"] / row["n"] for row in rates])
    coverage.bar(x, percentages, color=["#6c9290", TEAL, "#9cbab3", "#c8d8d1"], width=.54, zorder=3)
    for i, value in enumerate(percentages):
        coverage.text(i, value + 1, f"{value:.2f}%".replace(".", ","), ha="center", color=INK, fontsize=9)
    accuracy.set_ylim(75, 92)
    coverage.set_ylim(0, 34)
    accuracy.yaxis.set_major_locator(MultipleLocator(5))
    coverage.yaxis.set_major_locator(MultipleLocator(10))
    accuracy.legend(loc="lower right", frameon=False, fontsize=9)
    for ax, title in zip(axes, ("Acurácia por comprimento", "Resenhas do teste com corte"), strict=True):
        ax.set_facecolor(PAPER)
        ax.set_xticks(x, labels)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
        ax.grid(axis="y", color=RULE, linewidth=.65, zorder=0)
        ax.tick_params(length=0, pad=7)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=13, pad=13)
        ax.set_xlim(-.36, 3.36)
    destination = ROOT / "results/cnn_length/coverage_accuracy.png"
    fig.savefig(destination, dpi=180, facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    main()
