"""Plota a perda por época a partir do registro do ensaio de capacidade."""
import argparse
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
STUDIES = {
    "cnn_maior": {
        "textcnn": ("TextCNN · 2,71 mi", "#176c69"),
        "residual": ("CNN residual · taxa 0,001", "#a34d34"),
        "residual_lr_menor": ("CNN residual · taxa 0,0003", "#355e94"),
    },
    "textcnn_larga": {
        "textcnn": ("100 por largura · emb. 128", "#176c69"),
        "mais_filtros": ("1.024 por largura · emb. 128", "#a34d34"),
        "mais_filtros_embedding": ("1.024 por largura · emb. 384", "#355e94"),
    },
}
PAPER, INK, GRID = "#faf9f4", "#142637", "#d6e0e9"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estudo", choices=STUDIES, default="cnn_maior")
    parser.add_argument("--dados", type=Path, help="Pasta com selection.json e destino de curvas.png.")
    args = parser.parse_args()
    folder = args.dados or ROOT / "results" / args.estudo
    labels = STUDIES[args.estudo]
    results = json.loads((folder / "selection.json").read_text())["records"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), sharex=True, facecolor=PAPER)
    fig.subplots_adjust(left=.10, right=.98, top=.76, bottom=.22, wspace=.25)
    titles = ("Treino durante os lotes", "Validação ao fim da época")
    keys = ("train_online", "validation")
    axes[0].set_ylabel("Entropia cruzada", color=INK)
    for axis, key, title in zip(axes, keys, titles, strict=True):
        axis.set_facecolor(PAPER)
        axis.grid(axis="y", linewidth=.7, color=GRID)
        axis.set_axisbelow(True)
        axis.axvline(5, color="#a9b5b9", linewidth=1.1, linestyle=":", zorder=0)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
        axis.tick_params(colors=INK, length=0, pad=5)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_xlabel("Época", color=INK)
        axis.set_title(title, color=INK, loc="left", fontsize=12, weight="bold", pad=10)
        for record in results:
            label, color = labels[record["model"]]
            epochs = record["history"]
            line = "-" if record["seed"] == 17 else "--"
            axis.plot([row["epoch"] for row in epochs], [row[key]["ce"] for row in epochs],
                      color=color, linestyle=line, linewidth=1.85,
                      label=label if record["seed"] == 17 else "_nolegend_")
            if key == "validation":
                best = record["selected"]["melhor"]["epoch"]
                axis.scatter(best, epochs[best - 1][key]["ce"], s=33, color=color, zorder=4)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncols=3,
               frameon=False, labelcolor=INK, bbox_to_anchor=(.53, .93), fontsize=10)
    fig.text(.10, .036, "Traço contínuo (17) e tracejado (29). Os pontos marcam os checkpoints. A linha vertical indica a época 5.",
             color=INK, fontsize=8.7)
    fig.savefig(folder / "curvas.png", dpi=180, facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    main()
