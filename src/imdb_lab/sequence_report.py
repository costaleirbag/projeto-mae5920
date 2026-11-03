"""Didactic CNN/encoder figures, report, reading notebook and presentation data."""
from __future__ import annotations
import base64
import html
import json
from pathlib import Path

from .common import ROOT, read_json, save_json, file_sha256

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

BASE = ROOT / "results/sequences"
FIG = BASE / "figures"
PDF = ROOT / "output/pdf/relatorio_cnn_encoders.pdf"
HTML = ROOT / "relatorio_cnn_encoders.html"
NB = ROOT / "notebooks/cnn_encoders.ipynb"
COLORS = ["#167D92", "#DF8B2F", "#7957A0", "#CE5351", "#4F9A65"]
NAMES = {"bert": "BERT-base", "distilbert": "DistilBERT", "modernbert": "ModernBERT-base", "tfidf": "TF-IDF + logística"}
OPT = {"sgd": "SGD", "momentum": "SGD + momentum", "adam": "Adam"}
ARCH = {"base": "3/4/5, 100 filtros", "kernel_3": "Janela 3, 100 filtros", "kernel_5": "Janela 5, 100 filtros", "multiscale_32": "3/4/5, 32 filtros"}


def all_runs():
    return [read_json(p) for p in sorted((BASE / "runs").glob("*.json"))]


def curve_data(records):
    return pd.DataFrame([{**h, "run_id": r["run_id"], "family": r["spec"]["family"], "group": r["spec"]["group"],
                          "variant": r["spec"]["variant"], "optimizer": r["spec"]["optimizer"], "lr": r["spec"]["lr"], "seed": r["seed"]} for r in records for h in r["history"]])


def band(ax, rows, x, y, label, color, linestyle="-"):
    grouped = rows.groupby(x)[y]; c = grouped.mean(); low = grouped.min(); high = grouped.max()
    ax.plot(c.index, c.to_numpy(), lw=2, color=color, label=label, ls=linestyle)
    ax.fill_between(c.index, low.to_numpy(), high.to_numpy(), color=color, alpha=.12, lw=0)


def savefig(fig, name):
    for ext in ["png", "svg"]: fig.savefig(FIG / f"{name}.{ext}", dpi=180, facecolor="white", bbox_inches="tight", metadata={"Date": None} if ext == "svg" else None)
    plt.close(fig)


def selected(records, family, policy):
    frozen = read_json(BASE / "selection_frozen.json"); rate = frozen["selections"][f"{family}_{policy}"]
    return [r for r in records if r["spec"]["family"] == family and r["spec"]["lr"] == rate and
            (r["spec"]["group"] == "optimizer" and r["spec"]["optimizer"] == policy if family == "cnn" else r["spec"]["group"] == policy)]


def summarize():
    records = all_runs(); by_id = {r["run_id"]: r for r in records}
    audit = read_json(BASE / "audit_metrics.json"); verified = read_json(BASE / "verification.json")
    metrics = {r["run_id"]: r for r in verified["audit_metrics"]}
    groups = []
    for opt in OPT: groups.append((f"cnn_{opt}", OPT[opt], "cnn_optimizer", selected(records, "cnn", opt)))
    base = selected(records, "cnn", "adam")
    groups.append(("cnn_base", ARCH["base"], "cnn_architecture", base))
    for name, label in ARCH.items():
        if name != "base": groups.append((f"cnn_{name}", label, "cnn_architecture", [r for r in records if r["spec"]["variant"] == name]))
    groups.append(("dropout_0.5", "Dropout 0,5", "cnn_dropout", base))
    for drop in [0, .8]: groups.append((f"dropout_{drop:g}", f"Dropout {str(drop).replace('.', ',')}", "cnn_dropout", [r for r in records if r["spec"]["variant"] == f"dropout_{drop:g}"]))
    for family in ["bert", "distilbert", "modernbert"]:
        for policy in ["probe", "full"]: groups.append((f"{family}_{policy}", NAMES[family] + (" congelado" if policy == "probe" else " ajuste completo"), "encoder", selected(records, family, policy)))
    rows = []
    for key, label, group, runs in groups:
        assert len(runs) == 3
        row = {"key": key, "label": label, "group": group, "run_ids": [r["run_id"] for r in runs], "seeds": [r["seed"] for r in runs],
               "parameters": runs[0]["parameters"], "trainable_parameters": runs[0]["trainable_parameters"], "lr": runs[0]["spec"]["lr"],
               "best_epochs": [r["best_epoch"] for r in runs], "training_seconds_mean": float(np.mean([r["training_seconds"] for r in runs])),
               "total_seconds_mean": float(np.mean([r["total_seconds"] for r in runs]))}
        for metric in ["accuracy", "ce", "f1_macro", "brier"]:
            values = [metrics[r["run_id"]][metric] for r in runs]
            row.update({f"{metric}_mean": float(np.mean(values)), f"{metric}_min": float(np.min(values)), f"{metric}_max": float(np.max(values))})
        row["validation_ce_mean"] = float(np.mean([r["best_validation_ce"] for r in runs]))
        am = {m["run_id"]: m for m in audit["metrics"]}
        row["audit_inference_seconds_mean"] = float(np.mean([am[r["run_id"]]["inference_seconds"] for r in runs]))
        family = runs[0]["spec"]["family"]
        if runs[0]["spec"]["group"] == "probe":
            row["parameters"] = read_json(BASE / f"{family}_architecture.json")["parameters"]
            row["feature_extraction_seconds"] = read_json(BASE / f"{family}_feature_extraction.json")["train_validation_seconds"]
            row["fit_plus_features_seconds"] = row["training_seconds_mean"] + row["feature_extraction_seconds"]
            row["audit_end_to_end_seconds"] = row["audit_inference_seconds_mean"] + audit["frozen_encoder_extraction_seconds"][family]
        else:
            row["fit_plus_features_seconds"] = row["training_seconds_mean"]
            row["audit_end_to_end_seconds"] = row["audit_inference_seconds_mean"]
        rows.append(row)
    cfg = read_json(ROOT / "sequence_config.json")
    s = {"protocol": "CNN/encoder extension; common 128-word excerpts; fresh internal audit 2048; official test unused",
         "training_runs": len(records), "training_seconds_sum": sum(r["training_seconds"] for r in records),
         "total_run_seconds_sum": sum(r["total_seconds"] for r in records),
         "groups": rows, "baseline": metrics["tfidf"], "choices": read_json(BASE / "selection_frozen.json")["selections"],
         "candidate_validation_means": read_json(BASE / "selection_frozen.json")["validation_candidate_means"],
         "encoder_architectures": {k: read_json(BASE / f"{k}_architecture.json") for k in ["bert", "distilbert", "modernbert"]},
         "tokenization": {k: {**read_json(BASE / f"{k}_tokenization.json"), "audit": audit["tokenization"][k]} for k in ["cnn", "bert", "distilbert", "modernbert"]},
         "config": cfg, "verification_sha256": file_sha256(BASE / "verification.json")}
    save_json(BASE / "summary.json", s); pd.DataFrame(rows).drop(columns=["run_ids", "seeds", "best_epochs"]).to_csv(BASE / "comparison.csv", index=False)
    # Literal data for editable presentation charts, with no independent calculation of metrics.
    curves = curve_data(records); chart_data = {}
    for key, _, _, runs in groups:
        part = curves[curves.run_id.isin([r["run_id"] for r in runs])]
        chart_data[key] = {metric: {"epochs": list(map(int, part.groupby("epoch")[metric].mean().index)),
                                    "mean": part.groupby("epoch")[metric].mean().tolist(), "min": part.groupby("epoch")[metric].min().tolist(),
                                    "max": part.groupby("epoch")[metric].max().tolist()} for metric in ["train_ce", "validation_ce", "validation_accuracy"]}
    for opt in OPT:
        part = curves[(curves.family == "cnn") & (curves.group == "optimizer") & (curves.optimizer == opt) & (curves.lr == cfg["cnn"]["common_lr"])]
        chart_data[f"cnn_common_{opt}"] = {"epochs": part.groupby("epoch").validation_ce.mean().index.tolist(), "mean": part.groupby("epoch").validation_ce.mean().tolist()}
    save_json(BASE / "presentation_data.json", {"summary": s, "curves": chart_data})
    return s


def figures(s):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .15, "legend.fontsize": 9, "svg.fonttype": "none", "svg.hashsalt": "mae5920-sequences"})
    records = all_runs(); curves = curve_data(records)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for i, opt in enumerate(OPT):
        common = curves[(curves.family == "cnn") & (curves.group == "optimizer") & (curves.optimizer == opt) & (curves.lr == .001)]
        tuned = curves[curves.run_id.isin([r["run_id"] for r in selected(records, "cnn", opt)])]
        band(axes[0], common, "epoch", "validation_ce", OPT[opt], COLORS[i])
        band(axes[1], tuned, "epoch", "validation_ce", f"{OPT[opt]}, taxa {s['choices']['cnn_' + opt]:g}", COLORS[i])
    for ax in axes: ax.set(xlabel="Época", ylabel="Entropia cruzada na validação"); ax.legend(frameon=False)
    axes[0].set_title("CNN: mesma taxa 0,001"); axes[1].set_title("CNN: três taxas candidatas por método")
    savefig(fig, "cnn_optimizers")
    fig, ax = plt.subplots(figsize=(9.2, 3.6), constrained_layout=True)
    for i, opt in enumerate(OPT):
        rows = pd.DataFrame([{"lr": r["spec"]["lr"], "best_ce": r["best_validation_ce"]} for r in records if r["spec"]["family"] == "cnn" and r["spec"]["group"] == "optimizer" and r["spec"]["optimizer"] == opt])
        band(ax, rows, "lr", "best_ce", OPT[opt], COLORS[i])
        ax.scatter(rows.lr, rows.best_ce, s=16, color=COLORS[i], alpha=.6)
    ax.set(xscale="log", xlabel="Taxa de aprendizado", ylabel="CE da época escolhida", title="Seleção de taxa por validação, sem escolher a melhor semente"); ax.legend(frameon=False)
    savefig(fig, "cnn_learning_rates")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for i, row in enumerate([r for r in s["groups"] if r["group"] == "cnn_architecture"]):
        part = curves[curves.run_id.isin(row["run_ids"])]
        band(axes[0], part, "epoch", "validation_ce", row["label"], COLORS[i])
        axes[1].errorbar(100 * row["accuracy_mean"], i, xerr=[[100 * (row["accuracy_mean"] - row["accuracy_min"])], [100 * (row["accuracy_max"] - row["accuracy_mean"])]], fmt="o", color=COLORS[i], capsize=4)
    rows = [r for r in s["groups"] if r["group"] == "cnn_architecture"]
    axes[0].set(xlabel="Época", ylabel="CE de validação", title="Janelas e número de filtros"); axes[0].legend(frameon=False)
    axes[1].set(yticks=range(len(rows)), yticklabels=[r["label"] for r in rows], xlabel="Acurácia na nova auditoria (%)", title="Taxa de Adam reutilizada, sem retuning")
    savefig(fig, "cnn_architectures")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for i, row in enumerate(sorted([r for r in s["groups"] if r["group"] == "cnn_dropout"], key=lambda x: x["key"])):
        part = curves[curves.run_id.isin(row["run_ids"])]
        band(axes[0], part, "epoch", "train_ce", row["label"], COLORS[i]); band(axes[1], part, "epoch", "validation_ce", row["label"], COLORS[i])
    axes[0].set(xlabel="Época", ylabel="CE de treino em avaliação", title="Dropout desligado ao medir o treino")
    axes[1].set(xlabel="Época", ylabel="CE de validação", title="Overfitting e intensidade do dropout"); axes[1].legend(frameon=False)
    savefig(fig, "cnn_dropout")
    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    x = np.arange(3); offset = .18
    for i, policy in enumerate(["probe", "full"]):
        rows = [next(r for r in s["groups"] if r["key"] == f"{name}_{policy}") for name in ["bert", "distilbert", "modernbert"]]
        v = np.array([r["accuracy_mean"] for r in rows]) * 100
        low = v - np.array([r["accuracy_min"] for r in rows]) * 100; high = np.array([r["accuracy_max"] for r in rows]) * 100 - v
        ax.bar(x + (-offset if i == 0 else offset), v, .34, color=COLORS[i], label="Encoder congelado" if i == 0 else "Ajuste completo", yerr=[low, high], capsize=4)
    ax.axhline(100 * s["baseline"]["accuracy"], color="#546b76", ls="--", label="TF-IDF + logística")
    cnn = next(r for r in s["groups"] if r["key"] == "cnn_adam")
    ax.axhline(100 * cnn["accuracy_mean"], color=COLORS[2], ls=":", label="CNN + Adam")
    ax.set(xticks=x, xticklabels=["BERT-base", "DistilBERT", "ModernBERT-base"], ylabel="Acurácia na nova auditoria (%)", ylim=(0, 100), title="Mesmos exemplos e prefixos, regimes de treinamento distintos")
    ax.legend(frameon=False, loc="lower right"); savefig(fig, "encoder_comparison")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    for i, family in enumerate(["bert", "distilbert", "modernbert"]):
        rows = curves[curves.run_id.isin([r["run_id"] for r in selected(records, family, "full")])]
        for j, metric in enumerate(["train_ce", "validation_ce"]): band(axes[i], rows, "epoch", metric, "Treino" if j == 0 else "Validação", COLORS[j])
        axes[i].set(title=NAMES[family], xlabel="Época", ylabel="Entropia cruzada"); axes[i].legend(frameon=False)
    savefig(fig, "encoder_learning_curves")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), constrained_layout=True)
    for i, family in enumerate(["bert", "distilbert", "modernbert"]):
        for j, policy in enumerate(["probe", "full"]):
            rows = pd.DataFrame([{"lr": r["spec"]["lr"], "best_ce": r["best_validation_ce"]} for r in records if r["spec"]["family"] == family and r["spec"]["group"] == policy])
            band(axes[j], rows, "lr", "best_ce", NAMES[family], COLORS[i]); axes[j].scatter(rows.lr, rows.best_ce, color=COLORS[i], s=16, alpha=.6)
    for ax in axes: ax.set(xscale="log", xlabel="Taxa", ylabel="CE da época escolhida"); ax.legend(frameon=False)
    axes[0].set_title("Só a cabeça: 40 épocas, Adam"); axes[1].set_title("Ajuste completo: 3 épocas, AdamW")
    savefig(fig, "encoder_learning_rates")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    rows = [r for r in s["groups"] if r["group"] == "encoder"] + [cnn]
    for row in rows:
        is_full = row["key"].endswith("full"); family = row["key"].split("_")[0]
        c = COLORS[["bert", "distilbert", "modernbert", "cnn"].index(family)]
        for j, key in enumerate(["fit_plus_features_seconds", "audit_end_to_end_seconds"]):
            axes[j].scatter(row[key], row["accuracy_mean"] * 100, s=65, color=c, marker="o" if is_full or family == "cnn" else "s")
            axes[j].annotate(("CNN" if family == "cnn" else NAMES[family].replace("-base", "")) + ("" if family == "cnn" else " completo" if is_full else " congelado"), (row[key], row["accuracy_mean"] * 100), xytext=(5, 5), textcoords="offset points", fontsize=8)
    for ax in axes: ax.set(xscale="log", ylabel="Acurácia na auditoria (%)"); ax.margins(.24, .2)
    axes[0].set(xlabel="Ajuste + extração congelada, segundos (log)", title="Custo do regime, exclui avaliação entre épocas")
    axes[1].set(xlabel="Inferência de 2.048 entradas, segundos (log)", title="Encoder incluído também no modo congelado")
    savefig(fig, "cost_accuracy")
    diagrams(s)


def diagrams(s):
    """Scientific model diagrams with dimensions, for the report and notebook."""
    def canvas(height=4):
        f, a = plt.subplots(figsize=(11, height)); a.set(xlim=(0, 11), ylim=(0, height)); a.axis("off"); return f, a
    def box(ax, x, y, w, h, text, color=COLORS[0], size=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=.06", facecolor=color + "15", edgecolor=color, linewidth=1.2))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=size, color="#173d4c")
    def arrow(ax, start, end, color="#617984"):
        if abs(start[1]-end[1]) > .03:
            mid=(start[0]+end[0])/2
            ax.plot([start[0],mid,mid],[start[1],start[1],end[1]],color=color,lw=1.3)
            start=(mid,end[1])
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": color, "lw": 1.3})
    f, ax = canvas(4.3)
    box(ax,.15,1.55,1.5,1.05,"Tokens\n[B, L]")
    box(ax,2.0,1.55,1.6,1.05,"Embedding\n[B, L, 128]")
    arrow(ax,(1.75,2.07),(1.91,2.07))
    for i, k in enumerate([3,4,5]):
        y=3.1-i*1.2; box(ax,4.2,y,2.0,.7,f"Conv1D {k} + ReLU\n[B, 100, L-{k-1}]",COLORS[i],9.5)
        box(ax,6.75,y,1.55,.7,"Máximo global\n[B, 100]",COLORS[i],9.5)
        arrow(ax,(3.7,2.07),(4.1,y+.35)); arrow(ax,(6.3,y+.35),(6.65,y+.35)); arrow(ax,(8.4,y+.35),(8.8,2.1))
    box(ax,8.9,1.55,1.75,1.05,"Concatenar: 300\nDropout\nLinear 300 × 2",COLORS[3],10)
    ax.text(.2,.18,"L ≤ 256 tokens. Filtros compartilham pesos entre posições. Janelas de padding não competem pelo máximo.",fontsize=10,color="#526a77")
    savefig(f,"cnn_diagram")
    f,ax=canvas(3.3)
    box(ax,.2,1.35,1.6,1.0,"Embedding\ndos tokens\n[B, L, 768]")
    box(ax,2.5,.85,3.1,2.0,"Bloco Transformer\n\nAutoatenção com 12 cabeças\nSoma residual e normalização\nMLP por posição e residual",COLORS[2],10)
    box(ax,6.35,1.35,1.9,1.0,"Média dos tokens\ncom máscara\n[B, 768]")
    box(ax,8.95,1.35,1.65,1.0,"Dropout 0,1\nLinear 768 × 2\n[B, 2]",COLORS[1])
    arrow(ax,(1.9,1.85),(2.4,1.85));arrow(ax,(5.7,1.85),(6.25,1.85));arrow(ax,(8.35,1.85),(8.85,1.85))
    ax.text(4.05,.45,"Repetir 12 / 6 / 22 vezes",ha="center",fontsize=11,color=COLORS[2])
    ax.text(.2,.1,"Diagrama funcional. Ordem de normalização e atenção variam. ModernBERT alterna atenção local e global e usa RoPE.",fontsize=9.5,color="#526a77")
    savefig(f,"encoder_diagram")
    f,ax=canvas(3.6)
    for i,label in enumerate(["Encoder congelado","Ajuste completo"]):
        y=2.3-i*1.5
        ax.text(.2,y+.4,label,fontsize=11,fontweight="bold",color="#173d4c")
        box(ax,2.5,y,2.45,.95,"Encoder pré-treinado\n"+("Pesos fixos, modo eval" if i==0 else "Pesos recebem gradientes"),COLORS[0] if i==0 else COLORS[3])
        box(ax,5.7,y,1.9,.95,"Média e dropout\n768 números",COLORS[1])
        box(ax,8.3,y,2.2,.95,"Cabeça linear\n1.538 parâmetros",COLORS[3])
        arrow(ax,(5.05,y+.48),(5.6,y+.48));arrow(ax,(7.7,y+.48),(8.2,y+.48))
    ax.text(.2,.1,"Congelado: vetor pré-computado, treina a cabeça. Completo: backward atravessa também todos os blocos do encoder.",fontsize=10,color="#526a77")
    savefig(f,"transfer_diagram")


def create_notebook(s):
    import nbformat as nbf
    nb=nbf.v4.new_notebook(); cells=[]
    md=lambda t: cells.append(nbf.v4.new_markdown_cell(t))
    code=lambda t: cells.append(nbf.v4.new_code_cell(t))
    md("# CNN, BERT e encoders modernos\n\nNestor, Letícia e Gabriel · MAE5920. Este notebook lê experimentos reais já concluídos. Não treina nem baixa modelos. O estudo anterior de MLPs e o steering do Qwen continuam nos notebooks próprios.")
    code("from pathlib import Path\nimport json, sys, numpy as np, pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nif not (ROOT / 'sequence_config.json').exists(): ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT / 'src'))\nfrom imdb_lab.common import read_json\nB = ROOT / 'results/sequences'\ns = read_json(B / 'summary.json')\nv = read_json(B / 'verification.json')\nassert v['status'] == 'passed'\nprint(v['training_runs'], 'treinos; auditoria de', v['audit_checkpoints'], 'checkpoints')")
    md("## Papéis dos dados\n\nO mesmo prefixo de até 128 palavras entra em todos os modelos. A auditoria de 2.048 exemplos exclui todos os papéis da extensão MLP. O teste oficial não foi reaberto. A validação escolhe a taxa e a época, antes da auditoria.")
    code("m = read_json(B / 'data_manifest.json')\nold = read_json(ROOT / 'results/didactic/data_manifest.json')\nassert not set(m['roles']['audit']['ids']) & set().union(*(set(x['ids']) for x in old['roles'].values()))\ndisplay(pd.DataFrame([{'papel': k, 'n': r['n'], 'classe 0': r['label_counts']['0'], 'classe 1': r['label_counts']['1']} for k,r in m['roles'].items()]))")
    md("## A CNN usada neste estudo\n\nEmbedding treinável 128, convoluções paralelas 3/4/5, cem filtros por janela, ReLU, max pooling com máscara, concatenação 300, dropout e saída linear binária. O vocabulário usa só o treino. A rede aprende embeddings e filtros do zero. A saída contém dois logits; o softmax os converte em probabilidades para leitura.")
    code("display(Image(filename=str(B / 'figures/cnn_diagram.png')))\nrows = pd.DataFrame(s['groups'])\ndisplay(rows[rows.group.eq('cnn_architecture')][['label','parameters','lr','best_epochs']])")
    md("## Otimizador e taxa formam uma comparação\n\nOs painéis mostram taxa comum e taxa escolhida entre três candidatas por método. Pesos e minibatches são pareados por semente. A linha representa a média e a faixa, mínimo e máximo das três sementes, sem interpretação de intervalo de confiança.")
    code("display(Image(filename=str(B / 'figures/cnn_optimizers.png')))\ndisplay(Image(filename=str(B / 'figures/cnn_learning_rates.png')))\ndisplay(rows[rows.group.eq('cnn_optimizer')][['label','lr','accuracy_mean','accuracy_min','accuracy_max','ce_mean']])")
    md("## Arquitetura e dropout\n\nAs variantes reutilizam a taxa de Adam escolhida para a CNN principal. Não há ajuste por variante. O total de parâmetros inclui o embedding; mudar o número de filtros afeta apenas parte da rede. A CE de treino é medida em avaliação, com dropout desligado.")
    code("display(Image(filename=str(B / 'figures/cnn_architectures.png')))\ndisplay(Image(filename=str(B / 'figures/cnn_dropout.png')))\ndisplay(rows[rows.group.eq('cnn_dropout')][['label','best_epochs','validation_ce_mean','accuracy_mean','ce_mean']])")
    md("## O que o encoder calcula\n\nCada token ganha uma representação contextual de 768 dimensões. Fazemos a média dos estados finais, excluindo padding e tokens especiais. BERT tem 12 blocos, DistilBERT 6 e ModernBERT-base 22. No ModernBERT, a atenção alterna janelas locais e blocos globais, com posições rotativas. A ordem das operações difere entre arquiteturas.")
    code("display(Image(filename=str(B / 'figures/encoder_diagram.png')))\ndisplay(pd.DataFrame(s['encoder_architectures']).T[['num_hidden_layers','hidden_size','num_attention_heads','backbone_parameters','parameters']])\nprint('Revisões fixadas:', {k:r['revision'] for k,r in read_json(B / 'models_pinned.json').items()})")
    md("## Congelar e ajustar integralmente\n\nO modo congelado treina apenas 1.538 parâmetros sobre vetores pré-computados. O ajuste completo usa gradientes em todas as camadas. As cabeças começam iguais para a mesma semente. As receitas têm otimizadores e orçamentos diferentes; não isolam destravar pesos como único fator. Encoder congelado fica em avaliação; no ajuste completo, o dropout do encoder funciona durante o treino.")
    code("display(Image(filename=str(B / 'figures/transfer_diagram.png')))\ncfg = s['config']['encoders']\ndisplay(pd.DataFrame([{'regime': k, 'épocas': cfg[k]['epochs'], 'batch': cfg[k]['batch_size'], 'taxas': cfg[k]['learning_rates'], 'otimizador': cfg[k]['optimizer']} for k in ['probe','full']]))")
    md("## Resultados na nova auditoria interna\n\nOs encoders usam os mesmos exemplos, prefixos e formato de cabeça linear neste estudo. Pré-treinamento, vocabulários, número de parâmetros e orçamentos diferem. Estes resultados não devem ser misturados com os números no teste oficial dos relatórios anteriores.")
    code("display(Image(filename=str(B / 'figures/encoder_comparison.png')))\ndisplay(rows[rows.group.eq('encoder')][['label','trainable_parameters','lr','accuracy_mean','accuracy_min','accuracy_max','ce_mean','f1_macro_mean']])\nprint('TF-IDF:', s['baseline'])")
    md("## Hiperparâmetros dos Transformers\n\nCada taxa é testada nas três sementes. A seleção minimiza a média da melhor CE de validação, sem selecionar a melhor semente. As três épocas do ajuste completo são um orçamento fixo, sem garantia de convergência.")
    code("display(Image(filename=str(B / 'figures/encoder_learning_rates.png')))\ndisplay(Image(filename=str(B / 'figures/encoder_learning_curves.png')))\ndisplay(pd.DataFrame([{'método': k, 'taxa escolhida': lr} for k,lr in s['choices'].items()]))")
    md("## Tempo também é uma variável\n\nO ajuste exclui avaliações entre épocas. No modo congelado, adicionamos a extração de vetores de treino e validação. Na inferência, contamos também o encoder congelado, para não comparar só a cabeça com um modelo completo. Downloads, carregamento e tokenização ficam fora. Os tempos refletem esta implementação no NVIDIA GB10, sem prometer a melhor eficiência de cada família.")
    code("display(Image(filename=str(B / 'figures/cost_accuracy.png')))\ndisplay(rows[rows.group.eq('encoder')][['label','fit_plus_features_seconds','total_seconds_mean','audit_end_to_end_seconds']])\ndisplay(pd.DataFrame([{'tokenizador': k, 'truncadas no treino': r['train']['truncated_n'], 'truncadas na auditoria': r['audit']['truncated_n']} for k,r in s['tokenization'].items()]))")
    md("## Mesmos textos em modelos diferentes\n\nOs quatro textos próprios abaixo foram definidos antes de observar suas predições e avaliados após o congelamento dos modelos. Usam sempre a semente 17, não a melhor semente da auditoria. Os rótulos pretendidos pelo autor não constituem um benchmark. As probabilidades não foram calibradas.")
    code("examples = read_json(B / 'synthetic_examples.json')\nfor e in examples['examples']:\n    print(e['id'], '| intenção:', e['intended'])\n    print(e['text'])\n    display(pd.DataFrame([{'modelo': k, 'P(positivo)': p} for k,p in e['probabilities'].items()]))")
    md("## O que chega à última camada da CNN\n\nCom dropout desligado, o logit positivo menos o negativo é uma soma exata dos máximos dos filtros vezes os pesos da saída, mais o viés. Agrupamos filtros que venceram na mesma janela. Isso descreve a leitura linear do vetor já calculado. Não é uma atribuição causal às palavras: editar o texto muda convoluções e máximos de outras posições.")
    code("e = examples['examples'][2]\nr = e['cnn_readout']\nprint(e['text'])\nprint('Margem:', r['logit_margin'], '| soma reconstruída:', r['reconstructed_margin'])\nassert abs(r['logit_margin'] - r['reconstructed_margin']) < 1e-5\ndisplay(pd.DataFrame(r['positive_windows'] + r['negative_windows']))")
    md("## Reconstrução independente e ligação com as aulas\n\nO verificador refaz 217 métricas a partir dos logits, confirma as escolhas de validação e o pareamento de partidas. Uma segunda avaliação na GPU reconstrói todos os 43 checkpoints. Os capítulos 6–9 de Prince explicam treinamento e seleção, o capítulo 10 trata convoluções, o 11 os resíduos e o 12 os Transformers. O double descent e o steering permanecem demonstrações próprias nos outros notebooks.")
    code("from importlib.util import spec_from_file_location, module_from_spec\nsp = spec_from_file_location('verify_sequences', ROOT / 'verify_sequences.py')\nmod = module_from_spec(sp); sp.loader.exec_module(mod)\nmod.verify()\ndisplay(pd.DataFrame(read_json(ROOT / 'docs/REFERENCIAS_SEQUENCIAS.json')['references'])[['title','locators','url']].fillna(''))")
    nb.cells=cells; nb.metadata.kernelspec={"name":"python3","display_name":"Python 3","language":"python"}
    nbf.write(nb, NB)


def create_html(s):
    def image(name):
        encoded=base64.b64encode((FIG/f"{name}.png").read_bytes()).decode()
        return f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(name)}">'
    parts=['<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CNN e encoders · MAE5920</title><style>body{max-width:1120px;margin:50px auto;padding:0 24px;font:17px/1.6 system-ui;color:#243d49;background:#fafcfc}h1,h2{color:#125e70;line-height:1.2}h1{font-size:46px}section{margin:58px 0}img{width:100%;height:auto;background:white}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;text-align:left;border-bottom:1px solid #d7e2e7}th{background:#eaf3f5}blockquote{margin:20px 0;padding:12px 24px;border-left:3px solid #167d92}small{color:#526a77}details{margin:18px 0}a{color:#167d92}</style><h1>CNN, BERT e encoders modernos</h1><p>Nestor, Letícia e Gabriel · MAE5920 · Experimentos reais com três sementes</p><p>87 treinos: 42 CNNs, 27 cabeças com encoder congelado e 18 ajustes integrais. Nova auditoria interna com 2.048 resenhas. Mesmo prefixo de 128 palavras por modelo. O teste oficial não foi reaberto.</p>']
    for title,names,paragraph in [
        ("A CNN aprende embeddings e padrões locais",["cnn_diagram"],"Embedding 128, janelas 3/4/5, cem filtros por janela, máximo global e saída linear. Todo o vocabulário é ajustado no treino."),
        ("Otimizadores e taxas",["cnn_optimizers","cnn_learning_rates"],"Mesma partida e minibatches por semente. Três taxas candidatas por método. Cada execução escolhe sua época pela CE de validação."),
        ("Arquitetura e regularização",["cnn_architectures","cnn_dropout"],"As variantes reutilizam a taxa selecionada para Adam. Linhas e faixas representam média e mínimo/máximo das três sementes, não intervalos de confiança."),
        ("Representações contextuais",["encoder_diagram","transfer_diagram"],"Média dos estados finais de 768 dimensões, excluindo padding e tokens especiais. O encoder congelado deixa apenas 1.538 parâmetros treináveis. As receitas congelada e completa usam orçamentos e otimizadores diferentes."),
        ("Resultados dos encoders",["encoder_comparison","encoder_learning_rates","encoder_learning_curves"],"BERT, DistilBERT e ModernBERT partem de pré-treinamento público, sem ajuste prévio no IMDb. Comparações entre famílias não isolam arquitetura, escala e pré-treinamento."),
        ("Custo e qualidade",["cost_accuracy"],"O custo congelado inclui a extração dos vetores. O cronômetro de ajuste exclui avaliações entre épocas; a inferência inclui o encoder. Tempos refletem esta implementação e hardware.")]:
        parts.append(f'<section><h2>{title}</h2><p>{paragraph}</p>'+''.join(image(n) for n in names)+'</section>')
    df=pd.DataFrame(s["groups"]); columns=["label","lr","parameters","accuracy_mean","accuracy_min","accuracy_max","ce_mean","f1_macro_mean"]
    parts.append('<section><h2>Todas as condições selecionadas</h2>'+df[columns].to_html(index=False,float_format=lambda x:f"{x:.5g}")+f'<p>TF-IDF + logística: acurácia {100*s["baseline"]["accuracy"]:.2f}%, CE {s["baseline"]["ce"]:.4f}. Uma execução determinística, sem faixa entre sementes.</p></section>')
    examples=read_json(BASE/"synthetic_examples.json")
    parts.append('<section><h2>Quatro textos próprios, mesmos modelos</h2><p>Textos próprios definidos antes de observar as predições e avaliados após o congelamento dos modelos. Semente 17 em todos os modelos neurais. Probabilidades sem calibração. Não constituem um benchmark.</p>')
    for e in examples["examples"]:
        parts.append(f'<details open><summary>{html.escape(e["id"])} · intenção: {e["intended"]}</summary><blockquote>{html.escape(e["text"])}</blockquote>'+pd.DataFrame([{"modelo":k,"P(positivo)":p} for k,p in e["probabilities"].items()]).to_html(index=False,float_format=lambda x:f"{x:.4f}"))
        parts.append('<p>Contribuições à margem linear da CNN, agrupadas por janela. Descrição da leitura final, sem interpretação causal.</p>'+pd.DataFrame(e["cnn_readout"]["positive_windows"]+e["cnn_readout"]["negative_windows"]).to_html(index=False,float_format=lambda x:f"{x:.3f}")+'</details>')
    refs=read_json(ROOT/"docs/REFERENCIAS_SEQUENCIAS.json")
    parts.append('</section><section><h2>Fontes e limites</h2><p>O número de camadas, tokenização, pré-treinamento e quantidade de parâmetros variam entre famílias. As três sementes não representam toda a incerteza amostral. O regime de três épocas não garante convergência. O exemplo de double descent usa outro problema sintético. Os resultados oficiais e o Qwen ficam nos relatórios anteriores.</p><ul>')
    for r in refs["references"]:
        label=html.escape(r["title"]); parts.append('<li>'+ (f'<a href="{r["url"]}">{label}</a>' if r.get("url") else label)+ (' · '+html.escape(r["locators"]) if r.get("locators") else '')+'</li>')
    parts.append('</ul></section></html>'); HTML.write_text(''.join(parts))


def create_pdf(s):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak
    from matplotlib import font_manager
    from PIL import Image as PILImage
    fonts=Path(font_manager.findfont("DejaVu Sans")).parent
    for n,f in [("SeqBody","DejaVuSans.ttf"),("SeqBold","DejaVuSans-Bold.ttf")]: pdfmetrics.registerFont(TTFont(n,str(fonts/f)))
    pdfmetrics.registerFontFamily("SeqBody",normal="SeqBody",bold="SeqBold")
    w,h=A4; usable=w-84
    body=ParagraphStyle("SeqBody",fontName="SeqBody",fontSize=9.1,leading=13.1,textColor=colors.HexColor("#263c47"),spaceAfter=8)
    title=ParagraphStyle("SeqTitle",parent=body,fontName="SeqBold",fontSize=22,leading=27,textColor=colors.HexColor("#165d70"),spaceAfter=13)
    small=ParagraphStyle("SeqSmall",parent=body,fontSize=7.7,leading=10.7,textColor=colors.HexColor("#526a77"),spaceAfter=6)
    sub=ParagraphStyle("SeqSub",parent=body,fontName="SeqBold",fontSize=12,leading=16,spaceBefore=6,spaceAfter=8)
    story=[]
    def p(t,style=body): story.append(Paragraph(t,style))
    def page(label,t):
        if story: story.append(PageBreak())
        p(label.upper(),small);p(t,title)
    def pic(name,maxheight=230):
        with PILImage.open(FIG/f"{name}.png") as a: iw,ih=a.size
        scale=min(usable/iw,maxheight/ih);story.append(Image(str(FIG/f"{name}.png"),width=iw*scale,height=ih*scale));story.append(Spacer(1,7))
    def table(headers,rows,widths=None):
        data=[[Paragraph(str(x),small) for x in headers]]+[[Paragraph(str(x),body) for x in row] for row in rows]
        t=Table(data,colWidths=widths or [usable/len(headers)]*len(headers),repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eaf3f5")),("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LINEBELOW",(0,0),(-1,0),.8,colors.HexColor("#a9c6d0")),("LINEBELOW",(0,1),(-1,-1),.3,colors.HexColor("#dbe5e9")),
            ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
        story.append(t);story.append(Spacer(1,9))
    def row(key): return next(r for r in s["groups"] if r["key"]==key)
    pct=lambda v:f"{100*v:.2f}%".replace('.',',')
    page("MAE5920 / CNN e encoders", "Arquitetura, treinamento e transferência")
    p("<b>Nestor, Letícia e Gabriel</b><br/>Introdução a Deep Learning - extensão do projeto IMDb.")
    p("Este estudo treina CNNs e ajusta BERT, DistilBERT e ModernBERT-base para relacionar resultados com convoluções, atenção, otimização e generalização. São 87 treinos reais com três sementes: 42 CNNs, 27 cabeças sobre encoders congelados e 18 ajustes completos. Um baseline TF-IDF completa a comparação.")
    table(["Pergunta", "Experimento", "Material da disciplina"],[
        ["Como filtros locais classificam texto?","CNN 1D, janelas e largura","Prince 10; slides 10"],
        ["Adam e SGD aprendem igual?","Taxa comum e três taxas por método","Prince 6; slides 06"],
        ["Quando o ajuste se torna overfitting?","Dropout e curvas treino/validação","Prince 8-9; slides 08-09"],
        ["O que muda com pré-treinamento?","Encoder congelado e ajuste integral","Prince 11-12; slides 11-12"]],[usable*.32,usable*.36,usable*.32])
    p("Os papéis dos dados",sub)
    p("Treino: <b>6.144</b> resenhas. Validação: <b>1.536</b>. Nova auditoria interna: <b>2.048</b>, excluindo todos os papéis da extensão MLP. As amostras vêm apenas do treino oficial limpo. O teste oficial de 25 mil exemplos não foi reaberto. Esta auditoria é interna, não um novo benchmark externo.")
    p("Todos os modelos recebem o mesmo prefixo de até 128 palavras separadas por espaço. A CNN e os encoders admitem no máximo 256 tokens; a truncagem adicional é medida. O TF-IDF usa o prefixo completo, sem esse limite adicional. O rótulo continua sendo o da resenha completa, mesmo quando o prefixo omite a avaliação final.")
    p("Seleção antes da auditoria",sub)
    p("Cada treino escolhe a época com menor CE de validação. Para cada método, escolhemos a taxa que minimiza a média dessas perdas nas três sementes. Congelamos código, revisões, pesos, taxas e épocas antes de transferir a entrada da nova auditoria para a execução.")
    p("Os relatórios originais, o estudo MLP e o steering do Qwen permanecem separados. Não misturamos métricas de testes, prefixos ou divisões diferentes.",small)
    page("1 / Convoluções no texto", "A CNN usada neste estudo")
    pic("cnn_diagram",235)
    base=row("cnn_adam")
    p("A entrada é uma sequência de índices. Um embedding treinável associa a cada índice 128 números. Cada filtro combina todos esses canais em uma janela local de 3, 4 ou 5 tokens. Os mesmos pesos deslizam pelas posições. A ReLU introduz não linearidade; o máximo global retém a maior resposta de cada filtro.")
    p(f"Cem filtros em cada uma das três janelas produzem 300 máximos. Concatenamos esses valores, aplicamos dropout e uma camada linear com duas saídas. O modelo principal tem <b>{format(base['parameters'], ',').replace(',', '.')}</b> parâmetros treináveis, incluindo o embedding. Ele aprende do zero, sem vetores pré-treinados. A saída contém dois logits; usamos softmax para exibir probabilidades.")
    table(["Operação", "Formato por minibatch"],[["Índices","B × L"],["Embedding","B × L × 128"],["Conv1D de janela k","B × 100 × (L-k+1)"],["Máximo de cada ramo","B × 100"],["Concatenação e saída","B × 300; depois B × 2"]],[usable*.53,usable*.47])
    p("No padding, janelas inválidas não competem pelo máximo. Para entradas menores que uma janela, o código mantém uma janela mínima com zeros de padding. A convolução preserva ordem local; o máximo perde a localização exata do padrão vencedor. A rede rasa não modela todas as relações distantes do texto.")
    p("Fontes: Prince 10; Redes Convolucionais, páginas do arquivo 11-22 e 36; inspiração arquitetural em Kim (2014), sem reproduzir integralmente seu protocolo.",small)
    page("2 / Otimização da CNN", "A taxa muda a comparação")
    pic("cnn_optimizers",220);pic("cnn_learning_rates",160)
    table(["Método", "Taxa escolhida", "Acurácia interna", "CE interna"],[[OPT[o],f"{s['choices']['cnn_'+o]:g}",pct(row('cnn_'+o)['accuracy_mean']),f"{row('cnn_'+o)['ce_mean']:.3f}"] for o in OPT],[usable*.31,usable*.22,usable*.25,usable*.22])
    p("As três taxas de cada algoritmo estão fixadas no protocolo. Cada condição usa os mesmos pesos e minibatches por semente, doze épocas e clipping de norma 1. O painel com 0,001 responde o que ocorre mantendo a taxa. O outro inclui a escolha de validação, uma comparação diferente. Faixas mostram mínimo/máximo das três sementes.")
    p("Os métodos usam escalas de atualização diferentes. Um método lento na taxa comum pode melhorar com outra taxa. Os dados descrevem este orçamento e grade; não estabelecem superioridade universal de um otimizador. Fonte: Prince 6.1-6.5 e slides Ajuste de Modelos.",small)
    page("3 / Capacidade e regularização", "Filtros e dropout na CNN")
    pic("cnn_architectures",195);pic("cnn_dropout",195)
    table(["Variante de dropout", "Épocas escolhidas", "CE interna", "Acurácia interna"],[[r['label'],', '.join(map(str,r['best_epochs'])),f"{r['ce_mean']:.3f}",pct(r['accuracy_mean'])] for r in s['groups'] if r['group']=='cnn_dropout'],[usable*.29,usable*.25,usable*.21,usable*.25])
    p("A grade de arquitetura compara janelas 3, 5 ou 3/4/5 e 32 ou cem filtros por janela. Todas as variantes reutilizam a taxa de Adam escolhida para a CNN principal. O embedding ocupa grande parte dos parâmetros, por isso reduzir filtros não reduz a rede inteira na mesma proporção.")
    p("O treino continua por doze épocas para mostrar a trajetória. A avaliação usa o checkpoint escolhido pela CE de validação. Curvas de treino com dropout desligado permitem distinguir ajuste dos dados e o ruído da regularização. Fonte: Prince 8-10 e slides Regularização.",small)
    page("4 / Atenção e contexto", "BERT, DistilBERT e ModernBERT")
    pic("encoder_diagram",195)
    p("Um encoder Transformer transforma cada token usando o contexto. Na autoatenção, projeções aprendidas produzem queries, keys e values. A compatibilidade entre query e key determina os pesos da combinação dos values. Há múltiplas cabeças, transformações por posição, resíduos e normalização. A informação de posição permite distinguir ordens diferentes.")
    table(["Encoder", "Blocos", "Dimensão / cabeças", "Parâmetros do classificador completo"],[[NAMES[k],str(s['encoder_architectures'][k]['num_hidden_layers']),"768 / 12",format(s['encoder_architectures'][k]['parameters'], ',').replace(',', '.')] for k in ['bert','distilbert','modernbert']],[usable*.27,usable*.12,usable*.23,usable*.38])
    p("BERT-base usa atenção bidirecional global e posições aprendidas. DistilBERT reduz a pilha para seis blocos e incorpora destilação em seu pré-treinamento. ModernBERT-base combina 22 blocos, posições rotativas e atenção local intercalada com global. Não usamos sua capacidade máxima de contexto: todos recebem o mesmo prefixo curto neste ensaio.")
    p("A saída de cada token tem 768 dimensões. A média dos estados finais exclui padding e tokens especiais. Acrescentamos dropout 0,1 e Linear(768,2). O pooler original de BERT é removido; por isso a contagem difere de tabelas que incluem pooler ou cabeça de pré-treinamento.")
    p("O diagrama é funcional. A ordem de normalização e dos resíduos varia entre as famílias. Fontes: Prince 11-12; Transformers, páginas 7-18, 21-31 e 35-38; Devlin et al., Sanh et al. e Warner et al.",small)
    page("5 / Transferência de aprendizagem", "Congelar ou ajustar todos os pesos")
    pic("transfer_diagram",210)
    table(["Regime", "Pesos treináveis", "Receita", "Grade"],[
        ["Encoder congelado","Cabeça: 1.538","40 épocas, batch 128, Adam","0,001; 0,01; 0,1"],
        ["Ajuste completo","Encoder e cabeça","3 épocas, batch 32, AdamW","0,00001; 0,00003"]],[usable*.22,usable*.25,usable*.29,usable*.24])
    p("No modo congelado, o encoder fica em avaliação e gera vetores uma única vez. Nenhum gradiente atualiza seus pesos. A cabeça recebe gradientes e aprende a separar esses vetores. No ajuste completo, o backward atravessa também o encoder e adapta suas representações à tarefa.")
    p("As cabeças começam com os mesmos pesos na mesma semente. O ajuste completo usa warmup de 10%, redução linear da taxa, decaimento desacoplado de 0,01 nos pesos matriciais e clipping 1. Vieses e parâmetros de normalização ficam sem decaimento.")
    p("Esta comparação avalia duas receitas práticas. Orçamento, otimizador, dropout do encoder e atualização dos pesos mudam juntos. Portanto, não isolamos destravar o encoder como única causa de um ganho. A cabeça mantém dropout 0,1 nos dois regimes; no congelado, o encoder permanece determinístico em avaliação.")
    p("Os modelos partem de pré-treinamento público, sem checkpoints já ajustados no IMDb. Revisões e hashes estão fixados. Encoders usam BF16 com parâmetros e estado do otimizador FP32, atenção SDPA e TF32 desligado. A CNN usa FP32. Fonte: documentação oficial e protocolo executável.",small)
    page("6 / Generalização interna", "Os resultados dos encoders")
    pic("encoder_comparison",240)
    table(["Modelo / regime", "Taxa", "Acurácia média", "Mínimo - máximo", "CE"],[[NAMES[k]+(' congelado' if policy=='probe' else ' completo'),f"{row(k+'_'+policy)['lr']:g}",pct(row(k+'_'+policy)['accuracy_mean']),pct(row(k+'_'+policy)['accuracy_min'])+' - '+pct(row(k+'_'+policy)['accuracy_max']),f"{row(k+'_'+policy)['ce_mean']:.3f}"] for k in ['bert','distilbert','modernbert'] for policy in ['probe','full']],[usable*.33,usable*.12,usable*.17,usable*.25,usable*.13])
    p(f"No mesmo recorte, o baseline TF-IDF + logística tem acurácia {pct(s['baseline']['accuracy'])} e CE {s['baseline']['ce']:.3f}. A CNN + Adam tem acurácia média {pct(row('cnn_adam')['accuracy_mean'])}. São controles úteis para verificar se a complexidade adicional ajuda neste orçamento.")
    p("A amostra é a nova auditoria interna de 2.048 exemplos. A faixa entre três sementes não é intervalo de confiança sobre a população. A diferença entre famílias também inclui pré-treinamento, tokenização e escala. As probabilidades não foram calibradas e a CE pode divergir da acurácia.",small)
    page("7 / Hiperparâmetros dos encoders", "Trajetórias e seleção na validação")
    pic("encoder_learning_rates",205);pic("encoder_learning_curves",205)
    p("A grade congelada contém três taxas; a completa contém duas. Cada ponto usa três sementes. A escolha usa a média da CE de cada checkpoint selecionado, sem consultar a auditoria. As curvas inferiores mostram treino e validação com a taxa selecionada para o ajuste completo.")
    p("Três épocas são um orçamento, não uma prova de convergência. Um encoder pode precisar de outra taxa, regularização ou mais passos para alcançar seu melhor resultado. O objetivo é observar aprendizagem e custo com um protocolo explícito, mantendo todas as tentativas publicadas.")
    p("A curva da época zero mostra a cabeça aleatória e não participa da escolha de checkpoint. As condições congeladas não recalculam o encoder a cada época. O notebook permite consultar todas as trajetórias e taxas descartadas.",small)
    page("8 / Eficiência", "Qualidade e custo medidos juntos")
    pic("cost_accuracy",245)
    table(["Modelo / regime", "Ajuste + vetores (s)", "Inferência / 2.048 (s)"],[[r['label'],f"{r['fit_plus_features_seconds']:.2f}",f"{r['audit_end_to_end_seconds']:.2f}"] for r in s['groups'] if r['group']=='encoder'],[usable*.51,usable*.25,usable*.24])
    p("No congelado, o ajuste inclui a extração dos vetores de treino e validação, mais a média do treino da cabeça. O ajuste completo e a CNN contam apenas os minibatches de treino. As avaliações entre épocas, o carregamento e o download ficam fora desse cronômetro. O tempo total por execução também está nos resultados.")
    p("Na inferência, somamos o encoder congelado ao custo da cabeça. Assim, não tratamos a cabeça isolada como se fosse o modelo completo. A medição usa batch 64, exclui tokenização e carregamento. Não há repetição sistemática de benchmark de latência.")
    p("A implementação usa atenção SDPA e não tenta extrair a máxima eficiência de cada arquitetura. BF16, formato de entrada e kernels afetam tempo. Valores medidos no NVIDIA GB10 não devem ser generalizados para outro hardware.",small)
    page("9 / Exemplos didáticos", "O mesmo texto em modelos diferentes")
    examples=read_json(BASE/'synthetic_examples.json')['examples']
    for e in [examples[2],examples[3]]:
        p(e['intended'].capitalize()+': exemplo próprio',sub)
        p(html.escape(e['text']))
        keys=['cnn','bert_full','distilbert_full','modernbert_full']
        table(['Modelo','P(positivo), sem calibração'],[[('CNN' if k=='cnn' else NAMES[k.split('_')[0]]),f"{e['probabilities'][k]:.3f}"] for k in keys],[usable*.57,usable*.43])
    p("Definimos estes textos antes de observar suas predições e os avaliamos após o congelamento, usando a semente 17 em todos os casos. Ilustram negação, reversão de expectativa e avaliação mista. A intenção do autor é uma referência didática, sem substituir um conjunto de avaliação anotado.")
    p("O notebook e o HTML mostram os quatro exemplos nos oito classificadores, além das janelas que contribuem à margem linear da CNN. Essa decomposição é exata na última camada, mas não estabelece causalidade das palavras sobre a decisão.",small)
    page("10 / Fontes, limites e apresentação", "Ligação com a disciplina")
    table(['Tema visto no curso','O que os experimentos mostram'],[
        ['Convolução e viés indutivo','Pesos compartilhados, janelas locais, pooling e filtros treináveis.'],
        ['Gradientes e otimizadores','Taxa comum e taxa selecionada respondem perguntas diferentes.'],
        ['Generalização e regularização','Curvas de validação escolhem checkpoints; dropout pode mudar o compromisso entre ajuste e generalização.'],
        ['Resíduos e Transformers','Atenção e MLPs por posição usam caminhos residuais, com variantes entre famílias.'],
        ['Espaço de representações','Congelar o encoder treina uma fronteira sobre vetores existentes. Ajuste integral modifica também os vetores.'],
        ['Double descent e steering','Permanecem estudos separados: regressão sintética e intervenção nas ativações do Qwen, respectivamente.']],[usable*.34,usable*.66])
    p("Roteiro sugerido",sub)
    p("Para uma fala curta, selecione o diagrama CNN, a comparação de otimizadores, as curvas de dropout, a distinção congelado/completo e a comparação dos encoders. Use double descent ou steering como uma demonstração adicional, conforme o tempo disponível. Os slides incluem notas e material de apoio para não transformar a fala numa lista de métricas.")
    p("Verificação e limites",sub)
    p("O verificador reconstrói 217 métricas de treino/validação e auditoria, confirma as escolhas de taxa, as épocas e as partidas pareadas. Uma nova avaliação na GPU reconstrói os 43 checkpoints. O protocolo usa uma única divisão, três sementes e grades pequenas. Não mede superioridade universal nem isola o efeito da arquitetura do pré-treinamento. Não auditamos a presença de resenhas nos corpora de pré-treinamento.")
    p("Referências",sub)
    p("Prince, <i>Understanding Deep Learning</i>, capítulos 6-12. Slides MAE5920: <i>Redes Convolucionais</i>, páginas do arquivo 11-22, 36 e 51-52; <i>Transformers</i>, páginas 7-18, 21-31 e 35-38. Os hashes do acervo ficam no catálogo de referências.",small)
    p("Kim (2014), ACL D14-1181. Devlin et al. (2019), ACL N19-1423. Sanh et al. (2019), arXiv:1910.01108. Warner et al. (2024), arXiv:2412.13663. Model card e implementação oficial de ModernBERT no Hugging Face. URLs no HTML e no catálogo docs/REFERENCIAS_SEQUENCIAS.json.",small)
    PDF.parent.mkdir(parents=True,exist_ok=True)
    def decorate(c,doc):
        c.saveState();c.setStrokeColor(colors.HexColor('#167d92'));c.setLineWidth(2);c.line(42,h-31,w-42,h-31)
        c.setFont('SeqBody',7.5);c.setFillColor(colors.HexColor('#607985'));c.drawString(42,25,'MAE5920 - CNN e encoders');c.drawRightString(w-42,25,str(doc.page));c.restoreState()
    doc=SimpleDocTemplate(str(PDF),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=47,bottomMargin=43,title='CNN e encoders - MAE5920',author='Nestor, Letícia e Gabriel')
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)


def build():
    assert read_json(BASE/'verification.json')['status']=='passed'
    assert (BASE/'synthetic_examples.json').exists()
    s=summarize();figures(s);create_notebook(s);create_html(s);create_pdf(s)
    save_json(BASE/'report_manifest.json',{'pdf':PDF.relative_to(ROOT).as_posix(),'pdf_sha256':file_sha256(PDF),'html':HTML.relative_to(ROOT).as_posix(),'html_sha256':file_sha256(HTML),
        'notebook':NB.relative_to(ROOT).as_posix(),'notebook_execution':'pending','visual_review':'pending','selection_sha256':file_sha256(BASE/'selection_frozen.json'),
        'summary_sha256':file_sha256(BASE/'summary.json'),'verification_sha256':file_sha256(BASE/'verification.json'),
        'figures':{p.relative_to(ROOT).as_posix():file_sha256(p) for p in sorted(FIG.glob('*'))}})
    print('CNN/encoder report, figures, HTML and reading notebook created')
