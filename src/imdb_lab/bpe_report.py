"""BPE evidence and didactic sections for the existing report and notebook."""
from __future__ import annotations
import base64
import html
from pathlib import Path
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from .common import ROOT, read_json, save_json, file_sha256
from .models import Vocabulary
from .bpe_study import canonicalize, metrics, word_record

BASE = ROOT / "results/bpe"
FIG = BASE / "figures"
COLORS = ["#167D92", "#DF8B2F"]
LABELS = {"word": "Palavras", "bpe": "BPE 20 mil"}


def merge_trace(word, tokenizer):
    """Replay the actual learned merge ranks on one canonicalized word."""
    import json
    model = json.loads(tokenizer.to_str())["model"]
    ranks = {tuple(pair): i for i, pair in enumerate(model["merges"])}
    pieces = list(word[:-1]) + [word[-1] + "</w>"]
    rows = [{"step": 0, "rank": None, "pair": None, "tokens": pieces.copy()}]
    while len(pieces) > 1:
        candidates = [(ranks[(a, b)], (a, b)) for a, b in zip(pieces, pieces[1:]) if (a, b) in ranks]
        if not candidates: break
        rank, pair = min(candidates); merged = []; i = 0
        while i < len(pieces):
            if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == pair:
                merged.append(pieces[i] + pieces[i + 1]); i += 2
            else: merged.append(pieces[i]); i += 1
        pieces = merged
        rows.append({"step": len(rows), "rank": rank, "pair": list(pair), "tokens": pieces.copy()})
    assert pieces == tokenizer.encode(word).tokens
    return rows


def prepare_evidence():
    assert read_json(BASE / "verification.json")["status"] == "passed"
    cfg = read_json(ROOT / "bpe_config.json"); audit = read_json(BASE / "audit_metrics.json")
    records = {"word": [word_record(s) for s in cfg["seeds"]],
        "bpe": [read_json(BASE / "runs" / f"cnn_bpe20k__s{s}.json") for s in cfg["seeds"]]}
    groups = []
    for kind, runs in records.items():
        rows = [r for r in audit["metrics"] if r["kind"] == kind]
        group = {"kind": kind, "label": LABELS[kind], "parameters": runs[0]["parameters"],
            "best_epochs": [r["best_epoch"] for r in runs], "seeds": cfg["seeds"],
            "validation_ce_mean": float(np.mean([r["best_validation_ce"] for r in runs]))}
        for key in ["accuracy", "ce", "f1_macro", "roc_auc", "brier"]:
            values = [r[key] for r in rows]
            group.update({key + "_mean": float(np.mean(values)), key + "_min": float(np.min(values)), key + "_max": float(np.max(values))})
        groups.append(group)
    z = np.load(BASE / "audit_predictions.npz"); pairs = []
    for seed in cfg["seeds"]:
        w = z[f"word__s{seed}"].argmax(1) == z["labels"]
        b = z[f"bpe__s{seed}"].argmax(1) == z["labels"]
        pairs.append({"seed": seed, "both_correct": int((w & b).sum()), "word_only_correct": int((w & ~b).sum()),
            "bpe_only_correct": int((~w & b).sum()), "both_wrong": int((~w & ~b).sum()),
            "accuracy_difference_bpe_minus_word_pp": float(100 * (b.mean() - w.mean()))})
    stats = read_json(BASE / "tokenization_train_validation.json")
    for kind in ["word", "bpe"]: stats[kind]["audit"] = audit["tokenization"][kind]
    tokenizer = Tokenizer.from_file(str(BASE / "tokenizer.json"))
    vocab = Vocabulary(read_json(ROOT / "results/sequences/vocabulary.json"))
    examples = [{"word": word, "word_tokens": [word] if word in vocab.token_to_id else ["<unk>"],
        "word_ids": vocab.encode(word, 256), "bpe_tokens": tokenizer.encode(word).tokens,
        "bpe_ids": tokenizer.encode(word).ids} for word in ["pallbearer", "unwatchable", "wasn't", "cinematography"]]
    trace = merge_trace("pallbearer", tokenizer)
    curves = {kind: {key: {"epochs": list(range(13)), "mean": np.mean([[h[key] for h in r["history"]] for r in runs], axis=0).tolist(),
            "min": np.min([[h[key] for h in r["history"]] for r in runs], axis=0).tolist(),
            "max": np.max([[h[key] for h in r["history"]] for r in runs], axis=0).tolist()}
        for key in ["train_ce", "validation_ce", "validation_accuracy"]} for kind, runs in records.items()}
    result = {"groups": groups, "paired_counts": pairs, "audit_n": len(z["labels"]), "new_training_runs": 3,
        "reused_word_checkpoints": 3, "config": cfg, "tokenization": stats, "examples": examples,
        "merge_trace": trace, "curves": curves, "tokenizer_training": read_json(BASE / "tokenizer_training.json"),
        "difference_bpe_minus_word_pp": 100 * (groups[1]["accuracy_mean"] - groups[0]["accuracy_mean"]),
        "verification_sha256": file_sha256(BASE / "verification.json"),
        "interpretation": "BPE removes observed unknown tokens but does not improve mean accuracy in this fixed CNN recipe. This is a three-seed, one-holdout teaching comparison with a learning rate inherited from the word model."}
    save_json(BASE / "summary.json", result)
    pd.DataFrame(groups).to_csv(BASE / "comparison.csv", index=False)
    pd.DataFrame(pairs).to_csv(BASE / "paired_comparison.csv", index=False)
    save_json(BASE / "presentation_data.json", result)
    figures(result)
    save_json(BASE / "report_evidence_manifest.json", {"summary_sha256": file_sha256(BASE / "summary.json"),
        "builder_sha256": file_sha256(Path(__file__)), "figures": {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sorted(FIG.glob("*"))}})
    return result


def figures(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": .15, "svg.fonttype": "none", "svg.hashsalt": "mae5920-bpe"})
    def save(fig, name):
        for ext in ["png", "svg"]:
            fig.savefig(FIG / f"{name}.{ext}", dpi=180, bbox_inches="tight", facecolor="white", metadata={"Date": None} if ext == "svg" else None)
            if ext == "svg":
                path = FIG / f"{name}.{ext}"
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        plt.close(fig)
    f, axes = plt.subplots(1, 2, figsize=(10.4, 3.5), constrained_layout=True)
    for i, kind in enumerate(["word", "bpe"]):
        for ax, key in zip(axes, ["train_ce", "validation_ce"]):
            c = s["curves"][kind][key]
            ax.plot(c["epochs"], c["mean"], color=COLORS[i], lw=2, label=LABELS[kind])
            ax.fill_between(c["epochs"], c["min"], c["max"], color=COLORS[i], alpha=.12)
            ax.set(xlabel="Época", ylabel="Entropia cruzada")
    axes[0].set_title("Treino medido sem dropout"); axes[1].set_title("Validação escolhe o checkpoint")
    axes[1].legend(frameon=False); save(f, "learning_curves")
    f, axes = plt.subplots(1, 2, figsize=(10.4, 3.1), constrained_layout=True)
    x = np.arange(2)
    for i, kind in enumerate(["word", "bpe"]):
        vals = [s["tokenization"][kind][r] for r in ["validation", "audit"]]
        axes[0].bar(x + (i - .5) * .34, [100 * r["unknown_rate"] for r in vals], .34, color=COLORS[i], label=LABELS[kind])
        axes[1].bar(x + (i - .5) * .34, [r["mean_tokens"] for r in vals], .34, color=COLORS[i])
    for ax in axes: ax.set(xticks=x, xticklabels=["Validação", "Nova auditoria"])
    axes[0].set(ylabel="Tokens <unk> (%)", title="Cobertura do vocabulário", ylim=(0, 4)); axes[0].legend(frameon=False)
    axes[1].set(ylabel="Tokens médios por resenha", title="Sequências um pouco mais longas", ylim=(0, 160))
    save(f, "coverage_lengths")
    f, axes = plt.subplots(1, 2, figsize=(10.4, 3.1), constrained_layout=True)
    for i, row in enumerate(s["groups"]):
        mean = 100 * row["accuracy_mean"]
        axes[0].errorbar(i, mean, yerr=[[mean - 100 * row["accuracy_min"]], [100 * row["accuracy_max"] - mean]], fmt="o", color=COLORS[i], capsize=5, ms=7)
        axes[0].annotate(f"{mean:.2f}%", (i, mean), xytext=(14, 0), textcoords="offset points", fontsize=10)
    axes[0].set(xticks=[0, 1], xticklabels=["Palavras", "BPE 20 mil"], xlim=(-.5, 1.8), ylim=(80.5, 83), ylabel="Acurácia (%)", title="Média e mínimo/máximo das três sementes")
    axes[1].bar(range(3), [r["accuracy_difference_bpe_minus_word_pp"] for r in s["paired_counts"]], color=COLORS[1], width=.55)
    axes[1].axhline(0, color="#526a77", lw=.8)
    axes[1].set(xticks=range(3), xticklabels=[str(r["seed"]) for r in s["paired_counts"]], xlabel="Semente pareada", ylabel="BPE menos palavras (p.p.)", title="Diferença na mesma amostra")
    save(f, "audit_comparison")


def notebook_cells():
    import nbformat as nbf
    md = nbf.v4.new_markdown_cell; code = nbf.v4.new_code_cell
    return [
        md("## BPE aprendido no treino: outro experimento controlado\n\nAcrescentamos três treinos de CNN com BPE de 20 mil tokens e reutilizamos os três checkpoints por palavras. Mantemos a limpeza por regex, a arquitetura, os pesos numéricos iniciais, os minibatches e Adam 0,001. O BPE aprende regras estatísticas sem rótulos, fica congelado e depois a CNN aprende embeddings e filtros por gradiente. É BPE de caracteres com fim de palavra, não BPE de bytes. O tamanho de uma janela agora conta subpalavras. [Sennrich et al. (2016)](https://aclanthology.org/P16-1162/) e [Tokenizers](https://huggingface.co/docs/tokenizers/quicktour)."),
        code("from tokenizers import Tokenizer\nfrom imdb_lab.bpe_study import canonicalize, metrics\nfrom imdb_lab.bpe_report import merge_trace\nBP = ROOT / 'results/bpe'\nbs = read_json(BP / 'summary.json')\ntok = Tokenizer.from_file(str(BP / 'tokenizer.json'))\ntext = \"The movie wasn't good! The pallbearer was unwatchable.\"\nprint('Texto:', text)\nprint('Normalizado:', canonicalize(text))\nprint('BPE:', tok.encode(canonicalize(text)).tokens)\ndisplay(pd.DataFrame(bs['examples'])[['word','word_tokens','bpe_tokens']])\ndisplay(pd.DataFrame(merge_trace('pallbearer', tok)))"),
        md("### Cobertura e comprimento\n\nO tokenizer vê somente os 6.144 textos de treino, sem consultar rótulos ou validação. O marcador `</w>` identifica o final de uma palavra. Uma palavra rara pode ser composta de peças conhecidas. Isso não garante uma decomposição linguística nem ausência de UNK em alfabetos nunca vistos. A CNN começa novamente do zero, porque os índices passaram a representar outras unidades."),
        code("display(Image(filename=str(BP / 'figures/coverage_lengths.png')))\ndisplay(pd.DataFrame([{'tokenização': k, 'papel': role, **stats} for k,roles in bs['tokenization'].items() for role,stats in roles.items()])[['tokenização','papel','unknown_rate','mean_tokens','truncated_documents']])\nprint('Treino do tokenizer:', bs['tokenizer_training']['documents'], 'textos, sem rótulos')"),
        md("### Curvas da mesma CNN\n\nTreinamos 12 épocas e escolhemos a menor CE de validação entre as épocas 1 e 12. As faixas são mínimo/máximo de três sementes. A taxa veio da CNN por palavras e não foi reajustada para favorecer o BPE. Os parâmetros iniciais são numericamente idênticos, mas cada linha de embedding tem um significado diferente."),
        code("display(Image(filename=str(BP / 'figures/learning_curves.png')))\ndisplay(pd.DataFrame(bs['groups'])[['label','parameters','best_epochs','validation_ce_mean']])\nfor seed in bs['config']['seeds']:\n    w = read_json(B / 'runs' / f'cnn_optimizer__adam__lr0.001__s{seed}.json')\n    b = read_json(BP / 'runs' / f'cnn_bpe20k__s{seed}.json')\n    assert w['initial_state_sha256'] == b['initial_state_sha256']\n    assert w['batch_order_sha256'] == b['batch_order_sha256']\nprint('Partidas e minibatches pareados nas três sementes')"),
        md("### Uma nova auditoria para os seis checkpoints\n\nOs 2.048 exemplos desta comparação excluem todos os papéis dos estudos MLP e CNN/encoders. Abrimos a entrada somente após congelar os seis checkpoints. Não compare diretamente estes números com os resultados dos encoders acima: as auditorias são diferentes. O BPE eliminou os UNKs observados, mas a acurácia média ficou um pouco menor nesta receita fixa. Isso mostra por que cobertura e qualidade de classificação são perguntas distintas. Não concluímos superioridade universal de um tokenizer."),
        code("display(Image(filename=str(BP / 'figures/audit_comparison.png')))\ndisplay(pd.DataFrame(bs['groups'])[['label','accuracy_mean','accuracy_min','accuracy_max','ce_mean','f1_macro_mean']])\ndisplay(pd.DataFrame(bs['paired_counts']))\nbm = read_json(BP / 'data_manifest.json')\nexcluded = set().union(*(set(v['ids']) for m0 in [m,old] for v in m0['roles'].values()))\nassert not set(bm['roles']['audit']['ids']) & excluded\nz = np.load(BP / 'audit_predictions.npz')\nfor r in read_json(BP / 'audit_metrics.json')['metrics']:\n    measured = metrics(z[r['run_id']], z['labels'])\n    assert all(abs(measured[k] - r[k]) < 1e-12 for k in measured)\nprint('18 conjuntos de métricas conferidos pelo verificador, seis checkpoints reconstruídos na GPU')\nprint('Diferença média BPE menos palavras:', bs['difference_bpe_minus_word_pp'], 'p.p.')")]


def html_section():
    s = read_json(BASE / "summary.json")
    def image(name):
        data = base64.b64encode((FIG / f"{name}.png").read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}" alt="{html.escape(name)}">'
    parts = ['<section><h2>BPE aprendido no treino</h2><p>Três novos treinos da mesma CNN, BPE 20 mil e três checkpoints por palavras reutilizados. A normalização por regex é igual. O tokenizer aprende sem rótulos e fica congelado antes do treino neural. BPE de caracteres com marcador de fim de palavra, sem byte fallback.</p>',
        pd.DataFrame(s["examples"])[["word", "word_tokens", "bpe_tokens"]].to_html(index=False),
        '<p>Em pallbearer, aplicamos as regras aprendidas na ordem de prioridade abaixo. O marcador &lt;/w&gt; indica o final da palavra. O kernel de 3 tokens pode agora cobrir apenas parte de uma palavra.</p>',
        pd.DataFrame(s["merge_trace"]).to_html(index=False), image("coverage_lengths"), image("learning_curves"),
        '<h2>Resultado em uma auditoria diferente da seção dos encoders</h2><p>2.048 resenhas internas não usadas nos dois estudos didáticos anteriores. Mesmo treino e validação, Adam 0,001, 12 épocas, três sementes e 2.714.502 parâmetros. Mesmas matrizes iniciais e ordens de minibatches. Congelamos as seis escolhas antes de liberar a nova auditoria.</p>',
        image("audit_comparison"), pd.DataFrame(s["groups"])[["label", "accuracy_mean", "accuracy_min", "accuracy_max", "ce_mean", "f1_macro_mean"]].to_html(index=False, float_format=lambda x: f"{x:.5f}"),
        pd.DataFrame(s["paired_counts"]).to_html(index=False),
        '<p>O BPE eliminou os UNKs observados, mas não melhorou a acurácia média nesta receita. A taxa veio da CNN por palavras e não foi retestada. Faixas descrevem três sementes, sem representar toda a incerteza amostral. O experimento altera também o alcance das janelas em palavras.</p>',
        '<p>Fontes: <a href="https://aclanthology.org/P16-1162/">Sennrich et al. (2016)</a> e <a href="https://huggingface.co/docs/tokenizers/quicktour">documentação Tokenizers</a>. Algoritmo de subpalavras e exemplo de tradução não garantem melhoria na classificação de sentimento.</p></section>']
    return ''.join(parts)


def pdf_pages(page, p, table, pic, body, small, sub, usable):
    s = read_json(BASE / "summary.json"); pct = lambda x: f"{100*x:.2f}%".replace('.', ',')
    page("BPE / 1 - Representação", "Aprender as peças do texto")
    p("O tokenizer por palavras monta um vocabulário de frequência no treino. O BPE também aprende no treino, mas começa com caracteres e combina pares frequentes. As regras aprendidas permitem representar palavras raras por partes. Depois congelamos essas regras e treinamos a CNN do zero.")
    table(["Etapa", "O que aprende", "Como aprende"], [
        ["Tokenizer BPE", "Vocabulário e regras de união", "Contagem de pares no texto de treino, sem rótulos"],
        ["CNN", "Embeddings, filtros e saída", "Gradientes da classificação usando os rótulos"],
        ["Validação e auditoria", "Nenhum parâmetro", "Aplicam as regras e os pesos já aprendidos"]], [usable*.23, usable*.35, usable*.42])
    p("Preservamos a limpeza da CNN: minúsculas, regex de palavras e números e espaços entre tokens. Usamos WhitespaceSplit, BPE de caracteres e marcador de fim de palavra &lt;/w&gt;. Não usamos BPE de bytes nem os tokenizers dos Transformers. PAD é 0 e UNK é 1. O vocabulário final tem exatamente 20 mil entradas.")
    p("Exemplos do tokenizer aprendido", sub)
    table(["Palavra", "Vocabulário por palavras", "BPE"], [[html.escape(e["word"]), html.escape(' '.join(e["word_tokens"])), html.escape(' + '.join(e["bpe_tokens"]))] for e in s["examples"]], [usable*.24, usable*.27, usable*.49])
    p("Uma inferência passo a passo", sub)
    learned = s['tokenizer_training']['first_merges'][:3]
    p("As primeiras regras aprendidas no treino foram " + '; '.join(html.escape(' + '.join(pair)) for pair in learned) + ". Abaixo, aplicamos as regras já congeladas a uma palavra.")
    table(["Passo", "Estado da palavra pallbearer"], [[r["step"], html.escape(' + '.join(r["tokens"]))] for r in s["merge_trace"]], [usable*.13, usable*.87])
    p("A sequência aplica as prioridades das regras já aprendidas. Ela não reaprende BPE na inferência. As peças não precisam corresponder a morfemas. Caracteres ausentes do alfabeto de treino ainda podem gerar UNK.", small)
    p("Fontes: Sennrich, Haddow e Birch (2016), ACL P16-1162; documentação Tokenizers, Quicktour. O artigo trata tradução e não garante melhoria na nossa tarefa. URLs no HTML e em docs/PROTOCOLO_BPE.md.", small)
    page("BPE / 2 - Controles", "A mesma CNN, outras unidades")
    p("Acrescentamos três treinos BPE e reutilizamos três checkpoints da CNN por palavras. Ambos usam os mesmos 6.144 exemplos de treino e 1.536 de validação, com os mesmos prefixos de 128 palavras. O BPE vê apenas o texto de treino. Os rótulos são usados depois, pela rede.")
    table(["Controle", "Valor nos dois modelos"], [["Vocabulário e embedding", "20.000 entradas, 128 dimensões"], ["Convoluções e saída", "3/4/5, 100 filtros por janela, dropout 0,5, saída 300 × 2"], ["Otimização", "Adam 0,001, batch 64, clipping 1, 12 épocas"], ["Partidas pareadas", "Sementes 17, 29, 43; matrizes iniciais e minibatches iguais"], ["Seleção", "Menor CE de validação por semente, épocas 1 a 12"], ["Parâmetros treináveis", "2.714.502 em cada modelo"]], [usable*.32, usable*.68])
    p("A taxa veio da seleção anterior da CNN por palavras e não foi ajustada para o BPE. O limite é de 256 tokens em ambos. Matrizes iniciais iguais não tornam os embeddings semanticamente iguais: cada índice aponta para uma palavra ou uma subpalavra diferente. As janelas 3/4/5 agora medem unidades diferentes.")
    pic("bpe/coverage_lengths", 215)
    a = s["tokenization"]["word"]["audit"]; b = s["tokenization"]["bpe"]["audit"]
    p(f"Na nova auditoria, palavras produzem <b>{pct(a['unknown_rate'])}</b> de tokens UNK e BPE, <b>{pct(b['unknown_rate'])}</b>. O comprimento médio passa de {a['mean_tokens']:.1f} para {b['mean_tokens']:.1f} tokens (+{100*(b['mean_tokens']/a['mean_tokens']-1):.1f}%). Nenhuma resenha foi truncada nos três papéis. Essa cobertura é uma observação deste corpus, sem garantia para qualquer texto futuro.")
    p("Auditoria nova para os seis checkpoints", sub)
    p("Reservamos 2.048 resenhas balanceadas do treino oficial limpo, excluindo todos os papéis dos estudos didáticos MLP e CNN/encoders. O arquivo de entrada ficou fora do treino e foi liberado só após congelar os seis checkpoints. O teste oficial não foi reaberto. Esta amostra é diferente daquela usada na tabela dos encoders; os números não devem ser misturados.")
    page("BPE / 3 - Resultado", "Cobertura e classificação")
    pic("bpe/learning_curves", 205)
    pic("bpe/audit_comparison", 190)
    table(["Tokenização", "Acurácia média", "CE média", "F1 macro"], [[r["label"], pct(r["accuracy_mean"]), f"{r['ce_mean']:.3f}", f"{r['f1_macro_mean']:.3f}"] for r in s["groups"]], [usable*.28, usable*.27, usable*.23, usable*.22])
    p(f"Nesta receita, BPE fica {abs(s['difference_bpe_minus_word_pp']):.2f} ponto percentual abaixo em média. As três diferenças pareadas são negativas. Retirar UNKs preserva mais detalhes lexicais, mas isso não basta para melhorar uma CNN com este conjunto de janelas, taxa e orçamento. Não concluímos que BPE seja pior de modo geral.")
    p("Faixas mostram mínimo/máximo das três sementes. Uma divisão e três partidas não quantificam toda a incerteza. A comparação não procura a melhor receita para cada tokenizer. Os seis checkpoints reproduziram exatamente seus logits em outro processo; 18 conjuntos de métricas foram reconstruídos. Tempos brutos contêm efeitos de aquecimento e não sustentam uma comparação de velocidade.", small)


def build():
    prepare_evidence()
    from .sequence_report import build as build_integrated
    build_integrated()
