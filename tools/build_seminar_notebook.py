#!/usr/bin/env python3
"""Constrói o notebook de entrega com os dados e o código da execução."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile

import nbformat as nbf
import numpy as np
import pandas as pd

from seminar_report_cells import EVIDENCE, datasets, diagnostics, external, failures, official

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/seminario_completo.ipynb"
RESULT = ROOT / "results/seminar"
REPORT = "https://github.com/costaleirbag/projeto-mae5920/blob/main/output/pdf/relatorio_seminario.pdf"
SLIDES = "https://costaleirbag.github.io/projeto-mae5920/"
CELLS = []
SOURCE_EXTRACTS = []

EVIDENCE_CORE = (
    "data/manifest.json",
    "results/comparison.csv",
    "results/didactic/data_manifest.json",
    "results/sequences/data_manifest.json",
    "results/sequences/vocabulary.json",
    "results/sequences/models_pinned.json",
    "results/sequences/summary.json",
    "results/sequences/presentation_data.json",
    "results/sequences/audit_metrics.json",
    "results/sequences/runtime.json",
    "results/bpe/data_manifest.json",
    "results/bpe/tokenizer.json",
    "results/bpe/tokenizer_training.json",
    "results/bpe/summary.json",
    "results/bpe/audit_metrics.json",
    "results/bpe/audit_predictions.npz",
    "results/qwen/classification/metrics.json",
    "results/qwen_improved/presentation_examples.json",
    "results/qwen_improved/selected_training_features.json",
    "seminario/dados/animacoes.json",
)
RUNTIME = (
    "config.json", "sequence_config.json", "bpe_config.json", "didactic_config.json",
    "qwen_config.json", "qwen_improve_config.json", "pyproject.toml", "requirements-lock.txt",
    "run.py", "run_sequences.py", "run_qwen.py", "run_qwen_improve.py",
    "tools/seminar_retrain.py", "tools/seminar_recompute.py", "tools/seminar_steering.py",
    "tools/fetch_sequence_models.py", "tools/sequence_audit_input.py",
    "seminario/casos.json",
    *(f"seminario/{name}.py" for name in (
        "demo", "medir_fora_dominio", "medir_pares", "medir_animacoes", "medir_passo",
        "comparar_erros", "auditar_baseline", "medir_pooling_regularizacao",
        "medir_tfidf_slide", "medir_tfidf_youtube", "medir_recorte", "medir_recorte_bert",
        "medir_comprimentos", "medir_qwen_youtube",
    )),
    *(p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "src/imdb_lab").glob("*.py"))),
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def array_bytes(**arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def md(text, key=None):
    cell = nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())
    cell.id = key or f"leitura-{len(CELLS):03}"
    CELLS.append(cell)


def code(text, key=None, *, hidden=False):
    cell = nbf.v4.new_code_cell(textwrap.dedent(text).strip())
    cell.id = key or f"codigo-{len(CELLS):03}"
    if hidden:
        cell.metadata.jupyter = {"source_hidden": True}
    CELLS.append(cell)


def extract(path, names):
    source = (ROOT / path).read_text(encoding="utf-8")
    nodes = {getattr(n, "name", None): n for n in ast.parse(source).body}
    selected = []
    for name in names:
        node = nodes[name]
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        segment = "\n".join(source.splitlines()[start - 1:node.end_lineno])
        selected.append(segment)
        SOURCE_EXTRACTS.append({"source": path, "symbol": name, "source_sha256": sha(source.encode()),
                                "definition_sha256": sha(segment.encode())})
    return "\n\n\n".join(selected)


def official_predictions(content, origins):
    models = {"tfidf": "tfidf", "cnn": "cnn", "bert": "bert", "qwen": "qwen/classification"}
    frames = {}
    for name, folder in models.items():
        path = f"results/{folder}/predictions.parquet"
        data = (ROOT / path).read_bytes()
        frames[name] = pd.read_parquet(io.BytesIO(data)).set_index("id", verify_integrity=True)
        origins[f"source_archive/oficial_{name}"] = {"source": path, "sha256": sha(data)}
    reference = frames["tfidf"]
    splits = {
        role: {"ids": frame.index.tolist(), "labels": frame["label"].astype(int).tolist()}
        for role, frame in reference.groupby("split", sort=False)
    }
    content["derived/papeis_originais.json"] = json.dumps(splits, ensure_ascii=False).encode()
    test = reference[reference["split"] == "test"]
    arrays = {"ids": test.index.to_numpy(dtype=str), "labels": test["label"].to_numpy(dtype=np.int8),
              "word_counts": test["word_count"].to_numpy(dtype=np.int32)}
    for name, frame in frames.items():
        held_out = frame[frame["split"] == "test"]
        assert set(held_out.index) == set(test.index)
        aligned = held_out.loc[test.index]
        np.testing.assert_array_equal(aligned["label"], arrays["labels"])
        np.testing.assert_array_equal(aligned["word_count"], arrays["word_counts"])
        arrays[name + "_logits"] = aligned["logit"].to_numpy(dtype=np.float64)
        arrays[name + "_calibrated"] = aligned["p_positive"].to_numpy(dtype=np.float64)
    content["derived/oficial.npz"] = array_bytes(**arrays)


def snapshot():
    content = {name: (ROOT / name).read_bytes() for name in set(EVIDENCE_CORE) | set(EVIDENCE) | set(RUNTIME)}
    origins = {}
    official_predictions(content, origins)
    winner = json.loads(content["results/tfidf_regularization/summary.json"])["selected_by_validation"]
    for folder, keys in (
        ("tfidf_ablation", ("test_labels", "test_integral_unigramas_bigramas", "test_400_unigramas_bigramas")),
        ("tfidf_regularization", ("test_labels", f"test_{winner}")),
    ):
        source = f"results/{folder}/predictions.npz"
        raw = (ROOT / source).read_bytes()
        with np.load(io.BytesIO(raw), allow_pickle=False) as values:
            content[f"derived/{folder}.npz"] = array_bytes(**{key: values[key] for key in keys})
        origins[f"source_archive/{folder}"] = {"source": source, "sha256": sha(raw)}
    features = "results/qwen_improved/selected_training_features.npz"
    raw = (ROOT / features).read_bytes()
    with np.load(io.BytesIO(raw), allow_pickle=False) as values:
        vectors, labels = values["train_L28"].astype(np.float64), values["train_labels_L28"]
        means = {f"media_{label}": vectors[labels == label].mean(0) for label in (0, 1)}
    with np.load(ROOT / "results/qwen_improved/directions.npz", allow_pickle=False) as directions:
        content["derived/direcao_qwen.npz"] = array_bytes(direction=directions["intent_256_L28_means"], **means)
    origins["source_archive/estados_qwen"] = {"source": features, "sha256": sha(raw)}
    direction_source = "results/qwen_improved/directions.npz"
    origins["source_archive/direcao_qwen"] = {"source": direction_source,
                                           "sha256": sha((ROOT / direction_source).read_bytes())}
    assets = {name: {"sha256": sha(data), "bytes": len(data)} for name, data in content.items()}
    assets.update(origins)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(content.items()):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            archive.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return buffer.getvalue(), assets


def cover():
    md(f"""
    <a id="introducao"></a>
    # O que a CNN lê em uma resenha?

    Nestor, Letícia e Gabriel · MAE5920 · Notebook de entrega

    [Relatório de 8 páginas]({REPORT}) · [Apresentação]({SLIDES})

    | Relatório | Slides | Seção neste notebook |
    |---|---|---|
    | 1 | 1 e 2 | Introdução e métricas |
    | 2 | 3 | Dados |
    | 3 | 4 e 5 | Arquitetura da TextCNN |
    | 4 | 6 e 11 | Experimentos e resultados |
    | 5 | 7 a 9 e apoio 16 | Teste oficial do IMDb |
    | 6 | 9, 10, 14 e apoio 17 | Diagnósticos e custo |
    | 7 | 2, 12 e apoio 19 | Quando a CNN falha |
    | 8 | 13 e 15 | Fora do IMDb e conclusão |
    | | Apoio 18 | Intervenção no Qwen |

    **Conferir em CPU.** Salve este arquivo na pasta de execução e rode todas as células. Elas reconstroem
    métricas a partir das previsões incluídas, apresentam os diagnósticos do relatório e executam as
    operações centrais em exemplos pequenos. Esse caminho funciona com o arquivo sozinho.

    **Reproduzir os experimentos.** No fim do notebook, selecione o modo REPRODUZIR e execute
    todas as células. O notebook inclui a implementação, as configurações e os identificadores das
    divisões. O treino baixa o IMDb na revisão fixada e os modelos nas revisões registradas, exige CUDA
    para o Qwen e grava novos resultados em uma pasta própria. Métricas e tempos de uma nova execução
    são confrontados com os números registrados.

    Para conferir em CPU, use Python 3.11+ com Jupyter, NumPy, pandas, SciPy, scikit-learn,
    Matplotlib, PyTorch e Tokenizers 0.22.x. Para refazer os experimentos, instale também
    pyarrow, datasets, Transformers, accelerate e huggingface-hub nas versões indicadas no final.
    """, "capa")
    code(r'''
    from pathlib import Path
    from dataclasses import dataclass
    from collections import Counter
    from contextlib import nullcontext
    import ast, base64, copy, hashlib, html, io, json, math, os, re, subprocess, sys, time, zipfile
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torch.nn import functional as F
    from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from IPython.display import display, Code, HTML, Image

    torch.set_num_threads(4)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .15, "figure.dpi": 105})
    pd.set_option("display.max_colwidth", 100)
    CORES = ("#355e8f", "#a0462c", "#166b64")
    CHECKS = []

    def conferir(nome, condicao):
        assert bool(condicao), nome
        CHECKS.append(nome)

    candidatos = [Path("seminario_completo.ipynb"), Path("notebooks/seminario_completo.ipynb"),
                  *Path.cwd().glob("*.ipynb")]
    for arquivo_notebook in dict.fromkeys(candidatos):
        if arquivo_notebook.is_file():
            documento = json.loads(arquivo_notebook.read_text(encoding="utf-8"))
            meta = documento.get("metadata", {}).get("mae5920_seminar", {})
            if meta.get("format") == 2:
                break
    else:
        raise FileNotFoundError("Salve este .ipynb na pasta de execução ou em notebooks/.")
    pacote = base64.b64decode(documento["cells"][0]["attachments"]["evidencias.zip"]["application/zip"])
    conferir("Integridade do anexo", hashlib.sha256(pacote).hexdigest() == meta["payload_sha256"])
    ARQUIVOS = zipfile.ZipFile(io.BytesIO(pacote))

    def read_json(nome):
        return json.loads(ARQUIVOS.read(str(nome)))

    def read_npz(nome):
        return np.load(io.BytesIO(ARQUIVOS.read(str(nome))), allow_pickle=False)

    def read_csv(nome):
        return pd.read_csv(io.BytesIO(ARQUIVOS.read(str(nome))))

    def figura(nome):
        display(Image(data=ARQUIVOS.read(nome)))

    def mostrar_codigo(arquivo, *simbolos):
        """Exibe o código integral ou as funções nomeadas do anexo."""
        fonte = ARQUIVOS.read(arquivo).decode("utf-8")
        if simbolos:
            nos = {getattr(no, "name", None): no for no in ast.parse(fonte).body}
            linhas = fonte.splitlines()
            blocos = []
            for nome in simbolos:
                no = nos[nome]
                inicio = min([no.lineno] + [d.lineno for d in no.decorator_list])
                blocos.append("\n".join(linhas[inicio - 1:no.end_lineno]))
            fonte = "\n\n".join(blocos)
        display(Code(fonte, language="python"))

    seq = read_json("results/sequences/presentation_data.json")
    s = seq["summary"]
    bp = read_json("results/bpe/summary.json")
    cfg = read_json("sequence_config.json")
    SROOT = Path("results/sequences")

    def settings():
        return cfg

    print(f"{len(ARQUIVOS.namelist())} arquivos disponíveis no anexo local.")
    ''', "ambiente-e-anexo", hidden=True)
    md(r"""
    ## Introdução e métricas · relatório p. 1

    A variável de interesse é o sentimento da resenha, negativo (0) ou positivo (1).
    Acurácia mede a fração de acertos. F1 macro faz a média do F1 de cada classe,
    com \(F1=2PR/(P+R)\). A entropia cruzada penaliza a baixa probabilidade da classe verdadeira.
    O par com *but* reaparece na seção de falhas com as previsões dos modelos.
    """, "introducao")


def architecture():
    md(f"""
    <a id="cnn"></a>
    ## Arquitetura da TextCNN · relatório p. 3

    No [relatório, p. 3]({REPORT}#page=3) e nos slides 4 e 5, a mesma arquitetura
    recebe tokens por regex ou BPE. Uma janela aplica seus pesos aos embeddings das
    posições locais. Cada filtro entrega um máximo, e a camada linear recebe 300 valores.
    """, "cnn")
    code(extract("src/imdb_lab/data.py", ["clean_text"]) + "\n\n" +
         textwrap.dedent('''
         TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\\d+")
         ''') + "\n" + extract("src/imdb_lab/models.py", ["tokenize", "Vocabulary"]) + "\n\n" +
         extract("src/imdb_lab/bpe_study.py", ["canonicalize", "fit_bpe"]), "tokenizadores")
    code(extract("src/imdb_lab/bpe_report.py", ["merge_trace"]) + textwrap.dedent('''

    vocab = Vocabulary(read_json("results/sequences/vocabulary.json"))
    bpe_real = Tokenizer.from_str(ARQUIVOS.read("results/bpe/tokenizer.json").decode())
    palavra = "pallbearer"
    pecas = bpe_real.encode(palavra).tokens
    conferir("Palavra rara recebe UNK por regex", vocab.encode(palavra, 10) == [1])
    conferir("BPE usa as peças do slide 4", pecas == ["p", "all", "bearer</w>"])
    display(pd.DataFrame([{"etapa": r["step"], "peças": " + ".join(r["tokens"])}
                          for r in merge_trace(palavra, bpe_real)]))
    '''), "bpe-pallbearer")
    code(extract("src/imdb_lab/models.py", ["TextCNN", "PadBatch"]) + "\n\n" +
         extract("src/imdb_lab/sequences.py", ["initialize_cnn"]), "textcnn-codigo-real")
    code('''
    # Exemplo de dimensões com pesos iniciais. Os escores treinados vêm na p. 7.
    alvo_cpu = torch.device("cpu")
    modelo = initialize_cnn(cfg["cnn"], seed=17, target=alvo_cpu).eval()
    frase_a = "I thought this movie was terrible, but it was wonderful."
    ids_a = torch.tensor([vocab.encode(frase_a, 256)])
    tamanho = torch.tensor([ids_a.shape[1]])
    with torch.inference_mode():
        embedding = modelo.embedding(ids_a)
        x = embedding.transpose(1, 2)
        mapas = [torch.relu(conv(x)) for conv in modelo.convs]
        uniao = torch.cat([mapa.max(2).values for mapa in mapas], dim=1)
        logits = modelo(ids_a, tamanho)
        calcular = (x[0, :, :3] * modelo.convs[0].weight[0]).sum() + modelo.convs[0].bias[0]
        torch.testing.assert_close(calcular, modelo.convs[0](x)[0, 0, 0], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(modelo.classifier(uniao), logits)
        torch.testing.assert_close(modelo(F.pad(ids_a, (0, 12)), tamanho), logits)
    conferir("TextCNN com 2.714.502 parâmetros", sum(p.numel() for p in modelo.parameters()) == 2_714_502)
    display(pd.DataFrame([
        ("IDs", tuple(ids_a.shape)), ("Embedding", tuple(embedding.shape)),
        *((f"Mapa k={k}", tuple(mapa.shape)) for k, mapa in zip(modelo.kernels, mapas)),
        ("Máximos concatenados", tuple(uniao.shape)), ("Logits", tuple(logits.shape)),
    ], columns=["Etapa", "Dimensão"]))
    ''', "formas")
def training():
    md(f"""
    <a id="treino"></a>
    ## Experimentos e resultados · relatório p. 4

    A regra de seleção é a menor entropia cruzada na validação. O laço abaixo é o dos
    ensaios internos. A CNN do teste oficial tem receita própria no bloco seguinte.
    [Relatório, p. 4]({REPORT}#page=4) · [Curvas no slide 6]({SLIDES}#slide-6)
    """, "treino")
    helpers = ("array_hash", "state_hash", "snapshot", "sync", "autocast", "orders", "batches",
               "indexed", "optimizer_for", "predict", "select_rate")
    code(extract("src/imdb_lab/sequences.py", helpers), "treinador-auxiliares", hidden=True)
    code(extract("src/imdb_lab/sequences.py", ["train_one"]), "treinador-real")
    code('''
    display(pd.DataFrame([
        {"Otimizador": opt, "Taxas testadas": taxas, "Taxa escolhida": s["choices"]["cnn_" + opt]}
        for opt, taxas in cfg["cnn"]["learning_rates"].items()
    ]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.25), layout="constrained")
    for i, opt in enumerate(("sgd", "momentum", "adam")):
        curva = seq["curves"]["cnn_" + opt]["validation_ce"]
        axes[0].plot(curva["epochs"], curva["mean"], label=opt, color=CORES[i])
        axes[0].fill_between(curva["epochs"], curva["min"], curva["max"], color=CORES[i], alpha=.13)
    for i, particao in enumerate(("train_ce", "validation_ce")):
        curva = seq["curves"]["cnn_adam"][particao]
        axes[1].plot(curva["epochs"], curva["mean"], color=CORES[i],
                     label=("Treino", "Validação")[i])
        axes[1].fill_between(curva["epochs"], curva["min"], curva["max"], color=CORES[i], alpha=.13)
    axes[0].set(title="Otimizadores na validação", xlabel="Época", ylabel="Entropia cruzada")
    axes[1].set(title="Seleção de época com Adam", xlabel="Época", ylabel="Entropia cruzada")
    for ax in axes:
        ax.legend(frameon=False)
    plt.show()
    grupos = pd.DataFrame(s["groups"])
    colunas = ["label", "lr", "best_epochs", "accuracy_mean"]
    display(grupos.loc[grupos["group"].isin(["cnn_architecture", "cnn_dropout"]), colunas]
            .rename(columns=dict(zip(colunas, ["Condição", "Taxa", "Épocas escolhidas", "Acurácia de auditoria"])))
            .round(4).reset_index(drop=True))
    ''', "curvas-e-grade")
    code(extract("src/imdb_lab/bpe_study.py", ["metrics"]) + textwrap.dedent('''

    auditoria_bpe = read_json("results/bpe/audit_metrics.json")
    auditoria_logits = read_npz("results/bpe/audit_predictions.npz")
    y_audit = auditoria_logits["labels"]
    bpe_rows = []
    for row in auditoria_bpe["metrics"]:
        medido = metrics(auditoria_logits[row["run_id"]], y_audit)
        conferir(row["run_id"] + " acurácia/F1/CE refeitos", all(
            np.isclose(medido[k], row[k], atol=1e-8, rtol=0) for k in ("accuracy", "f1_macro", "ce")
        ))
        bpe_rows.append({"Representação": {"word": "Palavras", "bpe": "BPE"}[row["kind"]],
                         "Acurácia": medido["accuracy"], "F1 macro": medido["f1_macro"],
                         "CE": medido["ce"]})
    tabela_bpe = pd.DataFrame(bpe_rows).groupby("Representação", sort=False).mean()
    for grupo, kind in (("Palavras", "word"), ("BPE", "bpe")):
        tk = auditoria_bpe["tokenization"][kind]
        tabela_bpe.loc[grupo, "UNK (%)"] = 100 * tk["unknown_rate"]
        tabela_bpe.loc[grupo, "Tokens médios"] = tk["mean_tokens"]
    display(tabela_bpe.round(4))
    '''), "comparacao-pareada-bpe")


def original_code():
    md("""
    ### Implementação do ajuste no teste oficial

    A primeira função faz a busca de C na validação. A segunda executa o treino da
    CNN ou do BERT, guarda o checkpoint pelo F1 de validação e só depois avalia.
    A célula exibe as funções de produção que a etapa original executa.
    """)
    code('''
    mostrar_codigo("src/imdb_lab/baseline.py", "run")
    mostrar_codigo("src/imdb_lab/training.py", "run")
    ''', "treino-oficial-fonte", hidden=True)


def transfer_code():
    md("""
    ### Ajustar o BERT inteiro ou só a cabeça

    Nos ensaios internos, a média mascarada produz um vetor por texto. A cabeça
    aprende com esses vetores fixos ou com o gradiente passando pelos blocos.
    A função train_one da p. 4 é reutilizada nesses dois regimes.
    """)
    code(extract("src/imdb_lab/sequences.py", ["masked_mean", "MeanEncoder", "Probe"]), "cabeca-encoder")
    code('''
    mostrar_codigo("src/imdb_lab/training.py", "make_bert", "prepare_inputs")
    mostrar_codigo("src/imdb_lab/qwen_core.py", "classification_prompts", "forward_labels")
    ''', "classificacao-encoders-fonte", hidden=True)


def steering():
    md(f"""
    <a id="apoio"></a>
    ## Apoio · o mesmo enredo com três intervenções

    No [slide de apoio 18]({SLIDES}#slide-18) o Qwen recebe o mesmo enredo e o mesmo prompt.
    Reconstituímos a direção do bloco 28 com as duas médias de ativação do treino.
    A coluna REVIEW mostra as saídas salvas. O modo REPRODUZIR pode recapturar os vetores e gerar
    as três respostas. [Protocolo](
    https://github.com/costaleirbag/projeto-mae5920/blob/main/QWEN_EXPERIMENTO.md)
    """, "apoio")
    code('''
    estados = read_npz("derived/direcao_qwen.npz")
    diferenca = estados["media_1"] - estados["media_0"]
    escala = np.linalg.norm(diferenca)
    vetor = diferenca / escala
    np.testing.assert_allclose(vetor, estados["direction"], atol=1e-7, rtol=0)
    registrado = read_json("results/qwen_improved/selected_training_features.json")["directions"][0]
    conferir("Escala da direção do bloco 28", np.isclose(escala, registrado["natural_scale"]))
    exemplos = read_json("results/qwen_improved/presentation_examples.json")["cases"]
    sinal = next(caso for caso in exemplos if caso["id"] == "test_plot_000")
    conferir("As três gerações usam o mesmo prompt",
             len({r["prompt_sha256"] for r in sinal["outputs"]}) == 1)
    print(sinal["source"])
    for row in sorted(sinal["outputs"], key=lambda row: row["gain"]):
        review = row["generated"].partition("REVIEW:")[2].strip()
        print(f"α={row['gain']:+g}", review or row["generated"])
    ''', "direcao-e-tres-saidas")
    code('''
    mostrar_codigo("run_qwen_improve.py", "PackedSteering")
    ''', "hook-da-intervencao", hidden=True)


def reproduction():
    md("""
    <a id="reproduzir"></a>
    ## Reproduzir em uma pasta nova

    Escolha reproduzir na célula abaixo e rode desde o início. A ordem segue as dependências
    entre corpus, modelos e diagnósticos. Cada etapa imprime ou salva os resultados observados.
    A pasta recebe o código incluído no notebook. Os dados experimentais publicados são guardados
    em referencia/ e nunca são sobrescritos pelas novas execuções.

    Para o caminho completo, use Python 3.12 em um ambiente com CUDA e BF16, com memória
    para o Qwen3-8B e acesso às revisões públicas do IMDb e dos modelos. O ajuste dos encoders
    e a classificação com o Qwen podem levar mais de uma hora cada. O código e as configurações
    estão disponíveis pela função mostrar_codigo(arquivo). Os modelos podem mudar ligeiramente
    com hardware, bibliotecas e kernels diferentes. Os tempos são os da máquina que executou cada estudo.

    Dependências adicionais usadas nas execuções com GPU:
    ~~~
    %pip install "transformers==4.57.6" "tokenizers==0.22.1" "datasets==4.8.5" "accelerate>=1,<2" pyarrow joblib huggingface-hub
    ~~~
    Instale uma distribuição CUDA de PyTorch compatível com sua máquina. O estudo de encoders
    registrou Transformers 4.57.1 (anexo results/sequences/runtime.json), enquanto o BERT original
    e o Qwen usaram 4.57.6. O modo conferir usa os anexos e as dependências listadas na capa.
    Cada etapa concluída é registrada em logs/notebook. Para iniciar outra execução, escolha outra pasta.
    """, "reproduzir")
    code('''
    MODO = "conferir"  # Troque por "reproduzir" para refazer os experimentos.
    PASTA = Path("reproducao_mae5920")
    ETAPAS = (
        ("tools/seminar_retrain.py", "original"),
        ("tools/seminar_retrain.py", "encoders"),
        ("tools/seminar_retrain.py", "textcnn"),
        ("tools/seminar_retrain.py", "qwen"),
        ("tools/seminar_recompute.py", "original"),
        ("tools/seminar_recompute.py", "textcnn"),
        ("tools/seminar_recompute.py", "bert-cap"),
        ("tools/seminar_recompute.py", "qwen-youtube"),
        ("tools/seminar_steering.py",),
    )
    ''', "configuracao-execucao")
    code(r'''
    def preparar_execucao(pasta):
        pasta = pasta.resolve()
        marcador = pasta / ".mae5920_reproducao"
        if pasta.exists() and any(pasta.iterdir()):
            if not marcador.is_file() or marcador.read_text(encoding="utf-8") != meta["payload_sha256"]:
                raise ValueError("Escolha uma pasta nova ou de outra execução deste mesmo notebook.")
        pasta.mkdir(parents=True, exist_ok=True)
        marcador.write_text(meta["payload_sha256"], encoding="utf-8")
        fontes = set(meta["runtime_files"])
        for nome in ARQUIVOS.namelist():
            dados = ARQUIVOS.read(nome)
            assert hashlib.sha256(dados).hexdigest() == meta["assets"][nome]["sha256"], nome
            relativo = Path(nome)
            if relativo.is_absolute() or ".." in relativo.parts:
                raise ValueError(nome)
            destino = pasta / relativo if nome in fontes else pasta / "referencia" / relativo
            if not destino.resolve().is_relative_to(pasta):
                raise ValueError(f"Destino fora da execução: {destino}")
            if destino.exists() and destino.read_bytes() != dados:
                raise ValueError(f"Arquivo de origem mudou: {nome}")
            if not destino.exists():
                destino.parent.mkdir(parents=True, exist_ok=True)
                destino.write_bytes(dados)
        return pasta

    if MODO == "reproduzir":
        exige_gpu = {"encoders", "qwen", "bert-cap", "qwen-youtube", "tools/seminar_steering.py"}
        if any(etapa[-1] in exige_gpu for etapa in ETAPAS):
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("As etapas completas de BERT e Qwen exigem CUDA com BF16.")
        pasta = preparar_execucao(PASTA)
        env = dict(os.environ, PYTHONPATH=str(pasta / "src"), CUBLAS_WORKSPACE_CONFIG=":4096:8")
        logs = pasta / "logs/notebook"
        logs.mkdir(parents=True, exist_ok=True)
        for etapa in ETAPAS:
            chave = "-".join((Path(etapa[0]).stem, *etapa[1:]))
            feito = logs / (chave + ".concluido")
            if feito.exists():
                print("Já concluído:", chave)
                continue
            log = logs / (chave + ".log")
            print("Executando", chave, "· registro em", log.relative_to(pasta), flush=True)
            with log.open("w", encoding="utf-8") as log_file:
                processo = subprocess.run([sys.executable, *etapa], cwd=pasta, env=env,
                                           stdout=log_file, stderr=subprocess.STDOUT)
            print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-14:]))
            processo.check_returncode()
            feito.touch()
    elif MODO != "conferir":
        raise ValueError("Modo esperado: conferir ou reproduzir.")
    print(f"{len(CHECKS)} verificações no notebook passaram.")
    ''', "executar")


def build():
    cover()
    datasets(md, code)
    architecture()
    training()
    official(md, code)
    original_code()
    diagnostics(md, code)
    transfer_code()
    failures(md, code)
    external(md, code)
    steering()
    reproduction()
    for cell in CELLS:
        if cell.cell_type == "code":
            compile(cell.source, cell.id, "exec")
    payload, assets = snapshot()
    CELLS[0].attachments = {"evidencias.zip": {"application/zip": base64.b64encode(payload).decode()}}
    metadata = {"format": 2, "payload_sha256": sha(payload), "assets": assets,
                "runtime_files": sorted(set(RUNTIME)), "extracted_definitions": SOURCE_EXTRACTS}
    notebook = nbf.v4.new_notebook(cells=CELLS, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "mae5920_seminar": metadata,
        "language_info": {"name": "python", "file_extension": ".py"},
    })
    nbf.validate(notebook)
    nbf.write(notebook, OUTPUT)
    RESULT.mkdir(parents=True, exist_ok=True)
    record = {"notebook": OUTPUT.relative_to(ROOT).as_posix(), "builder_sha256": sha(Path(__file__).read_bytes()),
              "payload_sha256": sha(payload), "payload_bytes": len(payload), "cells": len(CELLS),
              "code_cells": sum(c.cell_type == "code" for c in CELLS), "execution": "pending"}
    (RESULT / "build_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    print(f"Notebook construído com {len(CELLS)} células e {len(payload) / 1e6:.2f} MB de anexos.")


if __name__ == "__main__":
    build()
