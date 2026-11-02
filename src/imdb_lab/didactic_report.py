"""Figures, an executed-reading notebook and an original course-linked supplement."""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

from .common import ROOT, read_json, save_json, file_sha256

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .didactic import DROOT, settings

FIG = DROOT / "figures"
PDF = ROOT / "output/pdf/relatorio_disciplina.pdf"
HTML = ROOT / "relatorio_disciplina.html"
NOTEBOOK = ROOT / "notebooks/conexoes_disciplina.ipynb"
COLORS = ["#167d92", "#df8b2f", "#7957a0", "#ce5351", "#4f9a65"]
OPT = {"sgd": "SGD", "momentum": "SGD + momentum", "adam": "Adam"}
ARCH = {"plain_1": "1 camada", "plain_4": "4 camadas", "plain_12": "12 camadas",
        "residual_12_raw": "12 + residual", "residual_12_scaled": "12 + residual escalado"}
REG = {"none": "Sem penalização", "dropout": "Dropout 0,5", "l2": "L2 explícita"}


def runs():
    return [read_json(p) for p in sorted((DROOT / "runs").glob("*.json"))]


def histories(records):
    return pd.DataFrame([{**row, "name": r["name"], "seed": r["seed"], "optimizer": r["optimizer"],
                          "lr": r["lr"], "group": r["group"]} for r in records for row in r["history"]])


def band(ax, rows, x, y, label, color, statistic="mean", linestyle="-"):
    grouped = rows.groupby(x)[y]
    if statistic == "median":
        center, low, high = grouped.median(), grouped.quantile(.25), grouped.quantile(.75)
    else:
        center, low, high = grouped.mean(), grouped.min(), grouped.max()
    xx = center.index.to_numpy()
    ax.plot(xx, center.to_numpy(), label=label, color=color, lw=2, ls=linestyle)
    ax.fill_between(xx, low.to_numpy(), high.to_numpy(), color=color, alpha=.12, lw=0)


def save_figure(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figures():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "axes.labelsize": 10.5, "legend.fontsize": 9.5,
                         "axes.grid": True, "grid.alpha": .18, "svg.fonttype": "none"})
    records = runs(); curve = histories(records); frozen = read_json(DROOT / "selection_frozen.json")
    quad = pd.read_csv(DROOT / "quadratic.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    gx, gy = np.meshgrid(np.linspace(-4.5, 4.5, 160), np.linspace(-3.6, 3.6, 160))
    ax[0].contour(gx, gy, (.5 * gx ** 2 + 20 * gy ** 2) / 2, levels=[.2, 1, 3, 10, 30, 60, 100], colors="#c6d7df", linewidths=.8)
    for i, name in enumerate(["gd_stable", "momentum", "adam", "gd_unstable"]):
        part = quad.loc[quad.condition == name]
        label = {"gd_stable": "GD, taxa 0,08", "momentum": "Momentum, taxa 0,03", "adam": "Adam, taxa 0,1", "gd_unstable": "GD, taxa 0,102"}[name]
        if name != "gd_unstable": ax[0].plot(part.x, part.y, color=COLORS[i], label=label, lw=1.3)
        ax[1].semilogy(part.step, part.loss, color=COLORS[i], label=label)
    ax[0].scatter([0], [0], color="#173040", marker="*", s=90)
    ax[0].set(xlabel="Parâmetro x", ylabel="Parâmetro y", title="Uma perda com duas escalas")
    ax[1].axhline(.01, color="#758591", lw=.7, ls=":")
    ax[1].set(xlabel="Passo", ylabel="Perda (log)", title="Estabilidade depende da taxa")
    ax[1].legend(frameon=False); save_figure(fig, "quadratic")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    part = curve.loc[curve.group == "optimizers"]
    for i, name in enumerate(OPT):
        common = part.loc[(part.optimizer == name) & (part.lr == settings()["optimizers"]["common_lr"])]
        chosen = part.loc[(part.optimizer == name) & (part.lr == frozen["optimizer_choices"][name]["lr"])]
        band(axes[0], common, "epoch", "validation_ce", OPT[name], COLORS[i])
        band(axes[1], chosen, "epoch", "validation_ce", f"{OPT[name]}: {frozen['optimizer_choices'][name]['lr']:g}", COLORS[i])
    for axis in axes:
        axis.set(xlabel="Época", ylabel="CE na validação"); axis.legend(frameon=False)
    axes[0].set_title("Mesma taxa: 0,003")
    axes[1].set_title("Taxa escolhida em validação")
    save_figure(fig, "optimizers")
    fig, ax = plt.subplots(figsize=(9.2, 3.7), constrained_layout=True)
    for i, name in enumerate(OPT):
        part = curve.loc[(curve.group == "optimizers") & (curve.optimizer == name) & (curve.epoch == settings()["optimizers"]["epochs"])]
        band(ax, part, "lr", "validation_ce", OPT[name], COLORS[i])
        for lr, rows in part.groupby("lr"):
            ax.scatter([lr] * len(rows), rows.validation_ce, s=15, color=COLORS[i], alpha=.55)
    ax.set(xscale="log", xlabel="Taxa de aprendizado", ylabel="CE final na validação", title="Três taxas por método, três sementes por taxa")
    ax.legend(frameon=False); save_figure(fig, "learning_rates")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    for i, (name, label) in enumerate(ARCH.items()):
        rows = curve.loc[(curve.group == "architectures") & (curve.name == name)]
        band(axes[0], rows, "epoch", "train_ce", label, COLORS[i])
        band(axes[1], rows, "epoch", "validation_ce", label, COLORS[i])
    axes[0].set(xlabel="Época", ylabel="CE de treino (log)", yscale="log", title="Capacidade de ajustar")
    axes[1].set(xlabel="Época", ylabel="CE na validação (log)", yscale="log", title="Generalizar é outra pergunta")
    axes[1].legend(frameon=False); save_figure(fig, "architectures")
    ini = pd.read_csv(DROOT / "initialization.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    for i, gain in enumerate([.1, 1., 3.]):
        rows = ini.loc[ini.gain == gain]
        label = "He" if gain == 1 else f"{gain:g} × escala He"
        band(axes[0], rows, "layer", "activation_rms", label, COLORS[i])
        band(axes[1], rows, "layer", "gradient_rms", label, COLORS[i])
    for axis in axes: axis.set(xlabel="Camada: entrada → saída", yscale="log"); axis.legend(frameon=False)
    axes[0].set(ylabel="RMS das ativações (log)", title="Propagação do sinal")
    axes[1].set(ylabel="RMS do gradiente (log)", title="Backward: da direita para a esquerda")
    save_figure(fig, "initialization")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    for i, (name, label) in enumerate(REG.items()):
        rows = curve.loc[(curve.group == "regularization") & (curve.name == name)]
        band(axes[0], rows, "epoch", "train_ce", label, COLORS[i])
        band(axes[1], rows, "epoch", "validation_ce", label, COLORS[i])
    stopping = np.median([r["virtual_stop_epoch"] for r in records if r["group"] == "regularization" and r["name"] == "none"])
    axes[1].axvline(stopping, color="#596c78", ls=":", label="Parada virtual: mediana, sem penalização")
    axes[0].set(xlabel="Época", ylabel="CE de treino (modo avaliação)", title="512 resenhas para ajustar os pesos")
    axes[1].set(xlabel="Época", ylabel="CE na validação", title="Overfitting e custo da regularização")
    axes[1].legend(frameon=False, fontsize=9); save_figure(fig, "regularization")
    dd = pd.read_csv(DROOT / "double_descent.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    for i, lam in enumerate(settings()["double_descent"]["lambdas"]):
        rows = dd.loc[dd["lambda"] == lam]
        label = "Sem ridge" if lam == 0 else "Ridge, lambda=0,001"
        band(axes[0], rows, "p_over_n", "train_mse", label, COLORS[i], "median")
        band(axes[1], rows, "p_over_n", "test_mse", label, COLORS[i], "median")
    for axis in axes:
        axis.set_xscale("log", base=2); axis.axvline(1, color="#596c78", ls=":")
        axis.set_xticks([1/16, 1/4, 1, 4, 8], ["1/16", "1/4", "1", "4", "8"])
        axis.set_xlabel("p / n: pesos ajustados / exemplos")
        axis.legend(frameon=False)
    axes[0].set(ylabel="MSE de treino", title="Interpolação: o erro chega a zero")
    axes[1].set(yscale="log", ylabel="MSE em novas entradas (log)", title="O pico e o segundo descenso")
    axes[1].axhline(settings()["double_descent"]["noise_sd"] ** 2, color="#758591", ls="--", lw=.9)
    save_figure(fig, "double_descent")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    for i, lam in enumerate(settings()["double_descent"]["lambdas"]):
        rows = dd.loc[dd["lambda"] == lam]; label = "Sem ridge" if lam == 0 else "Ridge, lambda=0,001"
        band(axes[0], rows, "p_over_n", "coefficient_norm", label, COLORS[i], "median")
        band(axes[1], rows, "p_over_n", "effective_df", label, COLORS[i], "median")
    for axis in axes:
        axis.set_xscale("log", base=2); axis.axvline(1, color="#596c78", ls=":"); axis.set_xlabel("p / n"); axis.legend(frameon=False)
    axes[0].set(yscale="log", ylabel="Norma dos coeficientes (log)", title="Instabilidade perto da interpolação")
    axes[1].set(ylabel="Graus de liberdade efetivos", title="Capacidade nominal e efetiva")
    save_figure(fig, "double_descent_mechanism")


def summary():
    audit = pd.read_csv(DROOT / "audit_metrics.csv"); dd = pd.read_csv(DROOT / "double_descent.csv")
    stats = []
    for key, rows in audit.groupby(["group", "name", "checkpoint"], sort=False):
        stats.append({"group": key[0], "name": key[1], "checkpoint": key[2], "parameters": int(rows.parameters.iloc[0]),
                      "accuracy_mean": float(rows.accuracy.mean()), "accuracy_min": float(rows.accuracy.min()),
                      "accuracy_max": float(rows.accuracy.max()), "ce_mean": float(rows.ce.mean())})
    medians = dd.groupby(["lambda", "p"])[["train_mse", "test_mse", "coefficient_norm"]].median()
    records = runs()
    out = {"internal_audit": stats, "training_runs": len(records),
           "training_seconds_sum": sum(r["training_seconds"] for r in records),
           "optimizer_choices": read_json(DROOT / "selection_frozen.json")["optimizer_choices"],
           "double_descent": {str(p): {"train_mse_median": float(medians.loc[0., p].train_mse),
                                       "test_mse_median": float(medians.loc[0., p].test_mse)} for p in [24, 64, 512]},
           "ridge_at_interpolation_median_mse": float(medians.loc[.001, 64].test_mse),
           "uncertainty": "IMDb: means and min-max across three fixed initialization seeds; synthetic: median and IQR across twenty seeds. Neither is a population confidence interval."}
    save_json(DROOT / "summary.json", out)
    return out


def create_notebook(s):
    import nbformat as nb
    cells = []
    def md(text): cells.append(nb.v4.new_markdown_cell(text))
    def code(text): cells.append(nb.v4.new_code_cell(text))
    md("# Do modelo aos mecanismos: conexões com a disciplina\n\n"
       "Nestor, Letícia e Gabriel - MAE5920. Este notebook lê execuções concluídas e confere os artefatos, sem repetir os treinos. "
       "As novas figuras são originais. Fontes: Prince, capítulos 6-9 e 11, e os slides correspondentes disponíveis ao grupo. "
       "O material do curso sustenta as conexões; os exemplos abaixo são elaboração didática própria.\n\n"
       "**Roteiro:** otimização → profundidade e inicialização → regularização → double descent. "
       "Os ensaios sintéticos e a classificação IMDb aparecem identificados separadamente.")
    code("from pathlib import Path\nimport json\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import display, Image\n"
         "ROOT=next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'didactic_config.json').exists())\nQ=ROOT/'results/didactic'\n"
         "read=lambda p: json.loads((ROOT/p).read_text())\n"
         "def plot(name): display(Image(filename=str(Q/'figures'/f'{name}.png')))\n"
         "cfg=read('didactic_config.json')\nfrozen=read('results/didactic/selection_frozen.json')\n"
         "records=[json.loads(p.read_text()) for p in sorted((Q/'runs').glob('*.json'))]\n"
         "audit=pd.read_csv(Q/'audit_metrics.csv')\nprint(len(records),'treinos; ',len(audit),'checkpoints na auditoria interna')")
    md("## Dados e regras antes da comparação\n\n"
       "Somente o antigo treino limpo do IMDb é repartido: 6.144 para ajuste, 1.536 para validação e 2.048 para auditoria interna. "
       "Não consultamos novamente o teste oficial. Vocabulário TF-IDF, IDF, SVD de 128 componentes e padronização usam somente os 6.144 exemplos de treino. "
       "As redes densas recebem a mesma representação. Este não é um novo leaderboard CNN/BERT/Qwen.\n\n"
       "As sementes 17, 29 e 43 são fixas. Cada comparação de otimizadores compartilha pesos iniciais e ordem de minibatches. "
       "A auditoria foi transferida para o ambiente de execução apenas após o congelamento; uma correção técnica posterior de precisão, sem retuning, está documentada.")
    code("manifest=read('results/didactic/data_manifest.json')\n"
         "display(pd.DataFrame([{'papel':k,'n':v['n'],'negativos':v['label_counts']['0'],'positivos':v['label_counts']['1']} for k,v in manifest['roles'].items()]))\n"
         "for a,aa in manifest['roles'].items():\n    for b,bb in manifest['roles'].items():\n        if a!=b: assert set(aa['text_sha256']).isdisjoint(bb['text_sha256'])\n"
         "assert manifest['official_test_opened'] is False")
    md("## 1. Otimização: taxa, memória e adaptação\n\n"
       "**Prince 6.1-6.5; slides Ajuste de Modelos, páginas do arquivo 26-39.** "
       "No exemplo quadrático, os autovalores da Hessiana são 0,5 e 20: GD é estável para taxa entre 0 e 0,1. "
       "O painel é determinístico, sem minibatches; suas taxas ilustram trajetórias e não constituem um ranking justo.")
    code("plot('quadratic')\nq=pd.read_csv(Q/'quadratic.csv')\ndisplay(q.loc[q.step==q.step.max(),['condition','lr','loss']])")
    md("### A mesma pergunta em resenhas reais\n\n"
       "MLP com duas camadas ocultas de 64 unidades, sem dropout/L2; 32 épocas. "
       "Primeiro, todos usam taxa 0,003. Depois, cada método recebe três candidatas, com a escolha pela menor CE final média de validação nas três sementes. "
       "Comparar somente uma taxa comum mistura o algoritmo com sua adequação à escala do problema. "
       "O momentum do PyTorch usa v ← 0,9v + g; a convenção difere da média móvel normalizada por um fator constante na taxa.")
    code("plot('optimizers')\ndisplay(pd.DataFrame({k:{'taxa escolhida':v['lr'],**v['mean_final_validation_ce_by_lr']} for k,v in frozen['optimizer_choices'].items()}).T)")
    code("plot('learning_rates')\n"
         "for seed in cfg['seeds']:\n    rr=[r for r in records if r['group']=='optimizers' and r['seed']==seed]\n    assert len({r['initial_state_sha256'] for r in rr})==1\n    assert len({r['minibatch_order_sha256'] for r in rr})==1\n"
         "print('Partidas e minibatches idênticos por semente confirmados.')")
    md("## 2. Mais camadas, atalhos e escala\n\n"
       "**Prince 11.1-11.3; slides Redes Residuais, páginas 5-18.** "
       "Comparamos 1, 4 e 12 camadas ocultas e duas variantes de 12 camadas com atalho. "
       "As três redes de 12 camadas têm exatamente os mesmos parâmetros e pesos iniciais. "
       "Sem atalho: h ← f(ReLU(h)); com atalho: h ← h + s f(ReLU(h)). "
       "A comparação com s=1 isola a soma identidade. Em seguida, s=1/√11 controla a escala do ramo residual. "
       "Não há BatchNorm/LayerNorm nessas ablações. Todas usam a taxa Adam já escolhida, sem retuning por arquitetura.")
    code("plot('architectures')\ndisplay(pd.DataFrame([{'arquitetura':r['name'],'parâmetros':r['parameters'],'CE treino final':r['history'][-1]['train_ce'],'CE validação final':r['history'][-1]['validation_ce']} for r in records if r['group']=='architectures']).groupby('arquitetura').mean())")
    md("## 3. A inicialização deixa o sinal sobreviver?\n\n"
       "**Prince 7.5; slides Gradientes e Inicialização, páginas 31-42.** "
       "Diagnóstico sintético, sem treino: 32 camadas ReLU de largura 64, entradas gaussianas e uma sonda linear fixa. "
       "A variância dos pesos é g²·2/64, com g em {0,1; 1; 3}. Para cada semente, reutilizamos os mesmos pesos gaussianos de base e o mesmo gradiente na saída. "
       "Os eixos são logarítmicos. A retropropagação percorre as camadas da direita para a esquerda. "
       "He é uma aproximação de preservação de escala, não a promessa de variância exatamente constante numa rede finita.")
    code("plot('initialization')\nini=pd.read_csv(Q/'initialization.csv')\ndisplay(ini.loc[ini.layer.isin([0,32])].groupby(['gain','layer'])[['activation_rms','gradient_rms']].mean())")
    md("## 4. Ajustar melhor o treino pode piorar a validação\n\n"
       "**Prince 9.1 e 9.3; slides Regularização, páginas 7-17 e 29-36.** "
       "Agora só 512 exemplos recebem supervisão: MLP de três camadas de 128 unidades, 120 épocas e a taxa Adam já congelada. "
       "Comparamos nenhuma penalização, dropout 0,5 e L2 explícita. A representação sem rótulos continua sendo aquela ajustada nos 6.144 exemplos do treino comum. "
       "L2 entra como lambda/2 vezes a soma dos pesos ao quadrado, excluindo vieses; Adam usa weight_decay=0. "
       "As curvas de CE são medidas em modo avaliação, com dropout desligado, e não incluem a penalização L2.\n\n"
       "A parada usa paciência 15 e melhora mínima de 0,0001 na validação. Continuamos o treino para visualizar a trajetória contrafactual, preservando o checkpoint escolhido até a parada. "
       "Isso não mede economia real de tempo. A escolha por CE pode reduzir a confiança errada sem maximizar a acurácia; dropout também pode subajustar com este orçamento.")
    code("plot('regularization')\n"
         "display(pd.DataFrame([{'variante':r['name'],'semente':r['seed'],'época selecionada':r['early_epoch'],'parada virtual':r['virtual_stop_epoch']} for r in records if r['group']=='regularization']))\n"
         "display(audit.loc[audit.group=='regularization'].groupby(['name','checkpoint'])[['ce','accuracy']].mean())")
    md("## 5. Double descent em um caso controlado\n\n"
       "**Prince 8.4-8.5; slides Desempenho e Generalização, páginas 30-37; Belkin et al.** "
       "Aqui mudamos explicitamente para regressão sintética: 64 exemplos de treino, 2.048 de teste, ruído de desvio 0,3 e 20 sementes fixadas. "
       "Uma camada ReLU aleatória fica congelada; ajustamos só p coeficientes de saída por mínimos quadrados de norma mínima ou ridge. "
       "As larguras são aninhadas dentro de cada semente. O limiar p/n=1 conta apenas os coeficientes ajustados, não todos os pesos congelados. "
       "Os gráficos mostram mediana e intervalo interquartil; todos os casos e os valores extremos ficam no CSV. "
       "Este experimento não estabelece double descent na CNN do IMDb e não é uma reprodução integral da rede treinada no livro.")
    code("plot('double_descent')\ndd=pd.read_csv(Q/'double_descent.csv')\n"
         "display(dd.loc[(dd['lambda']==0)&dd.p.isin([24,64,512])].groupby('p')[['train_mse','test_mse','coefficient_norm']].median())\n"
         "assert len(dd)==880 and set(dd.seed)==set(cfg['double_descent']['seeds'])")
    code("plot('double_descent_mechanism')\n"
         "print('Noise variance:',cfg['double_descent']['noise_sd']**2)\n"
         "display(dd.loc[dd.p==64].groupby('lambda')[['test_mse','effective_df']].median())")
    md("### Reconstruir um interpolador real, sem escolher uma semente favorável\n\n"
       "Usamos a primeira semente do protocolo e p=n. O verificador externo repete a conferência para os 880 casos. "
       "O erro quase nulo de treino e o erro em novas entradas respondem a perguntas diferentes.")
    code("import sys\nsys.path.insert(0,str(ROOT/'src'))\nfrom imdb_lab.didactic_toys import synthetic_case\n"
         "spec=cfg['double_descent'];seed=spec['seeds'][0];p=spec['train_n']\n"
         "x,xt,y,yt,f,ft,w=synthetic_case(seed,spec)\n"
         "with np.load(Q/'double_descent_weights.npz') as a: beta=a[f'seed{seed}_p{p}_lambda0']\n"
         "prediction=np.maximum(x@w[:,:p],0)@beta/np.sqrt(p)\n"
         "np.testing.assert_allclose(prediction,y,atol=1e-8)\nprint('MSE de treino reconstruído:',float(np.mean((prediction-y)**2)))")
    md("## Auditoria interna e limites\n\n"
       "Os 42 checkpoints foram avaliados depois de congelar as escolhas. "
       "As taxas dos otimizadores ficaram próximas em acurácia; profundidade adicional ajustou mais o treino e também elevou a CE de validação. "
       "A rede residual escalada melhorou em relação ao residual sem escala neste protocolo, mas a rede rasa continuou competitiva. "
       "Na regularização, parar por CE reduziu CE sem necessariamente elevar acurácia. "
       "São observações condicionais a esta representação, orçamento e três sementes, sem inferência de superioridade universal. "
       "As faixas mín.-máx. de três sementes não são intervalos de confiança populacionais.")
    code("display(audit.groupby(['group','name','checkpoint'])[['accuracy','ce']].agg(['mean','min','max']))\n"
         "evidence=read('results/didactic/verification.json')\nassert evidence['status']=='passed'\ndisplay(evidence['evidence'])\n"
         "correction_path=Q/'numerical_audit_correction.json'\n"
         "if correction_path.exists(): display(json.loads(correction_path.read_text()))")
    md("## Fontes e apresentação\n\n"
       "As referências e os localizadores estão abaixo. Os PDFs dos slides/livro permanecem no acervo do grupo, sem redistribuição no repositório. "
       "As figuras PNG e SVG em `results/didactic/figures` podem ser usadas diretamente na apresentação. "
       "Para uma fala de 8-10 minutos, escolha um gráfico de otimização, um de profundidade/inicialização, um de overfitting e o de double descent. "
       "O experimento de steering do Qwen permanece no notebook próprio e funciona como outra forma de intervenção em representações internas.")
    code("display(pd.DataFrame(read('results/didactic/source_references.json')['references'])[['id','title','locators','url']].fillna(''))\n"
         "print('Reproduzir a verificação: python verify_didactic.py')\nprint('Treino não repetido por este notebook.')")
    notebook = nb.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    nb.write(notebook, NOTEBOOK)


def create_html(s):
    titles = [("quadratic", "A geometria da otimização", "Prince 6.1-6.5: exemplo quadrático determinístico; não é um ranking de otimizadores."),
              ("optimizers", "Mesma taxa e taxas escolhidas", "IMDb: mesma inicialização e minibatches; média e faixa mín.-máx. de três sementes."),
              ("learning_rates", "A taxa muda a comparação", "Três candidatas por método, avaliadas pelo mesmo critério de validação."),
              ("architectures", "Profundidade, atalho e escala", "IMDb: três ablações de 12 camadas têm 54.146 parâmetros e pesos iniciais iguais."),
              ("initialization", "Propagação e retropropagação", "Diagnóstico sintético de inicialização: 32 camadas ReLU, sem treinamento."),
              ("regularization", "Overfitting e regularização", "512 resenhas supervisionadas. A linha de parada é contrafactual; o treino foi continuado para exibir a curva."),
              ("double_descent", "Double descent controlado", "Regressão sintética, camada aleatória congelada, só a saída ajustada. Mediana e IQR de 20 sementes."),
              ("double_descent_mechanism", "O mecanismo perto da interpolação", "Coeficientes grandes e graus de liberdade efetivos; não é evidência de double descent no IMDb.")]
    parts = ["<!doctype html><html lang='pt-BR'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>MAE5920 - Conexões com a disciplina</title>",
             "<style>body{font:17px/1.6 system-ui,sans-serif;color:#20333e;background:#f5f8f9;margin:0}main{max-width:1040px;margin:40px auto;padding:30px;background:white;border-radius:18px}h1,h2{color:#165d70}img{width:100%;height:auto}section{margin:44px 0}.note{padding:18px;background:#edf6f6;border-left:4px solid #167d92}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #d5e1e7;text-align:left}a{color:#167d92}</style><main>",
             "<h1>Do modelo aos mecanismos</h1><p>Nestor, Letícia e Gabriel - MAE5920</p>",
             "<p class='note'>Extensão didática: otimização, profundidade, inicialização, regularização e double descent. Os dados IMDb e os exemplos sintéticos estão identificados separadamente. Todas as figuras foram geradas por este experimento.</p>",
             "<p>Treino/validação/auditoria interna: 6.144 / 1.536 / 2.048 resenhas do antigo treino limpo. A representação TF-IDF/SVD usa somente o novo treino. O teste oficial não foi consultado nesta extensão. Três sementes nos modelos IMDb e vinte no exemplo de double descent.</p>"]
    for name, title, caption in titles:
        image = base64.b64encode((FIG / f"{name}.png").read_bytes()).decode()
        parts.append(f"<section><h2>{html.escape(title)}</h2><img alt='{html.escape(title)}' src='data:image/png;base64,{image}'><p>{html.escape(caption)}</p></section>")
    parts.append("<h2>Auditoria interna</h2>" + pd.DataFrame(s["internal_audit"])[["group", "name", "checkpoint", "accuracy_mean", "ce_mean"]].to_html(index=False))
    parts.append("<p>Early stopping seleciona pela CE de validação, não pela acurácia da auditoria. Dropout pode subajustar. Uma rede mais profunda pode ajustar o treino e generalizar pior. A auditoria é interna e as faixas de sementes não são intervalos de confiança populacionais. A correção da política de precisão está documentada nos artefatos.</p><h2>Fontes</h2><ul>")
    for ref in read_json(DROOT / "source_references.json")["references"]:
        title = html.escape(ref["title"]); loc = html.escape(ref.get("locators", ""))
        link = f"<a href='{html.escape(ref['url'])}'>{title}</a>" if ref.get("url") else title
        parts.append(f"<li>{link}. {loc}</li>")
    parts.append("</ul><p>PDFs das fontes não redistribuídos. Para reconstruir as verificações: <code>python verify_didactic.py</code>.</p></main></html>")
    HTML.write_text("".join(parts))


def create_pdf(s):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from matplotlib import font_manager
    base = Path(font_manager.findfont("DejaVu Sans")).parent
    pdfmetrics.registerFont(TTFont("UDLBody", str(base / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("UDLBold", str(base / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("UDLBody", normal="UDLBody", bold="UDLBold")
    width, height = A4; usable = width - 84
    body = ParagraphStyle("Body", fontName="UDLBody", fontSize=9.1, leading=13.2, textColor=colors.HexColor("#263c47"), spaceAfter=8)
    title = ParagraphStyle("Title", parent=body, fontName="UDLBold", fontSize=22, leading=27, textColor=colors.HexColor("#165d70"), spaceAfter=14)
    sub = ParagraphStyle("Sub", parent=body, fontName="UDLBold", fontSize=12, leading=16, spaceBefore=8, spaceAfter=8)
    small = ParagraphStyle("Small", parent=body, fontSize=7.8, leading=11, textColor=colors.HexColor("#526a77"))
    story = []
    def p(text, style=body): story.append(Paragraph(text, style))
    def page(label, heading):
        if story: story.append(PageBreak())
        p(label.upper(), small); p(heading, title)
    def pic(name, max_height=245):
        from PIL import Image as PILImage
        with PILImage.open(FIG / f"{name}.png") as image: w, h = image.size
        scale = min(usable / w, max_height / h)
        story.append(Image(str(FIG / f"{name}.png"), width=w * scale, height=h * scale)); story.append(Spacer(1, 6))
    def table(headers, rows, widths=None):
        data = [[Paragraph(str(x), small) for x in headers]] + [[Paragraph(str(x), body) for x in row] for row in rows]
        t = Table(data, colWidths=widths or [usable / len(headers)] * len(headers), repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf3f5")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LINEBELOW", (0, 0), (-1, 0), .8, colors.HexColor("#a9c6d0")),
                               ("LINEBELOW", (0, 1), (-1, -1), .3, colors.HexColor("#dbe5e9")),
                               ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                               ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        story.append(t); story.append(Spacer(1, 10))
    def entry(group, name, checkpoint="final"):
        return next(r for r in s["internal_audit"] if r["group"] == group and r["name"] == name and r["checkpoint"] == checkpoint)
    def pct(v): return f"{100 * v:.2f}%".replace(".", ",")
    page("MAE5920 / Suplemento didático", "Do modelo aos mecanismos")
    p("<b>Nestor, Letícia e Gabriel</b><br/>Introdução a Deep Learning - extensão do projeto IMDb e Qwen3-8B.")
    p("Além de treinar classificadores, podemos perguntar por que uma configuração aprende, como uma alteração muda a trajetória e quando ajustar melhor o treino prejudica a generalização. Este suplemento transforma essas perguntas em experimentos ligados ao livro e aos slides disponíveis no acervo do grupo.")
    table(["Pergunta para a apresentação", "Experimento", "Referência Prince"], [
        ["Adam é sempre melhor que SGD?", "Mesma taxa e taxas escolhidas em validação", "6.1-6.5"],
        ["O que acontece com mais camadas?", "Profundidade, atalho e escala residual", "7.5; 11.1-11.3"],
        ["Quando o treino começa a atrapalhar?", "Dropout, L2 e parada por validação", "9.1; 9.3"],
        ["Mais capacidade pode voltar a ajudar?", "Double descent em regressão sintética", "8.4-8.5"]], [usable*.38, usable*.42, usable*.20])
    p("O desenho", sub)
    p("IMDb: 6.144 exemplos para ajuste, 1.536 para validação e 2.048 para auditoria interna, todos extraídos do antigo treino limpo. Vocabulário TF-IDF, IDF, SVD de 128 componentes e padronização são ajustados somente nos 6.144 exemplos. As novas redes densas compartilham essas entradas. O teste oficial do projeto não foi consultado novamente.")
    p("<b>51 treinos e três sementes:</b> 27 trajetórias para otimizadores, 15 para arquiteturas e nove para regularização. As escolhas e os checkpoints são congelados antes da auditoria. O exemplo sintético usa outras vinte sementes e 880 condições. Não há promessa de que o mesmo fenômeno apareça na CNN ou no Qwen.")
    p("As faixas dos gráficos IMDb são mínimo e máximo das três sementes; no exemplo sintético, mediana e intervalo interquartil. Essas faixas não são intervalos de confiança populacionais.", small)
    page("1 / Otimização", "O algoritmo e a taxa interagem")
    p("Uma MLP de duas camadas ocultas de 64 unidades recebe os mesmos dados, pesos iniciais e minibatches por semente. O orçamento é de 32 épocas. Primeiro fixamos a taxa em 0,003 para todos; depois permitimos três candidatas por método e selecionamos pela CE final média na validação.")
    pic("optimizers", 220)
    table(["Método", "Taxa escolhida", "Acurácia na auditoria"], [[OPT[name], f"{v['lr']:g}", pct(entry("optimizers", f"{name}_lr{v['lr']:g}")["accuracy_mean"])] for name, v in s["optimizer_choices"].items()])
    p("As três escolhas ficam próximas em acurácia. O resultado depende da taxa, do orçamento e do critério: uma taxa que reduz muito a perda de treino pode elevar a CE de validação por excesso de confiança. O ensaio não estabelece superioridade universal de um otimizador.")
    p("<b>Momentum:</b> a implementação PyTorch usa v = 0,9v + g e passo proporcional a v. A convenção de média móvel normalizada usada em algumas derivações difere por um fator constante na taxa.")
    p("O notebook também mostra a perda quadrática com autovalores 0,5 e 20. Nela, GD é estável somente com taxa entre 0 e 0,1; a trajetória acima desse limite é preservada. É um exemplo determinístico de geometria, separado do SGD em minibatches.")
    p("Fontes: Prince 6.1-6.5; slides Ajuste de Modelos, páginas do arquivo 26-39; Kingma e Ba (Adam).", small)
    page("2 / Arquitetura", "Profundidade, identidade e escala")
    pic("architectures", 225)
    table(["Arquitetura", "Parâmetros", "Acurácia interna"], [[label, f"{entry('architectures', name)['parameters']:,}".replace(",", "."), pct(entry("architectures", name)["accuracy_mean"])] for name, label in ARCH.items()], [usable*.48, usable*.24, usable*.28])
    p("As três redes com 12 camadas possuem 54.146 parâmetros, os mesmos pesos iniciais e os mesmos minibatches. O atalho altera h = f(ReLU(h)) para h = h + s f(ReLU(h)). Com s=1, isolamos a soma identidade; com s=1/raiz(11), estudamos também o controle da escala residual. Não acrescentamos normalização.")
    p("A maior profundidade ajustou mais o treino, mas também elevou a CE de validação. Portanto, o efeito dominante observado aqui inclui overfitting; não podemos chamar esse resultado simplesmente de falha em otimizar. A variante residual escalada melhora em relação ao residual sem escala, mas não supera automaticamente a rede rasa.")
    p("A taxa Adam é a já escolhida no experimento anterior, sem retuning por arquitetura. Trata-se de uma ablação sob orçamento fixo, não do melhor desempenho possível de cada arquitetura.", small)
    p("Fontes: Prince 11.1-11.3; slides Redes Residuais, páginas do arquivo 5-18.", small)
    page("3 / Inicialização", "O sinal precisa sobreviver à profundidade")
    pic("initialization", 235)
    p("Este é um diagnóstico sintético sem treinamento: 32 camadas ReLU de largura 64, entradas gaussianas e uma sonda linear fixa na saída. Para cada semente, os três ganhos reutilizam as mesmas matrizes gaussianas de base. A variância dos pesos é g² vezes 2/64; g=1 corresponde à escala He.")
    p("Com ganho menor, as ativações se atenuam ao atravessar as camadas; com ganho maior, amplificam-se. No backward, a leitura segue da direita para a esquerda. Os eixos logarítmicos tornam visíveis as diferenças de muitas ordens de magnitude.")
    p("O verificador confere a homogeneidade positiva da ReLU neste caso sem vieses: na camada k, o sinal escala como g elevado a k e o gradiente como g elevado a (32-k). Essa identidade é específica do diagnóstico construído; não descreve toda a dinâmica de uma rede treinada.")
    p("He busca preservar a escala sob hipóteses de independência e simetria. Numa realização finita, a variância pode oscilar; a linha central não precisa ser perfeitamente horizontal. Conexões residuais também exigem atenção à escala da soma.")
    p("Fontes: Prince 7.5; slides Gradientes e Inicialização, páginas do arquivo 31-42; conexão com Prince 11.3.", small)
    page("4 / Regularização", "Treinar mais pode piorar a confiança")
    pic("regularization", 225)
    p("Reduzimos a supervisão para 512 exemplos e usamos uma MLP de três camadas de 128 unidades por 120 épocas. A representação continua sendo aquela aprendida, sem rótulos, nos 6.144 exemplos comuns. Comparamos ausência de penalização, dropout 0,5 e L2 explícita. As curvas medem CE em modo avaliação, com dropout desligado, e excluem a penalização.")
    table(["Variante", "CE: parada", "CE: final", "Acurácia: parada / final"], [[label, f"{entry('regularization',name,'early')['ce_mean']:.3f}", f"{entry('regularization',name)['ce_mean']:.3f}", pct(entry('regularization',name,'early')['accuracy_mean'])+" / "+pct(entry('regularization',name)['accuracy_mean'])] for name,label in REG.items()], [usable*.29, usable*.17, usable*.17, usable*.37])
    p("A parada virtual usa paciência 15 e melhora mínima de 0,0001 na CE de validação. O checkpoint é retido quando essa regra para de observar melhora; o treino continua apenas para revelar a trajetória contrafactual. Não medimos economia real de tempo.")
    p("Parar pela CE reduz a CE interna, mas pode reduzir a acurácia: perda probabilística e acerto no limiar respondem a objetivos diferentes. Dropout subajusta neste orçamento e L2 não garante melhoria em toda métrica. A penalização é explícita, lambda/2 vezes a soma dos pesos ao quadrado, excluindo vieses; Adam usa weight_decay=0.")
    p("Fontes: Prince 9.1 e 9.3; slides Regularização, páginas do arquivo 7-17 e 29-36.", small)
    page("5 / Generalização", "Double descent em um caso controlado")
    pic("double_descent", 245)
    p("Mudamos explicitamente para regressão sintética: n=64 exemplos de treino, 2.048 novas entradas e ruído gaussiano de desvio 0,3. Uma camada ReLU aleatória fica congelada. Ajustamos somente p coeficientes de saída por mínimos quadrados de norma mínima ou ridge. O eixo p/n conta esses coeficientes ajustados, não os pesos congelados.")
    table(["p", "MSE de treino: mediana", "MSE de teste: mediana"], [[str(p0), f"{s['double_descent'][str(p0)]['train_mse_median']:.3g}", f"{s['double_descent'][str(p0)]['test_mse_median']:.3f}"] for p0 in [24,64,512]])
    p(f"Na interpolação, a mediana do MSE de teste sem ridge é {s['double_descent']['64']['test_mse_median']:.2f}; com lambda=0,001, {s['ridge_at_interpolation_median_mse']:.3f}. Ao aumentar a largura para 512, a curva sem ridge desce novamente. O notebook liga o pico ao aumento da norma dos coeficientes e mostra os graus de liberdade efetivos.")
    p("Todas as vinte sementes e as 880 condições são preservadas. Mediana e intervalo interquartil são usados porque os erros perto da interpolação têm caudas fortes. O experimento é uma demonstração com características aleatórias congeladas; não estabelece double descent na CNN do IMDb e não reproduz integralmente a rede treinada do livro.")
    p("Fontes: Prince 8.4-8.5; slides Desempenho e Generalização, páginas do arquivo 30-37; Belkin et al. (2019).", small)
    page("6 / Como apresentar", "Quatro perguntas para uma fala de 8-10 minutos")
    table(["Tempo sugerido", "Gráfico e pergunta", "Conclusão que os dados permitem"], [
        ["2 minutos", "Otimização: e se mantivermos a mesma taxa?", "A taxa integra a comparação; taxas escolhidas tornam os métodos próximos neste ensaio."],
        ["2 minutos", "Arquitetura e inicialização: mais camadas ajudam?", "Capacidade, overfitting e escala têm papéis distintos; o atalho não garante o melhor resultado."],
        ["2 minutos", "Regularização: quando parar?", "CE e acurácia podem discordar; regularização também pode subajustar."],
        ["2 minutos", "Double descent: mais parâmetros voltam a ajudar?", "O fenômeno aparece no exemplo controlado; o mecanismo e seus limites ficam explícitos."]], [usable*.18, usable*.35, usable*.47])
    p("Ponte com o restante do projeto", sub)
    p("O relatório principal continua mostrando TF-IDF, CNN e BERT. O suplemento usa redes densas com uma representação comum para isolar alterações menores. O Qwen3-8B e o steering continuam no relatório próprio: eles ilustram uma intervenção nas ativações de um modelo congelado, distinta de trocar o otimizador ou a arquitetura no treinamento.")
    p("O que pode ser compartilhado", sub)
    p("Este PDF, o notebook conexoes_disciplina.ipynb e os gráficos PNG/SVG em results/didactic/figures. O HTML correspondente é autocontido. Código, dados vetoriais selecionados, trajetórias, pesos pequenos e verificadores permitem conferir os resultados sem carregar o Qwen.")
    p("Evite transformar a fala numa competição de métricas. A contribuição didática é formular uma hipótese, controlar o que muda e observar onde a explicação encontra limites. As referências localizam cada tema no material disponível ao grupo.")
    page("7 / Fontes e reprodutibilidade", "Evidência, limites e referências")
    p("Verificação", sub)
    p("Os testes cobrem partições disjuntas, ablações com parâmetros iguais, L2 sem vieses, regra causal de parada, minibatches pareados e solução ridge/norma mínima. O verificador científico reconstrói probabilidades dos 42 checkpoints, confere as 51 trajetórias e repete os 880 cálculos sintéticos.")
    if (DROOT / "numerical_audit_correction.json").exists():
        p("Uma verificação independente detectou inicialmente uma diferença entre a precisão do treino e a da avaliação na GPU. A política foi uniformizada com TF32 desligado; a execução foi repetida com as mesmas configurações e sementes. A correção e a comparação dos parâmetros repetidos estão documentadas em numerical_audit_correction.json. Não houve retuning por causa da auditoria.")
    else:
        p("Treino e auditoria usam float32 com TF32 desligado. O verificador reconstrói os resultados em CPU a partir dos pesos salvos; novas execuções mantêm os papéis separados de treino, validação e auditoria.")
    p("Limites", sub)
    p("As redes IMDb usam TF-IDF/SVD, não embeddings de um LLM, e compartilham uma única divisão interna. Três sementes de inicialização não abrangem toda a incerteza amostral. As taxas de arquitetura e regularização não foram ajustadas separadamente. O diagnóstico de inicialização e o double descent são sintéticos. A disponibilidade dos slides não comprova que cada página foi ministrada.")
    p("Referências", sub)
    p("Prince, Simon J. D. <i>Understanding Deep Learning</i>. Versão local com 541 páginas: seções 6.1-6.5, 7.5, 8.4-8.5, 9.1, 9.3 e 11.1-11.3. Site: udlbook.github.io/udlbook/.")
    p("Slides MAE5920 do acervo fornecido: <i>Ajuste de Modelos</i> (pp. 26-39); <i>Gradientes e Inicialização</i> (pp. 31-42); <i>Desempenho e Generalização</i> (pp. 30-37); <i>Regularização</i> (pp. 7-17 e 29-36); <i>Redes Residuais</i> (pp. 5-18). Localizadores são páginas do arquivo PDF.")
    p("Kingma, D. P.; Ba, J. <i>Adam: A Method for Stochastic Optimization</i>. arXiv:1412.6980. Belkin, M. et al. <i>Reconciling modern machine learning practice and the bias-variance trade-off</i>. PNAS, 2019; arXiv:1812.11118.")
    p("Os hashes das fontes estão em source_references.json. Todas as figuras deste suplemento foram produzidas pelos experimentos; páginas e figuras dos PDFs de referência não são redistribuídas. Para conferir: <b>python verify_didactic.py</b>.", small)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    def decorate(canvas, doc):
        canvas.saveState(); canvas.setStrokeColor(colors.HexColor("#167d92")); canvas.setLineWidth(2); canvas.line(42, height-31, width-42, height-31)
        canvas.setFont("UDLBody", 7.5); canvas.setFillColor(colors.HexColor("#607985")); canvas.drawString(42, 25, "MAE5920 - Conexões com a disciplina")
        canvas.drawRightString(width-42, 25, str(doc.page)); canvas.restoreState()
    document = SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=47, bottomMargin=43,
                                 title="MAE5920 - Do modelo aos mecanismos", author="Nestor, Letícia e Gabriel")
    document.build(story, onFirstPage=decorate, onLaterPages=decorate)


def run():
    assert read_json(DROOT / "verification.json")["status"] == "passed", "Verify the scientific results before reporting"
    if not (DROOT / "source_references.json").exists():
        save_json(DROOT / "source_references.json", read_json(ROOT / "docs/REFERENCIAS_DISCIPLINA.json"))
    figures(); s = summary(); create_notebook(s); create_html(s); create_pdf(s)
    save_json(DROOT / "report_manifest.json", {"pdf": PDF.relative_to(ROOT).as_posix(), "pdf_sha256": file_sha256(PDF),
              "html": HTML.relative_to(ROOT).as_posix(), "html_sha256": file_sha256(HTML),
              "notebook": NOTEBOOK.relative_to(ROOT).as_posix(), "notebook_execution": "pending",
              "verification_sha256": file_sha256(DROOT / "verification.json"),
              "selection_sha256": file_sha256(DROOT / "selection_frozen.json"),
              "figures": {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sorted(FIG.glob("*"))}})
    print("Didactic figures, PDF, HTML and notebook built from verified results")
