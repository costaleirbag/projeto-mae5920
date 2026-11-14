"use strict";

const byId = (id) => document.getElementById(id);
const { slides, evidence, architecture, contrast, position, youtubeQwen, cut, tfidf, lengthBands } =
  JSON.parse(byId("deck-data").textContent);
const number = new Intl.NumberFormat("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

function makeTransport(name, draw, nextView) {
  const slider = byId(name + "-step");
  const button = byId(name + "-play");
  const back = byId(name + "-back");
  const forward = byId(name + "-forward");
  const count = byId(name + "-count");
  const forwardText = forward.textContent;
  const forwardLabel = forward.getAttribute("aria-label");
  let timer = null;

  function render(step) {
    slider.value = step;
    const label = draw(step);
    count.textContent = label;
    slider.setAttribute("aria-valuetext", label);
    back.disabled = step === 0;
    const last = step === Number(slider.max);
    forward.disabled = last && !nextView;
    if (nextView) {
      forward.textContent = last ? nextView.label : forwardText;
      forward.setAttribute("aria-label", last ? nextView.label : forwardLabel);
    }
  }

  function pause() {
    window.clearInterval(timer);
    timer = null;
    if (button) button.textContent = "▶ Reproduzir";
  }

  function move(direction) {
    pause();
    const step = Number(slider.value) + direction;
    if (step < 0 || step > Number(slider.max)) return false;
    render(step);
    return true;
  }

  back.addEventListener("click", () => move(-1));
  forward.addEventListener("click", () => {
    if (!move(1)) nextView?.go();
  });

  button?.addEventListener("click", () => {
    if (timer !== null) {
      pause();
      return;
    }
    if (slider.value === slider.max) render(0);
    button.textContent = "Ⅱ Pausar";
    timer = window.setInterval(() => {
      const step = Number(slider.value) + 1;
      render(step);
      if (step === Number(slider.max)) pause();
    }, 900);
  });
  slider.addEventListener("input", () => {
    pause();
    render(Number(slider.value));
  });

  return {
    move,
    pause,
    reset(max, step = 0) {
      pause();
      slider.max = max;
      render(step);
    },
  };
}

function token(text, suffix = "") {
  const element = document.createElement("code");
  if (suffix && text.endsWith(suffix)) {
    element.textContent = text.slice(0, -suffix.length);
    const boundary = document.createElement("small");
    boundary.textContent = suffix;
    element.append(boundary);
  } else {
    element.textContent = text;
  }
  return element;
}

const bpe = evidence.bpe;
const bpeLast = bpe.etapas.length - 1;
const [left, right] = bpe.frase.split(bpe.palavra);
const focusWord = document.createElement("strong");
focusWord.textContent = bpe.palavra;
byId("bpe-sentence").append(left, focusWord, right);
byId("bpe-result").append(...bpe.etapas[bpeLast].pecas.map((text) => token(text, bpe.sufixo)));

const bpeTransport = makeTransport("bpe", (step) => {
  const stage = bpe.etapas[step];
  byId("bpe-before").hidden = step === 0;
  if (step > 0) {
    const before = bpe.etapas[step - 1].pecas.map((text, i) => {
      const piece = token(text, bpe.sufixo);
      piece.classList.toggle("source-pair", i === stage.indice || i === stage.indice + 1);
      return piece;
    });
    byId("bpe-before-pieces").replaceChildren(...before);
  }
  const pieces = stage.pecas.map((text, i) => {
    const piece = token(text, bpe.sufixo);
    piece.classList.toggle("fused", i === stage.indice);
    return piece;
  });
  byId("bpe-pieces").replaceChildren(...pieces);
  byId("bpe-stage").textContent = step === 0 ? "Caracteres de partida" : "Fusão " + step + " de " + bpeLast;
  byId("bpe-event").textContent = step === 0
    ? "O tokenizer começa pelos caracteres e pela marca de fim."
    : stage.par.join(" + ") + " → " + stage.pecas[stage.indice] + " · regra " + stage.regra.toLocaleString("pt-BR");
  return step + " / " + bpeLast + " fusões";
});
bpeTransport.reset(bpeLast);

const architectureViewport = byId("architecture-viewport");
const diagram = architectureViewport.querySelector("svg");
const architectureGroups = [...diagram.querySelectorAll("[data-architecture-step]")];

function centerArchitectureStage() {
  if (architectureViewport.clientWidth === 0 || diagram.clientWidth <= architectureViewport.clientWidth) return;
  const stage = diagram.querySelector(".current").getBBox();
  const center = (stage.x + stage.width / 2) * diagram.clientWidth / diagram.viewBox.baseVal.width;
  architectureViewport.scrollLeft = center - architectureViewport.clientWidth / 2;
}

function sourceBlock({ path, context, lines }) {
  const block = document.createElement("div");
  const link = document.createElement("a");
  const source = "https://github.com/costaleirbag/projeto-mae5920/blob/main/" + path;
  const first = lines[0].number;
  const last = lines.at(-1).number;
  link.className = "source-label";
  link.href = source + "#L" + first + (last === first ? "" : "-L" + last);
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = [context, path.split("/").at(-1), last === first ? "L" + first : "L" + first + "–" + last]
    .filter(Boolean).join(" · ");
  const listing = document.createElement("div");
  listing.className = "source-lines";
  for (const { number, text } of lines) {
    const gutter = document.createElement("a");
    gutter.className = "source-number";
    gutter.href = source + "#L" + number;
    gutter.target = "_blank";
    gutter.rel = "noopener";
    gutter.setAttribute("aria-label", "Abrir a linha " + number + " de " + path);
    gutter.textContent = number;
    listing.append(gutter, token(text));
  }
  block.append(link, listing);
  return block;
}

const architectureTransport = makeTransport("architecture", (step) => {
  const stage = architecture[step];
  architectureGroups.forEach((group, index) => {
    group.classList.toggle("future", index > step);
    group.classList.toggle("current", index === step);
  });
  byId("architecture-label").textContent = stage.label;
  byId("architecture-title").textContent = stage.title;
  byId("architecture-reason").textContent = stage.reason;
  byId("architecture-detail").textContent = stage.detail;
  byId("architecture-code").replaceChildren(...stage.code.map(sourceBlock));
  centerArchitectureStage();
  return (step + 1) + " / " + architecture.length + " etapas";
}, { label: "Ver varredura →", go: () => showSlide(slideIndex, "scan") });
architectureTransport.reset(architecture.length - 1);
window.addEventListener("resize", centerArchitectureStage);

const cnn = evidence.cnn.casos;
const scan = { name: "A", width: 3, stride: 1, alternate: false, tokens: [], bars: [], trace: null,
  scale: 1, edges: [], visited: [], winner: 0 };
const scanWindow = byId("cnn-window");
const cnnTransport = makeTransport("cnn", (step) => {
  const { width, stride, trace, tokens, bars, scale, edges, visited, winner } = scan;
  const atMax = step === visited.length;
  const position = atMax ? winner : visited[step];
  scanWindow.style.left = edges[position] + "%";
  scanWindow.style.width = (edges[position + width] - edges[position]) + "%";
  tokens.forEach((piece, i) => piece.classList.toggle("active", i >= position && i < position + width));
  bars.forEach((bar, i) => {
    const sampled = i % stride === 0;
    const seen = sampled && (atMax || i <= position);
    bar.style.height = (seen ? 100 * trace.ativacoes[i] / scale : 0) + "%";
    bar.classList.toggle("future", sampled && !seen);
    bar.classList.toggle("skipped", !sampled);
    bar.classList.toggle("current", i === position);
  });
  const current = number.format(trace.ativacoes[position]);
  byId("cnn-raw").textContent = number.format(trace.pre_relu[position]);
  byId("cnn-relu").textContent = current;
  byId("cnn-reading").textContent = atMax
    ? "O máximo global guarda " + current + " para este filtro."
    : "A janela avança " + stride + (stride === 1 ? " token." : " tokens.");
  return atMax ? "Máximo global" : (step + 1) + " / " + visited.length + " janelas";
});

function traceFor(name) {
  return scan.alternate ? evidence.cnn.filtro_fim_a[name] : cnn[name].janelas[scan.width];
}

function setupScan(showPeak = false) {
  const current = cnn[scan.name];
  scan.trace = traceFor(scan.name);
  scan.visited = scan.trace.ativacoes.map((_, i) => i).filter((i) => i % scan.stride === 0);
  scan.winner = scan.visited.reduce((best, i) =>
    scan.trace.ativacoes[i] > scan.trace.ativacoes[best] ? i : best, 0);
  scan.scale = 1.12 * Math.max(...Object.keys(cnn).flatMap((name) => traceFor(name).ativacoes), 0.01);
  scan.tokens = current.tokens.map((text) => token(text));
  const weights = current.tokens.map((text) => Math.max(4, text.length));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  scan.edges = [0];
  weights.forEach((weight, index) => scan.edges.push(scan.edges[index] + 100 * weight / total));
  scan.bars = scan.trace.ativacoes.map((value, index) => {
    const bar = document.createElement("span");
    bar.className = "bar";
    bar.style.left = ((scan.edges[index] + scan.edges[index + scan.width]) / 2) + "%";
    return bar;
  });
  byId("cnn-tokens").style.gridTemplateColumns = weights.map((weight) => "minmax(0, " + weight + "fr)").join(" ");
  byId("cnn-tokens").replaceChildren(scanWindow, ...scan.tokens);
  byId("cnn-bars").replaceChildren(...scan.bars);
  byId("cnn-filter").textContent = "Filtro " + scan.trace.filtro + " · k = " + scan.width;
  byId("cnn-other-filter").setAttribute("aria-pressed", scan.alternate);
  byId("cnn-other-filter").textContent = scan.alternate ? "Voltar aos filtros do exemplo" : "Ver o filtro 6 com wonderful";
  const finalPeaks = Object.values(evidence.cnn.picos_fim_a).reduce((total, count) => total + count, 0);
  byId("cnn-selection-note").textContent = scan.alternate
    ? "Na frase A, " + finalPeaks + " dos 300 filtros têm pico positivo na última janela."
    : "Um filtro por largura, escolhido pela maior variação na frase A.";
  cnnTransport.reset(scan.visited.length, showPeak ? scan.visited.length : 0);
}

const caseButtons = [...document.querySelectorAll("[data-case]")];
const widthButtons = [...document.querySelectorAll("[data-width]")];
const strideButtons = [...document.querySelectorAll("[data-stride]")];
caseButtons.forEach((button) => button.addEventListener("click", () => {
  scan.name = button.dataset.case;
  caseButtons.forEach((option) => option.setAttribute("aria-pressed", option === button));
  setupScan(scan.alternate);
}));
widthButtons.forEach((button) => button.addEventListener("click", () => {
  scan.width = Number(button.dataset.width);
  scan.alternate = false;
  widthButtons.forEach((option) => option.setAttribute("aria-pressed", option === button));
  setupScan();
}));
strideButtons.forEach((button) => button.addEventListener("click", () => {
  scan.stride = Number(button.dataset.stride);
  strideButtons.forEach((option) => option.setAttribute("aria-pressed", option === button));
  setupScan();
}));
byId("cnn-other-filter").addEventListener("click", () => {
  scan.alternate = !scan.alternate;
  if (scan.alternate) {
    scan.width = 3;
    scan.stride = 1;
    widthButtons.forEach((button) => button.setAttribute("aria-pressed", Number(button.dataset.width) === 3));
    strideButtons.forEach((button) => button.setAttribute("aria-pressed", Number(button.dataset.stride) === 1));
  }
  setupScan(scan.alternate);
});
setupScan();


const swapButtons = [...document.querySelectorAll("[data-swap]")];
const swapSlots = [byId("swap-first"), byId("swap-last")];
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const percent = new Intl.NumberFormat("pt-BR", {
  style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1,
});
const swapWords = contrast.adjectives.map((word, index) => {
  const chip = document.createElement("strong");
  chip.className = "swap-adjective";
  chip.dataset.tone = contrast.expected[index === 0 ? "b" : "a"];
  chip.textContent = word;
  return chip;
});
byId("swap-start").textContent = contrast.start;
byId("swap-connector").textContent = contrast.connector;

const swapRows = contrast.models.map(({ name, predictions, seeds }) => {
  const row = document.createElement("div");
  row.className = "swap-model";
  const label = document.createElement("span");
  label.className = "swap-model-name";
  label.textContent = name === "BPE" ? "CNN BPE" : "CNN por palavras";
  const track = document.createElement("div");
  track.className = "swap-track";
  track.setAttribute("aria-hidden", "true");
  const previous = document.createElement("span");
  previous.className = "swap-previous";
  previous.style.setProperty("--score", 100 * predictions.a.score + "%");
  const dot = document.createElement("span");
  dot.className = "swap-dot";
  track.append(previous, dot);
  const outcome = document.createElement("div");
  outcome.className = "swap-outcome";
  const score = document.createElement("strong");
  const category = document.createElement("span");
  category.className = "class-label";
  outcome.append(score, category);
  row.append(label, track, outcome);
  byId("swap-models").append(row);
  return { predictions, seeds, previous, dot, score, category };
});

const seeds = swapRows[0].seeds;
const runs = swapRows.length * seeds.length;
byId("swap-sample").textContent = swapRows.length + " tokenizadores × " + seeds.length
  + " treinos (sementes " + seeds.join(", ") + ") = " + runs + " treinos";
let swapState;
function showContrast(state, animate = true) {
  if (state === swapState) return false;
  const shouldMove = animate && !reducedMotion.matches && !byId("contrast").hidden;
  swapWords.forEach((chip) => chip.getAnimations().forEach((motion) => motion.cancel()));
  const from = shouldMove ? swapWords.map((chip) => chip.getBoundingClientRect()) : [];
  const order = state === "a" ? swapWords : [...swapWords].reverse();
  swapSlots.forEach((slot, index) => slot.replaceChildren(order[index]));
  if (shouldMove) {
    swapWords.forEach((chip, index) => {
      const to = chip.getBoundingClientRect();
      chip.animate([{ transform: "translate(" + (from[index].left - to.left) + "px, "
        + (from[index].top - to.top) + "px)" }, { transform: "translate(0, 0)" }],
      { duration: 430, easing: "cubic-bezier(.25,.8,.3,1)" });
    });
  }
  swapButtons.forEach((button) => button.setAttribute("aria-pressed", button.dataset.swap === state));
  const expected = byId("swap-expected");
  expected.dataset.tone = contrast.expected[state];
  expected.textContent = expected.dataset.tone[0].toUpperCase() + expected.dataset.tone.slice(1);
  for (const { predictions, seeds, previous, dot, score, category } of swapRows) {
    const result = predictions[state];
    previous.hidden = state === "a";
    dot.style.setProperty("--score", 100 * result.score + "%");
    dot.dataset.tone = result.label;
    score.textContent = percent.format(result.score);
    category.dataset.tone = result.label;
    const label = result.label[0].toUpperCase() + result.label.slice(1);
    category.textContent = label + (result.label === "dividida" ? "" : " · " + result.agreement + "/" + seeds.length);
  }
  byId("swap-note").textContent = state === "b"
    ? "○ indica os escores de A." : "Mesmas palavras, outra ordem.";
  byId("swap-summary").textContent = state === "b"
    ? contrast.reversed + " de " + runs + " previram negativa em A e positiva em B."
    : "Cada um dos " + runs + " treinos previu negativa para A.";
  swapState = state;
  return true;
}
swapButtons.forEach((button) => button.addEventListener("click", () => showContrast(button.dataset.swap)));
showContrast("a", false);
const contrastTransport = { move: (direction) => showContrast(direction > 0 ? "b" : "a") };

const pooling = evidence.pooling;
const poolButtons = [...document.querySelectorAll("[data-pool-case]")];
const poolBlocks = [byId("pool-first"), byId("pool-second")];
const poolCases = [pooling.casos.A, pooling.casos.B];
const poolMaxima = poolCases.map(({ ativacoes, max_indice }) => ativacoes[max_indice]);
const poolScale = 1.14 * Math.max(...poolMaxima);
const poolCount = poolCases[0].ativacoes.length;
const poolChart = byId("pool-bars");
poolChart.style.setProperty("--window-count", poolCount);
const poolBars = Array.from({ length: poolCount }, (_, index) => {
  const bar = document.createElement("span");
  bar.className = "bar";
  bar.style.left = (100 * (index + .5) / poolCount) + "%";
  return bar;
});
poolChart.append(...poolBars);

byId("pool-separator-text").textContent = "“" + pooling.separador + "”";
byId("pool-window").append(...pooling.janela.map((word) => token(word)));
byId("pool-positions").textContent = poolCases.map(({ max_indice }) => max_indice + 1).join(" → ");
byId("pool-maxima").textContent = poolMaxima.map((value) => number.format(value)).join(" = ");
byId("pool-evidence").textContent = "CNN por palavras, semente " + pooling.semente + ". Filtro "
  + pooling.filtro + ", janelas de " + pooling.largura + " tokens.";

let poolState;
function showPooling(state) {
  if (state === poolState) return false;
  const order = state === "A" ? ["elogio", "critica"] : ["critica", "elogio"];
  poolBlocks.forEach((block, index) => {
    block.textContent = pooling[order[index]];
    block.dataset.tone = order[index] === "elogio" ? "positiva" : "negativa";
  });
  const { ativacoes, max_indice } = pooling.casos[state];
  poolBars.forEach((bar, index) => {
    bar.style.height = (100 * ativacoes[index] / poolScale) + "%";
    bar.classList.toggle("current", index === max_indice);
  });
  byId("pool-current").textContent = "Em " + state + ", pico no token " + (max_indice + 1);
  poolButtons.forEach((button) => button.setAttribute("aria-pressed", button.dataset.poolCase === state));
  poolState = state;
  return true;
}
poolButtons.forEach((button) => button.addEventListener("click", () => showPooling(button.dataset.poolCase)));
showPooling("A");
const poolingTransport = { move: (direction) => showPooling(direction > 0 ? "B" : "A") };

document.querySelectorAll("[data-position-order]").forEach((table) => {
  const byValue = table.dataset.positionOrder === "value";
  table.querySelectorAll("[data-position-case]").forEach((row) => {
    const peaks = pooling.casos[row.dataset.positionCase].picos_por_metade;
    const values = byValue ? [...peaks].sort((a, b) => a - b) : peaks;
    row.querySelectorAll("td").forEach((cell, index) => {
      cell.textContent = number.format(values[index]);
      cell.classList.toggle("peak", values[index] === Math.max(...peaks));
    });
  });
});
byId("position-pairs").textContent = position.pairs;
for (const order of ["value", "half"]) byId("position-rate-" + order).textContent = percent.format(position.rates[order]);

document.querySelectorAll("[data-youtube-qwen]").forEach((row) => {
  const modes = youtubeQwen[row.dataset.youtubeQwen];
  row.querySelectorAll("[data-mode]").forEach((result) => {
    const votes = modes[result.dataset.mode];
    const badge = result.querySelector(".class-label");
    const [first] = votes;
    if (first && votes.every((vote) => vote === first)) {
      badge.dataset.tone = first;
      badge.textContent = first[0].toUpperCase() + first.slice(1) + " " + votes.length + "/" + votes.length;
    } else {
      badge.textContent = [[ "positiva", "pos." ], [ "negativa", "neg." ], [ null, "sem rótulo" ]]
        .map(([kind, label]) => [votes.filter((vote) => vote === kind).length, label])
        .filter(([count]) => count > 0)
        .map(([count, label]) => count + " " + label)
        .join(" · ");
    }
  });
});

const cutButtons = [...document.querySelectorAll("[data-cut]")];
const cutResults = [...document.querySelectorAll("[data-cut-model]")];
const cutRuler = document.querySelector(".cut-ruler");
cutRuler.style.setProperty("--cut-ratio", 100 * cut.cap / cut.total + "%");
byId("cut-cap").textContent = cut.cap;
byId("cut-limit").textContent = cut.cap;
byId("cut-total").textContent = cut.total + " tokens";
byId("cut-label").textContent = cut.label;
byId("cut-label").dataset.tone = cut.label.toLowerCase();

function tokenRange(excerpt) {
  return "tokens " + excerpt.primeiro_token + "–" + excerpt.ultimo_token;
}
byId("cut-start-range").textContent = "Dentro do recorte · " + tokenRange(cut.excerpts.prefixo);
byId("cut-start-quote").textContent = "“" + cut.excerpts.prefixo.texto + "…”";
byId("cut-end-quote").textContent = "“" + cut.excerpts.final.texto + "”";
byId("cut-omitted").textContent = (cut.total - cut.cap) + " tokens ficaram fora.";
let cutScope;
function showCut(scope) {
  if (scope === cutScope) return false;
  const full = scope === "inteiro";
  cutButtons.forEach((button) => button.setAttribute("aria-pressed", button.dataset.cut === scope));
  cutRuler.classList.toggle("full", full);
  byId("cut-end-range").textContent = full
    ? "Depois do corte · " + tokenRange(cut.excerpts.final) : "Depois do token " + cut.cap;
  byId("cut-end-quote").hidden = !full;
  byId("cut-omitted").hidden = full;
  cutResults.forEach((row) => {
    const model = row.dataset.cutModel;
    const original = full ? model === "tfidf" : model === "cnn";
    const score = cut.scores[scope][model];
    const category = row.querySelector(".class-label");
    row.querySelector(".cut-value").textContent = percent.format(score);
    row.querySelector(".cut-original").textContent = original ? "Usado no teste original" : "Reavaliação";
    category.textContent = score >= .5 ? "Positiva" : "Negativa";
    category.dataset.tone = category.textContent.toLowerCase();
  });
  byId("cut-outcome").textContent = full ? "Os dois acertaram o rótulo negativo." : "Os dois previram positiva com o recorte.";
  cutScope = scope;
  return true;
}
cutButtons.forEach((button) => button.addEventListener("click", () => showCut(button.dataset.cut)));
showCut("prefixo");
const cutTransport = { move: (direction) => showCut(direction > 0 ? "inteiro" : "prefixo") };

const lengthControls = [...document.querySelectorAll("[data-length-metric]")];
const lengthReplay = byId("length-replay");
const lengthPercent = new Intl.NumberFormat("pt-BR", { style: "percent", minimumFractionDigits: 2 });
const lengthCount = new Intl.NumberFormat("pt-BR");
let lengthMetric = "accuracy";

function drawLengths() {
  const columns = ["tfidf", lengthReplay.checked ? "cnn_integral" : "cnn", "bert"];
  const metricLabel = lengthControls.find((button) => button.dataset.lengthMetric === lengthMetric).textContent;
  byId("length-caption").textContent = metricLabel + " por faixa de comprimento no teste oficial";
  byId("length-cnn-heading").lastElementChild.textContent = lengthReplay.checked ? "Completa" : "400 tokens";
  lengthControls.forEach((button) => button.setAttribute("aria-pressed", button.dataset.lengthMetric === lengthMetric));
  const rows = lengthBands.map((band) => {
    const row = document.createElement("tr");
    const label = document.createElement("th");
    label.scope = "row";
    const { min_words: lo, max_words: hi } = band;
    label.textContent = lo === 0 ? "Até " + hi : hi === null ? "501 ou mais" : lo + "–" + hi;
    row.append(label);
    row.insertCell().textContent = lengthCount.format(band.n);
    for (const model of columns) {
      const value = band.metrics[model][lengthMetric];
      const cell = row.insertCell();
      cell.className = "length-metric";
      const score = document.createElement("span");
      score.textContent = lengthPercent.format(value);
      const dot = document.createElement("span");
      dot.className = "length-dot";
      dot.setAttribute("aria-hidden", "true");
      dot.style.setProperty("--length-position", (100 * (value - .8) / .14) + "%");
      cell.append(score, dot);
    }
    return row;
  });
  byId("length-rows").replaceChildren(...rows);
  const long = lengthBands.at(-1).metrics;
  byId("length-readout").textContent = lengthPercent.format(long.cnn[lengthMetric]) +
    " → " + lengthPercent.format(long.cnn_integral[lengthMetric]);
}
lengthControls.forEach((button) => button.addEventListener("click", () => {
  lengthMetric = button.dataset.lengthMetric;
  drawLengths();
}));
lengthReplay.addEventListener("change", drawLengths);
drawLengths();

const picker = byId("slide-picker");
const previous = byId("previous");
const next = byId("next");
let slideIndex = -1;
let cnnView = "build";
const tfidfOrders = document.querySelectorAll("[data-tfidf-order]");
tfidfOrders.forEach((button) => {
  button.addEventListener("click", () => {
    tfidfOrders.forEach((option) => option.setAttribute("aria-pressed", option === button));
    byId("tfidf-bigram").textContent = button.textContent;
  });
});

byId("tfidf-quote").textContent = "“" + tfidf.text + "”";
const tfidfNumber = new Intl.NumberFormat("pt-BR", { minimumFractionDigits: 3, maximumFractionDigits: 3 });
const tfidfCount = new Intl.NumberFormat("pt-BR");
const tfidfPercent = new Intl.NumberFormat("pt-BR", { style: "percent", minimumFractionDigits: 3, maximumFractionDigits: 3 });

function tfidfRow(table, label, values, tone = "") {
  const row = byId(table).insertRow();
  const title = document.createElement("th");
  title.scope = "row";
  title.append(label);
  row.append(title);
  values.forEach((value) => { row.insertCell().textContent = value; });
  row.lastElementChild.className = tone;
}
for (const { term, tf, document_frequency, idf, tfidf: feature, coefficient, contribution } of tfidf.terms) {
  tfidfRow("tfidf-weight-rows", token(term), [tfidfCount.format(tf), tfidfCount.format(document_frequency),
    tfidfNumber.format(idf), tfidfNumber.format(feature)]);
  tfidfRow("tfidf-sum", token(term), [tfidfNumber.format(feature), tfidfNumber.format(coefficient),
    tfidfNumber.format(contribution)], contribution < 0 ? "tfidf-negative" : "tfidf-positive");
}
tfidfRow("tfidf-sum", "Viés", ["", "", "+" + tfidfNumber.format(tfidf.bias)], "tfidf-positive");
const awful = tfidf.terms.find(({ term }) => term === "awful");
byId("tfidf-training").textContent = tfidfCount.format(tfidf.train_documents) + " resenhas no treino";
const tfidfExample = token("1 × " + tfidfNumber.format(awful.idf) + " ÷ " + tfidfNumber.format(tfidf.normalization) +
  " ≈ " + tfidfNumber.format(awful.tfidf));
byId("tfidf-worked").append("Para awful, ", tfidfExample, ". O divisor é a norma L2 dos cinco produtos TF × IDF.");
byId("tfidf-logit").textContent = tfidfNumber.format(tfidf.logit);
byId("tfidf-positive").textContent = tfidfPercent.format(tfidf.positive_raw);

const tfidfStages = [...document.querySelectorAll("[data-tfidf-step]")];
let tfidfStep = 0;
function showTfidf(step) {
  if (step < 0 || step >= tfidfStages.length) return false;
  tfidfStep = step;
  tfidfStages.forEach((button, index) => {
    button.setAttribute("aria-pressed", index === step);
    byId(button.getAttribute("aria-controls")).hidden = index !== step;
  });
  byId("tfidf-step-note").textContent = "Etapa " + (step + 1) + " de " + tfidfStages.length;
  byId("tfidf-next").textContent = step === tfidfStages.length - 1 ? "Próximo slide →" : "Próxima etapa →";
  return true;
}
const tfidfTransport = { move: (direction) => showTfidf(tfidfStep + direction) };
tfidfStages.forEach((button, index) => button.addEventListener("click", () => showTfidf(index)));
byId("tfidf-next").addEventListener("click", () => {
  if (!tfidfTransport.move(1)) showSlide(slideIndex + 1);
});

const labs = {
  bpe: bpeTransport, cnn: cnnTransport, tfidf: tfidfTransport,
  contrast: contrastTransport, pooling: poolingTransport, cut: cutTransport,
};
const ending = slides.findIndex((slide) => slide.interactive === "closing");
const mainGroup = document.createElement("optgroup");
const supportGroup = document.createElement("optgroup");
mainGroup.label = "Apresentação";
supportGroup.label = "Apoio";
slides.forEach((slide, index) => {
  const group = index <= ending ? mainGroup : supportGroup;
  const number = index <= ending ? index + 1 : index - ending;
  group.append(new Option(number + " · " + slide.title, index));
});
picker.append(mainGroup, supportGroup);

function showSlide(index, view = "build") {
  const target = Math.min(Math.max(0, index), slides.length - 1);
  const slide = slides[target];
  if (target === slideIndex && (slide.interactive !== "cnn" || view === cnnView)) return;
  slideIndex = target;
  cnnView = view;
  bpeTransport.pause();
  cnnTransport.pause();
  for (const id of ["bpe", "lengths", "contrast", "pooling", "position", "youtube", "closing", "cut", "tfidf"]) {
    byId(id).hidden = slide.interactive !== id;
  }
  byId("cnn-build").hidden = slide.interactive !== "cnn" || cnnView !== "build";
  byId("cnn").hidden = slide.interactive !== "cnn" || cnnView !== "scan";
  centerArchitectureStage();
  byId("static-slide").hidden = Boolean(slide.interactive);
  if (!slide.interactive) {
    byId("slide-image").src = "data:image/png;base64," + slide.png;
    byId("transcript").textContent = slide.text;
    byId("slide-links").replaceChildren(...slide.links.map(({ url, label, rect }) => {
      const link = document.createElement("a");
      link.className = "slide-link";
      link.href = url;
      if (!url.startsWith("#")) {
        link.target = "_blank";
        link.rel = "noopener";
      }
      link.setAttribute("aria-label", label);
      ["left", "top", "width", "height"].forEach((key, i) => {
        link.style[key] = rect[i] + "%";
      });
      return link;
    }));
  }
  picker.value = target;
  previous.disabled = target === 0;
  next.disabled = target === ending || target === slides.length - 1;
  byId("slide-count").textContent = target <= ending
    ? (target + 1) + " / " + (ending + 1)
    : "Apoio " + (target - ending) + " / " + (slides.length - ending - 1);
  const suffix = slide.interactive === "cnn" && cnnView === "scan" ? "-varredura" : "";
  history.replaceState(null, "", "#slide-" + (target + 1) + suffix);
}

const fullscreen = byId("fullscreen");
fullscreen.hidden = !document.fullscreenEnabled;
fullscreen.addEventListener("click", async () => {
  if (document.fullscreenElement) {
    await document.exitFullscreen();
  } else {
    await document.documentElement.requestFullscreen();
  }
});
document.addEventListener("fullscreenchange", () => {
  fullscreen.textContent = document.fullscreenElement ? "Sair da tela cheia" : "Tela cheia";
});

picker.addEventListener("change", () => showSlide(Number(picker.value)));
previous.addEventListener("click", () => showSlide(slideIndex - 1));
next.addEventListener("click", () => showSlide(slideIndex + 1));
document.querySelectorAll("[data-cnn-view]").forEach((button) => {
  button.addEventListener("click", () => showSlide(slideIndex, button.dataset.cnnView));
});
window.addEventListener("keydown", (event) => {
  if (event.altKey || event.ctrlKey || event.metaKey || event.target.closest?.("input, select, textarea")) return;
  if (event.key === " " && event.target.closest?.("button, a")) return;
  const moves = { ArrowRight: 1, PageDown: 1, " ": 1, ArrowLeft: -1, PageUp: -1 };
  const move = moves[event.key];
  if (move) {
    event.preventDefault();
    const interactive = slides[slideIndex].interactive;
    const building = interactive === "cnn" && cnnView === "build";
    const lab = building ? architectureTransport : labs[interactive];
    if (lab?.move(move)) return;
    if (building && move === 1) showSlide(slideIndex, "scan");
    else if (interactive === "cnn" && cnnView === "scan" && move === -1) showSlide(slideIndex);
    else if (move < 0 || slideIndex !== ending) showSlide(slideIndex + move);
  } else if (event.key === "Home" || event.key === "End") {
    event.preventDefault();
    showSlide(event.key === "Home" ? 0 : ending);
  } else if (event.key.toLowerCase() === "f" && !fullscreen.hidden) {
    fullscreen.click();
  }
});
function openFromHash() {
  const match = location.hash.match(/^#slide-(\d+)(-varredura)?$/);
  const shortcuts = { "#encerramento": "closing", "#recorte": "cut", "#tf-idf": "tfidf",
    "#comprimento": "lengths", "#contraste": "contrast", "#maximo-global": "pooling", "#posicao": "position" };
  const named = slides.findIndex((slide) => slide.interactive && slide.interactive === shortcuts[location.hash]);
  const index = location.hash === "#apoio" ? ending + 1 : Math.max(0, named);
  showSlide(match ? Number(match[1]) - 1 : index, match?.[2] ? "scan" : "build");
}
window.addEventListener("hashchange", openFromHash);
openFromHash();
