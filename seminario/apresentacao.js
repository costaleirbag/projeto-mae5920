"use strict";

const byId = (id) => document.getElementById(id);
const { slides, evidence, architecture, contrast, cut } = JSON.parse(byId("deck-data").textContent);
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
    reset(max) {
      pause();
      slider.max = max;
      render(0);
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
const scan = { name: "A", width: 3, stride: 1, tokens: [], bars: [], trace: null,
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

function setupScan() {
  const current = cnn[scan.name];
  scan.trace = current.janelas[scan.width];
  scan.visited = scan.trace.ativacoes.map((_, i) => i).filter((i) => i % scan.stride === 0);
  scan.winner = scan.visited.reduce((best, i) =>
    scan.trace.ativacoes[i] > scan.trace.ativacoes[best] ? i : best, 0);
  scan.scale = 1.12 * Math.max(...Object.values(cnn).flatMap((sample) => sample.janelas[scan.width].ativacoes), 0.01);
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
  cnnTransport.reset(scan.visited.length);
}

const caseButtons = [...document.querySelectorAll("[data-case]")];
const widthButtons = [...document.querySelectorAll("[data-width]")];
const strideButtons = [...document.querySelectorAll("[data-stride]")];
caseButtons.forEach((button) => button.addEventListener("click", () => {
  scan.name = button.dataset.case;
  caseButtons.forEach((option) => option.setAttribute("aria-pressed", option === button));
  setupScan();
}));
widthButtons.forEach((button) => button.addEventListener("click", () => {
  scan.width = Number(button.dataset.width);
  widthButtons.forEach((option) => option.setAttribute("aria-pressed", option === button));
  setupScan();
}));
strideButtons.forEach((button) => button.addEventListener("click", () => {
  scan.stride = Number(button.dataset.stride);
  strideButtons.forEach((option) => option.setAttribute("aria-pressed", option === button));
  setupScan();
}));
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

const picker = byId("slide-picker");
const previous = byId("previous");
const next = byId("next");
let slideIndex = -1;
let cnnView = "build";
const labs = { bpe: bpeTransport, cnn: cnnTransport, contrast: contrastTransport, cut: cutTransport };
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
  for (const id of ["bpe", "contrast", "youtube", "closing", "cut"]) byId(id).hidden = slide.interactive !== id;
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
      link.target = "_blank";
      link.rel = "noopener";
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
  const cutIndex = slides.findIndex((slide) => slide.interactive === "cut");
  const shortcuts = { "#encerramento": ending, "#apoio": ending + 1, "#recorte": cutIndex };
  showSlide(match ? Number(match[1]) - 1 : (shortcuts[location.hash] ?? 0), match?.[2] ? "scan" : "build");
}
window.addEventListener("hashchange", openFromHash);
openFromHash();
