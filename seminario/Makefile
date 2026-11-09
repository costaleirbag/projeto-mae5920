BUILD_DIR := ../tmp/seminario/build
PDF_DIR := ../output/pdf
LATEXMK ?= latexmk
export SOURCE_DATE_EPOCH := 1794182400
export FORCE_SOURCE_DATE := 1

.PHONY: all
all:
	mkdir -p $(BUILD_DIR) $(PDF_DIR)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD_DIR) relatorio.tex slides.tex
	cp $(BUILD_DIR)/relatorio.pdf $(PDF_DIR)/relatorio_seminario.pdf
	cp $(BUILD_DIR)/slides.pdf $(PDF_DIR)/slides_15_min.pdf
