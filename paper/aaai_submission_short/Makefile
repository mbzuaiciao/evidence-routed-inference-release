LATEXMK ?= latexmk
PYTHON ?= python3

.PHONY: main supplement checklist all prism-sources prism-main prism-supplement archives clean check

main:
	$(LATEXMK) -pdf -pdflatex="pdflatex %O %S" -interaction=nonstopmode -halt-on-error main.tex

supplement:
	$(LATEXMK) -pdf -pdflatex="pdflatex %O %S" -interaction=nonstopmode -halt-on-error supplement.tex

checklist:
	$(LATEXMK) -pdf -pdflatex="pdflatex %O %S" -interaction=nonstopmode -halt-on-error ReproducibilityChecklist.tex

all: main supplement checklist

prism-sources:
	$(PYTHON) scripts/prepare_prism.py

prism-main: prism-sources
	$(LATEXMK) -pdf -pdflatex="pdflatex %O %S" -interaction=nonstopmode -halt-on-error eri_aaai27_main.tex

prism-supplement: prism-sources
	$(LATEXMK) -pdf -pdflatex="pdflatex %O %S" -interaction=nonstopmode -halt-on-error eri_aaai27_supplement.tex

archives: prism-sources

clean:
	$(LATEXMK) -C main.tex
	$(LATEXMK) -C supplement.tex
	$(LATEXMK) -C ReproducibilityChecklist.tex
	$(LATEXMK) -C eri_aaai27_main.tex
	$(LATEXMK) -C eri_aaai27_supplement.tex

check:
	@test -f main.tex
	@test -f supplement.tex
	@test -f references.bib
	@test -f aaai2027.sty
	@test -f aaai2027.bst
	@grep -qF '\usepackage[submission]{aaai2027}' main.tex
	@grep -qF '\usepackage[submission]{aaai2027}' supplement.tex
	@! grep -R -nE '\\(bibliographystyle|clearpage|newpage|pagebreak|vspace\\{-|vskip\\{-|resizebox)' --include='*.tex' .
	@! grep -R -nE '\\usepackage.*\\{(authblk|balance|float|flushend|fullpage|geometry|hyperref|navigator|setspace|stfloats|titlesec|ulem|wrapfig)\\}' --include='*.tex' .
	@! grep -R -nE '/Users/|/private/tmp/' --include='*.tex' .
	@! grep -R -nE '\\documentclass\\[[^]]*(9pt|8pt)|\\fontsize|\\linespread|\\setstretch' --include='*.tex' .
