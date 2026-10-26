# H-potential: install, test, lint, figures.
#
# make install    editable install with every extra
# make test       fast suite, skipping the published-value regressions
# make test-all   everything, including the regressions (several minutes)
# make lint       ruff and black in check mode
# make format     ruff --fix and black, writing changes
# make figures    regenerate every figure at published settings
# make all        install, lint, test-all, figures
# make clean      remove build artefacts, caches and generated outputs

PYTHON ?= python
OUTDIR ?= out
FIGDIR ?= figs

.PHONY: all install test test-all lint format figures quick clean

all: install lint test-all figures

install:
	$(PYTHON) -m pip install -e ".[full,dev]"

test:
	$(PYTHON) -m pytest -q

test-all:
	$(PYTHON) -m pytest -q
	$(PYTHON) -m pytest -q -m slow

lint:
	ruff check src tests scripts
	black --check src tests scripts

format:
	ruff check src tests scripts --fix
	black src tests scripts

figures:
	$(PYTHON) scripts/regenerate_figures.py --all --outdir $(FIGDIR)

quick:
	$(PYTHON) scripts/run_all.py --quick --outdir $(OUTDIR)

clean:
	rm -rf build dist .pytest_cache .ruff_cache .coverage htmlcov coverage.xml
	rm -rf src/*.egg-info $(OUTDIR) $(FIGDIR)
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
