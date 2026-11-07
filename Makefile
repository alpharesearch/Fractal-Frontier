# Makefile for Fractal Frontier
#
# Uses a miniconda environment for the project. The default interpreter is
# the system python3, which is often too new for numba (needs 3.10-3.12);
# conda gives the project its own pinned interpreter. Override name and
# Python version on the command line, e.g.:
#
#   make conda-env                          # create env + install deps
#   make test                               # uses env 'fractal-frontier'
#   make test CONDA_ENV=bigcodebench        # use an existing env
#   make conda-env CONDA_ENV=ff CONDA_PY=3.11
#
# If conda is not installed, every target falls back to python3/venv.

CONDA_ENV ?= fractal-frontier
CONDA_PY  ?= 3.12

CONDA := $(shell command -v conda 2>/dev/null)
# Interpreter inside the conda env; empty when conda or the env is missing.
CONDA_PY_PATH := $(if $(CONDA),$(shell $(CONDA) run -n $(CONDA_ENV) python -c "import sys; print(sys.executable)" 2>/dev/null))
PY := $(if $(CONDA_PY_PATH),$(CONDA_PY_PATH),python3)

.PHONY: help conda-env env-check install lint format test check clean run

help:
	@echo "Available targets:" \
	&& echo "  conda-env  - Create miniconda env '$(CONDA_ENV)' (python=$(CONDA_PY)) and install deps" \
	&& echo "  install    - Install all dependencies (runtime & dev) into the active env" \
	&& echo "  env-check  - Show interpreter and runtime deps the targets will use" \
	&& echo "  lint       - Run flake8 linting" \
	&& echo "  format     - Auto-format code with black and isort" \
	&& echo "  test       - Run pytest unit tests" \
	&& echo "  run        - Start the application" \
	&& echo "  check      - Run format + lint" \
	&& echo "  clean      - Remove build artifacts (pycache, .coverage)" \
	&& echo "" \
	&& echo "Interpreter in use: $(PY)"

# Create the project conda env (idempotent) and install requirements into it.
conda-env:
	@if [ -z "$(CONDA)" ]; then \
		echo "conda not found - install miniconda, or use a venv:"; \
		echo "  python3 -m venv .venv && source .venv/bin/activate && make install"; \
		exit 1; \
	fi
	@if $(CONDA) run -n $(CONDA_ENV) true >/dev/null 2>&1; then \
		echo "conda env '$(CONDA_ENV)' already exists"; \
	else \
		$(CONDA) create -y -n $(CONDA_ENV) python=$(CONDA_PY); \
	fi
	@$(MAKE) --no-print-directory install

env-check:
	@$(PY) -c "import sys; print('interpreter:', sys.version.split()[0], '->', sys.executable)"
	@$(PY) -c "import numpy, numba, PIL; print('runtime deps ok:', 'numpy', numpy.__version__, '| numba', numba.__version__)"

install:
	$(PY) -m pip install -r requirements.txt

lint:
	$(PY) -m flake8 .

format:
	# Black and isort handle most formatting
	$(PY) -m black .
	$(PY) -m isort .

test:
	$(PY) -m pytest

check:
	$(MAKE) format
	$(MAKE) lint

run:
	$(PY) Fractal_Frontier.py

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} + ; \
	find . -type f -name '*.pyc' -delete ; \
	find . -type f -name '.coverage' -delete
