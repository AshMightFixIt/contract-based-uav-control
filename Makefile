# Test entry points. Set up an environment first:
#   python -m pip install -r requirements-test.txt                      (make test)
#   python -m pip install -r requirements-test.txt -r requirements-legacy.txt  (make test-pacti)
# Every target exits non-zero when a test fails.

PYTHON ?= python3
PYTEST := $(PYTHON) -m pytest
PYTEST_ARGS ?=

# Headless, and no bytecode or cache files in the working tree.
export MPLBACKEND := Agg
export SDL_VIDEODRIVER := dummy
export SDL_AUDIODRIVER := dummy
export PYTHONDONTWRITEBYTECODE := 1

.DEFAULT_GOAL := help
.PHONY: help test golden golden-update test-pacti golden-update-pacti test-all

help:
	@echo "make test                 default suite: module smoke tests + requirements-only goldens (~1 min)"
	@echo "make golden               only the requirements-only golden tests"
	@echo "make golden-update        rewrite the requirements-only goldens (reviewed behaviour change only)"
	@echo "make test-pacti           slow opt-in: pacti reproduction of RESULTS.md section 1 (~9 min)"
	@echo "make golden-update-pacti  rewrite the pacti golden (reviewed behaviour change only)"
	@echo "make test-all             everything, slow tests included"
	@echo "Options: PYTHON=<interpreter> (default python3), PYTEST_ARGS=<extra pytest args>"

test:
	$(PYTEST) $(PYTEST_ARGS)

golden:
	$(PYTEST) -m "golden and not slow" $(PYTEST_ARGS)

golden-update:
	$(PYTEST) -m "golden and not slow" --update-goldens $(PYTEST_ARGS)

test-pacti:
	$(PYTEST) -m slow $(PYTEST_ARGS)

golden-update-pacti:
	$(PYTEST) -m "golden and slow" --update-goldens $(PYTEST_ARGS)

test-all:
	$(PYTEST) -m "slow or not slow" $(PYTEST_ARGS)
