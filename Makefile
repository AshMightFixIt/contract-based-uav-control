# Test entry points. Set up an environment first, from the repository root:
#   python -m pip install -e ./contract_uav_core -r requirements-test.txt                             (make test)
#   python -m pip install -e ./contract_uav_core -r requirements-test.txt -r requirements-legacy.txt  (make test-pacti)
# The tests check that contract_uav_core resolves to this checkout (editable install).
# Every target exits non-zero when a test fails. The pacti targets (test-pacti,
# golden-update-pacti, test-all) also exit non-zero when pacti is missing, so
# they cannot pass by skipping the reproduction.

PYTHON ?= python3
PYTEST := $(PYTHON) -m pytest
PYTEST_ARGS ?=

# Exits 1 with a message on stderr when $(PYTHON) cannot find pacti.
REQUIRE_PACTI = @$(PYTHON) -c "import importlib.util, sys; importlib.util.find_spec('pacti') or sys.exit('ERROR: pacti is not installed for $(PYTHON); this target needs it. Run: $(PYTHON) -m pip install -r requirements-test.txt -r requirements-legacy.txt')"

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
	@echo "make test-all             everything, slow tests included (needs pacti)"
	@echo "Options: PYTHON=<interpreter> (default python3), PYTEST_ARGS=<extra pytest args>"

test:
	$(PYTEST) $(PYTEST_ARGS)

golden:
	$(PYTEST) -m "golden and not slow" $(PYTEST_ARGS)

golden-update:
	$(PYTEST) -m "golden and not slow" --update-goldens $(PYTEST_ARGS)

test-pacti:
	$(REQUIRE_PACTI)
	$(PYTEST) -m slow $(PYTEST_ARGS)

golden-update-pacti:
	$(REQUIRE_PACTI)
	$(PYTEST) -m "golden and slow" --update-goldens $(PYTEST_ARGS)

test-all:
	$(REQUIRE_PACTI)
	$(PYTEST) -m "slow or not slow" $(PYTEST_ARGS)
