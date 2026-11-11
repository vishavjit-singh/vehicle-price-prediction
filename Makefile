PY ?= python3
export PYTHONPATH := src

.PHONY: install train report test lint all

install:
	$(PY) -m pip install -r requirements-dev.txt

train:
	$(PY) -m vehicle_price train

report:
	$(PY) -m vehicle_price report

all: train report

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests
