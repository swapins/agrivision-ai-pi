.PHONY: check fetch-models run sim test

test:
	pytest -q

check:
	python -m py_compile scripts/fetch_models.py scripts/coral_smoke_test.py scripts/hardware_selftest.py
	pytest -q

fetch-models:
	python scripts/fetch_models.py

sim:
	AGRIVISION_SIMULATION=1 PYTHONPATH=src python -m agrivision.app

run:
	PYTHONPATH=src python -m agrivision.app
