.PHONY: test sim run

test:
	pytest -q

sim:
	AGRIVISION_SIMULATION=1 PYTHONPATH=src python -m agrivision.app

run:
	PYTHONPATH=src python -m agrivision.app
