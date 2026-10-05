UV := uv
CONFIG := index

.PHONY: install run debug clean lint lint-strict

install:
	$(UV) sync

run:
	@$(UV) run python -m src $(CONFIG)

debug:
	$(UV) run python -m pdb -m src $(CONFIG)

clean:
	rm -rf __pycache__ .mypy_cache
	rm -rf dist build *.egg-info
	find . -name "*.pyc" -delete

lint:
	$(UV) run flake8 .
	$(UV) run mypy . --warn-return-any \
	--warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict
