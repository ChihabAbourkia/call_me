NAME = src

install:
	@uv sync

run:
	@uv run python -m $(NAME)

debug:
	@uv run python -m pdb -m $(NAME)

clean:
	@rm -rf */__pycache__ */.mypy_cache .mypy_cache __pycache__ data/output

lint:
	flake8 $(NAME)
	mypy $(NAME) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 $(NAME)
	mypy $(NAME) --strict
