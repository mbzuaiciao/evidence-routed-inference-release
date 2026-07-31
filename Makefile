.PHONY: install test lint typecheck verify-artifacts benchmark-smoke paper supplement checklist check

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run pyright

verify-artifacts:
	uv run python scripts/verify_release_artifacts.py

benchmark-smoke:
	uv run python scripts/run_phase3a_benchmark.py \
		--config configs/benchmark/phase3a-smoke.yaml \
		--output-directory /tmp/eri-release-phase3a-smoke

paper:
	$(MAKE) -C paper/aaai_submission_short main

supplement:
	$(MAKE) -C paper/aaai_submission_short supplement

checklist:
	$(MAKE) -C paper/aaai_submission_short checklist

check: test lint typecheck verify-artifacts
