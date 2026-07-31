"""Offline integration tests for the Phase 4B matched benchmark."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from eri.agents.provider import ProviderError
from eri.benchmark.phase4b_runner import OUTPUT_FILES, run_phase4b
from eri.benchmark.phase4b_schemas import Phase4BSystem

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/benchmark/phase4b-smoke.yaml"


@pytest.fixture(scope="module")
def run_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("phase4b") / "run"
    result = run_phase4b(
        repo_root=ROOT,
        config_path=CONFIG,
        output_directory=directory,
        run_id="phase4b-test",
    )
    assert result.output_count == result.cell_count * 6
    return directory


def _json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text("utf-8").splitlines()
    ]


@pytest.mark.parametrize(
    "regime",
    ["full-information", "budgeted-observation", "open-world-provenance"],
)
def test_each_primary_regime_smokes(run_directory: Path, regime: str) -> None:
    ledger = _jsonl(run_directory / "ledger.jsonl")
    assert any(row["regime"] == regime for row in ledger)


def test_exact_output_contract(run_directory: Path) -> None:
    assert {item.name for item in run_directory.iterdir()} == set(OUTPUT_FILES)


def test_all_systems_share_each_cell(run_directory: Path) -> None:
    outputs = _jsonl(run_directory / "system_outputs.jsonl")
    cells = _json(run_directory / "cell_manifest.json")
    for cell in cells:
        systems = {
            row["system"] for row in outputs if row["cell_id"] == cell["cell_id"]
        }
        assert systems == {item.value for item in Phase4BSystem}
        assert set(cell["fingerprints"]) == {
            "returns",
            "initial_evidence",
            "prose_metadata",
            "allowed_tools",
            "budget",
            "method_metadata",
            "inference_conventions",
        }
        matched = [row for row in outputs if row["cell_id"] == cell["cell_id"]]
        assert all(row["input_fingerprints"] == cell["fingerprints"] for row in matched)


def test_static_router_cannot_access_hidden_diagnostics(run_directory: Path) -> None:
    outputs = _jsonl(run_directory / "system_outputs.jsonl")
    static = [
        row
        for row in outputs
        if row["system"] == Phase4BSystem.STATIC_DETERMINISTIC.value
    ]
    assert all(not row["diagnostic_requests"] for row in static)
    assert all(not row["acquired_evidence_ids"] for row in static)


def test_adaptive_policy_respects_cell_budget(run_directory: Path) -> None:
    outputs = _jsonl(run_directory / "system_outputs.jsonl")
    cells = {row["cell_id"]: row for row in _json(run_directory / "cell_manifest.json")}
    adaptive = [
        row
        for row in outputs
        if row["system"] == Phase4BSystem.ADAPTIVE_DETERMINISTIC.value
    ]
    assert all(
        len(row["diagnostic_requests"]) <= cells[row["cell_id"]]["diagnostic_budget"]
        for row in adaptive
    )


def test_agent_systems_have_matched_provider_budget(run_directory: Path) -> None:
    providers = _json(run_directory / "provider_manifest.json")
    agent_names = {
        Phase4BSystem.MONOLITHIC_AGENT.value,
        Phase4BSystem.UNCONSTRAINED_AGENTS.value,
        Phase4BSystem.CONSTRAINED_AGENTS.value,
    }
    agent_rows = [row for row in providers if row["system"] in agent_names]
    for cell_id in {row["cell_id"] for row in agent_rows}:
        matched = [row for row in agent_rows if row["cell_id"] == cell_id]
        assert len(matched) == 3
        assert (
            len(
                {
                    (
                        row["model"],
                        row["temperature"],
                        row["maximum_response_tokens"],
                        row["maximum_input_tokens"],
                        row["maximum_model_calls"],
                        row["diagnostic_budget"],
                        row["sensitivity_budget"],
                        row["allowed_tools_fingerprint"],
                    )
                    for row in matched
                }
            )
            == 1
        )


def test_smoke_counts_and_summaries_reconstruct(run_directory: Path) -> None:
    cells = _json(run_directory / "cell_manifest.json")
    outputs = _jsonl(run_directory / "system_outputs.jsonl")
    ledger = _jsonl(run_directory / "ledger.jsonl")
    summary = _json(run_directory / "summary.json")
    assert len(cells) == 6
    assert len(outputs) == len(ledger) == 36
    assert summary["cell_count"] == len(cells)
    assert summary["output_count"] == len(ledger)
    assert len({row["cell_id"] for row in ledger}) == len(cells)
    assert all(float(row["routing_regret"]) >= -1e-12 for row in ledger)
    assert summary["primary_outcomes"] == [
        "invalid_primary_method_selection",
        "harmful_nonabstention",
        "routing_regret",
    ]


def test_regime_and_scenario_summaries_do_not_mix_systems(
    run_directory: Path,
) -> None:
    import csv

    with (run_directory / "metrics_by_regime.csv").open(newline="") as handle:
        regime_rows = list(csv.DictReader(handle))
    with (run_directory / "metrics_by_scenario.csv").open(newline="") as handle:
        scenario_rows = list(csv.DictReader(handle))
    assert len(regime_rows) == len(Phase4BSystem) * 3
    assert len(scenario_rows) == len(Phase4BSystem) * 2
    assert all(row["system"] and row["regime"] for row in regime_rows)


def test_mock_run_cannot_emit_performance_claim(run_directory: Path) -> None:
    summary = _json(run_directory / "summary.json")
    claims = _json(run_directory / "result_claims.json")
    outputs = _jsonl(run_directory / "system_outputs.jsonl")
    assert summary["claim_level"] == "implementation-only"
    assert not summary["scientific_interpretation_permitted"]
    assert all(not row["scientific_agent_reasoning"] for row in outputs)
    assert (
        next(item for item in claims if item["claim_id"] == "agent-performance")[
            "supported"
        ]
        is False
    )


def test_live_provider_requires_explicit_confirmation(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="confirm-live"):
        run_phase4b(
            repo_root=ROOT,
            config_path=ROOT / "configs/benchmark/phase4b-live-pilot.yaml",
            output_directory=tmp_path / "live",
        )


def test_confirmed_live_run_requires_injected_provider(tmp_path: Path) -> None:
    with pytest.raises(ProviderError, match="injected typed provider"):
        run_phase4b(
            repo_root=ROOT,
            config_path=ROOT / "configs/benchmark/phase4b-live-pilot.yaml",
            output_directory=tmp_path / "live",
            confirm_live=True,
        )


def test_generated_runs_are_gitignored() -> None:
    completed = subprocess.run(
        ["git", "check-ignore", "runs/phase4b/example/summary.json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip().startswith("runs/")


def test_personal_obsidian_paths_are_outside_benchmark_outputs(
    run_directory: Path,
) -> None:
    assert "inference-obsidian" not in str(run_directory)
    assert all(
        "mynotes" not in item.read_text("utf-8") for item in run_directory.iterdir()
    )
