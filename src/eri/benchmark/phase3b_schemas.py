"""Typed contracts for Phase 3B benchmark validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, FiniteFloat, JsonValue, model_validator

from eri.benchmark.schemas import BenchmarkModel

EvaluationNamespace = Literal[
    "development-evaluation",
    "hackathon-evaluation",
    "aaai-evaluation",
]
ClaimLevel = Literal[
    "smoke-only",
    "preliminary-descriptive",
    "benchmark-supported",
    "confirmation-supported",
]


class SeedNamespaceConfig(BenchmarkModel):
    """Versioned, disjoint deterministic seed namespace allocation."""

    version: str = Field(min_length=1)
    namespace_width: int = Field(ge=10_000)
    oracle_training_start: int = Field(ge=0)
    development_evaluation_start: int = Field(ge=0)
    hackathon_evaluation_start: int = Field(ge=0)
    aaai_evaluation_start: int = Field(ge=0)
    frozen_confirmation_start: int = Field(ge=0)
    development_evaluation_count: int = Field(ge=1)
    hackathon_evaluation_count: int = Field(ge=1)
    aaai_evaluation_count: int = Field(ge=1)
    frozen_confirmation_count: int = Field(ge=1)

    @model_validator(mode="after")
    def disjoint_ranges(self) -> Self:
        starts = self.starts()
        ordered = sorted(starts.items(), key=lambda item: item[1])
        for (_, left), (_, right) in zip(ordered, ordered[1:], strict=False):
            if right - left < self.namespace_width:
                raise ValueError("scientific seed namespaces must not overlap")
        counts = (
            self.development_evaluation_count,
            self.hackathon_evaluation_count,
            self.aaai_evaluation_count,
            self.frozen_confirmation_count,
        )
        if max(counts) > self.namespace_width:
            raise ValueError("seed count exceeds namespace width")
        return self

    def starts(self) -> dict[str, int]:
        return {
            "oracle-training": self.oracle_training_start,
            "development-evaluation": self.development_evaluation_start,
            "hackathon-evaluation": self.hackathon_evaluation_start,
            "aaai-evaluation": self.aaai_evaluation_start,
            "frozen-confirmation": self.frozen_confirmation_start,
        }


class Phase3BConfig(BenchmarkModel):
    """Configuration for one two-profile Phase 3B run tier."""

    benchmark_version: str = Field(min_length=1)
    run_label: str = Field(min_length=1)
    output_root: Path = Path("runs/phase3b")
    scenario_source: Path
    loss_profiles_source: Path = Path("configs/benchmark/loss-profiles.yaml")
    loss_profiles: tuple[str, ...] = Field(min_length=2)
    confidence_level: FiniteFloat = Field(gt=0.0, lt=1.0)
    hac_bandwidth: int = Field(ge=0)
    bootstrap_block_length: int = Field(ge=1)
    bootstrap_replications: int = Field(ge=2)
    bootstrap_seed: int
    oracle_training_draws: int = Field(ge=1)
    evaluation_draws: int = Field(ge=1)
    evaluation_namespace: EvaluationNamespace
    oracle_stability_sizes: tuple[int, ...] = Field(min_length=1)
    oracle_stability_blocks: int = Field(ge=2)
    oracle_weak_margin: FiniteFloat = Field(ge=0.0)
    seed_namespaces: SeedNamespaceConfig
    generate_figures: bool = True
    independent_review_complete: bool = False
    max_projected_hackathon_seconds: FiniteFloat = Field(gt=0.0)
    max_projected_hackathon_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_design(self) -> Self:
        if set(self.loss_profiles) != {"validity-first", "efficiency-balanced"}:
            raise ValueError("Phase 3B must retain exactly both registered profiles")
        sizes = self.oracle_stability_sizes
        if tuple(sorted(set(sizes))) != sizes:
            raise ValueError("oracle stability sizes must be unique and increasing")
        if sizes[-1] > self.seed_namespaces.namespace_width:
            raise ValueError("oracle stability size exceeds its namespace")
        required_oracle_seeds = sizes[-1] * self.oracle_stability_blocks
        if required_oracle_seeds > self.seed_namespaces.namespace_width:
            raise ValueError("oracle stability blocks exceed the seed namespace")
        if self.oracle_training_draws > sizes[-1]:
            raise ValueError("final oracle training exceeds stability support")
        expected_evaluation = {
            "development-evaluation": (
                self.seed_namespaces.development_evaluation_count
            ),
            "hackathon-evaluation": self.seed_namespaces.hackathon_evaluation_count,
            "aaai-evaluation": self.seed_namespaces.aaai_evaluation_count,
        }[self.evaluation_namespace]
        if self.evaluation_draws != expected_evaluation:
            raise ValueError("evaluation draw count must match its seed namespace")
        return self


class SeedManifest(BenchmarkModel):
    namespace_version: str
    namespace_width: int
    oracle_training: tuple[int, ...]
    oracle_stability_blocks: tuple[tuple[int, ...], ...]
    development_evaluation: tuple[int, ...]
    hackathon_evaluation: tuple[int, ...]
    aaai_evaluation: tuple[int, ...]
    frozen_confirmation: tuple[int, ...]
    active_evaluation_namespace: str
    active_evaluation: tuple[int, ...]

    @model_validator(mode="after")
    def all_namespaces_disjoint(self) -> Self:
        oracle: set[int] = set()
        for block in self.oracle_stability_blocks:
            oracle.update(block)
        named: dict[str, set[int]] = {
            "oracle": oracle,
            "development": set(self.development_evaluation),
            "hackathon": set(self.hackathon_evaluation),
            "aaai": set(self.aaai_evaluation),
            "confirmation": set(self.frozen_confirmation),
        }
        names = tuple(named)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                if named[left] & named[right]:
                    raise ValueError(f"seed namespaces overlap: {left} and {right}")
        if not set(self.oracle_training) <= named["oracle"]:
            raise ValueError("final oracle seeds must be within oracle namespace")
        return self


class IntervalEstimate(BenchmarkModel):
    estimate: FiniteFloat
    lower: FiniteFloat
    upper: FiniteFloat
    standard_error: FiniteFloat | None = Field(default=None, ge=0.0)
    confidence_level: FiniteFloat = Field(gt=0.0, lt=1.0)
    method: str
    numerator: int | None = Field(default=None, ge=0)
    denominator: int | None = Field(default=None, ge=0)
    clusters: int | None = Field(default=None, ge=0)


class ConfirmationValidation(BenchmarkModel):
    valid: bool
    manifest_hash: str | None = None
    changed_paths: tuple[str, ...] = ()
    reason: str


class Phase3BRunResult(BenchmarkModel):
    profile: str
    run_id: str
    run_directory: Path
    record_count: int
    permitted_claim_level: ClaimLevel
    summary: dict[str, JsonValue]
