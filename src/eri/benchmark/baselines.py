"""Fixed, routed, and oracle benchmark execution adapters."""

from __future__ import annotations

from math import isfinite
from time import perf_counter

from eri.benchmark.schemas import (
    BaselineName,
    BenchmarkConfig,
    MethodExecution,
    SimulationResult,
)
from eri.config import Phase2Config
from eri.inference.block_bootstrap import circular_block_bootstrap
from eri.inference.common import normal_quantile
from eri.inference.gaussian import gaussian_sharpe_confidence_interval
from eri.inference.hac import hac_sharpe_standard_error
from eri.inference.mertens_psr import probabilistic_sharpe_ratio
from eri.schemas.inference import InferenceResult
from eri.workflows.deterministic import run_deterministic_workflow


def run_baseline(
    baseline: BaselineName,
    simulation: SimulationResult,
    phase2_config: Phase2Config,
    benchmark_config: BenchmarkConfig,
    *,
    oracle_method: str,
) -> MethodExecution:
    """Evaluate one baseline on a shared simulated draw."""
    if baseline is BaselineName.DETERMINISTIC_ROUTER:
        return _run_router(simulation, phase2_config, benchmark_config)
    method_id = {
        BaselineName.ALWAYS_GAUSSIAN: "iid-gaussian-sharpe",
        BaselineName.ALWAYS_HAC: "hac-sharpe",
        BaselineName.ORACLE: oracle_method,
    }[baseline]
    if method_id == "abstention":
        return abstention_execution()
    return execute_method(method_id, simulation, benchmark_config)


def execute_method(
    method_id: str,
    simulation: SimulationResult,
    config: BenchmarkConfig,
) -> MethodExecution:
    """Execute an existing Phase 1 method and normalize its interval output."""
    started = perf_counter()
    try:
        estimate, interval, warnings = _method_values(method_id, simulation, config)
        runtime = perf_counter() - started
        return MethodExecution(
            selected_method=method_id,
            abstained=False,
            estimate=estimate,
            confidence_interval=interval,
            routing_runtime_seconds=0.0,
            inference_runtime_seconds=runtime,
            warnings=warnings,
        )
    except (ArithmeticError, ValueError) as error:
        runtime = perf_counter() - started
        return MethodExecution(
            selected_method=method_id,
            abstained=False,
            execution_failure=f"{type(error).__name__}: {error}",
            routing_runtime_seconds=0.0,
            inference_runtime_seconds=runtime,
        )


def abstention_execution(
    *, reason: str = "Oracle registered abstention."
) -> MethodExecution:
    return MethodExecution(
        selected_method=None,
        abstained=True,
        routing_runtime_seconds=0.0,
        inference_runtime_seconds=0.0,
        warnings=(reason,),
    )


def _method_values(
    method_id: str,
    simulation: SimulationResult,
    config: BenchmarkConfig,
) -> tuple[float, tuple[float, float], tuple[str, ...]]:
    returns = simulation.returns
    if method_id == "iid-gaussian-sharpe":
        result = gaussian_sharpe_confidence_interval(
            returns, confidence_level=config.confidence_level
        )
        return result.estimate, (result.lower, result.upper), result.warnings
    if method_id == "mertens-psr":
        result = probabilistic_sharpe_ratio(returns, benchmark=0.0)
        z_value = normal_quantile((1.0 + config.confidence_level) / 2.0)
        half_width = z_value * result.standard_error
        return (
            result.sample_sharpe,
            (result.sample_sharpe - half_width, result.sample_sharpe + half_width),
            (
                *result.warnings,
                "Benchmark CI applies the existing Mertens standard error "
                "with a normal quantile.",
            ),
        )
    if method_id == "hac-sharpe":
        result = hac_sharpe_standard_error(returns, bandwidth=config.hac_bandwidth)
        z_value = normal_quantile((1.0 + config.confidence_level) / 2.0)
        half_width = z_value * result.standard_error
        return (
            result.estimate,
            (result.estimate - half_width, result.estimate + half_width),
            result.warnings,
        )
    if method_id == "circular-block-bootstrap":
        result = circular_block_bootstrap(
            returns,
            block_length=config.bootstrap_block_length,
            replications=config.bootstrap_replications,
            seed=config.bootstrap_seed + simulation.seed,
            confidence_level=config.confidence_level,
        )
        return result.estimate, (result.lower, result.upper), result.warnings
    if method_id == "abstention":
        raise ValueError("abstention is an action, not an inference method")
    raise KeyError(method_id)


def _run_router(
    simulation: SimulationResult,
    phase2_config: Phase2Config,
    benchmark_config: BenchmarkConfig,
) -> MethodExecution:
    experiment = phase2_config.experiment.model_copy(
        update={
            "confidence_level": benchmark_config.confidence_level,
            "hac_bandwidth": benchmark_config.hac_bandwidth,
            "bootstrap_block_length": benchmark_config.bootstrap_block_length,
            "bootstrap_replications": benchmark_config.bootstrap_replications,
            "bootstrap_seed": benchmark_config.bootstrap_seed + simulation.seed,
        }
    )
    configured = phase2_config.model_copy(update={"experiment": experiment})
    started = perf_counter()
    try:
        result = run_deterministic_workflow(simulation.returns, configured)
    except (ArithmeticError, ValueError) as error:
        runtime = perf_counter() - started
        return MethodExecution(
            selected_method=None,
            abstained=False,
            execution_failure=f"{type(error).__name__}: {error}",
            routing_runtime_seconds=runtime,
            inference_runtime_seconds=0.0,
        )
    runtime = perf_counter() - started
    decision = result.decision
    if decision.selected_method is None:
        reasons = decision.abstention.reasons if decision.abstention else ()
        return MethodExecution(
            selected_method=None,
            sensitivity_methods=decision.sensitivity_methods,
            abstained=True,
            evidence_ids=tuple(item.evidence_id for item in result.evidence),
            unresolved_conflicts=decision.unresolved_conflicts,
            routing_runtime_seconds=runtime,
            inference_runtime_seconds=0.0,
            warnings=(*decision.warnings, *reasons),
        )
    selected = next(
        item
        for item in result.inference_results
        if item.method_id == decision.selected_method
    )
    interval = _interval_from_workflow(selected, benchmark_config.confidence_level)
    if not isfinite(float(selected.estimate)):
        raise ArithmeticError("router produced a non-finite estimate")
    return MethodExecution(
        selected_method=decision.selected_method,
        sensitivity_methods=decision.sensitivity_methods,
        abstained=False,
        evidence_ids=tuple(item.evidence_id for item in result.evidence),
        unresolved_conflicts=decision.unresolved_conflicts,
        confidence_interval=interval,
        estimate=float(selected.estimate),
        routing_runtime_seconds=runtime,
        inference_runtime_seconds=0.0,
        warnings=(*decision.warnings, *selected.warnings),
    )


def _interval_from_workflow(
    result: InferenceResult, confidence_level: float
) -> tuple[float, float] | None:
    if result.confidence_interval is not None:
        return (
            float(result.confidence_interval[0]),
            float(result.confidence_interval[1]),
        )
    if result.standard_error is None:
        return None
    z_value = normal_quantile((1.0 + confidence_level) / 2.0)
    half_width = z_value * float(result.standard_error)
    estimate = float(result.estimate)
    return estimate - half_width, estimate + half_width
