"""Versioned benchmark truth labels independent of router output."""

from __future__ import annotations

from eri.benchmark.schemas import (
    EligibilityLabel,
    EligibilityStatus,
    LabelBasis,
    ProcessFamily,
    ScenarioConfig,
    SimulationResult,
)

METHOD_IDS: tuple[str, ...] = (
    "iid-gaussian-sharpe",
    "mertens-psr",
    "hac-sharpe",
    "circular-block-bootstrap",
    "abstention",
)
REGISTRY_VERSION = "phase3a-eligibility-1.0"


def eligibility_for(simulation: SimulationResult) -> tuple[EligibilityLabel, ...]:
    """Return truth labels without using any diagnostic or router output."""
    family = simulation.process_family
    raw_ar_phi = simulation.parameters.get("phi", 0.0)
    ar_phi = (
        float(raw_ar_phi)
        if family is ProcessFamily.AR1 and isinstance(raw_ar_phi, int | float)
        else None
    )
    if family is ProcessFamily.IID_GAUSSIAN or ar_phi == 0.0:
        return (
            _label(
                "iid-gaussian-sharpe",
                EligibilityStatus.ELIGIBLE,
                "IID Gaussian assumptions hold.",
                ("IID Gaussian returns",),
            ),
            _label(
                "mertens-psr",
                EligibilityStatus.ELIGIBLE,
                "IID finite-moment assumptions hold.",
                ("IID returns", "finite fourth moment"),
            ),
            _label(
                "hac-sharpe",
                EligibilityStatus.ELIGIBLE,
                "Stationary IID data satisfy weak-dependence conditions.",
                ("stationarity",),
            ),
            _label(
                "circular-block-bootstrap",
                EligibilityStatus.ELIGIBLE,
                "IID data are a valid special case for circular blocks.",
                ("stationarity",),
            ),
            _label(
                "abstention",
                EligibilityStatus.SENSITIVITY_ONLY,
                "Abstention is policy-available but unnecessary when data "
                "are adequate.",
                (),
                basis=LabelBasis.POLICY,
            ),
        )
    if family in (ProcessFamily.AR1, ProcessFamily.GARCH):
        dependence = (
            "linear dependence"
            if family is ProcessFamily.AR1
            else "squared-return dependence"
        )
        finite_fourth_moment = bool(
            simulation.inference_truth.get("finite_fourth_moment", True)
        )
        robust_status = (
            EligibilityStatus.ELIGIBLE
            if finite_fourth_moment
            else EligibilityStatus.INELIGIBLE
        )
        robust_reason = (
            "Stable weak dependence and finite fourth moment are within the "
            "registered {method} regime."
            if finite_fourth_moment
            else "The configured GARCH process lacks the finite fourth moment "
            "required by the registered {method} regime."
        )
        return (
            _label(
                "iid-gaussian-sharpe",
                EligibilityStatus.INELIGIBLE,
                f"{dependence.capitalize()} violates temporal IID inference.",
                ("temporal IID returns",),
            ),
            _label(
                "mertens-psr",
                EligibilityStatus.INELIGIBLE,
                f"{dependence.capitalize()} violates the IID basis of Mertens/PSR.",
                ("temporal IID returns",),
            ),
            _label(
                "hac-sharpe",
                robust_status,
                robust_reason.format(method="HAC target"),
                ("weak dependence", "stationarity", "finite fourth moment"),
            ),
            _label(
                "circular-block-bootstrap",
                robust_status,
                robust_reason.format(method="block-bootstrap"),
                ("stationarity", "finite fourth moment", "adequate block length"),
            ),
            _label(
                "abstention",
                (
                    EligibilityStatus.SENSITIVITY_ONLY
                    if finite_fourth_moment
                    else EligibilityStatus.ELIGIBLE
                ),
                (
                    "Abstention is policy-available when finite-sample evidence "
                    "is inadequate."
                    if finite_fourth_moment
                    else "Abstention protects a regime with no registered "
                    "primary-valid inference method."
                ),
                (),
                basis=LabelBasis.POLICY,
            ),
        )
    return (
        *(
            _label(
                method_id,
                EligibilityStatus.TARGET_NOT_WELL_DEFINED,
                "No single stable full-sample population Sharpe is defined.",
                ("stable full-sample target",),
            )
            for method_id in METHOD_IDS[:-1]
        ),
        _label(
            "abstention",
            EligibilityStatus.ELIGIBLE,
            "Abstention protects an undefined stationary full-sample target.",
            (),
            basis=LabelBasis.POLICY,
        ),
    )


def label_for(labels: tuple[EligibilityLabel, ...], method_id: str) -> EligibilityLabel:
    for label in labels:
        if label.method_id == method_id:
            return label
    raise KeyError(method_id)


def eligible_method_ids(labels: tuple[EligibilityLabel, ...]) -> tuple[str, ...]:
    return tuple(
        label.method_id
        for label in labels
        if label.method_id != "abstention"
        and label.status is EligibilityStatus.ELIGIBLE
    )


def generate_scenario(scenario: ScenarioConfig, *, seed: int) -> SimulationResult:
    """Dispatch one of the four fixed Phase 3A simulation families."""
    from eri.simulation.ar1 import simulate_ar1
    from eri.simulation.garch import simulate_garch
    from eri.simulation.iid_gaussian import simulate_iid_gaussian
    from eri.simulation.structural_break import simulate_structural_break

    generator = {
        ProcessFamily.IID_GAUSSIAN: simulate_iid_gaussian,
        ProcessFamily.AR1: simulate_ar1,
        ProcessFamily.GARCH: simulate_garch,
        ProcessFamily.STRUCTURAL_BREAK: simulate_structural_break,
    }[scenario.family]
    return generator(
        sample_size=scenario.sample_size,
        seed=seed,
        parameters=scenario.parameters,
    )


def _label(
    method_id: str,
    status: EligibilityStatus,
    reason: str,
    assumptions: tuple[str, ...],
    *,
    basis: LabelBasis = LabelBasis.ANALYTICAL,
) -> EligibilityLabel:
    return EligibilityLabel(
        method_id=method_id,
        status=status,
        reason=reason,
        assumptions=assumptions,
        version=REGISTRY_VERSION,
        basis=basis,
    )
