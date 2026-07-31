"""Summary helpers for Phase 4B run artifacts."""

from __future__ import annotations

from collections.abc import Sequence

from eri.benchmark.phase4b_schemas import ClaimLevel


def summary_markdown(summary: dict[str, object]) -> str:
    return (
        "# Phase 4B Run Summary\n\n"
        f"- Tier: `{summary['tier']}`\n"
        f"- Claim level: `{summary['claim_level']}`\n"
        f"- Matched cells: {summary['cell_count']}\n"
        f"- System outputs: {summary['output_count']}\n"
        f"- Live model used: {str(summary['live_model_used']).lower()}\n\n"
        "Mock outputs validate implementation only and cannot support "
        "agent-performance claims.\n"
    )


def result_claims(
    *, claim_level: ClaimLevel, live_model_used: bool
) -> Sequence[dict[str, object]]:
    return (
        {
            "claim_id": "phase4b-implementation",
            "claim": "The benchmark contracts and matched-cell runner executed.",
            "level": ClaimLevel.IMPLEMENTATION_ONLY.value,
            "supported": True,
        },
        {
            "claim_id": "agent-performance",
            "claim": "An agent system improves procedure selection.",
            "level": claim_level.value,
            "supported": False,
            "reason": (
                "No reviewed live scientific run."
                if not live_model_used
                else "Requires paired analysis and independent review."
            ),
        },
    )
