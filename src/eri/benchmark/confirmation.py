"""Freeze and validate Phase 3B confirmation-critical state."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from eri.benchmark.phase3b_schemas import ConfirmationValidation

FREEZE_VERSION = "phase3b-confirmation-freeze-1.0"


def critical_paths(repo_root: Path, config_path: Path) -> tuple[Path, ...]:
    """Return exact policy, configuration, simulation, and benchmark sources."""
    root = repo_root.resolve()
    candidates = [
        config_path.resolve(),
        root / "configs/benchmark/phase3a-mini.yaml",
        root / "configs/benchmark/loss-profiles.yaml",
        root / "configs/diagnostics/default.yaml",
        root / "configs/routing/conservative.yaml",
        root / "configs/experiments/phase2-smoke.yaml",
        root / "src/eri/routing/router.py",
    ]
    candidates.extend(sorted((root / "src/eri/benchmark").glob("*.py")))
    candidates.extend(sorted((root / "src/eri/simulation").glob("*.py")))
    resolved = tuple(dict.fromkeys(path.resolve() for path in candidates))
    if any(not path.is_file() or not path.is_relative_to(root) for path in resolved):
        raise ValueError("confirmation critical path is missing or outside repository")
    return resolved


def create_freeze_manifest(
    repo_root: Path, config_path: Path, output_path: Path
) -> dict[str, Any]:
    """Write a deterministic critical-state hash before confirmation seeds run."""
    root = repo_root.resolve()
    files = {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in critical_paths(root, config_path)
    }
    critical = {
        "freeze_version": FREEZE_VERSION,
        "configuration": config_path.resolve().relative_to(root).as_posix(),
        "files": files,
        "confirmation_seed_policy": (
            "frozen-confirmation namespace only; seeds unseen before freeze"
        ),
    }
    manifest_hash = _canonical_hash(critical)
    manifest = {
        **critical,
        "manifest_hash": manifest_hash,
        "created_at": datetime.now(UTC).isoformat(),
        "confirmation_executed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_freeze_manifest(
    repo_root: Path,
    manifest_path: Path,
    *,
    expected_config_path: Path | None = None,
) -> ConfirmationValidation:
    """Reject confirmation when any frozen critical input changed."""
    root = repo_root.resolve()
    raw: object = json.loads(manifest_path.read_text("utf-8"))
    if not isinstance(raw, dict):
        return ConfirmationValidation(valid=False, reason="manifest is not an object")
    manifest = cast(dict[str, object], raw)
    files_raw = manifest.get("files")
    expected_hash = manifest.get("manifest_hash")
    configuration = manifest.get("configuration")
    freeze_version = manifest.get("freeze_version")
    seed_policy = manifest.get("confirmation_seed_policy")
    if (
        not isinstance(files_raw, dict)
        or not isinstance(expected_hash, str)
        or not isinstance(configuration, str)
        or freeze_version != FREEZE_VERSION
        or not isinstance(seed_policy, str)
    ):
        return ConfirmationValidation(valid=False, reason="manifest fields are missing")
    if expected_config_path is not None:
        expected_config = expected_config_path.resolve()
        if (
            not expected_config.is_relative_to(root)
            or expected_config.relative_to(root).as_posix() != configuration
        ):
            return ConfirmationValidation(
                valid=False,
                manifest_hash=expected_hash,
                reason="manifest is frozen for a different benchmark configuration",
            )
    files = cast(dict[object, object], files_raw)
    critical: dict[str, object] = {
        key: manifest[key]
        for key in (
            "freeze_version",
            "configuration",
            "files",
            "confirmation_seed_policy",
        )
        if key in manifest
    }
    if _canonical_hash(critical) != expected_hash:
        return ConfirmationValidation(
            valid=False,
            manifest_hash=expected_hash,
            reason="manifest hash does not match its recorded contents",
        )
    changed: list[str] = []
    for relative, digest in files.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            changed.append(str(relative))
            continue
        path = (root / relative).resolve()
        if (
            not path.is_file()
            or not path.is_relative_to(root)
            or _sha256_file(path) != digest
        ):
            changed.append(relative)
    return ConfirmationValidation(
        valid=not changed,
        manifest_hash=expected_hash,
        changed_paths=tuple(sorted(changed)),
        reason=(
            "frozen critical state is unchanged"
            if not changed
            else "critical state changed after freeze"
        ),
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
