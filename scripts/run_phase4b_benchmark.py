"""Run one explicitly configured Phase 4B benchmark tier."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import cast

from eri.agents.provider import TypedProvider
from eri.benchmark.phase4b_runner import run_phase4b


def load_provider_factory(specification: str) -> TypedProvider:
    """Load an explicitly supplied zero-argument typed-provider factory."""
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("provider factory must use module:function syntax")
    factory_object: object = getattr(import_module(module_name), attribute)
    if not callable(factory_object):
        raise TypeError("provider factory target must be callable")
    factory = cast(Callable[[], TypedProvider], factory_object)
    return factory()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--confirm-live", action="store_true")
    parser.add_argument(
        "--provider-factory",
        default="",
        help="explicit module:function returning a TypedProvider",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    provider = (
        load_provider_factory(args.provider_factory) if args.provider_factory else None
    )
    result = run_phase4b(
        repo_root=root,
        config_path=args.config,
        confirm_live=args.confirm_live,
        provider=provider,
    )
    print(result.run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
