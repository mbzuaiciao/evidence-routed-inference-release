# Selected release configurations

This snapshot includes deterministic, mock-provider, and scientific benchmark
configurations needed by the non-live tests and examples. Private provider,
workspace, authorization, and live-execution-control records are excluded.

The files named `live` or `deadline` describe historical benchmark interfaces
and are retained because typed validation tests cover them. They contain no
credentials or concrete network client and are not invoked by `make check`.
