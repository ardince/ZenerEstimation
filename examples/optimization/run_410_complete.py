"""Run the complete dataset-410 evidence workflow.

This orchestrator deliberately keeps two scientific stages separate:

1. Frozen benchmark:
   development through 2023-12-01, evaluated on measured 2024-03-01
   through 2025-06-01 holdout observations.

2. Operational extrapolation:
   refit the same frozen configurations through 2025-06-01 without
   reoptimization, then forecast six unvalidated quarters through 2026-12-01.

Run from repository root:
    python -m examples.optimization.run_410_complete
"""
from __future__ import annotations

from pathlib import Path
from zenerestimation.visualization.benchmark_to_operational_transition import (
    BenchmarkToOperationalTransitionPlotter,
)

from examples.optimization import run_410_optimized_benchmark as benchmark_run
from examples.optimization import run_410_operational_extrapolation as operational_run


def main() -> None:
    """Generate all accepted 410 benchmark and operational artifacts."""
    print()
    print("=" * 78)
    print("DATASET 410 — COMPLETE EVIDENCE RUN")
    print("=" * 78)

    print()
    print("[STAGE 1/2] Frozen measured-holdout benchmark")
    print(
        "Training boundary: "
        f"{benchmark_run.DEVELOPMENT_END}; "
        "holdout: "
        f"{benchmark_run.BENCHMARK_START} through "
        f"{benchmark_run.BENCHMARK_END}"
    )
    benchmark_evidence = benchmark_run.run_benchmark()

    print()
    print("=" * 78)
    print("[STAGE 2/2] Operational extrapolation")
    print(
        "The same frozen model configurations are now refitted through "
        f"{operational_run.CUTOFF:%Y-%m-%d} without reoptimization."
    )
    print(
        "These future predictions are unvalidated and are not benchmark "
        "metrics or model-selection evidence."
    )
    operational_evidence = operational_run.run_operational_extrapolation()

    # Presentation-only: use the in-memory predictions from both stages.
    # This section must not fit or predict any model.
    transition_plotter = BenchmarkToOperationalTransitionPlotter(
        benchmark_evidence,
        operational_evidence,
        development_end=benchmark_run.DEVELOPMENT_END,
        operational_cutoff=operational_run.CUTOFF,
        operational_dates=operational_run.EXPECTED_DATES,
    )
    transition_path = transition_plotter.save(
        Path("results") / benchmark_run.BATTERY /
        "transition_diagnostics" / "benchmark_to_operational_transition.png"
    )
    print(f"Transition diagnostic figure: {transition_path}")

    print()
    print("=" * 78)
    print("DATASET 410 — COMPLETE EVIDENCE RUN FINISHED")
    print("=" * 78)
    print(f"Frozen benchmark artifacts : {benchmark_run.RESULT_DIRECTORY}")
    print(f"Operational artifacts      : {operational_run.OUTPUT_DIR}")
    print(
        "Scientific separation preserved: measured holdout evaluation and "
        "future operational extrapolation remain independent evidence sets."
    )


if __name__ == "__main__":
    main()
