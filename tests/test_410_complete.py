"""Tests for the complete dataset-410 evidence orchestrator."""
from __future__ import annotations

from examples.optimization import run_410_complete

from types import SimpleNamespace


def test_complete_runner_calls_benchmark_then_operational(
    monkeypatch,
):
    calls = []

    benchmark_result = SimpleNamespace(
        dataset=object(),
        evaluations=object(),
        forecast_results=object(),
        training_dates=object(),
        artifact_paths={},
    )

    operational_result = SimpleNamespace(
        training=object(),
        forecasts=object(),
        specs=(),
        artifact_paths={},
    )

    def fake_benchmark():
        calls.append("benchmark")
        return benchmark_result

    def fake_operational():
        calls.append("operational")
        return operational_result

    def fake_transition(*args, **kwargs):
        calls.append("transition")

    monkeypatch.setattr(
        run_410_complete.benchmark_run,
        "run_benchmark",
        fake_benchmark,
    )

    monkeypatch.setattr(
        run_410_complete.operational_run,
        "run_operational_extrapolation",
        fake_operational,
    )

    # Patch the transition plotting entry point used by the
    # complete runner. Adjust this target to match the
    # exact name imported in run_410_complete.py.

    class FakeTransitionPlotter:
        def __init__(self, *args, **kwargs):
            calls.append("transition")

        def plot(self, *args, **kwargs):
            return None

        def save(self, *args, **kwargs):
            return "mock_transition.png"


    monkeypatch.setattr(
        run_410_complete,
        "BenchmarkToOperationalTransitionPlotter",
        FakeTransitionPlotter,
    )

    run_410_complete.main()

    assert calls == [
        "benchmark",
        "operational",
        "transition",
    ]


def test_complete_runner_uses_distinct_artifact_directories():
    assert (
        run_410_complete.benchmark_run.RESULT_DIRECTORY.resolve()
        != run_410_complete.operational_run.OUTPUT_DIR.resolve()
    )
