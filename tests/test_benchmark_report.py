"""Tests for the human-readable benchmark report."""

from __future__ import annotations

from pathlib import Path

import pytest

from zenerestimation.comparison import (
    ComparisonResult,
)
from zenerestimation.optimization import (
    BenchmarkReport,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def comparison_result() -> ComparisonResult:
    return ComparisonResult(
        battery="BAT001",
        metrics={
            "ARIMA": {
                "rmse": 0.8,
                "mae": 0.6,
                "mape": 0.4,
            },
            "Kalman": {
                "rmse": 0.5,
                "mae": 0.4,
                "mape": 0.3,
            },
            "LSTM": {
                "rmse": 0.7,
                "mae": 0.5,
                "mape": 0.35,
            },
        },
        rankings={
            "rmse": [
                {
                    "model": "Kalman",
                    "value": 0.5,
                    "rank": 1,
                },
                {
                    "model": "LSTM",
                    "value": 0.7,
                    "rank": 2,
                },
                {
                    "model": "ARIMA",
                    "value": 0.8,
                    "rank": 3,
                },
            ],
            "mae": [
                {
                    "model": "Kalman",
                    "value": 0.4,
                    "rank": 1,
                },
                {
                    "model": "LSTM",
                    "value": 0.5,
                    "rank": 2,
                },
                {
                    "model": "ARIMA",
                    "value": 0.6,
                    "rank": 3,
                },
            ],
            "mape": [
                {
                    "model": "Kalman",
                    "value": 0.3,
                    "rank": 1,
                },
                {
                    "model": "LSTM",
                    "value": 0.35,
                    "rank": 2,
                },
                {
                    "model": "ARIMA",
                    "value": 0.4,
                    "rank": 3,
                },
            ],
        },
        best_models={
            "rmse": "Kalman",
            "mae": "Kalman",
            "mape": "Kalman",
        },
        metadata={
            "source": "optimized_benchmark",
            "evaluation_steps": 5,
        },
    )


# ============================================================
# Construction
# ============================================================


def test_requires_comparison_result():

    with pytest.raises(
        TypeError,
        match="ComparisonResult",
    ):
        BenchmarkReport(
            "not a comparison result"
        )


def test_preserves_underlying_result(
    comparison_result,
):

    report = BenchmarkReport(
        comparison_result
    )

    assert report.result is comparison_result


# ============================================================
# Rendering
# ============================================================


def test_render_returns_string(
    comparison_result,
):

    report = BenchmarkReport(
        comparison_result
    )

    text = report.render()

    assert isinstance(text, str)
    assert text


def test_render_contains_title_and_battery(
    comparison_result,
):

    text = BenchmarkReport(
        comparison_result
    ).render()

    assert (
        "OPTIMIZED MULTI-MODEL BENCHMARK"
        in text
    )

    assert "Battery: BAT001" in text


def test_render_contains_models(
    comparison_result,
):

    text = BenchmarkReport(
        comparison_result
    ).render()

    assert "ARIMA" in text
    assert "Kalman" in text
    assert "LSTM" in text


def test_render_contains_standard_metrics(
    comparison_result,
):

    text = BenchmarkReport(
        comparison_result
    ).render()

    assert "RMSE" in text
    assert "MAE" in text
    assert "MAPE (%)" in text

    assert "0.800000" in text
    assert "0.500000" in text
    assert "0.350000" in text


def test_render_preserves_model_order(
    comparison_result,
):

    text = BenchmarkReport(
        comparison_result
    ).render()

    arima = text.index("ARIMA")
    kalman = text.index("Kalman")
    lstm = text.index("LSTM")

    assert arima < kalman < lstm


def test_render_uses_stored_best_models(
    comparison_result,
):

    text = BenchmarkReport(
        comparison_result
    ).render()

    assert (
        "BEST BY STORED METRIC"
        in text
    )

    assert "RMSE: Kalman" in text
    assert "MAE: Kalman" in text
    assert "MAPE: Kalman" in text


def test_render_contains_metadata(
    comparison_result,
):

    text = BenchmarkReport(
        comparison_result
    ).render()

    assert "METADATA" in text

    assert (
        "source: optimized_benchmark"
        in text
    )

    assert (
        "evaluation_steps: 5"
        in text
    )


def test_render_does_not_mutate_result(
    comparison_result,
):

    original = comparison_result.to_dict()

    BenchmarkReport(
        comparison_result
    ).render()

    assert (
        comparison_result.to_dict()
        == original
    )


# ============================================================
# Persistence
# ============================================================


def test_save_creates_report(
    comparison_result,
    tmp_path,
):

    filename = (
        tmp_path / "benchmark.txt"
    )

    report = BenchmarkReport(
        comparison_result
    )

    path = report.save(
        filename
    )

    assert path == filename
    assert path.exists()


def test_save_creates_parent_directory(
    comparison_result,
    tmp_path,
):

    filename = (
        tmp_path
        / "nested"
        / "reports"
        / "benchmark.txt"
    )

    report = BenchmarkReport(
        comparison_result
    )

    report.save(
        filename
    )

    assert filename.exists()


def test_saved_report_matches_rendered_text(
    comparison_result,
    tmp_path,
):

    filename = (
        tmp_path / "benchmark.txt"
    )

    report = BenchmarkReport(
        comparison_result
    )

    expected = report.render()

    report.save(
        filename
    )

    actual = filename.read_text(
        encoding="utf-8"
    )

    assert actual == expected


# ============================================================
# Representation
# ============================================================


def test_repr_contains_useful_summary(
    comparison_result,
):

    report = BenchmarkReport(
        comparison_result
    )

    text = repr(report)

    assert "BenchmarkReport" in text
    assert "BAT001" in text
    assert "models=3" in text

# Sprint 14.9F.1: optional forecast-detail reporting.
from datetime import datetime
from zenerestimation.evaluation.result import EvaluationResult


def _evaluation(name, predicted=(9.75, 10.25)):
    return EvaluationResult(
        model=name, evaluation_steps=2,
        rmse=0.5, mae=0.5, mape=5.0,
        actual=(10.0, 10.0), predicted=predicted,
        dates=(datetime(2024, 3, 1), datetime(2024, 6, 1)),
    )


def _evidence():
    return {
        "ARIMA": _evaluation("ARIMA"),
        "Kalman": _evaluation("Kalman"),
        "LSTM": _evaluation("LSTM"),
    }


def test_legacy_report_has_no_forecast_section(comparison_result):
    assert "DETAILED HOLDOUT FORECASTS" not in BenchmarkReport(comparison_result).render()


def test_detailed_report_has_all_models_and_dates(comparison_result):
    text = BenchmarkReport(comparison_result, evaluations=_evidence()).render()
    section = text.split("DETAILED HOLDOUT FORECASTS", 1)[1]
    assert section.index("ARIMA:") < section.index("Kalman:") < section.index("LSTM:")
    assert section.count("2024-03-01") == 3
    assert section.count("2024-06-01") == 3
    assert "Delta (uV) = Actual - Predicted" in section


def test_detailed_report_signed_delta_and_precision(comparison_result):
    text = BenchmarkReport(comparison_result, evaluations=_evidence()).render()
    assert "+0.250000" in text
    assert "-0.250000" in text
    assert "9.750000" in text
    assert "10.250000" in text


def test_detailed_report_uses_unrounded_values(comparison_result):
    evidence = _evidence()
    evidence["ARIMA"] = _evaluation("ARIMA", (9.9999996, 10.0))
    text = BenchmarkReport(comparison_result, evaluations=evidence).render()
    assert "+0.000000" in text  # 0.0000004 rounded only for display


def test_detailed_report_saves_exact_render(comparison_result, tmp_path):
    report = BenchmarkReport(comparison_result, evaluations=_evidence())
    path = report.save(tmp_path / "nested" / "benchmark.txt")
    assert path.read_text(encoding="utf-8") == report.render()


def test_detailed_report_rejects_missing_model(comparison_result):
    with pytest.raises(ValueError, match="comparison models"):
        BenchmarkReport(comparison_result, evaluations={"ARIMA": _evaluation("ARIMA")})


def test_detailed_report_rejects_inconsistent_dates(comparison_result):
    evidence = _evidence()
    other = _evaluation("Kalman")
    from dataclasses import replace
    evidence["Kalman"] = replace(other, dates=(datetime(2024, 3, 1), datetime(2024, 9, 1)))
    with pytest.raises(ValueError, match="inconsistent"):
        BenchmarkReport(comparison_result, evaluations=evidence)


def test_detailed_report_does_not_mutate_evidence(comparison_result):
    evidence = _evidence()
    before = {name: (e.actual, e.predicted, e.dates) for name, e in evidence.items()}
    BenchmarkReport(comparison_result, evaluations=evidence).render()
    assert before == {name: (e.actual, e.predicted, e.dates) for name, e in evidence.items()}
