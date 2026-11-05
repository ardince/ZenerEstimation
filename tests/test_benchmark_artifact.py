"""Tests for the standardized benchmark artifact contract."""

from __future__ import annotations

import csv
import json

import pytest

from zenerestimation.comparison import (
    ComparisonResult,
)
from zenerestimation.optimization import (
    BenchmarkArtifact,
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
        BenchmarkArtifact(
            "not a comparison result"
        )


def test_preserves_underlying_result(
    comparison_result,
):

    artifact = BenchmarkArtifact(
        comparison_result
    )

    assert artifact.result is comparison_result


# ============================================================
# Serialization contract
# ============================================================


def test_to_dict_preserves_benchmark_evidence(
    comparison_result,
):

    artifact = BenchmarkArtifact(
        comparison_result
    )

    payload = artifact.to_dict()

    assert payload["status"] == "benchmarked"
    assert payload["battery"] == "BAT001"

    assert payload["models"] == [
        "ARIMA",
        "Kalman",
        "LSTM",
    ]

    assert payload["metrics"] == (
        comparison_result.metrics
    )

    assert payload["rankings"] == (
        comparison_result.rankings
    )

    assert payload["best_models"] == (
        comparison_result.best_models
    )

    assert payload["metadata"] == (
        comparison_result.metadata
    )


def test_to_dict_does_not_mutate_result(
    comparison_result,
):

    original = comparison_result.to_dict()

    artifact = BenchmarkArtifact(
        comparison_result
    )

    artifact.to_dict()

    assert (
        comparison_result.to_dict()
        == original
    )


def test_returned_payload_is_defensively_independent(
    comparison_result,
):

    artifact = BenchmarkArtifact(
        comparison_result
    )

    payload = artifact.to_dict()

    payload["metrics"]["ARIMA"][
        "rmse"
    ] = 999.0

    payload["models"].append(
        "InjectedModel"
    )

    fresh = artifact.to_dict()

    assert (
        fresh["metrics"]["ARIMA"]["rmse"]
        == pytest.approx(0.8)
    )

    assert "InjectedModel" not in (
        fresh["models"]
    )


def test_payload_is_json_serializable(
    comparison_result,
):

    payload = BenchmarkArtifact(
        comparison_result
    ).to_dict()

    serialized = json.dumps(
        payload
    )

    assert isinstance(
        serialized,
        str,
    )


# ============================================================
# Persistence
# ============================================================


def test_save_creates_json_and_csv(
    comparison_result,
    tmp_path,
):

    artifact = BenchmarkArtifact(
        comparison_result
    )

    paths = artifact.save(
        tmp_path
    )

    assert paths["json"] == (
        tmp_path / "benchmark.json"
    )

    assert paths["csv"] == (
        tmp_path / "benchmark.csv"
    )

    assert paths["json"].exists()
    assert paths["csv"].exists()


def test_save_creates_directory(
    comparison_result,
    tmp_path,
):

    directory = (
        tmp_path
        / "nested"
        / "benchmark"
    )

    artifact = BenchmarkArtifact(
        comparison_result
    )

    artifact.save(
        directory
    )

    assert directory.exists()

    assert (
        directory / "benchmark.json"
    ).exists()

    assert (
        directory / "benchmark.csv"
    ).exists()


def test_saved_json_matches_contract(
    comparison_result,
    tmp_path,
):

    artifact = BenchmarkArtifact(
        comparison_result
    )

    paths = artifact.save(
        tmp_path
    )

    with paths["json"].open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(
            file
        )

    assert payload["status"] == "benchmarked"
    assert payload["battery"] == "BAT001"

    assert payload["metrics"][
        "Kalman"
    ]["rmse"] == pytest.approx(
        0.5
    )


def test_saved_csv_has_standardized_columns(
    comparison_result,
    tmp_path,
):

    artifact = BenchmarkArtifact(
        comparison_result
    )

    paths = artifact.save(
        tmp_path
    )

    with paths["csv"].open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        rows = list(
            reader
        )

    assert reader.fieldnames == [
        "model",
        "rmse",
        "mae",
        "mape",
    ]

    assert len(rows) == 3


def test_saved_csv_preserves_model_order(
    comparison_result,
    tmp_path,
):

    artifact = BenchmarkArtifact(
        comparison_result
    )

    paths = artifact.save(
        tmp_path
    )

    with paths["csv"].open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    assert [
        row["model"]
        for row in rows
    ] == [
        "ARIMA",
        "Kalman",
        "LSTM",
    ]


def test_saved_csv_preserves_metrics(
    comparison_result,
    tmp_path,
):

    artifact = BenchmarkArtifact(
        comparison_result
    )

    paths = artifact.save(
        tmp_path
    )

    with paths["csv"].open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = {
            row["model"]: row
            for row in csv.DictReader(
                file
            )
        }

    assert float(
        rows["ARIMA"]["rmse"]
    ) == pytest.approx(
        0.8
    )

    assert float(
        rows["Kalman"]["mae"]
    ) == pytest.approx(
        0.4
    )

    assert float(
        rows["LSTM"]["mape"]
    ) == pytest.approx(
        0.35
    )


# ============================================================
# Representation
# ============================================================


def test_repr_contains_useful_summary(
    comparison_result,
):

    artifact = BenchmarkArtifact(
        comparison_result
    )

    text = repr(
        artifact
    )

    assert "BenchmarkArtifact" in text
    assert "BAT001" in text
    assert "models=3" in text