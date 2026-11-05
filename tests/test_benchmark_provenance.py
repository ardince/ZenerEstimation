"""Tests for benchmark evidence provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zenerestimation.optimization import (
    BenchmarkProvenance,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def provenance() -> BenchmarkProvenance:
    return BenchmarkProvenance(
        battery="BAT001",
        optimization={
            "ARIMA": Path(
                "optimization/arima/optimization.json"
            ),
            "Kalman": Path(
                "optimization/kalman/optimization.json"
            ),
            "LSTM": Path(
                "optimization/lstm/optimization.json"
            ),
        },
        stability={
            "LSTM": Path(
                "stability/lstm/stability.json"
            ),
            "GRU": Path(
                "stability/gru/stability.json"
            ),
        },
        benchmark=Path(
            "benchmark.json"
        ),
        metadata={
            "benchmark_steps": 5,
            "seed_policy": "fixed_seed_42",
        },
    )


# ============================================================
# Construction
# ============================================================


def test_requires_non_empty_battery():

    with pytest.raises(
        ValueError,
        match="battery",
    ):
        BenchmarkProvenance(
            battery="",
            optimization={},
            stability={},
            benchmark="benchmark.json",
        )


def test_requires_optimization_mapping():

    with pytest.raises(
        TypeError,
        match="optimization",
    ):
        BenchmarkProvenance(
            battery="BAT001",
            optimization=[],
            stability={},
            benchmark="benchmark.json",
        )


def test_requires_stability_mapping():

    with pytest.raises(
        TypeError,
        match="stability",
    ):
        BenchmarkProvenance(
            battery="BAT001",
            optimization={},
            stability=[],
            benchmark="benchmark.json",
        )


def test_requires_benchmark_path():

    with pytest.raises(
        TypeError,
        match="benchmark",
    ):
        BenchmarkProvenance(
            battery="BAT001",
            optimization={},
            stability={},
            benchmark=123,
        )


def test_requires_metadata_mapping():

    with pytest.raises(
        TypeError,
        match="metadata",
    ):
        BenchmarkProvenance(
            battery="BAT001",
            optimization={},
            stability={},
            benchmark="benchmark.json",
            metadata=[],
        )


def test_rejects_invalid_optimization_path():

    with pytest.raises(
        TypeError,
        match="optimization",
    ):
        BenchmarkProvenance(
            battery="BAT001",
            optimization={
                "ARIMA": 123,
            },
            stability={},
            benchmark="benchmark.json",
        )


def test_rejects_invalid_stability_path():

    with pytest.raises(
        TypeError,
        match="stability",
    ):
        BenchmarkProvenance(
            battery="BAT001",
            optimization={},
            stability={
                "LSTM": 123,
            },
            benchmark="benchmark.json",
        )


# ============================================================
# Stored contract
# ============================================================


def test_preserves_battery(
    provenance,
):

    assert provenance.battery == "BAT001"


def test_preserves_model_order(
    provenance,
):

    assert list(
        provenance.optimization
    ) == [
        "ARIMA",
        "Kalman",
        "LSTM",
    ]

    assert list(
        provenance.stability
    ) == [
        "LSTM",
        "GRU",
    ]


def test_normalizes_paths(
    provenance,
):

    assert provenance.optimization[
        "ARIMA"
    ] == Path(
        "optimization/arima/optimization.json"
    )

    assert provenance.stability[
        "LSTM"
    ] == Path(
        "stability/lstm/stability.json"
    )

    assert provenance.benchmark == Path(
        "benchmark.json"
    )


def test_metadata_is_defensively_copied():

    metadata = {
        "benchmark_steps": 5,
    }

    provenance = BenchmarkProvenance(
        battery="BAT001",
        optimization={},
        stability={},
        benchmark="benchmark.json",
        metadata=metadata,
    )

    metadata[
        "benchmark_steps"
    ] = 999

    assert provenance.metadata[
        "benchmark_steps"
    ] == 5


def test_returned_metadata_is_defensive(
    provenance,
):

    metadata = provenance.metadata

    metadata[
        "benchmark_steps"
    ] = 999

    assert provenance.metadata[
        "benchmark_steps"
    ] == 5


# ============================================================
# Serialization
# ============================================================


def test_to_dict_has_standardized_contract(
    provenance,
):

    payload = provenance.to_dict()

    assert payload[
        "status"
    ] == "linked"

    assert payload[
        "battery"
    ] == "BAT001"

    assert payload[
        "optimization"
    ] == {
        "ARIMA":
            "optimization/arima/optimization.json",
        "Kalman":
            "optimization/kalman/optimization.json",
        "LSTM":
            "optimization/lstm/optimization.json",
    }

    assert payload[
        "stability"
    ] == {
        "LSTM":
            "stability/lstm/stability.json",
        "GRU":
            "stability/gru/stability.json",
    }

    assert payload[
        "benchmark"
    ] == "benchmark.json"


def test_to_dict_preserves_metadata(
    provenance,
):

    payload = provenance.to_dict()

    assert payload[
        "metadata"
    ] == {
        "benchmark_steps": 5,
        "seed_policy": "fixed_seed_42",
    }


def test_to_dict_is_json_serializable(
    provenance,
):

    json.dumps(
        provenance.to_dict()
    )


def test_to_dict_is_defensively_independent(
    provenance,
):

    payload = provenance.to_dict()

    payload[
        "optimization"
    ][
        "ARIMA"
    ] = "changed.json"

    payload[
        "metadata"
    ][
        "benchmark_steps"
    ] = 999

    assert provenance.to_dict()[
        "optimization"
    ][
        "ARIMA"
    ] == (
        "optimization/arima/optimization.json"
    )

    assert provenance.to_dict()[
        "metadata"
    ][
        "benchmark_steps"
    ] == 5


# ============================================================
# Scientific boundary
# ============================================================


def test_contains_only_artifact_references(
    provenance,
):

    payload = provenance.to_dict()

    serialized = json.dumps(
        payload
    )

    forbidden = (
        '"rmse"',
        '"mae"',
        '"mape"',
        '"actual"',
        '"predicted"',
        '"predictions"',
        '"best_params"',
        '"best_score"',
        '"candidates"',
        '"seeds"',
        '"best_seed"',
    )

    for token in forbidden:
        assert token not in serialized


def test_does_not_require_stability_for_all_models():

    provenance = BenchmarkProvenance(
        battery="BAT001",
        optimization={
            "ARIMA": "arima/optimization.json",
            "LSTM": "lstm/optimization.json",
        },
        stability={
            "LSTM": "lstm/stability.json",
        },
        benchmark="benchmark.json",
    )

    assert list(
        provenance.stability
    ) == [
        "LSTM",
    ]


def test_allows_empty_stability_mapping():

    provenance = BenchmarkProvenance(
        battery="BAT001",
        optimization={
            "ARIMA": "arima/optimization.json",
        },
        stability={},
        benchmark="benchmark.json",
    )

    assert provenance.stability == {}


# ============================================================
# Persistence
# ============================================================


def test_filename_is_standardized():

    assert (
        BenchmarkProvenance.FILENAME
        == "provenance.json"
    )


def test_save_creates_provenance_json(
    provenance,
    tmp_path,
):

    path = provenance.save(
        tmp_path
    )

    assert path == (
        tmp_path
        / "provenance.json"
    )

    assert path.exists()
    assert path.is_file()


def test_save_creates_directory(
    provenance,
    tmp_path,
):

    directory = (
        tmp_path
        / "battery"
        / "benchmark"
        / "run"
    )

    path = provenance.save(
        directory
    )

    assert directory.exists()
    assert path.exists()


def test_saved_json_matches_contract(
    provenance,
    tmp_path,
):

    path = provenance.save(
        tmp_path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(
            file
        )

    assert payload == provenance.to_dict()


def test_save_accepts_string_path(
    provenance,
    tmp_path,
):

    path = provenance.save(
        str(tmp_path)
    )

    assert isinstance(
        path,
        Path,
    )

    assert path.exists()


def test_save_rejects_invalid_directory_type(
    provenance,
):

    with pytest.raises(
        TypeError,
        match="directory",
    ):
        provenance.save(
            123
        )


def test_save_rejects_existing_file_as_directory(
    provenance,
    tmp_path,
):

    existing_file = (
        tmp_path
        / "existing.txt"
    )

    existing_file.write_text(
        "not a directory",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="directory",
    ):
        provenance.save(
            existing_file
        )


# ============================================================
# Representation / public API
# ============================================================


def test_repr_contains_useful_summary(
    provenance,
):

    text = repr(
        provenance
    )

    assert "BenchmarkProvenance" in text
    assert "BAT001" in text
    assert "optimization=3" in text
    assert "stability=2" in text


def test_is_publicly_exported():

    from zenerestimation.optimization import (
        BenchmarkProvenance
        as PublicBenchmarkProvenance,
    )

    assert (
        PublicBenchmarkProvenance
        is BenchmarkProvenance
    )