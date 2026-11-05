"""Tests for benchmark evidence bundle utilities."""

from pathlib import Path

import pytest

from zenerestimation.optimization import (
    BenchmarkProvenance,
    copy_evidence_files,
    validate_provenance_references,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_file(path: Path, content: str = "{}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _make_complete_bundle(root: Path) -> BenchmarkProvenance:
    """Create a minimal complete provenance bundle for validation tests."""

    optimization = {
        "ARIMA": Path("optimization/arima/optimization.json"),
        "Kalman": Path("optimization/kalman/optimization.json"),
    }

    stability = {
        "LSTM": Path("stability/lstm/stability.json"),
        "GRU": Path("stability/gru/stability.json"),
    }

    benchmark = Path("benchmark.json")

    for reference in optimization.values():
        _write_file(root / reference)

    for reference in stability.values():
        _write_file(root / reference)

    _write_file(root / benchmark)

    return BenchmarkProvenance(
        battery="test-battery",
        optimization=optimization,
        stability=stability,
        benchmark=benchmark,
    )


# ---------------------------------------------------------------------------
# copy_evidence_files
# ---------------------------------------------------------------------------


def test_copy_evidence_files_copies_files(tmp_path):
    source = _write_file(
        tmp_path / "source" / "optimization.json",
        '{"model": "ARIMA"}',
    )

    sources = {
        "ARIMA": source,
    }

    destinations = {
        "ARIMA": Path("optimization/arima/optimization.json"),
    }

    copied = copy_evidence_files(
        sources,
        destinations,
        root=tmp_path / "bundle",
    )

    expected = (
        tmp_path
        / "bundle"
        / "optimization"
        / "arima"
        / "optimization.json"
    )

    assert expected.is_file()
    assert copied["ARIMA"] == expected


def test_copy_evidence_files_creates_nested_directories(tmp_path):
    source = _write_file(
        tmp_path / "source.json",
    )

    destination = Path(
        "optimization/linear_trend_lstm/optimization.json"
    )

    copy_evidence_files(
        {"LinearTrendLSTM": source},
        {"LinearTrendLSTM": destination},
        root=tmp_path / "bundle",
    )

    assert (tmp_path / "bundle" / destination).is_file()


def test_copy_evidence_files_preserves_source_bytes(tmp_path):
    source = tmp_path / "source.json"

    payload = b'{"value": 123}\r\n'

    source.write_bytes(payload)

    destination = Path(
        "optimization/arima/optimization.json"
    )

    copy_evidence_files(
        {"ARIMA": source},
        {"ARIMA": destination},
        root=tmp_path / "bundle",
    )

    copied = tmp_path / "bundle" / destination

    assert copied.read_bytes() == payload


def test_copy_evidence_files_does_not_modify_source(tmp_path):
    source = _write_file(
        tmp_path / "source.json",
        '{"original": true}',
    )

    before = source.read_bytes()

    copy_evidence_files(
        {"ARIMA": source},
        {
            "ARIMA": Path(
                "optimization/arima/optimization.json"
            )
        },
        root=tmp_path / "bundle",
    )

    assert source.read_bytes() == before


def test_copy_evidence_files_rejects_missing_source(tmp_path):
    missing = tmp_path / "does-not-exist.json"

    with pytest.raises(
        FileNotFoundError,
        match="Evidence source does not exist",
    ):
        copy_evidence_files(
            {"ARIMA": missing},
            {
                "ARIMA": Path(
                    "optimization/arima/optimization.json"
                )
            },
            root=tmp_path / "bundle",
        )


def test_copy_evidence_files_rejects_mismatched_keys(tmp_path):
    source = _write_file(
        tmp_path / "source.json",
    )

    with pytest.raises(
        ValueError,
        match="same evidence keys",
    ):
        copy_evidence_files(
            {"ARIMA": source},
            {
                "Kalman": Path(
                    "optimization/kalman/optimization.json"
                )
            },
            root=tmp_path / "bundle",
        )


def test_copy_evidence_files_rejects_absolute_destination(tmp_path):
    source = _write_file(
        tmp_path / "source.json",
    )

    absolute_destination = (
        tmp_path
        / "optimization"
        / "arima"
        / "optimization.json"
    ).resolve()

    with pytest.raises(
        ValueError,
        match="relative paths",
    ):
        copy_evidence_files(
            {"ARIMA": source},
            {"ARIMA": absolute_destination},
            root=tmp_path / "bundle",
        )


# ---------------------------------------------------------------------------
# validate_provenance_references
# ---------------------------------------------------------------------------


def test_validate_provenance_references_accepts_complete_bundle(
    tmp_path,
):
    provenance = _make_complete_bundle(tmp_path)

    validate_provenance_references(
        provenance,
        root=tmp_path,
    )


def test_validate_provenance_references_rejects_missing_optimization(
    tmp_path,
):
    provenance = _make_complete_bundle(tmp_path)

    missing = (
        tmp_path
        / "optimization"
        / "arima"
        / "optimization.json"
    )

    missing.unlink()

    with pytest.raises(
        FileNotFoundError,
        match="optimization/arima/optimization.json",
    ):
        validate_provenance_references(
            provenance,
            root=tmp_path,
        )


def test_validate_provenance_references_rejects_missing_stability(
    tmp_path,
):
    provenance = _make_complete_bundle(tmp_path)

    missing = (
        tmp_path
        / "stability"
        / "lstm"
        / "stability.json"
    )

    missing.unlink()

    with pytest.raises(
        FileNotFoundError,
        match="stability/lstm/stability.json",
    ):
        validate_provenance_references(
            provenance,
            root=tmp_path,
        )


def test_validate_provenance_references_rejects_missing_benchmark(
    tmp_path,
):
    provenance = _make_complete_bundle(tmp_path)

    (tmp_path / "benchmark.json").unlink()

    with pytest.raises(
        FileNotFoundError,
        match="benchmark.json",
    ):
        validate_provenance_references(
            provenance,
            root=tmp_path,
        )


def test_validate_provenance_references_does_not_modify_files(
    tmp_path,
):
    provenance = _make_complete_bundle(tmp_path)

    paths = [
        tmp_path
        / "optimization"
        / "arima"
        / "optimization.json",
        tmp_path
        / "optimization"
        / "kalman"
        / "optimization.json",
        tmp_path
        / "stability"
        / "lstm"
        / "stability.json",
        tmp_path
        / "stability"
        / "gru"
        / "stability.json",
        tmp_path / "benchmark.json",
    ]

    before = {
        path: path.read_bytes()
        for path in paths
    }

    validate_provenance_references(
        provenance,
        root=tmp_path,
    )

    after = {
        path: path.read_bytes()
        for path in paths
    }

    assert after == before