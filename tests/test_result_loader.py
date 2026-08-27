"""
Tests for ResultLoader.
"""

import json

import pytest

from zenerestimation.utils.result_loader import (
    ResultLoader,
    ResultPackage,
)


# ============================================================
# Helpers
# ============================================================


def create_run(
    root,
    battery="BAT001",
    model="arima",
    timestamp="20260819_150000",
    run_number=1,
    forecast=True,
    evaluation=True,
    experiment=True,
    report=True,
    figure=True,
    log=True,
):
    """
    Create a synthetic experiment result package.
    """

    directory = (
        root
        / battery
        / model
        / f"{timestamp}_{run_number}"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if forecast:

        (directory / "forecast.json").write_text(
            json.dumps(
                {
                    "model": model,
                    "horizon": 6,
                    "forecast": [1.0, 2.0, 3.0],
                }
            ),
            encoding="utf-8",
        )

    if evaluation:

        (directory / "evaluation.json").write_text(
            json.dumps(
                {
                    "rmse": 1.25,
                    "mae": 0.95,
                    "mape": 2.50,
                }
            ),
            encoding="utf-8",
        )

    if experiment:

        (directory / "experiment.json").write_text(
            json.dumps(
                {
                    "battery": battery,
                    "model": model,
                    "run_number": run_number,
                }
            ),
            encoding="utf-8",
        )

    if report:

        (directory / "report.txt").write_text(
            "Experiment Report",
            encoding="utf-8",
        )

    if figure:

        (directory / "forecast.png").write_bytes(
            b"fake image"
        )

    if log:

        (directory / "experiment.log").write_text(
            "Experiment completed.",
            encoding="utf-8",
        )

    return directory


# ============================================================
# Constructor
# ============================================================


def test_constructor(tmp_path):

    loader = ResultLoader(
        root=tmp_path
    )

    assert loader.root == tmp_path


# ============================================================
# Empty Directory
# ============================================================


def test_discover_empty_directory(
    tmp_path,
):

    loader = ResultLoader(
        root=tmp_path
    )

    assert loader.discover() == []


# ============================================================
# Discovery
# ============================================================


def test_discover_one_result(
    tmp_path,
):

    directory = create_run(
        tmp_path
    )

    loader = ResultLoader(
        root=tmp_path
    )

    results = loader.discover()

    assert len(results) == 1

    assert results[0] == directory


def test_discover_multiple_results(
    tmp_path,
):

    first = create_run(
        tmp_path,
        timestamp="20260819_150000",
        run_number=1,
    )

    second = create_run(
        tmp_path,
        timestamp="20260819_151000",
        run_number=2,
    )

    third = create_run(
        tmp_path,
        model="kalman",
        timestamp="20260819_152000",
        run_number=1,
    )

    loader = ResultLoader(
        root=tmp_path
    )

    results = loader.discover()

    assert len(results) == 3

    assert first in results
    assert second in results
    assert third in results


# ============================================================
# Run Directory Validation
# ============================================================


def test_invalid_run_directory_is_ignored(
    tmp_path,
):

    valid = create_run(
        tmp_path
    )

    invalid = (
        tmp_path
        / "BAT001"
        / "arima"
        / "invalid_directory"
    )

    invalid.mkdir(
        parents=True
    )

    loader = ResultLoader(
        root=tmp_path
    )

    results = loader.discover()

    assert valid in results

    assert invalid not in results


def test_invalid_run_directory_load_raises(
    tmp_path,
):

    invalid = (
        tmp_path
        / "BAT001"
        / "arima"
        / "invalid_directory"
    )

    invalid.mkdir(
        parents=True
    )

    loader = ResultLoader(
        root=tmp_path
    )

    with pytest.raises(ValueError):

        loader.load(
            invalid
        )


# ============================================================
# Load One Result
# ============================================================


def test_load_result(
    tmp_path,
):

    directory = create_run(
        tmp_path,
        battery="BAT001",
        model="arima",
        timestamp="20260819_150000",
        run_number=1,
    )

    loader = ResultLoader(
        root=tmp_path
    )

    result = loader.load(
        directory
    )

    assert isinstance(
        result,
        ResultPackage,
    )

    assert result.battery == "BAT001"

    assert result.model == "arima"

    assert result.timestamp == (
        "20260819_150000"
    )

    assert result.run_number == 1

    assert result.directory == directory


# ============================================================
# JSON Artifacts
# ============================================================


def test_load_forecast(
    tmp_path,
):

    directory = create_run(
        tmp_path
    )

    loader = ResultLoader(
        root=tmp_path
    )

    result = loader.load(
        directory
    )

    assert result.forecast is not None

    assert result.forecast["model"] == "arima"

    assert result.forecast["horizon"] == 6


def test_load_evaluation(
    tmp_path,
):

    directory = create_run(
        tmp_path
    )

    loader = ResultLoader(
        root=tmp_path
    )

    result = loader.load(
        directory
    )

    assert result.evaluation is not None

    assert result.evaluation["rmse"] == 1.25

    assert result.evaluation["mae"] == 0.95

    assert result.evaluation["mape"] == 2.50


def test_load_experiment(
    tmp_path,
):

    directory = create_run(
        tmp_path,
        battery="BAT001",
        model="arima",
    )

    loader = ResultLoader(
        root=tmp_path
    )

    result = loader.load(
        directory
    )

    assert result.experiment is not None

    assert result.experiment["battery"] == (
        "BAT001"
    )

    assert result.experiment["model"] == (
        "arima"
    )


# ============================================================
# Optional Artifacts
# ============================================================


def test_optional_artifacts(
    tmp_path,
):

    directory = create_run(
        tmp_path,
        report=False,
        figure=False,
        log=False,
    )

    loader = ResultLoader(
        root=tmp_path
    )

    result = loader.load(
        directory
    )

    assert result.report is None

    assert result.figure is None

    assert result.log is None


# ============================================================
# Missing JSON
# ============================================================


def test_missing_json_artifact(
    tmp_path,
):

    directory = create_run(
        tmp_path,
        forecast=False,
    )

    loader = ResultLoader(
        root=tmp_path
    )

    result = loader.load(
        directory
    )

    assert result.forecast is None

    assert result.evaluation is not None

    assert result.experiment is not None


# ============================================================
# Load Battery
# ============================================================


def test_load_battery(
    tmp_path,
):

    first = create_run(
        tmp_path,
        battery="BAT001",
        model="arima",
        timestamp="20260819_150000",
        run_number=1,
    )

    second = create_run(
        tmp_path,
        battery="BAT001",
        model="kalman",
        timestamp="20260819_151000",
        run_number=1,
    )

    other = create_run(
        tmp_path,
        battery="BAT002",
        model="arima",
        timestamp="20260819_152000",
        run_number=1,
    )

    loader = ResultLoader(
        root=tmp_path
    )

    results = loader.load_battery(
        "BAT001"
    )

    directories = {
        result.directory
        for result in results
    }

    assert directories == {
        first,
        second,
    }

    assert other not in directories


# ============================================================
# Load Model
# ============================================================


def test_load_model(
    tmp_path,
):

    arima = create_run(
        tmp_path,
        battery="BAT001",
        model="arima",
        timestamp="20260819_150000",
        run_number=1,
    )

    kalman = create_run(
        tmp_path,
        battery="BAT001",
        model="kalman",
        timestamp="20260819_151000",
        run_number=1,
    )

    other_battery = create_run(
        tmp_path,
        battery="BAT002",
        model="arima",
        timestamp="20260819_152000",
        run_number=1,
    )

    loader = ResultLoader(
        root=tmp_path
    )

    results = loader.load_model(
        battery="BAT001",
        model="arima",
    )

    directories = {
        result.directory
        for result in results
    }

    assert directories == {
        arima,
    }

    assert kalman not in directories

    assert other_battery not in directories


# ============================================================
# Latest
# ============================================================


def test_latest_returns_latest_run(
    tmp_path,
):

    first = create_run(
        tmp_path,
        timestamp="20260819_150000",
        run_number=1,
    )

    second = create_run(
        tmp_path,
        timestamp="20260819_151000",
        run_number=2,
    )

    loader = ResultLoader(
        root=tmp_path
    )

    latest = loader.latest(
        battery="BAT001",
        model="arima",
    )

    assert latest is not None

    assert latest.directory == second

    assert latest.directory != first


def test_latest_returns_none_when_missing(
    tmp_path,
):

    loader = ResultLoader(
        root=tmp_path
    )

    result = loader.latest(
        battery="BAT001",
        model="arima",
    )

    assert result is None


# ============================================================
# Error Handling
# ============================================================


def test_load_missing_directory(
    tmp_path,
):

    loader = ResultLoader(
        root=tmp_path
    )

    missing = (
        tmp_path
        / "does_not_exist"
    )

    with pytest.raises(
        FileNotFoundError
    ):

        loader.load(
            missing
        )


def test_load_file_instead_of_directory(
    tmp_path,
):

    filename = (
        tmp_path
        / "result.txt"
    )

    filename.write_text(
        "invalid",
        encoding="utf-8",
    )

    loader = ResultLoader(
        root=tmp_path
    )

    with pytest.raises(
        NotADirectoryError
    ):

        loader.load(
            filename
        )


# ============================================================
# Case Handling
# ============================================================


def test_model_matching_is_case_insensitive(
    tmp_path,
):

    directory = create_run(
        tmp_path,
        model="ARIMA",
    )

    loader = ResultLoader(
        root=tmp_path
    )

    results = loader.load_model(
        battery="BAT001",
        model="arima",
    )

    assert len(results) == 1

    assert results[0].directory == directory