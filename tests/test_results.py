from pathlib import Path
import json

from zenerestimation.utils.results import (
    ExperimentResult,
    create_result_files,
    save_metadata,
    save_report,
)


def test_create_result_files(tmp_path):

    files = create_result_files(
        battery="BAT001",
        model="arima",
        root=tmp_path,
    )

    assert isinstance(
        files,
        ExperimentResult,
    )


def test_result_directory_structure():

    files = create_result_files(
        battery="BAT001",
        model="arima",
    )

    assert files.directory.exists()

    assert files.directory.is_dir()

    assert files.directory.parent.name == "arima"

    assert files.directory.parent.parent.name == "BAT001"

    assert files.directory.parent.parent.parent.name == "results"


def test_standard_artifact_names():

    files = create_result_files(
        battery="BAT001",
        model="arima",
    )

    assert files.figure.name == "forecast.png"

    assert files.forecast.name == "forecast.json"

    assert files.evaluation.name == "evaluation.json"

    assert files.experiment.name == "experiment.json"

    assert files.report.name == "report.txt"

    assert files.log.name == "experiment.log"


def test_paths_are_path_objects():

    files = create_result_files(
        battery="BAT001",
        model="arima",
    )

    assert isinstance(files.figure, Path)

    assert isinstance(files.forecast, Path)

    assert isinstance(files.evaluation, Path)

    assert isinstance(files.experiment, Path)

    assert isinstance(files.report, Path)

    assert isinstance(files.log, Path)

    assert isinstance(files.directory, Path)


def test_battery_name_in_path():

    files = create_result_files(
        battery="BAT001",
        model="arima",
    )

    #assert files.directory.parent.name == "BAT001"
    assert files.directory.parent.parent.name == "BAT001"


def test_model_name_in_path():

    files = create_result_files(
        battery="BAT001",
        model="arima",
    )

    assert files.directory.parent.name == "arima"


def test_run_directory_is_timestamped(tmp_path):

    files = create_result_files(
        battery="BAT001",
        model="arima",
        root=tmp_path,
    )

    parts = files.directory.name.split("_")

    assert len(parts) == 3

    # YYYYMMDD
    assert len(parts[0]) == 8
    assert parts[0].isdigit()

    # HHMMSS
    assert len(parts[1]) == 6
    assert parts[1].isdigit()

    # Sequential run number
    assert parts[2].isdigit()
    assert int(parts[2]) >= 1


def test_suffix():

    files = create_result_files(
        battery="BAT001",
        model="arima",
        suffix="validation",
    )

    assert files.directory.name.endswith(
        "_validation"
    )


def test_different_runs_have_different_directories():

    first = create_result_files(
        battery="BAT001",
        model="arima",
    )

    second = create_result_files(
        battery="BAT001",
        model="arima",
    )

    assert first.directory != second.directory


def test_save_metadata(tmp_path):

    filename = tmp_path / "metadata.json"

    metadata = {
        "battery": "BAT001",
        "model": "arima",
    }

    save_metadata(
        filename,
        metadata,
    )

    assert filename.exists()

    loaded = json.loads(
        filename.read_text(
            encoding="utf-8"
        )
    )

    assert loaded["battery"] == "BAT001"

    assert loaded["model"] == "arima"


def test_save_report(tmp_path):

    filename = tmp_path / "report.txt"

    text = (
        "Forecast completed successfully."
    )

    save_report(
        filename,
        text,
    )

    assert filename.exists()

    assert (
        filename.read_text(
            encoding="utf-8"
        )
        == text
    )


def test_same_battery_model_share_common_parent():

    arima = create_result_files(
        battery="BAT001",
        model="arima",
    )

    lstm = create_result_files(
        battery="BAT001",
        model="lstm",
    )

    assert (
        arima.directory.parent.parent
        == lstm.directory.parent.parent
    )


def test_run_numbers_increment(tmp_path):

    first = create_result_files(
        battery="BAT001",
        model="arima",
        root=tmp_path,
    )

    second = create_result_files(
        battery="BAT001",
        model="arima",
        root=tmp_path,
    )

    assert first.run_number == 1
    assert second.run_number == 2


def test_models_have_independent_run_numbers(tmp_path):

    arima = create_result_files(
        battery="BAT001",
        model="arima",
        root=tmp_path,
    )

    kalman = create_result_files(
        battery="BAT001",
        model="kalman",
        root=tmp_path,
    )

    assert arima.run_number == 1
    assert kalman.run_number == 1