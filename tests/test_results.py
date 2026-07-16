from pathlib import Path
import json

from zenerestimation.utils.results import (
    ExperimentResult,
    create_result_files,
    save_metadata,
    save_report,
)

def test_create_result_files():

    files = create_result_files(
        battery="BAT001",
        model="arima",
    )

    assert isinstance(
        files,
        ExperimentResult,
    )

    assert files.figure.suffix == ".png"

    assert files.metadata.suffix == ".json"

    assert files.report.suffix == ".txt"

    assert files.log.suffix == ".log"


def test_results_directory_exists():

    files = create_result_files(
        battery="BAT001",
        model="arima",
    )

    assert files.directory.exists()

    assert files.directory.is_dir()


def test_battery_name():

    files = create_result_files(
        battery="BAT001",
        model="arima",
    )

    assert "BAT001" in files.figure.name

    assert "BAT001" in files.metadata.name


def test_model_name():

    files = create_result_files(
        battery="BAT001",
        model="arima",
    )

    assert "arima" in files.figure.name

    assert "arima" in files.report.name


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

    text = "Forecast completed successfully."

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


def test_paths_are_path_objects():

    files = create_result_files(
        battery="BAT001",
        model="arima",
    )

    assert isinstance(files.figure, Path)

    assert isinstance(files.metadata, Path)

    assert isinstance(files.report, Path)

    assert isinstance(files.log, Path)

