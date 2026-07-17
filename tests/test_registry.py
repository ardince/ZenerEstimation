import json

from zenerestimation.experiment import Experiment
from zenerestimation.utils.registry import ExperimentRegistry


def test_empty_registry(tmp_path):

    registry = ExperimentRegistry(
        tmp_path / "history.json"
    )

    assert registry.count() == 0

    assert registry.latest() is None


def test_register(tmp_path):

    registry = ExperimentRegistry(
        tmp_path / "history.json"
    )

    experiment = Experiment(

        battery="732B-5610410",

        model="ARIMA",

        version="0.5.0",

        execution_time=0.42,

        horizon=6,

    )

    registered = registry.register(
    experiment
    )

    assert registered.id == 1

    assert registered.battery == "732B-5610410"

    assert registered.model == "ARIMA"

    assert registry.count() == 1

    assert registered is experiment


def test_count(tmp_path):

    registry = ExperimentRegistry(
        tmp_path / "history.json"
    )

    for i in range(3):

        registry.register(

            Experiment(

                battery=f"Battery-{i}",

                model="ARIMA",

            )

        )

    assert registry.count() == 3


def test_latest(tmp_path):

    registry = ExperimentRegistry(
        tmp_path / "history.json"
    )

    registry.register(

        Experiment(

            battery="A",

            model="ARIMA",

        )

    )

    registry.register(

        Experiment(

            battery="B",

            model="Kalman",

        )

    )

    latest = registry.latest()

    assert latest["battery"] == "B"

    assert latest["model"] == "Kalman"

    assert latest["id"] == 2


def test_history_file_created(tmp_path):

    filename = tmp_path / "history.json"

    registry = ExperimentRegistry(filename)

    registry.register(

        Experiment(

            battery="732B-5610410",

            model="ARIMA",

        )

    )

    assert filename.exists()

    with open(

        filename,

        encoding="utf-8",

    ) as f:

        history = json.load(f)

    assert len(history) == 1

    assert history[0]["battery"] == "732B-5610410"
