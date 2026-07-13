from zenerestimation.forecasting import ForecastPipeline

from tests.dummy_model import DummyForecastModel

import pandas as pd

from zenerestimation.data import BatteryDataset


def make_dataset():

    df = pd.DataFrame(
        {
            "ds": [
                "01/01/2024",
                "01/04/2024",
            ],
            "microVolt": [
                180,
                179,
            ],
        }
    )

    return BatteryDataset(df)


def test_pipeline_creation():

    dataset = make_dataset()

    pipeline = ForecastPipeline(dataset)

    assert pipeline.dataset is dataset


def test_register_model():

    pipeline = ForecastPipeline(make_dataset())

    pipeline.register_model(
        "dummy",
        DummyForecastModel,
    )

    assert "dummy" in pipeline.available_models()


def test_run_dummy():

    pipeline = ForecastPipeline(make_dataset())

    pipeline.register_model(
        "dummy",
        DummyForecastModel,
    )

    result = pipeline.run("dummy", steps=5)

    assert result == [0.0] * 5