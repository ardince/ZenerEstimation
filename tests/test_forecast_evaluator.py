"""
Tests for ForecastEvaluator.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.evaluation import (
    EvaluationResult,
    ForecastEvaluator,
)

from zenerestimation.data.temporal import (
    TemporalPreprocessor,
)

# ------------------------------------------------------------
# Test model
# ------------------------------------------------------------


class DummyForecaster:
    """
    Minimal deterministic forecaster for evaluator tests.
    """

    def __init__(
        self,
        predictions=None,
    ):
        self.predictions = predictions
        self.dataset = None
        self.predict_steps = None

    def fit(
        self,
        dataset,
    ):
        self.dataset = dataset

        return self

    def predict(
        self,
        steps,
    ):
        self.predict_steps = steps

        if self.predictions is None:
            predictions = [
                float(
                    self.dataset.target.iloc[-1]
                )
            ] * steps
        else:
            predictions = self.predictions

        return SimpleNamespace(
            model="DummyForecaster",
            forecast=np.asarray(
                predictions,
                dtype=float,
            ),
        )


class RecordingPreprocessor:
    """
    Test double that records exactly which
    dataset was supplied for preprocessing.
    """

    def __init__(self):
        self.received_dataset = None

    def fit_transform(
        self,
        dataset,
    ):
        self.received_dataset = dataset

        return dataset


class RecordingModel:
    """
    Minimal forecasting model used to inspect
    the dataset received by fit().
    """

    def __init__(
        self,
        predictions,
    ):
        self.predictions = predictions
        self.fitted_dataset = None

    def fit(
        self,
        dataset,
    ):
        self.fitted_dataset = dataset
        return self

    def predict(
        self,
        steps,
    ):
        class Result:
            pass

        result = Result()

        result.forecast = pd.Series(
            self.predictions[:steps]
        )

        return result


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------


@pytest.fixture
def dataset():
    return BatteryDataset(
        pd.DataFrame(
            {
                "ds": pd.date_range(
                    "2020-03-01",
                    periods=8,
                    freq="3MS",
                ),
                "microVolt": [
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                    16.0,
                    17.0,
                ],
            }
        )
    )


# ------------------------------------------------------------
# Construction
# ------------------------------------------------------------


def test_evaluator_creation():
    evaluator = ForecastEvaluator(
        evaluation_steps=3
    )

    assert evaluator.evaluation_steps == 3


def test_default_evaluation_steps():
    evaluator = ForecastEvaluator()

    assert evaluator.evaluation_steps == 6


def test_zero_evaluation_steps_raises():
    with pytest.raises(
        ValueError,
        match=(
            "evaluation_steps must be "
            "greater than zero"
        ),
    ):
        ForecastEvaluator(
            evaluation_steps=0
        )


def test_negative_evaluation_steps_raises():
    with pytest.raises(ValueError):
        ForecastEvaluator(
            evaluation_steps=-1
        )


# ------------------------------------------------------------
# Split
# ------------------------------------------------------------


def test_split_dataset(dataset):
    evaluator = ForecastEvaluator(
        evaluation_steps=3
    )

    train, validation = (
        evaluator.split_dataset(
            dataset
        )
    )

    assert len(train) == 5
    assert len(validation) == 3


def test_split_preserves_training_values(
    dataset,
):
    evaluator = ForecastEvaluator(
        evaluation_steps=3
    )

    train, _ = (
        evaluator.split_dataset(
            dataset
        )
    )

    assert list(
        train.data["microVolt"]
    ) == [
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
    ]


def test_split_preserves_validation_values(
    dataset,
):
    evaluator = ForecastEvaluator(
        evaluation_steps=3
    )

    _, validation = (
        evaluator.split_dataset(
            dataset
        )
    )

    assert list(
        validation["microVolt"]
    ) == [
        15.0,
        16.0,
        17.0,
    ]


def test_split_does_not_modify_original(
    dataset,
):
    original = dataset.data.copy(
        deep=True
    )

    evaluator = ForecastEvaluator(
        evaluation_steps=3
    )

    evaluator.split_dataset(
        dataset
    )

    pd.testing.assert_frame_equal(
        dataset.data,
        original,
    )


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------


def test_evaluate_returns_result(dataset):
    evaluator = ForecastEvaluator(
        evaluation_steps=3
    )

    model = DummyForecaster(
        predictions=[
            15.0,
            16.0,
            17.0,
        ]
    )

    result = evaluator.evaluate(
        dataset,
        model,
    )

    assert isinstance(
        result,
        EvaluationResult,
    )


def test_evaluation_fits_training_only(
    dataset,
):
    evaluator = ForecastEvaluator(
        evaluation_steps=3
    )

    model = DummyForecaster(
        predictions=[
            15.0,
            16.0,
            17.0,
        ]
    )

    evaluator.evaluate(
        dataset,
        model,
    )

    assert len(model.dataset) == 5


def test_evaluation_predicts_holdout_steps(
    dataset,
):
    evaluator = ForecastEvaluator(
        evaluation_steps=3
    )

    model = DummyForecaster(
        predictions=[
            15.0,
            16.0,
            17.0,
        ]
    )

    evaluator.evaluate(
        dataset,
        model,
    )

    assert model.predict_steps == 3


def test_perfect_prediction_metrics(
    dataset,
):
    evaluator = ForecastEvaluator(
        evaluation_steps=3
    )

    model = DummyForecaster(
        predictions=[
            15.0,
            16.0,
            17.0,
        ]
    )

    result = evaluator.evaluate(
        dataset,
        model,
    )

    assert result.rmse == pytest.approx(
        0.0
    )

    assert result.mae == pytest.approx(
        0.0
    )

    assert result.mape == pytest.approx(
        0.0
    )


def test_evaluation_actual_values(
    dataset,
):
    evaluator = ForecastEvaluator(
        evaluation_steps=3
    )

    model = DummyForecaster(
        predictions=[
            14.0,
            15.0,
            16.0,
        ]
    )

    result = evaluator.evaluate(
        dataset,
        model,
    )

    assert result.actual == (
        15.0,
        16.0,
        17.0,
    )


def test_evaluation_predicted_values(
    dataset,
):
    evaluator = ForecastEvaluator(
        evaluation_steps=3
    )

    model = DummyForecaster(
        predictions=[
            14.0,
            15.0,
            16.0,
        ]
    )

    result = evaluator.evaluate(
        dataset,
        model,
    )

    assert result.predicted == (
        14.0,
        15.0,
        16.0,
    )


def test_evaluation_dates(
    dataset,
):
    evaluator = ForecastEvaluator(
        evaluation_steps=3
    )

    model = DummyForecaster(
        predictions=[
            15.0,
            16.0,
            17.0,
        ]
    )

    result = evaluator.evaluate(
        dataset,
        model,
    )

    expected = tuple(
        dataset.data[
            "ds"
        ].iloc[-3:]
    )

    assert result.dates == expected


def test_evaluation_metadata(
    dataset,
):
    evaluator = ForecastEvaluator(
        evaluation_steps=3
    )

    model = DummyForecaster(
        predictions=[
            15.0,
            16.0,
            17.0,
        ]
    )

    result = evaluator.evaluate(
        dataset,
        model,
    )

    assert (
        result.metadata[
            "training_points"
        ]
        == 5
    )

    assert (
        result.metadata[
            "validation_points"
        ]
        == 3
    )


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------


def test_rmse():
    actual = [
        1.0,
        2.0,
        3.0,
    ]

    predicted = [
        1.0,
        2.0,
        4.0,
    ]

    result = ForecastEvaluator.rmse(
        actual,
        predicted,
    )

    assert result == pytest.approx(
        np.sqrt(1 / 3)
    )


def test_mae():
    result = ForecastEvaluator.mae(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 4.0],
    )

    assert result == pytest.approx(
        1 / 3
    )


def test_mape():
    result = ForecastEvaluator.mape(
        [10.0, 20.0],
        [11.0, 18.0],
    )

    assert result == pytest.approx(
        10.0
    )


def test_mape_ignores_zero_actual():
    result = ForecastEvaluator.mape(
        [0.0, 10.0],
        [5.0, 11.0],
    )

    assert result == pytest.approx(
        10.0
    )


def test_mape_all_zero_actual_raises():
    with pytest.raises(
        ValueError,
        match=(
            "MAPE cannot be calculated"
        ),
    ):
        ForecastEvaluator.mape(
            [0.0, 0.0],
            [1.0, 2.0],
        )


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------


def test_dataset_too_short_raises(
    dataset,
):
    evaluator = ForecastEvaluator(
        evaluation_steps=len(
            dataset
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "evaluation_steps must be "
            "smaller than the dataset length"
        ),
    ):
        evaluator.evaluate(
            dataset,
            DummyForecaster(),
        )


def test_invalid_dataset_type_raises():
    evaluator = ForecastEvaluator(
        evaluation_steps=2
    )

    with pytest.raises(
        TypeError,
        match=(
            "dataset must be a "
            "BatteryDataset"
        ),
    ):
        evaluator.evaluate(
            [1, 2, 3],
            DummyForecaster(),
        )


def test_model_without_fit_raises(
    dataset,
):
    class InvalidModel:
        def predict(self, steps):
            return None

    evaluator = ForecastEvaluator(
        evaluation_steps=2
    )

    with pytest.raises(
        TypeError,
        match=(
            "model must implement "
            "fit"
        ),
    ):
        evaluator.evaluate(
            dataset,
            InvalidModel(),
        )


def test_model_without_predict_raises(
    dataset,
):
    class InvalidModel:
        def fit(self, dataset):
            pass

    evaluator = ForecastEvaluator(
        evaluation_steps=2
    )

    with pytest.raises(
        TypeError,
        match=(
            "model must implement "
            "predict"
        ),
    ):
        evaluator.evaluate(
            dataset,
            InvalidModel(),
        )


def test_wrong_prediction_length_raises(
    dataset,
):
    evaluator = ForecastEvaluator(
        evaluation_steps=3
    )

    model = DummyForecaster(
        predictions=[
            15.0,
            16.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match=(
            "model prediction length "
            "must match evaluation_steps"
        ),
    ):
        evaluator.evaluate(
            dataset,
            model,
        )


def test_metric_length_mismatch_raises():
    with pytest.raises(
        ValueError,
        match=(
            "actual and predicted must "
            "have the same length"
        ),
    ):
        ForecastEvaluator.rmse(
            [1.0, 2.0],
            [1.0],
        )


def test_non_finite_actual_raises():
    with pytest.raises(
        ValueError,
        match=(
            "actual must contain only "
            "finite values"
        ),
    ):
        ForecastEvaluator.rmse(
            [1.0, np.nan],
            [1.0, 2.0],
        )


def test_non_finite_prediction_raises():
    with pytest.raises(
        ValueError,
        match=(
            "predicted must contain only "
            "finite values"
        ),
    ):
        ForecastEvaluator.rmse(
            [1.0, 2.0],
            [1.0, np.inf],
        )


def test_repr():
    evaluator = ForecastEvaluator(
        evaluation_steps=6
    )

    assert repr(evaluator) == (
        "ForecastEvaluator("
        "evaluation_steps=6)"
    )


def test_preprocessor_receives_training_partition_only():

    data = pd.DataFrame(
        {
            "ds": pd.date_range(
                start="2023-03-01",
                periods=8,
                freq="QS-MAR",
            ),
            "microVolt": [
                30.0,
                31.0,
                32.0,
                33.0,
                34.0,
                35.0,
                40.0,
                41.0,
            ],
            "is_observed": [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
        }
    )

    dataset = BatteryDataset(
        data
    )

    preprocessor = (
        RecordingPreprocessor()
    )

    model = RecordingModel(
        predictions=[
            40.0,
            41.0,
        ]
    )

    evaluator = ForecastEvaluator(
        evaluation_steps=2,
        preprocessor=preprocessor,
    )

    evaluator.evaluate(
        dataset,
        model,
    )

    received = (
        preprocessor
        .received_dataset
        .data
    )

    assert len(
        received
    ) == 6

    assert (
        received["ds"].max()
        == pd.Timestamp(
            "2024-06-01"
        )
    )

    assert (
        pd.Timestamp(
            "2024-09-01"
        )
        not in received[
            "ds"
        ].values
    )

    assert (
        pd.Timestamp(
            "2024-12-01"
        )
        not in received[
            "ds"
        ].values
    )


def test_model_receives_preprocessed_training_data():

    data = pd.DataFrame(
        {
            "ds": pd.date_range(
                start="2023-03-01",
                periods=8,
                freq="QS-MAR",
            ),
            "microVolt": [
                30.0,
                31.0,
                None,
                33.0,
                34.0,
                35.0,
                40.0,
                41.0,
            ],
            "is_observed": [
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
            ],
        }
    )

    dataset = BatteryDataset(
        data
    )

    model = RecordingModel(
        predictions=[
            40.0,
            41.0,
        ]
    )

    evaluator = ForecastEvaluator(
        evaluation_steps=2,
        preprocessor=(
            TemporalPreprocessor()
        ),
    )

    evaluator.evaluate(
        dataset,
        model,
    )

    fitted_data = (
        model
        .fitted_dataset
        .data
    )

    assert not (
        fitted_data[
            "microVolt"
        ]
        .isna()
        .any()
    )

    assert (
        fitted_data.loc[
            2,
            "microVolt",
        ]
        == pytest.approx(
            32.0
        )
    )


def test_preprocessing_preserves_observation_provenance():

    data = pd.DataFrame(
        {
            "ds": pd.date_range(
                start="2023-03-01",
                periods=8,
                freq="QS-MAR",
            ),
            "microVolt": [
                30.0,
                31.0,
                None,
                33.0,
                34.0,
                35.0,
                40.0,
                41.0,
            ],
            "is_observed": [
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
            ],
        }
    )

    dataset = BatteryDataset(
        data
    )

    model = RecordingModel(
        predictions=[
            40.0,
            41.0,
        ]
    )

    evaluator = ForecastEvaluator(
        evaluation_steps=2,
        preprocessor=(
            TemporalPreprocessor()
        ),
    )

    evaluator.evaluate(
        dataset,
        model,
    )

    fitted_data = (
        model
        .fitted_dataset
        .data
    )

    assert (
        fitted_data.loc[
            2,
            "is_observed",
        ]
        == False
    )


def test_holdout_values_do_not_affect_training_preprocessing():

    common_dates = pd.date_range(
        start="2023-03-01",
        periods=8,
        freq="QS-MAR",
    )

    first = pd.DataFrame(
        {
            "ds": common_dates,
            "microVolt": [
                30.0,
                31.0,
                None,
                33.0,
                34.0,
                35.0,
                40.0,
                41.0,
            ],
            "is_observed": [
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
            ],
        }
    )

    second = first.copy(
        deep=True
    )

    second.loc[
        6,
        "microVolt",
    ] = 4000.0

    second.loc[
        7,
        "microVolt",
    ] = 5000.0

    dataset_a = BatteryDataset(
        first
    )

    dataset_b = BatteryDataset(
        second
    )

    model_a = RecordingModel(
        predictions=[
            40.0,
            41.0,
        ]
    )

    model_b = RecordingModel(
        predictions=[
            4000.0,
            5000.0,
        ]
    )

    evaluator_a = ForecastEvaluator(
        evaluation_steps=2,
        preprocessor=(
            TemporalPreprocessor()
        ),
    )

    evaluator_b = ForecastEvaluator(
        evaluation_steps=2,
        preprocessor=(
            TemporalPreprocessor()
        ),
    )

    evaluator_a.evaluate(
        dataset_a,
        model_a,
    )

    evaluator_b.evaluate(
        dataset_b,
        model_b,
    )

    pd.testing.assert_series_equal(
        model_a.fitted_dataset.data[
            "microVolt"
        ],
        model_b.fitted_dataset.data[
            "microVolt"
        ],
        check_names=False,
    )


def test_holdout_actual_values_are_not_preprocessed():

    data = pd.DataFrame(
        {
            "ds": pd.date_range(
                start="2023-03-01",
                periods=8,
                freq="QS-MAR",
            ),
            "microVolt": [
                30.0,
                31.0,
                None,
                33.0,
                34.0,
                35.0,
                40.0,
                41.0,
            ],
            "is_observed": [
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
            ],
        }
    )

    dataset = BatteryDataset(
        data
    )

    model = RecordingModel(
        predictions=[
            39.0,
            42.0,
        ]
    )

    evaluator = ForecastEvaluator(
        evaluation_steps=2,
        preprocessor=(
            TemporalPreprocessor()
        ),
    )

    result = evaluator.evaluate(
        dataset,
        model,
    )

    assert result.actual == (
        40.0,
        41.0,
    )


def test_evaluation_preprocessing_does_not_modify_original_dataset():

    data = pd.DataFrame(
        {
            "ds": pd.date_range(
                start="2023-03-01",
                periods=8,
                freq="QS-MAR",
            ),
            "microVolt": [
                30.0,
                31.0,
                None,
                33.0,
                34.0,
                35.0,
                40.0,
                41.0,
            ],
            "is_observed": [
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
            ],
        }
    )

    dataset = BatteryDataset(
        data
    )

    original = dataset.data.copy(
        deep=True
    )

    model = RecordingModel(
        predictions=[
            40.0,
            41.0,
        ]
    )

    evaluator = ForecastEvaluator(
        evaluation_steps=2,
        preprocessor=(
            TemporalPreprocessor()
        ),
    )

    evaluator.evaluate(
        dataset,
        model,
    )

    pd.testing.assert_frame_equal(
        dataset.data,
        original,
    )


def test_evaluator_preprocessor_defaults_to_none():

    evaluator = ForecastEvaluator(
        evaluation_steps=2
    )

    assert (
        evaluator.preprocessor
        is None
    )


def test_evaluation_without_preprocessor_remains_supported():

    data = pd.DataFrame(
        {
            "ds": pd.date_range(
                start="2023-03-01",
                periods=8,
                freq="QS-MAR",
            ),
            "microVolt": [
                30.0,
                31.0,
                32.0,
                33.0,
                34.0,
                35.0,
                40.0,
                41.0,
            ],
        }
    )

    dataset = BatteryDataset(
        data
    )

    model = RecordingModel(
        predictions=[
            40.0,
            41.0,
        ]
    )

    evaluator = ForecastEvaluator(
        evaluation_steps=2,
    )

    result = evaluator.evaluate(
        dataset,
        model,
    )

    assert (
        result.rmse
        == pytest.approx(
            0.0
        )
    )


def test_invalid_preprocessor_raises():

    with pytest.raises(
        TypeError,
        match=(
            "preprocessor must provide "
            "a callable fit_transform method"
        ),
    ):

        ForecastEvaluator(
            evaluation_steps=2,
            preprocessor=object(),
        )


def test_evaluation_metadata_records_no_preprocessing():

    data = pd.DataFrame(
        {
            "ds": pd.date_range(
                "2023-03-01",
                periods=8,
                freq="QS-MAR",
            ),
            "microVolt": [
                30.0,
                31.0,
                32.0,
                33.0,
                34.0,
                35.0,
                40.0,
                41.0,
            ],
        }
    )

    dataset = BatteryDataset(data)

    model = RecordingModel(
        predictions=[
            40.0,
            41.0,
        ]
    )

    evaluator = ForecastEvaluator(
        evaluation_steps=2,
    )

    result = evaluator.evaluate(
        dataset,
        model,
    )

    assert (
        result.metadata[
            "preprocessing"
        ]
        == {
            "enabled": False,
        }
    )


def test_evaluation_metadata_records_temporal_preprocessing():

    data = pd.DataFrame(
        {
            "ds": pd.date_range(
                "2023-03-01",
                periods=8,
                freq="QS-MAR",
            ),
            "microVolt": [
                30.0,
                31.0,
                None,
                33.0,
                34.0,
                35.0,
                40.0,
                41.0,
            ],
            "is_observed": [
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
            ],
        }
    )

    dataset = BatteryDataset(data)

    model = RecordingModel(
        predictions=[
            40.0,
            41.0,
        ]
    )

    evaluator = ForecastEvaluator(
        evaluation_steps=2,
        preprocessor=(
            TemporalPreprocessor(
                method="linear",
                fill_edges=True,
            )
        ),
    )

    result = evaluator.evaluate(
        dataset,
        model,
    )

    preprocessing = (
        result.metadata[
            "preprocessing"
        ]
    )

    assert preprocessing[
        "enabled"
    ]

    assert (
        preprocessing["name"]
        == "TemporalPreprocessor"
    )

    assert (
        preprocessing["method"]
        == "linear"
    )

    assert (
        preprocessing[
            "fill_edges"
        ]
        is True
    )


def test_evaluation_rejects_missing_validation_targets():

    data = pd.DataFrame(
        {
            "ds": pd.date_range(
                "2023-03-01",
                periods=8,
                freq="QS-MAR",
            ),
            "microVolt": [
                30.0,
                31.0,
                32.0,
                33.0,
                34.0,
                35.0,
                None,
                41.0,
            ],
            "is_observed": [
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
            ],
        }
    )

    dataset = BatteryDataset(data)

    model = RecordingModel(
        predictions=[
            40.0,
            41.0,
        ]
    )

    evaluator = ForecastEvaluator(
        evaluation_steps=2,
        preprocessor=(
            TemporalPreprocessor()
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "validation partition contains "
            "missing target values"
        ),
    ):
        evaluator.evaluate(
            dataset,
            model,
        )