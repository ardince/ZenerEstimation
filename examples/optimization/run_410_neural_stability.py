"""Repeated-seed neural stability analysis for dataset 732B-5610410."""

from __future__ import annotations

import numpy as np

from zenerestimation.data.dataset import BatteryDataset
from pathlib import Path

from zenerestimation.forecasting.hybrid import (
    KalmanLSTMForecaster,
    LinearTrendLSTMForecaster,
)
from zenerestimation.forecasting.neural import (
    GRUForecaster,
    LSTMForecaster,
)
from zenerestimation.optimization import (
    ExpandingWindowSplitter,
    OptimizationEvaluator,
    StabilityEvaluator,
    StabilityArtifact,
)

from zenerestimation.data.temporal.preprocessor import (
    TemporalPreprocessor,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BATTERY = "732B-5610410"

DATA_PATH = Path(
    "datasets/processed/732B-5610410.csv"
)

STABILITY_OUTPUT_ROOT = Path(
    "results/732B-5610410/optimized_benchmark/stability"
)

# ---------------------------------------------------------
# Load processed dataset
# ---------------------------------------------------------

dataset = BatteryDataset.from_processed_csv(
    DATA_PATH,
    battery=BATTERY,
)

development_dataset = dataset


STABILITY_SEEDS = (0, 1, 2, 3, 4)

EPOCHS = 100
BATCH_SIZE = 8

MODEL_CONFIGS = {
    "LSTM": {
        "window": 4,
        "units": 32,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
    },
    "GRU": {
        "window": 4,
        "units": 24,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
    },
    "LinearTrendLSTM": {
        "window": 12,
        "units": 32,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
    },
    "KalmanLSTM": {
        "window": 8,
        "units": 32,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
    },
}

STABILITY_DIRECTORIES = {
    "LSTM": "lstm",
    "GRU": "gru",
    "LinearTrendLSTM": "linear_trend_lstm",
    "KalmanLSTM": "kalman_lstm",
}


# ---------------------------------------------------------
# Leakage-safe training preprocessing
# ---------------------------------------------------------

preprocessor = TemporalPreprocessor(
    method="linear",
    fill_edges=True,
)

splitter = ExpandingWindowSplitter(
    folds=3,
    validation_steps=6,
)

evaluator = OptimizationEvaluator(
    splitter,
    metric="rmse",
    preprocessor=preprocessor,
)


def lstm_factory(**params):
    return LSTMForecaster(
        window=params["window"],
        units=params["units"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        seed=params["seed"],
    )


def gru_factory(**params):
    return GRUForecaster(
        window=params["window"],
        units=params["units"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        seed=params["seed"],
    )


def linear_trend_lstm_factory(**params):
    window = params["window"]

    residual_lstm = LSTMForecaster(
        window=window,
        units=params["units"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        seed=params["seed"],
    )

    return LinearTrendLSTMForecaster(
        lstm_model=residual_lstm,
        window=window,
    )


def kalman_lstm_factory(**params):
    window = params["window"]

    residual_lstm = LSTMForecaster(
        window=window,
        units=params["units"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        seed=params["seed"],
    )

    return KalmanLSTMForecaster(
        window=window,
        lstm_model=residual_lstm,
    )


def save_stability_artifact(
    result,
    *,
    model_name: str,
) -> Path:
    """Persist one completed stability result."""

    output_directory = (
        STABILITY_OUTPUT_ROOT
        / STABILITY_DIRECTORIES[model_name]
    )

    return StabilityArtifact(
        result
    ).save(
        output_directory
    )


RUNS = (
    (
        "LSTM",
        lstm_factory,
    ),
    (
        "GRU",
        gru_factory,
    ),
    (
        "LinearTrendLSTM",
        linear_trend_lstm_factory,
    ),
    (
        "KalmanLSTM",
        kalman_lstm_factory,
    ),
)


results = {}

for model_name, factory in RUNS:
    print()
    print("=" * 72)
    print(f"Stability analysis: {model_name}")
    print("=" * 72)

    stability = StabilityEvaluator(
        evaluator,
        model=model_name,
    )

    result = stability.evaluate(
        development_dataset,
        factory,
        params=MODEL_CONFIGS[
            model_name
        ],
        seeds=STABILITY_SEEDS,
    )

    results[model_name] = result


    artifact_path = save_stability_artifact(
        result,
        model_name=model_name,
    )


    for seed_result in result.seeds:
        print(
            f"seed={seed_result.seed:<2d} "
            f"RMSE={seed_result.metrics['rmse']:.6f} "
            f"MAE={seed_result.metrics['mae']:.6f} "
            f"MAPE={seed_result.metrics['mape']:.6f}%"
        )

    print()
    print(
        f"RMSE mean : {result.mean_score:.6f}"
    )
    print(
        f"RMSE std  : {result.std_score:.6f}"
    )
    print(
        f"RMSE min  : {result.min_score:.6f}"
    )
    print(
        f"RMSE max  : {result.max_score:.6f}"
    )


    cv = (
        result.std_score
        / result.mean_score
        if result.mean_score != 0.0
        else np.nan
    )

    print(
        f"RMSE CV   : {cv:.6f}"
    )

    print(
        f"Artifact  : {artifact_path}"
    )