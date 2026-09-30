"""Run the frozen optimized multi-model benchmark for dataset 410."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.data.temporal import TemporalPreprocessor
from zenerestimation.evaluation.benchmark import (
    BenchmarkModelSpec,
    OptimizedBenchmark,
)
from zenerestimation.evaluation.evaluator import ForecastEvaluator
from zenerestimation.forecasting.arima import ARIMAForecaster
from zenerestimation.forecasting.kalman import KalmanForecaster
from zenerestimation.forecasting.neural.lstm import LSTMForecaster
from zenerestimation.forecasting.neural.gru import GRUForecaster
from zenerestimation.forecasting.hybrid.linear_trend_lstm import (
    LinearTrendLSTMForecaster,
)
from zenerestimation.forecasting.hybrid.kalman_lstm import (
    KalmanLSTMForecaster,
)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------


DATA_PATH = Path(
    "datasets/processed/732B-5610110.csv"
)

BATTERY = "732B-5610110"

BENCHMARK_STEPS = 5

DEVELOPMENT_END = "2022-12-01"
BENCHMARK_START = "2023-03-01"
BENCHMARK_END = "2024-03-01"

EVALUATION_END = pd.Timestamp(
    year=2024,
    month=3,
    day=1,
)

# ---------------------------------------------------------------------
# Frozen neural training configuration
# ---------------------------------------------------------------------

EPOCHS = 100
BATCH_SIZE = 8
SEED = 42


# ---------------------------------------------------------------------
# Hybrid factories
# ---------------------------------------------------------------------


def create_linear_trend_lstm(
    window,
    units,
    epochs,
    batch_size,
    seed,
):
    """Create a fresh optimized LinearTrendLSTM model."""

    residual_model = LSTMForecaster(
        window=window,
        units=units,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )

    return LinearTrendLSTMForecaster(
        lstm_model=residual_model,
        window=window,
    )


def create_kalman_lstm(
    window,
    units,
    epochs,
    batch_size,
    seed,
):
    """Create a fresh optimized KalmanLSTM model."""

    residual_model = LSTMForecaster(
        window=window,
        units=units,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )

    # Preserve the established default Kalman baseline.
    return KalmanLSTMForecaster(
        window=window,
        lstm_model=residual_model,
    )


# ---------------------------------------------------------------------
# Frozen optimized specifications
# ---------------------------------------------------------------------


def create_model_specs():
    """Create the six frozen optimized 110 model specifications."""

    return (
        BenchmarkModelSpec(
            name="ARIMA",
            factory=ARIMAForecaster,
            params={
                "order": (
                    5,
                    1,
                    3,
                ),
            },
        ),
        BenchmarkModelSpec(
            name="Kalman",
            factory=KalmanForecaster,
            params={
                "dt": 0.25,
                "process_noise": 0.01,
                "drift_noise": 0.001,
                "adaptive": True,
                "regime_factor": 2.0,
                "regime_multiplier": 10.0,
            },
        ),
        BenchmarkModelSpec(
            name="LSTM",
            factory=LSTMForecaster,
            params={
                "window": 4,
                "units": 32,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
            },
        ),
        BenchmarkModelSpec(
            name="GRU",
            factory=GRUForecaster,
            params={
                "window": 4,
                "units": 32,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
            },
        ),
        BenchmarkModelSpec(
            name="LinearTrendLSTM",
            factory=create_linear_trend_lstm,
            params={
                "window": 12,
                "units": 24,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
            },
        ),
        BenchmarkModelSpec(
            name="KalmanLSTM",
            factory=create_kalman_lstm,
            params={
                "window": 12,
                "units": 24,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
            },
        ),
    )


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------


def load_dataset():
    """Load the canonical processed 110 dataset."""

    return BatteryDataset.from_processed_csv(
        DATA_PATH,
        battery=BATTERY,
    )


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------


def print_dataset_summary(
    dataset,
):
    """Print the fixed final-benchmark boundary."""

    print()
    print("=" * 72)
    print(
        "Dataset 110 — optimized multi-model benchmark"
    )
    print("=" * 72)

    print(
        f"Battery             : {BATTERY}"
    )
    
    print(
        f"Development end     : {DEVELOPMENT_END}"
    )
    print(
        f"Benchmark reserved  : "
        f"{BENCHMARK_START} through {BENCHMARK_END}"
    )
    print(
        f"Benchmark steps     : {BENCHMARK_STEPS}"
    )
    print(
        f"Neural seed         : {SEED}"
    )


def print_frozen_specs(
    specs,
):
    """Print frozen configurations before benchmark execution."""

    print()
    print("=" * 72)
    print(
        "Frozen optimized configurations"
    )
    print("=" * 72)

    for spec in specs:
        print()
        print(
            f"{spec.name}:"
        )

        for name, value in spec.params.items():
            print(
                f"  {name:<20}: {value}"
            )


def print_results(
    results,
):
    """Print standardized final benchmark metrics."""

    print()
    print("=" * 72)
    print(
        "Final benchmark results"
    )
    print("=" * 72)

    print(
        f"{'Model':<20}"
        f"{'RMSE':>12}"
        f"{'MAE':>12}"
        f"{'MAPE (%)':>14}"
    )

    print("-" * 58)

    for model, result in results.items():
        print(
            f"{model:<20}"
            f"{result.rmse:>12.6f}"
            f"{result.mae:>12.6f}"
            f"{result.mape:>14.6f}"
        )


def print_forecasts(
    results,
):
    """Print final holdout observations and model predictions."""

    print()
    print("=" * 72)
    print(
        "Final benchmark forecasts"
    )
    print("=" * 72)

    for model, result in results.items():
        print()
        print(
            f"{model}:"
        )

        for index, (
            actual,
            predicted,
        ) in enumerate(
            zip(
                result.actual,
                result.predicted,
            ),
            start=1,
        ):
            print(
                f"  step={index:<2d} "
                f"actual={actual:.6f} "
                f"predicted={predicted:.6f}"
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():

    dataset = load_dataset()

    specs = create_model_specs()

    print_dataset_summary(
        dataset
    )

    print_frozen_specs(
        specs
    )

    evaluator = ForecastEvaluator(
        evaluation_steps=BENCHMARK_STEPS,
        evaluation_end=EVALUATION_END,
        preprocessor=TemporalPreprocessor(),
    )

    benchmark = OptimizedBenchmark(
        evaluator=evaluator
    )

    results = benchmark.evaluate(
        dataset=dataset,
        specs=specs,
    )

    print_results(
        results
    )

    print_forecasts(
        results
    )

    print()
    print("=" * 72)
    print(
        "Dataset 110 optimized benchmark complete"
    )
    print("=" * 72)

    print(
        "All configurations were frozen before final "
        "benchmark evaluation."
    )

    print(
        "No benchmark result was used for parameter "
        "selection or seed selection."
    )


if __name__ == "__main__":
    main()