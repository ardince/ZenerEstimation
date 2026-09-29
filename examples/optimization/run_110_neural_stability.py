"""Repeated-seed neural stability analysis for dataset 732B-5610110.

Sprint 14.7F

This runner measures the sensitivity of already-selected neural and hybrid
configurations to random initialization/training stochasticity.

Important scientific rules
--------------------------
1. Hyperparameters are frozen. This script does not optimize them.
2. Seeds are replications, not candidate parameters.
3. No "best seed" is selected.
4. Only the development period through 2022-12 is used.
5. The final five-quarter benchmark remains untouched.
6. TemporalPreprocessor is applied inside temporal validation so that
   preprocessing remains training-only and leakage-safe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from zenerestimation.data import BatteryDataset
from zenerestimation.data.temporal.preprocessor import (
    TemporalPreprocessor,
)

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
    FailedSeedResult,
)


# =====================================================================
# Experiment identity
# =====================================================================

BATTERY = "732B-5610110"

# Use the same processed 110 dataset used by the successful real
# optimization runners.
DATA_PATH = Path(
    "datasets/processed/732B-5610110.csv"
)

DEVELOPMENT_END = pd.Timestamp(
    "2022-12-01"
)

BENCHMARK_START = pd.Timestamp(
    "2023-03-01"
)

BENCHMARK_END = pd.Timestamp(
    "2024-03-01"
)

BENCHMARK_STEPS = 5


# =====================================================================
# Stability protocol
# =====================================================================

STABILITY_SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

FOLDS = 3
VALIDATION_STEPS = 5

EPOCHS = 100
BATCH_SIZE = 8


# =====================================================================
# Frozen optimized configurations
# =====================================================================

MODEL_CONFIGS = {
    "LSTM": {
        "window": 4,
        "units": 32,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
    },
    "GRU": {
        "window": 4,
        "units": 32,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
    },
    "LinearTrendLSTM": {
        "window": 12,
        "units": 24,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
    },
    "KalmanLSTM": {
        "window": 12,
        "units": 24,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
    },
}


# =====================================================================
# Model factories
# =====================================================================


def lstm_factory(**params):
    """Create a fresh LSTM for one temporal fold."""

    return LSTMForecaster(
        window=params["window"],
        units=params["units"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        seed=params["seed"],
    )


def gru_factory(**params):
    """Create a fresh GRU for one temporal fold."""

    return GRUForecaster(
        window=params["window"],
        units=params["units"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        seed=params["seed"],
    )


def linear_trend_lstm_factory(**params):
    """Create a fresh LinearTrendLSTM hybrid."""

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
    """Create a fresh KalmanLSTM hybrid.

    The established hybrid Kalman baseline is intentionally preserved.
    Standalone Kalman optimization parameters are not injected here.
    """

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


MODEL_RUNS = (
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


# =====================================================================
# Dataset loading
# =====================================================================


def load_development_dataset() -> BatteryDataset:
    """Load 110 and isolate the development period.

    The benchmark period from 2023-03 through 2024-03 is deliberately
    excluded before repeated-seed temporal validation begins.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed dataset not found: "
            f"{DATA_PATH.resolve()}"
        )

    dataframe = pd.read_csv(
        DATA_PATH
    )

    if "ds" not in dataframe.columns:
        raise ValueError(
            "Processed dataset must contain a 'ds' column"
        )

    if "microVolt" not in dataframe.columns:
        raise ValueError(
            "Processed dataset must contain a "
            "'microVolt' column"
        )

    dataframe = dataframe.copy()

    dataframe["ds"] = pd.to_datetime(
        dataframe["ds"],
        errors="raise",
    )

    dataframe = (
        dataframe
        .sort_values("ds")
        .reset_index(drop=True)
    )

    development = dataframe.loc[
        dataframe["ds"]
        <= DEVELOPMENT_END
    ].copy()

    if development.empty:
        raise ValueError(
            "Development dataset is empty"
        )

    if development["ds"].max() != DEVELOPMENT_END:
        raise ValueError(
            "Unexpected development boundary: "
            f"{development['ds'].max()}. "
            f"Expected {DEVELOPMENT_END}."
        )

    # Critical leakage guard:
    #
    # None of the final benchmark observations may enter repeated-seed
    # internal validation.
    if (
        development["ds"]
        >= BENCHMARK_START
    ).any():
        raise RuntimeError(
            "Benchmark observations leaked into "
            "the development dataset"
        )

    return BatteryDataset(
        dataframe=development
    )


# =====================================================================
# Internal temporal evaluator
# =====================================================================


def create_optimization_evaluator() -> OptimizationEvaluator:
    """Create the same leakage-safe temporal validation contract."""

    splitter = ExpandingWindowSplitter(
        folds=FOLDS,
        validation_steps=VALIDATION_STEPS,
    )

    # Keep this configuration identical to the successful real
    # optimization runners.
    preprocessor = TemporalPreprocessor()

    return OptimizationEvaluator(
        splitter,
        metric="rmse",
        preprocessor=preprocessor,
    )


# =====================================================================
# Reporting
# =====================================================================


def print_dataset_summary(
    dataset: BatteryDataset,
) -> None:
    """Print the scientific boundary used by this run."""

    data = dataset.data

    print()
    print("=" * 72)
    print("Dataset 110 — repeated-seed stability analysis")
    print("=" * 72)

    print(
        f"Battery             : {BATTERY}"
    )

    print(
        f"Development rows    : {len(data)}"
    )

    print(
        "Development start   : "
        f"{pd.Timestamp(data['ds'].min()).date()}"
    )

    print(
        "Development end     : "
        f"{pd.Timestamp(data['ds'].max()).date()}"
    )

    print(
        "Benchmark reserved  : "
        f"{BENCHMARK_START.date()} "
        f"through {BENCHMARK_END.date()}"
    )

    print(
        f"Benchmark steps     : {BENCHMARK_STEPS}"
    )

    print(
        f"Validation folds    : {FOLDS}"
    )

    print(
        "Validation steps    : "
        f"{VALIDATION_STEPS}"
    )

    print(
        "Stability seeds     : "
        f"{STABILITY_SEEDS}"
    )


def print_stability_result(result) -> None:
    """Print per-seed and aggregate stability evidence."""

    print()

    successful_results = []
    failed_results = []

    for seed_result in result.seeds:

        if isinstance(
            seed_result,
            FailedSeedResult,
        ):
            failed_results.append(seed_result)

            print(
                f"seed={seed_result.seed:<2d} "
                f"FAILED "
                f"{seed_result.error_type}: "
                f"{seed_result.error_message}"
            )

            continue

        successful_results.append(seed_result)

        print(
            f"seed={seed_result.seed:<2d} "
            f"RMSE={seed_result.metrics['rmse']:.6f} "
            f"MAE={seed_result.metrics['mae']:.6f} "
            f"MAPE={seed_result.metrics['mape']:.6f}%"
        )

    successful_count = len(successful_results)
    failed_count = len(failed_results)
    run_count = len(result.seeds)

    print()

    print(
        f"Successful runs : "
        f"{successful_count}/{run_count}"
    )

    print(
        f"Failed runs     : "
        f"{failed_count}/{run_count}"
    )

    print()

    # Aggregate RMSE directly from successful SeedResult objects.
    #
    # This keeps the example runner compatible with the current
    # StabilityResult public contract and naturally excludes failed seeds.
    scores = np.asarray(
        [
            seed_result.score
            for seed_result in successful_results
        ],
        dtype=float,
    )

    if scores.size == 0:
        print(
            "No successful seed runs are available "
            "for aggregate statistics."
        )
        return

    mean_score = float(
        np.mean(scores)
    )

    std_score = float(
        np.std(
            scores,
            ddof=0,
        )
    )

    min_score = float(
        np.min(scores)
    )

    max_score = float(
        np.max(scores)
    )

    cv = (
        std_score / mean_score
        if mean_score != 0.0
        else np.nan
    )

    print(
        f"RMSE mean : {mean_score:.6f}"
    )

    print(
        f"RMSE std  : {std_score:.6f}"
    )

    print(
        f"RMSE min  : {min_score:.6f}"
    )

    print(
        f"RMSE max  : {max_score:.6f}"
    )

    print(
        f"RMSE CV   : {cv:.6f}"
    )


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    """Run repeated-seed analysis for all optimized 110 neural models."""

    development_dataset = (
        load_development_dataset()
    )

    print_dataset_summary(
        development_dataset
    )

    evaluator = (
        create_optimization_evaluator()
    )

    results = {}

    for model_name, model_factory in MODEL_RUNS:
        print()
        print("=" * 72)
        print(
            f"Stability analysis: {model_name}"
        )
        print("=" * 72)

        params = MODEL_CONFIGS[
            model_name
        ]

        stability = StabilityEvaluator(
            evaluator,
            model=model_name,
        )

        result = stability.evaluate(
            development_dataset,
            model_factory,
            params=params,
            seeds=STABILITY_SEEDS,
        )

        results[
            model_name
        ] = result

        print_stability_result(
            result
        )

    print()
    print("=" * 72)
    print("Dataset 110 stability analysis complete")
    print("=" * 72)

    print(
        "No seed was selected. "
        "All successful seed replications contribute "
        "to the reported stability distributions."
    )

    print(
        "The final benchmark observations remained "
        "outside repeated-seed validation."
    )


if __name__ == "__main__":
    main()