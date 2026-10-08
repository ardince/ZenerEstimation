"""Run the frozen optimized multi-model benchmark for dataset 410."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

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

from zenerestimation.evaluation.benchmark_integration import (
    comparison_from_evaluations,
)

from zenerestimation.optimization import (
    BenchmarkArtifact,
    BenchmarkProvenance,
    BenchmarkReport,
)
from zenerestimation.visualization import (
    BenchmarkPlotter,
    BenchmarkHistoryPlotter,
)
from zenerestimation.visualization.benchmark_best_worst_history import (
    BenchmarkBestWorstHistoryPlotter,
)

from zenerestimation.optimization.evidence import (
    copy_evidence_files,  validate_provenance_references,
)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------


DATA_PATH = Path(
    "datasets/processed/732B-5610410.csv"
)

BATTERY = "732B-5610410"

BENCHMARK_STEPS = 6

DEVELOPMENT_END = "2023-12-01"
BENCHMARK_START = "2024-03-01"
BENCHMARK_END = "2025-06-01"


RESULT_DIRECTORY = Path(
    "results"
) / BATTERY / "optimized_benchmark"


OPTIMIZATION_REFERENCES = {
    "ARIMA": Path(
        "optimization/arima/optimization.json"
    ),
    "Kalman": Path(
        "optimization/kalman/optimization.json"
    ),
    "LSTM": Path(
        "optimization/lstm/optimization.json"
    ),
    "GRU": Path(
        "optimization/gru/optimization.json"
    ),
    "LinearTrendLSTM": Path(
        "optimization/linear_trend_lstm/"
        "optimization.json"
    ),
    "KalmanLSTM": Path(
        "optimization/kalman_lstm/"
        "optimization.json"
    ),
}

OPTIMIZATION_SOURCES = {
    "ARIMA": Path(
        "results/732B-5610410/arima/"
        "20260923_161237_29/optimization.json"
    ),
    "Kalman": Path(
        "results/732B-5610410/kalman/"
        "20260923_160643_16/optimization.json"
    ),
    "LSTM": Path(
        "results/732B-5610410/lstm/"
        "20260924_110003_3/optimization.json"
    ),
    "GRU": Path(
        "results/732B-5610410/gru/"
        "20260924_130358_2/optimization.json"
    ),
    "LinearTrendLSTM": Path(
        "results/732B-5610410/lineartrendlstm/"
        "20260925_121224_1/optimization.json"
    ),
    "KalmanLSTM": Path(
        "results/732B-5610410/kalman_lstm/"
        "20260925_123331_3/optimization.json"
    ),
}


STABILITY_REFERENCES = {
    "LSTM": Path(
        "stability/lstm/stability.json"
    ),
    "GRU": Path(
        "stability/gru/stability.json"
    ),
    "LinearTrendLSTM": Path(
        "stability/linear_trend_lstm/stability.json"
    ),
    "KalmanLSTM": Path(
        "stability/kalman_lstm/stability.json"
    ),
}


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
    """Create the six frozen optimized 410 model specifications."""

    return (
        BenchmarkModelSpec(
            name="ARIMA",
            factory=ARIMAForecaster,
            params={
                "order": (
                    12,
                    1,
                    1,
                ),
            },
        ),
        BenchmarkModelSpec(
            name="Kalman",
            factory=KalmanForecaster,
            params={
                "dt": 0.25,
                "process_noise": 0.01,
                "drift_noise": 1e-5,
                "adaptive": True,
                "regime_factor": 3.0,
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
                "units": 24,
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
                "units": 32,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
            },
        ),
        BenchmarkModelSpec(
            name="KalmanLSTM",
            factory=create_kalman_lstm,
            params={
                "window": 8,
                "units": 32,
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
    """Load the canonical processed 410 dataset."""

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
        "Dataset 410 — optimized multi-model benchmark"
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



def print_forecasts(results):
    """Display date-indexed forecasts and signed errors."""

    print()
    print("=" * 88)
    print("Final benchmark forecasts")
    print("=" * 88)

    for model, result in results.items():
        print()
        print(f"{model}:")
        print(
            f"{'Step':<6}"
            f"{'Date':<14}"
            f"{'Actual (µV)':>16}"
            f"{'Predicted (µV)':>18}"
            f"{'Delta (µV)':>18}"
        )
        print("-" * 72)

        for step, (date, actual, predicted) in enumerate(
            zip(
                result.dates,
                result.actual,
                result.predicted,
                strict=True,
            ),
            start=1,
        ):
            actual = float(actual)
            predicted = float(predicted)

            # Signed forecast error:
            # positive = underestimation
            # negative = overestimation
            delta = actual - predicted

            date_text = date.strftime("%Y-%m-%d")

            print(
                f"{step:<6}"
                f"{date_text:<14}"
                f"{actual:>16.6f}"
                f"{predicted:>18.6f}"
                f"{delta:>+18.6f}"
            )


def save_benchmark_artifacts(
    dataset,
    results,
    forecast_results,
    training_dates,
):
    """Persist standardized dataset-410 benchmark evidence."""

    RESULT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison = (
        comparison_from_evaluations(
            BATTERY,
            results,
        )
    )

    benchmark_artifact = (
        BenchmarkArtifact(
            comparison
        )
    )

    benchmark_paths = (
        benchmark_artifact.save(
            RESULT_DIRECTORY
        )
    )

    report = BenchmarkReport(
        comparison,
        evaluations=results,
    )

    report_path = report.save(
        RESULT_DIRECTORY
        / "benchmark.txt"
    )

    plotter = BenchmarkPlotter(
        results
    )

    figure_path = plotter.save(
        RESULT_DIRECTORY
        / "benchmark_forecast.png"
    )

    history_plotter = BenchmarkHistoryPlotter(
        dataset,
        evaluations=results,
        forecasts=forecast_results,
        training_dates=training_dates,
    )

    history_plotter.plot()

    history_figure_path = history_plotter.save(
        RESULT_DIRECTORY / "benchmark_history.png"
    )

    best_worst_plotter = BenchmarkBestWorstHistoryPlotter(
        dataset,
        evaluations=results,
        forecasts=forecast_results,
        training_dates=training_dates,
    )

    best_worst_plotter.plot()

    best_worst_figure_path = best_worst_plotter.save(
        RESULT_DIRECTORY
        / "benchmark_best_worst_history.png"
    )


    provenance = BenchmarkProvenance(
        battery=BATTERY,
        optimization=(
            OPTIMIZATION_REFERENCES
        ),
        stability=(
            STABILITY_REFERENCES
        ),
        benchmark=Path(
            "benchmark.json"
        ),
        metadata={
            "benchmark_steps":
                BENCHMARK_STEPS,
            "development_end":
                DEVELOPMENT_END,
            "benchmark_start":
                BENCHMARK_START,
            "benchmark_end":
                BENCHMARK_END,
            "neural_seed":
                SEED,
            "seed_policy":
                "fixed_precommitted_seed",
            "reference_base":
                "provenance_directory",
            "optimization_evidence":
                "copied_unchanged",
        },
    )

    copy_evidence_files(
        OPTIMIZATION_SOURCES,
        OPTIMIZATION_REFERENCES,
        root=RESULT_DIRECTORY,
    )


    # Persist the provenance manifest.
    provenance_path = provenance.save(RESULT_DIRECTORY)

    # Verify that every referenced evidence artifact physically exists.
    validate_provenance_references(
        provenance,
        root=RESULT_DIRECTORY,
    )


    return {
        "benchmark_json":
            benchmark_paths["json"],
        "benchmark_csv":
            benchmark_paths["csv"],
        "report":
            report_path,
        "figure":
            figure_path,
        "history_figure":
            history_figure_path,
        "best_worst_figure":
            best_worst_figure_path,
        "provenance":
            provenance_path,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkRun:
    dataset: object
    evaluations: dict
    forecast_results: dict
    training_dates: object
    artifact_paths: dict


def run_benchmark():
    """Execute the frozen benchmark once and return existing evidence."""
    dataset = load_dataset()

    specs = create_model_specs()

    print_dataset_summary(
        dataset
    )

    print_frozen_specs(
        specs
    )

    preprocessor = TemporalPreprocessor()

    evaluator = ForecastEvaluator(
        evaluation_steps=BENCHMARK_STEPS,
        preprocessor=preprocessor,
    )

    # Recover the exact training-only timeline without refitting
    # any forecasting model or accessing holdout targets.
    training, _ = evaluator.split_dataset(dataset)
    prepared_training = preprocessor.fit_transform(training)
    training_dates = prepared_training.data["ds"].copy()

    benchmark = OptimizedBenchmark(
        evaluator=evaluator
    )

    results, forecast_results = benchmark.evaluate_with_forecasts(
        dataset=dataset,
        specs=specs,
    )

    print_results(
        results
    )

    print_forecasts(
        results
    )

    artifact_paths = (
        save_benchmark_artifacts(
            dataset,
            results,
            forecast_results,
            training_dates,
        )
    )

    print()
    print("=" * 72)
    print(
        "Benchmark artifacts"
    )
    print("=" * 72)

    for name, path in (
        artifact_paths.items()
    ):
        print(
            f"{name:<20}: {path}"
        )

    print()
    print("=" * 72)
    print(
        "Dataset 410 optimized benchmark complete"
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

    return BenchmarkRun(
        dataset=dataset,
        evaluations=results,
        forecast_results=forecast_results,
        training_dates=training_dates,
        artifact_paths=artifact_paths,
    )


def main():
    """Preserve standalone CLI behavior."""
    run_benchmark()


if __name__ == "__main__":
    main()