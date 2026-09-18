"""
============================================================

ZenerEstimation
Standardized LSTM Forecast Demonstration

Usage:

    python examples/demo_lstm_standardized.py \
        --battery 732B-5610410 \
        --horizon 6 \
        --evaluation-steps 6

============================================================
"""

from pathlib import Path
from time import perf_counter
import argparse

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.forecasting.neural.lstm import (
    LSTMForecaster,
)
from zenerestimation.visualization.forecast import ForecastPlot
from zenerestimation.experiment import Experiment
from zenerestimation.utils.registry import ExperimentRegistry

from zenerestimation.utils.console import Console

from zenerestimation.utils.results import (
    create_result_files,
    save_metadata,
)

from zenerestimation.utils.report_writer import ReportWriter

from zenerestimation.evaluation import ForecastEvaluator
from zenerestimation.data.temporal import TemporalPreprocessor


# ============================================================
# Configuration
# ============================================================

FRAMEWORK_VERSION = "0.12.0"

MODEL = "LSTM"


EVALUATION_DEFAULTS = {
    "732B-5610110": {
        "steps": 5,
        "end": "2024-03-01",
    },
    "732B-5610410": {
        "steps": 6,
        "end": None,
    },
}


LSTM_CONFIG = {
    "window": 4,
    "units": 32,
    "epochs": 100,
    "batch_size": 8,
    "seed": 42,
}


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="ZenerEstimation standardized LSTM demonstration"
    )

    parser.add_argument(
        "--battery",
        required=True,
        help="Battery dataset identifier",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=6,
        help="Forecast horizon in quarters",
    )

    parser.add_argument(
        "--evaluation-steps",
        type=int,
        default=None,
        help=(
            "Number of final observations reserved "
            "for holdout evaluation."
        ),
    )

    parser.add_argument(
        "--evaluation-end",
        default=None,
        help=(
            "Last date included in holdout evaluation "
            "(YYYY-MM-DD)."
        ),
    )

    return parser.parse_args()


# ============================================================
# Dataset
# ============================================================

def resolve_dataset(battery):

    path = Path(
        f"datasets/processed/{battery}.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return path


# ============================================================
# Model Factory
# ============================================================

def create_lstm_model():
    """
    Create a fresh LSTM model using the official
    demonstration configuration.

    Separate model instances are used for holdout
    evaluation and final full-data forecasting.
    """

    return LSTMForecaster(
        **LSTM_CONFIG,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    battery = args.battery
    horizon = args.horizon

    dataset_path = resolve_dataset(
        battery
    )

    defaults = EVALUATION_DEFAULTS.get(
        battery,
        {
            "steps": 6,
            "end": None,
        },
    )

    evaluation_steps = (
        args.evaluation_steps
        if args.evaluation_steps is not None
        else defaults["steps"]
    )

    evaluation_end = (
        args.evaluation_end
        if args.evaluation_end is not None
        else defaults["end"]
    )


    Console.header(
        "Official LSTM Forecast Demonstration"
    )

    start = perf_counter()


    # ========================================================
    # Load Dataset
    # ========================================================

    Console.section(
        "Loading Dataset"
    )

    dataset = BatteryDataset.from_processed_csv(
        dataset_path
    )

    summary = dataset.summary()

    Console.success(
        "Dataset loaded successfully."
    )

    print()

    print(
        f"Battery           : "
        f"{dataset.battery}"
    )

    print(
        f"Source Type       : "
        f"{dataset.source_type}"
    )

    print(
        f"Measurements      : "
        f"{summary['rows']}"
    )

    print(
        f"Observed Rows     : "
        f"{dataset.observed_rows}"
    )

    print(
        f"Time Span         : "
        f"{summary['start']:%Y-%m-%d}"
        f" → "
        f"{summary['end']:%Y-%m-%d}"
    )

    print(
        f"Frequency         : "
        f"{summary['frequency']}"
    )

    print(
        f"Missing Periods   : "
        f"{dataset.missing_period_count}"
    )

    print()


    # ========================================================
    # Model Configuration
    # ========================================================

    Console.section(
        "LSTM Configuration"
    )

    print(
        f"Window            : "
        f"{LSTM_CONFIG['window']}"
    )

    print(
        f"Units             : "
        f"{LSTM_CONFIG['units']}"
    )

    print(
        f"Epochs            : "
        f"{LSTM_CONFIG['epochs']}"
    )

    print(
        f"Batch Size        : "
        f"{LSTM_CONFIG['batch_size']}"
    )

    print(
        f"Seed              : "
        f"{LSTM_CONFIG['seed']}"
    )

    print()


    # ========================================================
    # Holdout Evaluation
    # ========================================================

    Console.section(
        "Evaluating LSTM Model"
    )

    evaluator = ForecastEvaluator(
        evaluation_steps=evaluation_steps,
        preprocessor=TemporalPreprocessor(),
        evaluation_end=evaluation_end,
    )

    # IMPORTANT:
    # Use a dedicated model instance for holdout evaluation.
    # This model is not reused for the final forecast.

    evaluation_model = create_lstm_model()

    evaluation_result = evaluator.evaluate(
        dataset=dataset,
        model=evaluation_model,
    )

    evaluation = evaluation_result.to_dict()


    Console.success(
        "Holdout evaluation completed."
    )

    print()

    print(
        f"Evaluation Steps  : "
        f"{evaluation_result.evaluation_steps}"
    )

    print(
        f"RMSE              : "
        f"{evaluation_result.rmse:.6f}"
    )

    print(
        f"MAE               : "
        f"{evaluation_result.mae:.6f}"
    )

    print(
        f"MAPE              : "
        f"{evaluation_result.mape:.6f}%"
    )

    print()


    # ========================================================
    # Forecast
    # ========================================================

    Console.section(
        "Training LSTM Model"
    )

    # The final forecast intentionally uses all available
    # observations. Persistent processed datasets retain
    # missing values, so preprocessing is applied to a copy
    # before fitting the final model.

    forecast_dataset = (
        TemporalPreprocessor()
        .fit_transform(dataset)
    )

    # IMPORTANT:
    # This is a fresh model instance, independent from the
    # holdout evaluation model.

    model = create_lstm_model()

    model.fit(
        forecast_dataset
    )

    result = model.predict(
        horizon
    )

    Console.success(
        "Forecast completed."
    )


    # ========================================================
    # Result Directory
    # ========================================================

    paths = create_result_files(
        battery=battery,
        model="lstm",
    )


    # ========================================================
    # Timing
    # ========================================================

    elapsed = (
        perf_counter() - start
    )


    # ========================================================
    # Experiment
    # ========================================================

    experiment = Experiment(

        battery=battery,

        model=MODEL,

        version=FRAMEWORK_VERSION,

        execution_time=elapsed,

        horizon=horizon,

        artifacts={

            "figure": str(paths.figure),

            "forecast": str(paths.forecast),

            "evaluation": str(paths.evaluation),

            "experiment": str(paths.experiment),

            "report": str(paths.report),

            "log": str(paths.log),

        },

        metadata=result.summary(),

    )


    # ========================================================
    # Register
    # ========================================================

    registry = ExperimentRegistry()

    experiment = registry.register(
        experiment
    )

    Console.success(
        f"Experiment #{experiment.id} registered."
    )


    # ========================================================
    # Save Experiment
    # ========================================================

    save_metadata(
        paths.experiment,
        {
            "experiment": experiment.to_dict(),
            "dataset": summary,
        },
    )


    # ========================================================
    # Save Forecast
    # ========================================================

    forecast_json = {

        "battery": battery,
        "model": "lstm",
        "experiment_id": experiment.id,
        "horizon": result.horizon,

        "dates": [
            str(date)
            for date in result.dates
        ],

        "forecast": [
            float(value)
            for value in result.forecast
        ],

        "metadata": result.summary(),

    }

    save_metadata(
        paths.forecast,
        forecast_json,
    )


    # ========================================================
    # Save Evaluation
    # ========================================================

    evaluation_json = {
        **evaluation,
        "battery": battery,
        "model": "lstm",
        "experiment_id": experiment.id,
    }


    # ========================================================
    # Save Evaluation Metadata
    # ========================================================

    save_metadata(
        paths.evaluation,
        evaluation_json,
    )


    # ========================================================
    # Save Log
    # ========================================================

    paths.log.write_text(
        (
            "ZenerEstimation Experiment Log\n"
            f"Experiment ID: {experiment.id}\n"
            f"Battery: {battery}\n"
            f"Model: {MODEL}\n"
            f"Horizon: {horizon}\n"
            f"Execution Time: {elapsed:.3f} s\n"
            "Status: completed\n"
        ),
        encoding="utf-8",
    )


    # ========================================================
    # Plot
    # ========================================================

    Console.section(
        "Creating Forecast Plot"
    )

    plot = ForecastPlot(
        dataset,
        result,
        experiment=experiment,
        evaluation=evaluation,
    )

    plot.plot(
        title=(
            f"{battery} - LSTM Forecast"
        )
    )

    plot.save(
        paths.figure
    )

    Console.success(
        "Figure saved."
    )


    # ========================================================
    # Report
    # ========================================================

    ReportWriter.save(

        filename=paths.report,

        dataset=dataset,

        result=result,

        experiment=experiment,

        evaluation=evaluation_result,

    )


    # ========================================================
    # Final Summary
    # ========================================================

    Console.section(
        "Experiment Summary"
    )

    print(
        f"Experiment ID     : "
        f"{experiment.id}"
    )

    print(
        f"Battery           : "
        f"{battery}"
    )

    print(
        f"Model             : "
        f"{MODEL}"
    )

    print(
        f"Horizon           : "
        f"{horizon} Quarters"
    )

    print(
        f"Execution Time    : "
        f"{elapsed:.2f} s"
    )

    print(
        f"Result Directory  : "
        f"{paths.directory}"
    )


    Console.footer(
        FRAMEWORK_VERSION
    )


if __name__ == "__main__":
    main()