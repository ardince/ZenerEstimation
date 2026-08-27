"""
============================================================

ZenerEstimation
Official ARIMA Forecast Demonstration

Usage:

    python examples/demo_arima.py \
        --battery 732B-5610410 \
        --horizon 6

============================================================
"""

from pathlib import Path
from time import perf_counter
import argparse
import numpy as np

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.forecasting.arima import ARIMAForecaster
from zenerestimation.visualization.forecast import ForecastPlot
from zenerestimation.experiment import Experiment
from zenerestimation.utils.registry import ExperimentRegistry

from zenerestimation.utils.console import Console

from zenerestimation.utils.results import (
    create_result_files,
    save_metadata,
)

from zenerestimation.utils.report_writer import ReportWriter


# ============================================================
# Configuration
# ============================================================

FRAMEWORK_VERSION = "0.10.0"

MODEL = "ARIMA"


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="ZenerEstimation ARIMA demonstration"
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
        default=6,
        help="Number of final observations reserved for holdout evaluation.",
    )

    return parser.parse_args()


# ============================================================
# Dataset
# ============================================================

def resolve_dataset(battery):

    path = Path(
        f"datasets/raw/{battery}.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return path

# ============================================================
# RMSE, MAE, MAPE Computation
# ============================================================

def compute_rmse(
    actual,
    predicted,
):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                (actual - predicted) ** 2
            )
        )
    )


def compute_mae(
    actual,
    predicted,
):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    return float(
        np.mean(
            np.abs(
                actual - predicted
            )
        )
    )


def compute_mape(
    actual,
    predicted,
):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    mask = actual != 0

    if not np.any(mask):
        return None

    return float(
        np.mean(
            np.abs(
                (
                    actual[mask]
                    - predicted[mask]
                )
                / actual[mask]
            )
        )
        * 100.0
    )

# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    battery = args.battery
    horizon = args.horizon
    evaluation_steps = args.evaluation_steps
    dataset_path = resolve_dataset(
        battery
    )

    Console.header(
        "Official ARIMA Forecast Demonstration"
    )

    start = perf_counter()


    # ========================================================
    # Load Dataset
    # ========================================================

    Console.section(
        "Loading Dataset"
    )

    dataset = BatteryDataset.from_csv(
        dataset_path
    )

    summary = dataset.summary()

    Console.success(
        "Dataset loaded successfully."
    )

    print()

    print(
        f"Battery           : "
        f"{dataset.metadata['battery_id']}"
    )

    print(
        f"Measurements      : "
        f"{summary['rows']}"
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
        f"{summary['missing_periods']}"
    )

    print()


    # ========================================================
    # Holdout Evaluation
    # ========================================================

    Console.section(
        "Evaluating ARIMA Model"
    )

    if evaluation_steps <= 0:

        raise ValueError(
            "evaluation-steps must be greater than zero."
    )

    if evaluation_steps >= len(dataset):

        raise ValueError(
            "evaluation-steps must be smaller "
            "than the dataset length."
    )


    train_df = (
        dataset.data
        .iloc[:-evaluation_steps]
        .copy()
    )

    validation_df = (
        dataset.data
        .iloc[-evaluation_steps:]
        .copy()
    )

    train_dataset = BatteryDataset(
        train_df
    )

    validation_actual = (
        validation_df["microVolt"]
        .to_numpy(dtype=float)
    )


    evaluation_model = (
        ARIMAForecaster()
    )

    evaluation_result = (
        evaluation_model.fit_predict(
            train_dataset,
            steps=evaluation_steps,
        )
    )

    validation_prediction = (
        np.asarray(
            evaluation_result.forecast,
            dtype=float,
        )
    )


    rmse = compute_rmse(
        validation_actual,
        validation_prediction,
    )

    mae = compute_mae(
        validation_actual,
        validation_prediction,
    )

    mape = compute_mape(
        validation_actual,
        validation_prediction,
    )


    Console.success(
    "Holdout evaluation completed."
    )

    print()

    print(
        f"Evaluation Steps  : "
        f"{evaluation_steps}"
    )

    print(
        f"RMSE              : "
        f"{rmse:.6f}"
    )

    print(
        f"MAE               : "
        f"{mae:.6f}"
    )

    if mape is not None:

        print(
            f"MAPE              : "
            f"{mape:.6f}%"
    )

    else:

        print(
            "MAPE              : N/A"
    )

    print()


    # ========================================================
    # Evaluation Dictionary
    # ========================================================

    evaluation = {

        "status":
        "evaluated",

        "method":
        "holdout",

        "evaluation_steps":
            evaluation_steps,

        "rmse":
            rmse,

        "mae":
            mae,

        "mape":
            mape,

    }   


    # ========================================================
    # Forecast
    # ========================================================

    Console.section(
        "Training ARIMA Model"
    )

    model = ARIMAForecaster()

    result = model.fit_predict(
        dataset,
        steps=horizon,
    )

    Console.success(
        "Forecast completed."
    )


    # ========================================================
    # Result Directory
    # ========================================================

    paths = create_result_files(
        battery=battery,
        model="arima",
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
        "model": "arima",
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
        forecast_json
    )

    # ========================================================
    # Save Evaluation
    # ========================================================

    evaluation_json = {

        **evaluation,

        "battery":
            battery,

        "model":
            "arima",

        "experiment_id":
            experiment.id,

        "training_points":
            len(train_dataset),
        
        "validation_points":
            len(validation_actual),

        "actual": [
            float(value)
            for value in validation_actual
        ],

        "predicted": [
            float(value)
            for value in validation_prediction
        ], 

        "dates": [
            str(date)
            for date in validation_df["ds"]
        ],

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
            f"{battery} - ARIMA Forecast"
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