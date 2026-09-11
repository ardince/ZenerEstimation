"""
============================================================

ZenerEstimation
Official ARIMA Forecast Demonstration

Usage:

    python examples/demo_arima.py \
        --battery 732B-5610410 \
        --horizon 6
        --evaluation-steps 6

============================================================
"""

from pathlib import Path
from time import perf_counter
import argparse

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

from zenerestimation.evaluation import ForecastEvaluator
from zenerestimation.data.temporal import TemporalPreprocessor

# ============================================================
# Configuration
# ============================================================

FRAMEWORK_VERSION = "0.12.0"

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
# Main
# ============================================================

def main():

    args = parse_args()

    battery = args.battery
    horizon = args.horizon
    evaluation_steps = args.evaluation_steps
    evaluation_end = args.evaluation_end
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
    # Holdout Evaluation
    # ========================================================

    Console.section(
        "Evaluating ARIMA Model"
    )

    evaluator = ForecastEvaluator(
        evaluation_steps=evaluation_steps,
        preprocessor=TemporalPreprocessor(),
        evaluation_end=evaluation_end,
    )

    evaluation_model = ARIMAForecaster()

    evaluation_result = evaluator.evaluate(
        dataset,
        evaluation_model,
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
        "Training ARIMA Model"
    )

    forecast_dataset = (
        TemporalPreprocessor()
        .fit_transform(dataset)
    )

    model = ARIMAForecaster()

    result = model.fit_predict(
        forecast_dataset,
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
        "battery": battery,
        "model": "arima",
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