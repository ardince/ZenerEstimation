"""
============================================================

ZenerEstimation
Official Kalman Demonstration

============================================================
"""

from pathlib import Path
from time import perf_counter
import argparse

from zenerestimation.data.dataset import BatteryDataset

from zenerestimation.data.temporal import TemporalPreprocessor

from zenerestimation.evaluation import ForecastEvaluator

from zenerestimation.forecasting.kalman import KalmanForecaster

from zenerestimation.visualization.forecast import ForecastPlot

from zenerestimation.experiment import Experiment

from zenerestimation.utils.registry import ExperimentRegistry

from zenerestimation.utils.results import (
    create_result_files,
    save_metadata,)

from zenerestimation.utils.report_writer import ReportWriter

from zenerestimation.utils.console import Console

# ============================================================
# Configuration
# ============================================================

FRAMEWORK_VERSION = "0.12.0"

DEFAULT_BATTERY = "732B-5610110"
DEFAULT_HORIZON = 6
DEFAULT_EVALUATION_STEPS = 5
DEFAULT_EVALUATION_END = None

MODEL = "Kalman"


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run the official ZenerEstimation "
            "Kalman forecasting demonstration."
        )
    )

    parser.add_argument(
        "--battery",
        default=DEFAULT_BATTERY,
        help="Battery dataset identifier.",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Forecast horizon in quarters.",
    )

    parser.add_argument(
        "--evaluation-steps",
        type=int,
        default=DEFAULT_EVALUATION_STEPS,
        help="Number of holdout quarters.",
    )

    parser.add_argument(
        "--evaluation-end",
        default=DEFAULT_EVALUATION_END,
        help=(
            "Last date included in holdout evaluation "
            "(YYYY-MM-DD)."
        ),
    )

    return parser.parse_args()

args = parse_args()

DATASET = Path(
    f"datasets/processed/{args.battery}.csv"
)

FORECAST_HORIZON = args.horizon
EVALUATION_STEPS = args.evaluation_steps
EVALUATION_END = args.evaluation_end

# ============================================================
# Start
# ============================================================

Console.header(
    "Official Kalman Demonstration"
)

start = perf_counter()

# ============================================================
# Load Processed Dataset
# ============================================================

Console.section("Loading Processed Dataset")

dataset = BatteryDataset.from_processed_csv(
    DATASET
)

Console.success(
    "Processed dataset loaded successfully."
)

summary = dataset.summary()

print()

print(f"Battery           : {dataset.battery}")
print(f"Source Type       : {dataset.source_type}")
print(f"Measurements      : {summary['rows']}")
print(f"Observed Rows     : {dataset.observed_rows}")
print(f"Missing Periods   : {dataset.missing_period_count}")

print(
    f"Time Span         : "
    f"{summary['start'].date()}  →  "
    f"{summary['end'].date()}"
)

print(f"Frequency         : {summary['frequency']}")

print()

# ============================================================
# Holdout Evaluation
# ============================================================

Console.section("Evaluating Kalman Model")

evaluator = ForecastEvaluator(
    evaluation_steps=EVALUATION_STEPS,
    preprocessor=TemporalPreprocessor(),
    evaluation_end=EVALUATION_END,
)

evaluation_model = KalmanForecaster()

evaluation_result = evaluator.evaluate(
    dataset,
    evaluation_model,
)

evaluation_payload = evaluation_result.to_dict()

Console.success("Holdout evaluation completed.")

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


# ============================================================
# Final Full-Data Forecast
# ============================================================

Console.section("Training Final Kalman Model")

forecast_dataset = (
    TemporalPreprocessor()
    .fit_transform(dataset)
)

model = KalmanForecaster()

result = model.fit_predict(
    forecast_dataset,
    steps=FORECAST_HORIZON,
)

Console.success("Forecast completed.")

# ============================================================
# Prepare Result Directory
# ============================================================

battery = DATASET.stem

paths = create_result_files(
    battery=battery,
    model="kalman",
)

# ============================================================
# Plot
# ============================================================

Console.section("Creating Forecast Plot")

plot = ForecastPlot(
    dataset,
    result,
    evaluation=evaluation_payload,
)

plot.plot(
    title=f"{battery} - Kalman Forecast"
)

plot.save(paths.figure)

Console.success("Figure saved.")

# ============================================================
# Finish Timing
# ============================================================

elapsed = perf_counter() - start

# ============================================================
# Experiment
# ============================================================

experiment = Experiment(

    battery=battery,

    model=MODEL,

    version=FRAMEWORK_VERSION,

    execution_time=elapsed,

    horizon=FORECAST_HORIZON,

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

# ============================================================
# Register Experiment
# ============================================================

registry = ExperimentRegistry()

experiment = registry.register(experiment)

Console.success(
    f"Experiment #{experiment.id} registered."
)

# ============================================================
# Save Log
# ============================================================

log_lines = [
    "ZenerEstimation",
    "Kalman Experiment Log",
    "=" * 60,
    f"Experiment ID     : {experiment.id}",
    f"Battery           : {battery}",
    f"Model             : {MODEL}",
    f"Framework Version : {FRAMEWORK_VERSION}",
    f"Forecast Horizon  : {FORECAST_HORIZON}",
    f"Evaluation Steps  : {evaluation_result.evaluation_steps}",
    (
        f"Evaluation End    : "
        f"{evaluation_result.metadata.get('evaluation_end')}"
    ),
    f"RMSE              : {evaluation_result.rmse:.6f}",
    f"MAE               : {evaluation_result.mae:.6f}",
    f"MAPE              : {evaluation_result.mape:.6f}%",
    f"Execution Time    : {experiment.execution_time:.3f} s",
    f"Source Type       : {dataset.source_type}",
    f"Rows              : {summary['rows']}",
    f"Observed Rows     : {dataset.observed_rows}",
    f"Missing Periods   : {dataset.missing_period_count}",
]

paths.log.write_text(
    "\n".join(log_lines) + "\n",
    encoding="utf-8",
)

# ============================================================
# Save Experiment
# ============================================================

experiment_json = {

    "experiment": experiment.to_dict(),

    "dataset": summary,

    "forecast": result.summary(),

    "evaluation": evaluation_payload,

}

save_metadata(
    paths.experiment,
    experiment_json,
)

# ============================================================
# Save Evaluation
# ============================================================

save_metadata(
    paths.evaluation,
    evaluation_payload,
)

# ============================================================
# Save Forecast
# ============================================================

forecast_payload = {
    "model": result.model,
    "horizon": result.horizon,
    "dates": [
        date.strftime("%Y-%m-%d")
        for date in result.dates
    ],
    "forecast": [
        float(value)
        for value in result.forecast
    ],
    "metadata": dict(
        result.metadata
    )
    if result.metadata
    else {},
}

save_metadata(
    paths.forecast,
    forecast_payload,
)

# ============================================================
# Save Report
# ============================================================

ReportWriter.save(

    filename=paths.report,

    dataset=dataset,

    result=result,

    experiment=experiment,

    evaluation=evaluation_result,

)

# ============================================================
# Dataset Summary
# ============================================================

Console.section("Dataset Summary")

print(f"Measurements      : {summary['rows']}")
print(f"Observed Rows     : {dataset.observed_rows}")
print(f"Columns           : {summary['columns']}")
print(f"Missing Values    : {summary['missing']}")
print(f"Missing Periods   : {dataset.missing_period_count}")

print(f"Time Span         : {summary['start'].date()} → {summary['end'].date()}")
print(f"Frequency         : {summary['frequency']}")

print()

print(f"Experiment ID     : {experiment.id}")
print(f"Battery           : {experiment.battery}")
print(f"Model             : {experiment.model}")
print(f"Horizon           : {experiment.horizon}")
print(f"Evaluation Steps  : {evaluation_result.evaluation_steps}")
print(f"RMSE              : {evaluation_result.rmse:.6f}")
print(f"MAE               : {evaluation_result.mae:.6f}")
print(f"MAPE              : {evaluation_result.mape:.6f}%")
print(f"Execution Time    : {experiment.execution_time:.2f} s")

# ============================================================
# Footer
# ============================================================

Console.footer(FRAMEWORK_VERSION)