"""
============================================================

ZenerEstimation
Official Kalman Demonstration

============================================================
"""

from pathlib import Path
from time import perf_counter

from zenerestimation.data.dataset import BatteryDataset
#from zenerestimation.data.smart_loader import SmartDatasetLoader
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

DATASET = Path(
    "datasets/processed/732B-5610410.csv"
)

FORECAST_HORIZON = 6

EVALUATION_STEPS = 6

MODEL = "Kalman"


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

save_metadata(
    paths.forecast,
    result.summary(),
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