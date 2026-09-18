"""
============================================================

ZenerEstimation
Official LSTM Forecast Demonstration

Legacy / Raw Dataset Workflow

============================================================
"""

from pathlib import Path
from time import perf_counter

from zenerestimation.data.dataset import BatteryDataset

from zenerestimation.forecasting.neural.lstm import (
    LSTMForecaster,
)

from zenerestimation.visualization.forecast import ForecastPlot

from zenerestimation.experiment import Experiment

from zenerestimation.utils.registry import ExperimentRegistry

from zenerestimation.utils.results import (
    create_result_files,
    save_metadata,
)

from zenerestimation.utils.report_writer import ReportWriter

from zenerestimation.utils.console import Console


# ============================================================
# Configuration
# ============================================================

FRAMEWORK_VERSION = "0.12.0"

DATASET = Path(
    "datasets/raw/732B-5610110.csv"
)

FORECAST_HORIZON = 6

MODEL = "LSTM"


# ============================================================
# Start
# ============================================================

Console.header(
    "Official LSTM Forecast Demonstration"
)

start = perf_counter()


# ============================================================
# Load Dataset
# ============================================================

Console.section(
    "Loading Dataset"
)

dataset = BatteryDataset.from_csv(
    DATASET
)

metadata = dataset.metadata

Console.success(
    "Dataset loaded successfully."
)

summary = dataset.summary()

print()

print(
    f"Battery           : "
    f"{metadata['battery_id']}"
)

print(
    f"Dataset Format    : "
    f"{metadata['format']}"
)

print(
    f"Measurements      : "
    f"{summary['rows']}"
)

print(
    f"Time Span         : "
    f"{summary['start'].date()}"
    f"  →  "
    f"{summary['end'].date()}"
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


# ============================================================
# Forecast
# ============================================================

Console.section(
    "Training LSTM Model"
)

model = LSTMForecaster(

    window=10,

    units=24,

    epochs=200,

)

result = model.fit_predict(

    dataset,

    steps=FORECAST_HORIZON,

)

Console.success(
    "Forecast completed."
)


# ============================================================
# Prepare Result Directory
# ============================================================

battery = DATASET.stem

paths = create_result_files(
    battery=battery,
    model="lstm",
)

print(
    "Battery:",
    dataset.battery,
)

print(
    "Metadata:",
    dataset.metadata,
)


# ============================================================
# Plot
# ============================================================

plot = ForecastPlot(
    dataset,
    result,
)

plot.plot(

    title=(
        f"{battery} - LSTM Forecast\n"
        f"Forecast Horizon: "
        f"{FORECAST_HORIZON} Quarters"
    )
)

plot.save(
    paths.figure
)

Console.success(
    "Figure saved."
)


# ============================================================
# Finish Timing
# ============================================================

elapsed = (
    perf_counter() - start
)


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

experiment = registry.register(
    experiment
)

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

    "workflow": {
        "type": "legacy_raw",
        "dataset_source": "raw",
        "evaluation": False,
    },

}

save_metadata(
    paths.experiment,
    experiment_json,
)


# ============================================================
# Save Forecast
# ============================================================

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

    "workflow": "legacy_raw",

}

save_metadata(
    paths.forecast,
    forecast_json,
)


# ============================================================
# Save Log
# ============================================================

paths.log.write_text(
    (
        "ZenerEstimation Experiment Log\n"
        f"Experiment ID: {experiment.id}\n"
        f"Battery: {battery}\n"
        f"Model: {MODEL}\n"
        "Workflow: legacy_raw\n"
        f"Horizon: {FORECAST_HORIZON}\n"
        f"Execution Time: {elapsed:.3f} s\n"
        "Status: completed\n"
    ),
    encoding="utf-8",
)


# ============================================================
# Save Report
# ============================================================

ReportWriter.save(

    filename=paths.report,

    dataset=dataset,

    result=result,

    experiment=experiment,

)


# ============================================================
# Neural Network Summary
# ============================================================

Console.section(
    "Neural Network Summary"
)

framework = result.metadata[
    "framework"
]

print(
    f"TensorFlow      : "
    f"{framework['tensorflow']}"
)

print(
    f"Keras           : "
    f"{framework['keras']}"
)

print(
    f"Window          : "
    f"{result.metadata['window']}"
)

print(
    f"LSTM Units      : "
    f"{result.metadata['units']}"
)

print(
    f"Epochs          : "
    f"{result.metadata['epochs']}"
)

print(
    f"Batch Size      : "
    f"{result.metadata['batch_size']}"
)

print(
    f"Seed            : "
    f"{result.metadata['seed']}"
)

print()

print(
    f"Execution Time  : "
    f"{elapsed:.2f} s"
)


# ============================================================
# Finished
# ============================================================

Console.header(
    "Demo Completed Successfully"
)

print(
    f"Framework Version : "
    f"{FRAMEWORK_VERSION}"
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
    f"Forecast Horizon  : "
    f"{FORECAST_HORIZON} Quarters"
)

print(
    f"Result Directory  : "
    f"{paths.directory}"
)