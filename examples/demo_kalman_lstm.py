"""
============================================================

ZenerEstimation
Official Kalman LSTM Forecast Demonstration

============================================================
"""

from pathlib import Path
from time import perf_counter

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.data.smart_loader import SmartDatasetLoader

from zenerestimation.forecasting.hybrid.kalman_lstm import KalmanLSTMForecaster
from zenerestimation.forecasting.neural.lstm import LSTMForecaster

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

FRAMEWORK_VERSION = "0.8.0"

DATASET = Path(
    "datasets/raw/732B-5610110.csv"
)

FORECAST_HORIZON = 6

MODEL = "KalmanLSTM"


# ============================================================
# Start
# ============================================================

Console.header(
    "Official Kalman LSTM Forecast Demonstration"
)

start = perf_counter()


# ============================================================
# Load Dataset
# ============================================================

Console.section("Loading Dataset")

dataset = BatteryDataset.from_csv(DATASET)

metadata = dataset.metadata

Console.success("Dataset loaded successfully.")

summary = dataset.summary()

print()

print(f"Battery           : {metadata['battery_id']}")
print(f"Dataset Format    : {metadata['format']}")
print(f"Measurements      : {summary['rows']}")
print(f"Time Span         : {summary['start'].date()}  →  {summary['end'].date()}")
print(f"Frequency         : {summary['frequency']}")
print(f"Missing Periods   : {summary['missing_periods']}")

print()


# ============================================================
# Forecast
# ============================================================

Console.section("Training Kalman LSTM Model")

model = KalmanLSTMForecaster(

    window=10,

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
    model="kalman_lstm",
)

print("Battery:", dataset.battery)
print("Metadata:", dataset.metadata)

# ============================================================
# Plot
# ============================================================

plot = ForecastPlot(
    dataset,
    result,
)

plot.plot(

    title=f"{battery} - Kalman LSTM Forecast\n"
          f"Forecast Horizon: {FORECAST_HORIZON} Quarters"
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

        "metadata": str(paths.metadata),

        "report": str(paths.report),

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
# Save Metadata
# ============================================================

metadata_json = {

    "experiment": experiment.to_dict(),

    "dataset": summary,

    "forecast": result.summary(),

}

save_metadata(
    paths.metadata,
    metadata_json,
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

Console.section("Hybrid Model Summary")

framework = result.metadata["framework"]

print(
    f"TensorFlow      : {framework['tensorflow']}"
)

print(
    f"Keras           : {framework['keras']}"
)

print(
    f"Window          : {result.metadata['window']}"
)

print(
    f"LSTM Units      : {result.metadata['units']}"
)

print(
    f"Epochs          : {result.metadata['epochs']}"
)

print(
    f"Batch Size      : {result.metadata['batch_size']}"
)

print(
    f"Seed            : {result.metadata['seed']}"
)

print()

summary = model.summary()

print(f"Architecture     : {summary['architecture']}")
print(f"Trend Model      : {summary['trend']}")
print(f"Residual Model   : {summary['residual_model']}")

print(
    f"Execution Time  : {elapsed:.2f} s"
)


# ============================================================
# Finished
# ============================================================

Console.header(
    "Demo Completed Successfully"
)

print(
    f"Framework Version : {FRAMEWORK_VERSION}"
)

print(
    f"Battery           : {battery}"
)

print(
    f"Model             : {MODEL}"
)

print(
    f"Forecast Horizon  : {FORECAST_HORIZON} Quarters"
)