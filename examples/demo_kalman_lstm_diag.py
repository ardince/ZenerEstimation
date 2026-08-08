"""
============================================================

ZenerEstimation
Official Kalman LSTM Forecast Diagnostic Demonstration

============================================================
"""

from pathlib import Path
from time import perf_counter

from zenerestimation.data.dataset import BatteryDataset
#from zenerestimation.data.smart_loader import SmartDatasetLoader

from zenerestimation.forecasting.hybrid.kalman_lstm import KalmanLSTMForecaster
from zenerestimation.diagnostics import HybridDiagnostics

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

FRAMEWORK_VERSION = "0.10.0"

DATASET = Path(
    "datasets/raw/732B-5610110.csv"
)

FORECAST_HORIZON = 6

MODEL = "KalmanLSTM"


# ============================================================
# Start
# ============================================================

Console.header(
    "Official Kalman LSTM Forecast Diagnostic Demonstration"
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
# Diagnostics
# ============================================================

Console.section("Running Hybrid Diagnostics")

diagnostics = (

    HybridDiagnostics(model)

        .run(dataset)

)

diagnostics_result = diagnostics.result()

Console.success(
    "Diagnostics completed."
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
# Plot
# ============================================================

plot = ForecastPlot(
    dataset,
    result,
    experiment=experiment,
)

plot.plot(

    title=f"{battery} - Kalman LSTM Forecast\n"
          f"Forecast Horizon: {FORECAST_HORIZON} Quarters"
)

plot.save(paths.figure)

Console.success("Figure saved.")


# ============================================================
# Diagnostics - Results
# ============================================================

Console.section("Hybrid Diagnostics")

diag = diagnostics_result.summary()

print()

print(

    f"Quality Score      : "

    f"{diagnostics_result.quality_score:.1f}/100"

)

print(

    f"Grade              : "

    f"{diagnostics_result.quality_grade}"

)

print()

print(

    f"Variance Explained : "

    f"{100*diag['variance_explained']:.1f}%"

)

print(

    f"Residual RMSE      : "

    f"{diag['residual_rmse']:.4f}"

)

print(

    f"Residual Std       : "

    f"{diag['residual_std']:.4f}"

)

print(

    f"Lag-1 Corr         : "

    f"{diag['lag1_autocorrelation']:.4f}"

)

print(

    f"Durbin-Watson      : "

    f"{diag['durbin_watson']:.4f}"

)

print(

    f"Ljung-Box p-value  : "

    f"{diag['ljung_box_pvalue']:.4f}"

)

print()

print("Recommendations")

for line in diagnostics_result.recommendations:

    print(f"  ✓ {line}")

print()

# ============================================================
# Save Metadata
# ============================================================

metadata_json = {

    "experiment": experiment.to_dict(),

    "dataset": summary,

    "forecast": result.summary(),

    "diagnostics":

        diagnostics_result.to_dict(),

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

    diagnostics=diagnostics_result,

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

print(

    f"Hybrid Quality    : "

    f"{diagnostics_result.quality_score:.1f}/100 "

    f"({diagnostics_result.quality_grade})"

)