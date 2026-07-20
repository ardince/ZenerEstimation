"""
============================================================

ZenerEstimation
Official ARIMA Demonstration

This example demonstrates the complete workflow:

    Smart Dataset Loader
            ↓
      BatteryDataset
            ↓
     ARIMAForecaster
            ↓
      ForecastResult
            ↓
      ForecastPlot
            ↓
   Save Experiment Files
            ↓
   Register Experiment

============================================================
"""

from pathlib import Path
from time import perf_counter

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.data.smart_loader import SmartDatasetLoader
from zenerestimation.forecasting.arima import ARIMAForecaster
from zenerestimation.visualization.forecast import ForecastPlot
from zenerestimation.experiment import Experiment
from zenerestimation.utils.registry import ExperimentRegistry

from zenerestimation.utils.console import Console

from zenerestimation.utils.results import (
    create_result_files,
    save_metadata,
    save_report,
)


# ============================================================
# Configuration
# ============================================================

FRAMEWORK_VERSION = "0.5.0"

DATASET = Path(
    "datasets/raw/732B-5610410.csv"
)

FORECAST_HORIZON = 6

MODEL = "ARIMA"

# ============================================================
# Start
# ============================================================

Console.header(
    "Official ARIMA Demonstration"
)

start = perf_counter()

# ============================================================
# Load Dataset
# ============================================================

Console.section("Loading Dataset")

loader = SmartDatasetLoader()

df, metadata = loader.load(DATASET)

dataset = BatteryDataset(df)

summary = dataset.summary()

Console.success("Dataset loaded successfully.")

print()

print(f"Battery           : {metadata['battery_id']}")
print(f"Dataset Format    : {metadata['format']}")
print(f"Measurements      : {summary['rows']}")
print(f"Time Span         : {summary['start']:%Y-%m-%d}  →  {summary['end']:%Y-%m-%d}")
print(f"Frequency         : {summary['frequency']}")
print(f"Missing Periods   : {summary['missing_periods']}")

print()

# ============================================================
# Forecast
# ============================================================

Console.section("Training ARIMA Model")

model = ARIMAForecaster()

result = model.fit_predict(

    dataset,

    steps=FORECAST_HORIZON,

)

Console.success("Forecast completed.")

# ============================================================
# Prepare Result Directory
# ============================================================

battery = DATASET.stem

paths = create_result_files(

    battery=battery,

    model="arima",

)

# ============================================================
# Plot
# ============================================================

Console.section("Creating Forecast Plot")

plot = ForecastPlot(

    dataset,

    result,

)

plot.plot(

    title=f"{battery} - ARIMA Forecast"

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

registry = ExperimentRegistry()

experiment = registry.register(experiment)

# ============================================================
# Save Metadata
# ============================================================

save_metadata(

    paths.metadata,

    {

        "experiment": experiment.to_dict(),

        "forecast": result.summary(),

        "dataset": summary,

    },

)

# ============================================================
# Save Report
# ============================================================

report = f"""
ZenerEstimation Experiment Report
=================================

Experiment ID : {experiment.id}

Battery       : {battery}

Model         : {MODEL}

Framework     : {FRAMEWORK_VERSION}

Execution     : {experiment.execution_time:.2f} s


Dataset

-------

Measurements  : {summary['rows']}

Frequency     : {summary['frequency']}

Missing       : {summary['missing']}

Missing Qtrs  : {summary['missing_periods']}


Forecast

--------

Horizon       : {result.horizon}

First Value   : {result.first_forecast:.4f}

Last Value    : {result.last_forecast:.4f}


Metadata

--------

{result.metadata}
"""

save_report(

    paths.report,

    report,

)

# ============================================================
# Summary
# ============================================================

Console.success(
    f"Experiment #{experiment.id} registered."
)

Console.section("Dataset Summary")

print(f"Measurements      : {summary['rows']}")
print(f"Columns           : {summary['columns']}")
print(f"Missing Values    : {summary['missing']}")
print(f"Missing Periods   : {summary['missing_periods']}")
print(f"Time Span         : {summary['start']:%Y-%m-%d} → {summary['end']:%Y-%m-%d}")
print(f"Frequency         : {summary['frequency']}")

print()

print(f"Experiment ID     : {experiment.id}")
print(f"Battery           : {experiment.battery}")
print(f"Model             : {experiment.model}")
print(f"Horizon           : {experiment.horizon}")
print(f"Execution Time    : {experiment.execution_time:.2f} s")


#print(result.fitted.head())

#print()

#print(result.forecast)

# ============================================================
# Footer
# ============================================================

Console.footer(FRAMEWORK_VERSION)