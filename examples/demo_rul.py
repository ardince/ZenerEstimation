"""
Official Remaining Useful Life demonstration.

Demonstrates the complete prognostics workflow.
"""

from pathlib import Path
from time import perf_counter

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.data.smart_loader import SmartDatasetLoader

from zenerestimation.forecasting.kalman import KalmanForecaster

from zenerestimation.prognostics import (
    ThresholdEstimator,
    MonteCarloRUL,
    RULAnalyzer,
)

from zenerestimation.visualization.rul_histogram import (
    RULHistogram,
)

from zenerestimation.utils.rul_report_writer import (
    RULReportWriter,
)

from zenerestimation.experiment import Experiment

from zenerestimation.utils.registry import ExperimentRegistry

from zenerestimation.utils.console import Console

from zenerestimation.utils.results import (
    create_result_files,
    save_metadata,)


from zenerestimation import __version__


DATASET = Path("datasets/raw/732B-5610110.csv")

MODEL = "Kalman + Monte Carlo"

FRAMEWORK_VERSION = __version__

SIMULATIONS = 3000


# ============================================================
# Start
# ============================================================

Console.header(
    "Official Remaining Useful Life Demonstration"
)

start = perf_counter()

# ============================================================
# Load Dataset
# ============================================================

Console.section("Loading Dataset")

loader = SmartDatasetLoader()

df, metadata = loader.load(DATASET)

dataset = BatteryDataset(df)

Console.success("Dataset loaded successfully.")

summary = dataset.summary()

Console.info(f"Battery           : {DATASET.stem}")
Console.info(f"Measurements      : {summary['rows']}")
Console.info(f"Time Span         : {summary['start']}  →  {summary['end']}")
Console.info(f"Frequency         : {summary['frequency']}")
Console.info(f"Missing Periods   : {summary['missing_periods']}")

# ============================================================
# Train Forecasting Model
# ============================================================

Console.section("Training Kalman Model")

model = KalmanForecaster()

model.fit(dataset)

Console.success("Model trained.")

# ============================================================
# Estimate Threshold
# ============================================================

Console.section("Estimating Threshold")

threshold = ThresholdEstimator.default(
    dataset.data["microVolt"]
)

Console.success(
    f"Threshold = {threshold:.3f} µV"
)

# ============================================================
# Run Monte Carlo
# ============================================================

Console.section("Running Monte Carlo")

result = model.rul(
    threshold=threshold,
    simulations=SIMULATIONS,
)

Console.success("Simulation completed.")

# ============================================================
# Prepare Result Directory
# ============================================================

battery = DATASET.stem

paths = create_result_files(
    battery=battery,
    model="rul",
)

# ============================================================
# Save Histogram
# ============================================================

Console.section("Creating RUL Histogram")

histogram = RULHistogram(result)

#histogram.save(
 #   paths.figure / "rul_histogram.png"
#)

histogram.save(paths.figure)

Console.success("Histogram saved.")

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

    horizon=None,

    artifacts={

        "histogram":
            str(paths.figure.parent / "rul_histogram.png"),

        "metadata":
            str(paths.metadata),

        "report":
            str(paths.report),

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

    "dataset": summary,

    "prognostics": result.summary(),

    "experiment": experiment.to_dict(),

}

save_metadata(
    paths.metadata,
    metadata_json,
)

Console.success("Metadata saved.")

# ============================================================
# Save Report
# ============================================================

RULReportWriter.save(

    filename=paths.report,

    dataset=dataset,

    result=result,

    experiment=experiment,

)

Console.success("Report saved.")

# ============================================================
# Final Summary
# ============================================================

Console.section("RUL Summary")

Console.info(f"Model             : {MODEL}")
Console.info(f"Threshold         : {result.threshold:.3f} µV")
Console.info(f"Mean              : {result.mean:.2f} years")
Console.info(f"Median            : {result.median:.2f} years")
Console.info(f"P10               : {result.p10:.2f} years")
Console.info(f"P90               : {result.p90:.2f} years")
Console.info(f"Simulations       : {result.simulations}")

Console.info("")
Console.info(f"Execution Time    : {elapsed:.2f} s")

# ============================================================
# Footer
# ============================================================

Console.footer(FRAMEWORK_VERSION)