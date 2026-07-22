"""
Official Remaining Useful Life demonstration.

This example demonstrates the complete prognostics workflow:

    Dataset
        ↓
    Kalman Forecaster
        ↓
    Threshold Estimation
        ↓
    Monte Carlo Simulation
        ↓
    Remaining Useful Life
"""

from pathlib import Path
from time import perf_counter

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.data.smart_loader import SmartDatasetLoader
from zenerestimation.forecasting.kalman import KalmanForecaster

from zenerestimation.prognostics import (
    ThresholdEstimator, MonteCarloRUL, RULAnalyzer,
)

from zenerestimation.utils.console import Console

from zenerestimation import __version__


FRAMEWORK_VERSION = __version__

DATASET = Path(
    "datasets/raw/732B-5610110.csv"
)

SIMULATIONS = 3000


# ============================================================
# Header
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

print()

print(f"Battery           : {DATASET.stem}")
print(f"Measurements      : {summary['rows']}")
print(f"Time Span         : {summary['start'].date()}  →  {summary['end'].date()}")
print(f"Frequency         : {summary['frequency']}")
print(f"Missing Periods   : {summary['missing_periods']}")

# ============================================================
# Train Kalman
# ============================================================

Console.section("Training Kalman Model")

model = KalmanForecaster()

model.fit(dataset)

Console.success("Model trained.")

# ============================================================
# Threshold
# ============================================================

Console.section("Estimating Threshold")

threshold = ThresholdEstimator.default(
    dataset.target
)

Console.success(
    f"Threshold = {threshold:.3f} µV"
)

# ============================================================
# Monte Carlo
# ============================================================

Console.section("Running Monte Carlo")

print(f"Time Step         : {model.dt:.2f} years")
print(f"Simulations       : {SIMULATIONS}")

engine = MonteCarloRUL(simulations=SIMULATIONS)

print(f"Fast Probability  : {engine.fast_probability:.0%}")
print(f"Fast Noise        : {engine.fast_noise}")
print(f"Slow Noise        : {engine.slow_noise}")

analyzer = RULAnalyzer(simulator=engine)

print()

rul = model.rul(
    threshold=threshold,
    simulations=SIMULATIONS,
)

Console.success("Simulation completed.")

elapsed = perf_counter() - start

# ============================================================
# Dataset Summary
# ============================================================

Console.section("Dataset Summary")

print(f"Measurements      : {summary['rows']}")
print(f"Columns           : {summary['columns']}")
print(f"Missing Values    : {summary['missing']}")
print(f"Missing Periods   : {summary['missing_periods']}")
print(f"Time Span         : {summary['start'].date()} → {summary['end'].date()}")
print(f"Frequency         : {summary['frequency']}")

# ============================================================
# RUL Summary
# ============================================================

Console.section("RUL Summary")

info = rul.summary()

print(f"Model             : Kalman + Monte Carlo")
print(f"Threshold         : {info['threshold']:.3f} µV")
print(f"Mean RUL          : {info['mean']:.2f} years")
print(f"Median RUL        : {info['median']:.2f} years")
print(f"P10 RUL           : {info['p10']:.2f} years")
print(f"P90 RUL           : {info['p90']:.2f} years")
print(f"Shortest RUL      : {info['minimum']:.2f} years")
print(f"Longest RUL       : {info['maximum']:.2f} years")
print(f"Simulations       : {info['simulations']}")
print(f"Threshold Method  : 99th percentile + 5 µV")

print()

print(f"Execution Time    : {elapsed:.2f} s")

# ============================================================
# Footer
# ============================================================

Console.footer(FRAMEWORK_VERSION)