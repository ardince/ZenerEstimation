# ZenerEstimation Architecture

**Version:** 0.5.0

---

# 1. System Overview

ZenerEstimation is a modular Python framework for battery voltage analysis,
forecasting and Remaining Useful Life (RUL) estimation.

The framework is designed around a common forecasting interface, allowing
different algorithms (ARIMA, Kalman, ETS, LSTM, GRU, etc.) to be added
without changing the surrounding infrastructure.

---

# 2. Package Structure

```
zenerestimation/

├── analysis/
├── data/
├── features/
├── forecasting/
├── models/
├── utils/
├── visualization/

examples/
tests/
datasets/
results/
```

Each package has a single responsibility and can evolve independently.

---

# 3. Forecast Pipeline

```
CSV Dataset
      │
      ▼
SmartDatasetLoader
      │
      ▼
BatteryDataset
      │
      ▼
ForecastModel
 (ARIMA, Kalman, ...)
      │
      ▼
ForecastResult
      │
      ├────────► ForecastPlot
      │
      ├────────► Report
      │
      └────────► ExperimentRegistry
```

The forecasting engine is completely separated from plotting,
report generation and experiment management.

---

# 4. Core Components

## SmartDatasetLoader

Automatically detects supported dataset formats and converts them into
a unified internal representation.

---

## BatteryDataset

Provides

- validation
- cleaning
- frequency detection
- missing period detection
- statistical summary

---

## BaseForecastModel

Defines the common interface implemented by every forecasting algorithm.

```
fit()

predict()

fit_predict()
```

---

## ForecastResult

Stores

- forecast values
- forecast dates
- model metadata

Returned by every forecasting model.

---

## ForecastPlot

Produces consistent visualizations independent of the forecasting model.

---

## Experiment

Stores experiment metadata such as

- battery
- model
- execution time
- forecast horizon

---

## ExperimentRegistry

Keeps a persistent history of executed experiments.

---

# 5. Extension Points

Adding a new forecasting algorithm only requires implementing
`BaseForecastModel`.

Example:

```
BaseForecastModel
        │
        ├── ARIMAForecaster
        ├── KalmanForecaster
        ├── ETSForecaster
        ├── LSTMForecaster
        └── HybridForecaster
```

No changes are required in

- plotting
- reporting
- experiment registry
- demonstrations

because every model returns the same `ForecastResult`.

---

# Current Implementation Status

## Infrastructure

✓ Dataset loading

✓ Dataset validation

✓ Missing data detection

✓ Feature extraction

✓ Forecast pipeline

✓ Experiment registry

✓ Forecast visualization

✓ Reporting

---

## Forecasting

✓ ARIMA

Planned

- Adaptive Kalman
- ETS
- Bayesian LSTM
- GRU–LSTM Hybrid

---

# Design Philosophy

The framework follows four principles:

1. Single Responsibility
2. Modularity
3. Extensibility
4. Reproducibility

Every forecasting algorithm should be interchangeable while sharing
the same workflow for visualization, reporting and experimentation.