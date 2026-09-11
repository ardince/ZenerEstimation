# ZenerEstimation Architecture

## 1. Overview

**ZenerEstimation** is a modular battery degradation forecasting and Remaining Useful Life (RUL) estimation framework.

The project is designed around a small number of architectural principles:

* deterministic and reproducible dataset preparation,
* strict separation between structural data processing and model-specific preprocessing,
* leakage-safe holdout evaluation,
* standardized forecast and evaluation artifacts,
* model-independent comparison,
* reusable diagnostics and prognostics,
* backward-compatible framework evolution,
* transparent experiment tracking.

The framework currently supports:

* ARIMA forecasting,
* Adaptive Kalman filtering,
* LSTM forecasting,
* GRU forecasting,
* hybrid trend-residual forecasting,
* threshold-based RUL estimation,
* Monte Carlo RUL estimation,
* hybrid diagnostics,
* standardized experiment storage,
* standardized holdout evaluation,
* multi-model comparison infrastructure.

---

# 2. High-Level Architecture

```text
                         RAW DATA
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Raw Dataset Ingest  │
                 │ / Schema Normalize  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  DatasetProcessor   │
                 │                     │
                 │ - date parsing      │
                 │ - duplicate policy  │
                 │ - frequency check   │
                 │ - canonical grid    │
                 │ - missing periods   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Processed Dataset   │
                 │                     │
                 │ ds                  │
                 │ microVolt           │
                 │ is_observed         │
                 └──────────┬──────────┘
                            │
                            ▼
              BatteryDataset.from_processed_csv()
                            │
                            ▼
                 ┌─────────────────────┐
                 │  BatteryDataset     │
                 │                     │
                 │ canonical timeline  │
                 │ observation mask    │
                 │ source metadata     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ ForecastEvaluator   │
                 │                     │
                 │ train / holdout     │
                 │ split               │
                 └──────────┬──────────┘
                            │
                            ▼
                TRAINING PARTITION ONLY
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Temporal / Model    │
                 │ Preprocessing       │
                 │                     │
                 │ interpolation       │
                 │ scaling             │
                 │ windows             │
                 │ etc.                │
                 └──────────┬──────────┘
                            │
                            ▼
            ┌───────────────┼────────────────┐
            │               │                │
            ▼               ▼                ▼
          ARIMA           Kalman         LSTM / GRU
            │               │                │
            └───────────────┼────────────────┘
                            │
                            ▼
                    ForecastResult
                            │
                            ▼
                    EvaluationResult
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
       Result Storage     Reports       ForecastPlot
             │
             ▼
        ResultLoader
             │
             ▼
     ForecastComparison
```

---

# 3. Package Structure

The principal framework structure is:

```text
zenerestimation/
│
├── data/
│   ├── dataset.py
│   │
│   └── processing/
│       ├── __init__.py
│       ├── processor.py
│       ├── result.py
│       └── writer.py
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py
│   └── result.py
│
├── models/
│   ├── arima/
│   ├── kalman/
│   ├── lstm/
│   ├── gru/
│   └── hybrid/
│
├── diagnostics/
│   └── hybrid.py
│
├── prognostics/
│   └── ...
│
├── comparison/
│   ├── __init__.py
│   ├── comparison.py
│   └── result.py
│
├── reporting/
│   └── ...
│
├── visualization/
│   └── ...
│
└── utils/
    ├── results.py
    └── result_loader.py
```

Example scripts are located under:

```text
examples/
```

including dataset processing, model demonstrations, forecast comparison, and processed-loader validation.

---

# 4. Data Architecture

## 4.1 Raw datasets

Raw source files are stored under:

```text
datasets/raw/
```

Raw datasets may use different source schemas.

Examples include:

```text
ds, microVolt
```

or:

```text
Month, Year, microVolt
```

Raw schema normalization occurs at the ingestion boundary.

The universal internal processing contract is:

```text
ds
microVolt
```

The `DatasetProcessor` does not contain battery-specific schema conversion logic.

---

# 5. Canonical Dataset Processing

Dataset processing is implemented under:

```text
zenerestimation/data/processing/
```

The core components are:

```text
DatasetProcessor
DatasetProcessingResult
ProcessedDatasetWriter
ProcessedDatasetPaths
```

## 5.1 DatasetProcessor

`DatasetProcessor` performs deterministic structural preparation.

Its responsibilities include:

* explicit date parsing,
* chronological sorting,
* duplicate timestamp handling,
* canonical-frequency validation,
* canonical timeline construction,
* insertion of missing periods,
* creation of the `is_observed` flag.

The default quarterly frequency is:

```text
QS-MAR
```

Supported duplicate policies are:

```text
error
first
last
mean
```

Production processing currently uses:

```text
duplicate_policy="mean"
```

for the validated battery datasets.

---

# 6. Processed Dataset Contract

Processed datasets are stored under:

```text
datasets/processed/
```

Every processed CSV contains:

```text
ds
microVolt
is_observed
```

Example:

```text
ds          microVolt    is_observed
2023-03-01  30.100       True
2023-06-01  30.400       True
2023-09-01               False
2023-12-01  31.000       True
```

The meaning of each column is:

### `ds`

Canonical timestamp on the configured temporal grid.

### `microVolt`

Measured battery voltage signal.

For inserted canonical periods:

```text
microVolt = NaN
```

### `is_observed`

Identifies whether the row originated from a real measurement.

```text
True
```

means an original observed measurement.

```text
False
```

means a canonical period inserted by the processing layer.

---

# 7. Missing-Period Accounting

Missing-period accounting is based on unique observed measurements, not raw source-row count.

The canonical relationship is:

```text
missing_periods
    =
processed_rows
    -
observed_rows
```

Duplicate-resolution accounting is:

```text
duplicate_rows_resolved
    =
source_rows
    -
observed_rows
```

This distinction is required because duplicate raw timestamps may be collapsed during processing.

---

# 8. Validated Dataset State

The current processed datasets have been validated as follows.

## 8.1 Battery 732B-5610110

```text
Source Rows       : 105
Processed Rows    : 109
Observed Rows     : 103
Missing Periods   : 6
Start Date        : 1998-03-01
End Date          : 2025-03-01
```

Inserted canonical periods:

```text
2002-09-01
2010-03-01
2010-06-01
2024-06-01
2024-09-01
2024-12-01
```

Duplicate rows resolved:

```text
2
```

## 8.2 Battery 732B-5610410

```text
Source Rows       : 109
Processed Rows    : 110
Observed Rows     : 107
Missing Periods   : 3
Start Date        : 1998-03-01
End Date          : 2025-06-01
```

Inserted canonical periods:

```text
2002-09-01
2010-03-01
2010-06-01
```

Duplicate rows resolved:

```text
2
```

---

# 9. Processed Dataset Persistence

`ProcessedDatasetWriter` persists canonical datasets.

For each battery it writes:

```text
datasets/processed/
    <battery>.csv
    <battery>.metadata.json
```

The metadata file records information including:

* schema name,
* schema version,
* battery identifier,
* source row count,
* processed row count,
* observed row count,
* missing-period count,
* frequency,
* processing version,
* processing policy,
* source file,
* column semantics.

Processed artifacts are overwrite-protected by default.

Explicit regeneration requires:

```text
--overwrite
```

---

# 10. BatteryDataset

`BatteryDataset` is the framework-level dataset abstraction.

Legacy loading remains available through:

```python
BatteryDataset.from_csv(...)
```

Canonical processed datasets are loaded through:

```python
BatteryDataset.from_processed_csv(...)
```

These APIs intentionally remain separate.

This prevents Sprint 12 processing behavior from silently changing existing raw-data workflows.

---

# 11. Processed Dataset Consumption

`BatteryDataset.from_processed_csv()`:

* loads canonical processed CSV files,
* parses `ds` as datetime,
* preserves `microVolt` missing values,
* preserves `is_observed`,
* rejects malformed observation relationships,
* rejects duplicate canonical timestamps,
* identifies the dataset as processed,
* records the source path,
* preserves battery identity.

No interpolation occurs during loading.

No scaling occurs during loading.

No model-specific preprocessing occurs during loading.

---

# 12. BatteryDataset Observation API

Processed datasets expose an explicit observation-aware interface.

## `observed_mask`

Returns a Boolean mask identifying genuine measurements.

```python
dataset.observed_mask
```

## `observed_rows`

Returns the number of real observed measurements.

```python
dataset.observed_rows
```

## `missing_period_count`

Returns the number of canonical periods inserted during structural processing.

```python
dataset.missing_period_count
```

## `is_processed`

Identifies whether the dataset originated from the canonical processed-data layer.

```python
dataset.is_processed
```

## `source_type`

For processed datasets:

```text
processed
```

## `source_path`

Records the processed CSV from which the dataset was loaded.

---

# 13. Backward Compatibility

The framework retains the existing callable:

```python
dataset.missing_periods()
```

because existing reporting, summary, and visualization code depends on it.

The new processed-dataset count therefore uses the separate API:

```python
dataset.missing_period_count
```

This preserves the existing framework contract while providing canonical Sprint 12 semantics.

---

# 14. Leakage-Safety Rule

A central architectural rule is:

> Structural dataset processing may occur before evaluation splitting. Any transformation that estimates or derives values from the target series must occur only after the holdout split.

Therefore the following operations are permitted in persistent processed datasets:

```text
date parsing
sorting
duplicate resolution
frequency validation
canonical grid construction
missing-period insertion
observation flagging
```

The following operations are explicitly excluded:

```text
target interpolation
MinMax scaling
standardization
LSTM window generation
GRU sequence generation
ARIMA differencing
Kalman state estimation
trend decomposition
residual decomposition
model-specific feature generation
```

---

# 15. Evaluation Architecture

Standardized evaluation is provided by:

```text
ForecastEvaluator
EvaluationResult
```

The evaluation contract is:

```text
canonical dataset
      │
      ▼
holdout split
      │
      ├── training partition
      │
      └── validation partition
      │
      ▼
model fit on training data only
      │
      ▼
holdout prediction
      │
      ▼
EvaluationResult
```

`ForecastEvaluator` owns metric calculation.

Demos should only orchestrate evaluation.

---

# 16. EvaluationResult

`EvaluationResult` standardizes:

```text
model
evaluation_steps
rmse
mae
mape
actual
predicted
dates
metadata
```

Its serialized evaluation artifact uses a common schema.

This enables downstream result loading and model comparison without model-specific parsing.

---

# 17. Forecast Evaluation Metrics

The current standard metrics are:

```text
RMSE
MAE
MAPE
```

Lower values are considered better.

MAPE excludes zero-valued actual observations from the denominator.

---

# 18. Forecasting Models

The framework currently includes:

## Classical models

```text
ARIMA
Adaptive Kalman Filter
```

## Neural models

```text
LSTM
GRU
```

## Hybrid models

```text
LinearTrendLSTMForecaster
KalmanLSTMForecaster
```

Hybrid forecasting follows the structure:

```text
observed series
      │
      ▼
trend model
      │
      ├──── trend
      │
      ▼
residual series
      │
      ▼
neural residual model
      │
      ▼
trend forecast
      +
residual forecast
      │
      ▼
hybrid forecast
```

---

# 19. Hybrid Diagnostics

`HybridDiagnostics` evaluates hybrid decomposition quality.

It provides:

* decomposition verification,
* residual mean,
* residual standard deviation,
* residual RMSE,
* residual autocorrelation,
* Durbin-Watson statistic,
* Ljung-Box statistics,
* quality score,
* quality grade,
* recommendations.

Results can be exported through:

```text
HybridDiagnosticsResult
```

---

# 20. Prognostics

The prognostics layer supports:

* deterministic threshold RUL,
* Monte Carlo RUL,
* prognostic summaries,
* estimated failure timing.

Forecasting and prognostics remain conceptually separate:

```text
forecasting
    → estimate future signal

prognostics
    → interpret future signal relative to
      degradation / failure criteria
```

---

# 21. Experiment Result Storage

Experiment outputs use the standardized structure:

```text
results/
  <battery>/
    <model>/
      <timestamp>_<run_number>/
        forecast.png
        forecast.json
        evaluation.json
        experiment.json
        report.txt
        experiment.log
```

Run numbering is maintained using a model-local counter.

---

# 22. Result Loading

`ResultLoader` provides read-only access to stored runs.

Supported operations include:

```text
discover
load
load_battery
load_model
latest
```

Loaded runs are represented by:

```text
ResultPackage
```

This allows comparison and reporting to operate entirely from persisted experiment artifacts.

---

# 23. Forecast Comparison

The comparison layer contains:

```text
ForecastComparison
ComparisonResult
```

It compares stored model evaluation results without retraining.

Supported comparison metrics are currently:

```text
rmse
mae
mape
```

The comparison layer can:

* construct metric tables,
* rank models,
* identify the best model per metric,
* generate standardized comparison summaries.

---

# 24. Reporting

`ReportWriter` produces human-readable experiment reports.

It supports:

* forecast information,
* evaluation information,
* hybrid diagnostics,
* residual statistics,
* quality scores,
* recommendations.

The reporting layer accepts optional diagnostics and optional evaluation information to preserve compatibility across model types.

---

# 25. Visualization

`ForecastPlot` provides forecast visualization.

Current capabilities include:

* historical observations,
* fitted/model values,
* future forecasts,
* experiment information,
* evaluation information.

The experiment information box may include:

```text
experiment identifier
missing periods
holdout length
RMSE
```

Detailed metrics remain available in reports and evaluation artifacts.

---

# 26. Sprint 12 Architecture

Sprint 12 focuses on:

> Multi-Model Evaluation Standardization

Its architectural pipeline is:

```text
Every forecasting model
        ↓
same canonical dataset representation
        ↓
same leakage-safe holdout protocol
        ↓
same evaluation schema
        ↓
ResultLoader
        ↓
ForecastComparison
```

---

# 27. Sprint 12 Milestone Status

## Sprint 12 — Standardized Forecast Evaluation

Sprint 12 introduced a common evaluation and result pipeline for
classical forecasting models.

```text
Processed Dataset
        ↓
ForecastEvaluator
        ↓
Temporal Holdout Split
        ↓
Train-Only TemporalPreprocessor
        ↓
Forecast Model
        ↓
EvaluationResult
        ↓
Standardized Result Artifacts
        ↓
ResultLoader / ForecastComparison


### Evaluation Architecture

Evaluation Principles

Processed datasets preserve the canonical temporal structure and
explicitly identify missing observations.
Temporal preprocessing is fitted only on the training partition.
Validation targets remain untouched and are never interpolated.
Historical benchmark windows may be defined using an explicit
evaluation endpoint.
Final forecasting uses the complete processed dataset independently
of the holdout evaluation window.
Forecast dates are generated centrally by BatteryDataset.
ARIMA and Kalman use the same standardized evaluation contract.

Standard Result Artifacts

Each experiment produces:

forecast.png
forecast.json
evaluation.json
experiment.json
report.txt
experiment.log

The forecast artifact contains the numerical forecast values and dates,
while the evaluation artifact contains holdout predictions and
standardized error metrics.

Classical Model Status

| Model  | Processed Data | Standard Evaluation | Standard Artifacts |
| ------ | -------------- | ------------------- | ------------------ |
| ARIMA  | ✓              | ✓                   | ✓                  |
| Kalman | ✓              | ✓                   | ✓                  |


```text
Processed Battery Dataset
        │
        ▼
ForecastEvaluator
        │
        ├── Temporal holdout split
        │
        ├── Training partition
        │       │
        │       ▼
        │   TemporalPreprocessor
        │       │
        │       ▼
        │   Model.fit()
        │
        └── Untouched validation partition
                │
                ▼
            Model.predict()
                │
                ▼
          EvaluationResult
                │
                ├── RMSE
                ├── MAE
                ├── MAPE
                ├── actual values
                ├── predicted values
                ├── dates
                └── preprocessing metadata
```

The holdout partition is never interpolated or otherwise transformed by the evaluator.

Training-only preprocessing is applied after the temporal split, preventing information from future validation periods from leaking into model training.

### Processed Dataset Layer

Canonical datasets are stored under:

```text
datasets/processed/
```

Processed datasets preserve the complete temporal structure of the battery measurements.

Required columns are:

```text
ds
microVolt
is_observed
```

Missing quarterly periods are inserted explicitly and retain:

```text
microVolt = NaN
is_observed = False
```

Persistent processed datasets are not interpolated.

Interpolation is performed only on model-training data through `TemporalPreprocessor`.

The standard loading path is:

```python
dataset = BatteryDataset.from_processed_csv(...)
```

### Temporal Preprocessing

`TemporalPreprocessor` performs universal temporal missing-value preparation.

Its responsibilities are intentionally limited to:

* internal interpolation of missing training targets;
* optional edge filling;
* preservation of `is_observed` provenance;
* preservation of dataset identity and source metadata.

It does not perform model-specific transformations such as scaling, sequence-window creation, ARIMA differencing, Kalman state estimation, or hybrid decomposition.

### Standard Forecasting Contract

Models evaluated by `ForecastEvaluator` expose the common interface:

```python
model.fit(dataset)
model.predict(steps)
```

`predict()` returns a `ForecastResult`.

This separates framework-facing forecasting models from lower-level numerical components.

For example:

```text
AdaptiveKalmanFilter
    → low-level Kalman filtering component

KalmanForecaster
    → framework forecasting model
    → fit(dataset)
    → predict(steps)
```

### Kalman Migration

`KalmanForecaster` is fully integrated with the standardized Sprint 12 evaluation pipeline.

The official Kalman workflow is now:

```text
BatteryDataset.from_processed_csv()
        │
        ├── ForecastEvaluator
        │       │
        │       ├── holdout split
        │       ├── train-only TemporalPreprocessor
        │       ├── KalmanForecaster.fit()
        │       └── EvaluationResult
        │
        └── full-data TemporalPreprocessor
                │
                ▼
          KalmanForecaster.fit()
                │
                ▼
          future ForecastResult
```

Evaluation fitting and final future forecasting use separate model instances.

The evaluation model is fitted only on the training partition.

The final forecasting model is fitted on all currently available historical data.

### Standard Result Artifacts

Each experiment produces the standardized result structure:

```text
results/
  <battery>/
    <model>/
      <timestamp>_<run_number>/
        forecast.png
        forecast.json
        evaluation.json
        experiment.json
        report.txt
        experiment.log
```

`evaluation.json` is generated from `EvaluationResult.to_dict()` and is consumed by `ResultLoader` and `ForecastComparison`.

### Current Sprint 12 Status

Completed:

* standardized `EvaluationResult`;
* standardized `ForecastEvaluator`;
* leakage-safe temporal holdout evaluation;
* train-only temporal preprocessing;
* processed dataset generation and persistence;
* processed dataset loading through `BatteryDataset`;
* ARIMA standardized evaluation;
* Kalman standardized evaluation;
* standardized evaluation artifacts;
* standardized comparison compatibility.

Current regression baseline:

```text
356 tests passing
```

Next model migrations:

```text
LSTM
GRU
Kalman-LSTM hybrid
```

---

# 28. Next Milestone — 2A.5

The next planned component is:

```text
Train-Only Temporal Preprocessing
```

Its purpose is to prepare model-ready training data after the evaluation split.

Expected architecture:

```text
BatteryDataset
      │
      ▼
ForecastEvaluator
      │
      ▼
train / holdout split
      │
      ▼
TRAIN ONLY
      │
      ▼
TemporalPreprocessor
      │
      ├── missing-value handling
      └── universal temporal preparation
      │
      ▼
model-specific preprocessing
```

For neural models this may subsequently feed:

```text
scaling
window generation
sequence construction
```

These operations must never be fitted using holdout values.

---

# 29. Planned Sprint 12 Sequence

```text
Milestone 1
    ✓ shared evaluation framework

Milestone 2A
    ✓ processed dataset standardization
    ✓ persistence
    ✓ consumption
    → train-only temporal preprocessing

Milestone 2B
    Adaptive Kalman evaluation migration

Milestone 3
    LSTM evaluation migration
    GRU evaluation migration

Milestone 4
    Kalman-LSTM evaluation migration

Milestone 5
    true multi-model comparison
```

---

# 30. Future Sprint — Model Optimization

Model optimization is intentionally separated from Sprint 12.

A later sprint will address:

* ARIMA order optimization,
* Kalman Q/R tuning,
* LSTM window optimization,
* GRU window optimization,
* neural unit-size search,
* scaling strategies,
* repeated neural runs,
* forecast stability,
* rolling-origin validation,
* hyperparameter ranking.

The separation is deliberate:

```text
Sprint 12
"What is the common data and evaluation protocol?"

Sprint 13
"What model configuration performs best under that protocol?"
```

---

# 31. Core Architectural Principle

The framework now distinguishes three fundamentally different stages:

```text
1. Structural Processing

raw measurements
    ↓
canonical timeline
```

```text
2. Evaluation-Safe Preparation

canonical timeline
    ↓
holdout split
    ↓
training-only transformations
```

```text
3. Model-Specific Learning

prepared training data
    ↓
forecasting model
    ↓
standard evaluation result
```

This separation is the foundation for reproducible and scientifically comparable battery degradation forecasting in ZenerEstimation.
