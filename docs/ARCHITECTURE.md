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
                 │ universal           │
                 │ train-only          │
                 │ temporal            │
                 │                     │
                 │ → scaling           │
                 │ → neural windows    │
                 │ →ARIMA/Kalman/hybrid│
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

# 26. Standardized Evaluation Architecture

ZenerEstimation uses a common leakage-safe evaluation architecture across classical, neural, and hybrid forecasting models.

The standardized workflow is:

```text
Processed BatteryDataset
        │
        ▼
ForecastEvaluator
        │
        ▼
Temporal Holdout Split
        │
        ├───────────────┐
        │               │
        ▼               ▼
Training Partition   Validation Partition
        │               │
        ▼               │
TemporalPreprocessor    │
        │               │
        ▼               │
Model.fit()             │
        │               │
        ▼               │
Model.predict() ────────┘
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
        └── evaluation metadata
```

The temporal split occurs before any target-derived preprocessing.

The validation partition remains untouched by interpolation, scaling, sequence construction, state estimation, or model fitting.

This rule applies equally to classical, neural, and hybrid forecasting models.

## 26.1 Evaluation and Final Forecasting Are Separate

Standardized demonstrations use separate model instances for evaluation and final forecasting.

```text
                    Processed Dataset
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
      Holdout Evaluation          Final Forecasting
              │                         │
      ForecastEvaluator          Full-data preparation
              │                         │
      training partition           fresh model
              │                         │
         evaluation fit              fit
              │                         │
      holdout prediction          future forecast
              │                         │
              ▼                         ▼
      EvaluationResult           ForecastResult
```

The evaluation model is fitted only on the training partition.

The final forecasting model is a fresh instance fitted independently using all currently available historical data after temporal preprocessing.

This prevents evaluation state from leaking into final forecasting and keeps the two workflows scientifically distinct.

## 26.2 Historical Evaluation Boundaries

`ForecastEvaluator` supports an explicit evaluation endpoint.

This is required when a processed dataset contains observations later than the intended historical benchmark window.

For example, battery `732B-5610110` uses the established evaluation interval ending:

```text
2024-03-01
```

with:

```text
evaluation_steps = 5
```

The final forecasting horizon is independent of the evaluation horizon.

Therefore:

```text
evaluation horizon != forecast horizon
```

is a valid and supported configuration.

---

# 27. Neural Forecasting Architecture

The standardized neural forecasting layer currently contains:

```text
LSTMForecaster
GRUForecaster
```

Both models follow the common neural lifecycle:

```text
BatteryDataset
      │
      ▼
BaseNeuralForecaster
      │
      ├── target extraction
      │
      ├── training-only scaling
      │
      └── WindowGenerator
              │
              ▼
        Neural Network
              │
              ▼
      Historical Fitted Values
              │
              ▼
       Recursive Forecast
              │
              ▼
        ForecastResult
```

## 27.1 Neural Scaling

Scaling is model-specific preprocessing.

The scaler is fitted only on the dataset supplied to the neural model.

During standardized holdout evaluation, `ForecastEvaluator` supplies only the training partition to `fit()`.

Therefore validation targets do not participate in scaler fitting.

## 27.2 Window Generation

Neural sequence construction is performed by `WindowGenerator`.

For a configured window length `w`, the first `w` historical positions do not have a complete input sequence.

Historical neural fitted values therefore intentionally preserve:

```text
NaN, NaN, ..., NaN, fitted values...
└──── w ────┘
```

These warm-up values are not filled or fabricated.

## 27.3 Recursive Forecasting

LSTM and GRU future forecasts are generated recursively.

Each predicted value becomes part of the input sequence used to generate the next forecast step.

Future dates are not generated independently by the neural models.

They are obtained through:

```python
dataset.forecast_dates(steps)
```

This centralizes temporal alignment across the framework.

---

# 28. Hybrid Forecasting Architecture

Hybrid forecasting separates the measured signal into a trend component and a residual component.

The common architecture is:

```text
Observed Signal
      │
      ▼
Trend Extraction
      │
      ├──────────────► Trend
      │
      ▼
Residual Series
      │
      ▼
Residual Neural Model
      │
      ▼
Residual Forecast
      │
      │
Trend Forecast
      │
      ▼
Additive Combination
      │
      ▼
Hybrid ForecastResult
```

The shared framework abstraction is:

```text
BaseHybridForecaster
```

The currently standardized hybrid implementations are:

```text
LinearTrendLSTMForecaster
KalmanLSTMForecaster
```

## 28.1 LinearTrendLSTM

`LinearTrendLSTMForecaster` uses:

```text
Trend Model       : Linear Regression
Residual Model    : LSTM
Combination       : Additive
```

The historical signal is decomposed as:

```text
measurement = linear trend + residual
```

The residual series is supplied to the LSTM model.

Future prediction is:

```text
linear trend forecast
        +
residual LSTM forecast
        =
hybrid forecast
```

## 28.2 KalmanLSTM

`KalmanLSTMForecaster` uses:

```text
Trend Model       : Adaptive Kalman Filter
Residual Model    : LSTM
Combination       : Additive
```

The Kalman filter supplies the historical trend estimate.

The residual is:

```text
measurement - Kalman trend
```

The residual series is supplied to the LSTM model.

Future prediction is:

```text
Kalman trend forecast
        +
residual LSTM forecast
        =
hybrid forecast
```

## 28.3 Shared Hybrid Contract

Both hybrid architectures expose the same framework-facing lifecycle:

```python
model.fit(dataset)
model.predict(steps)
model.diagnostics(dataset)
model.summary()
```

Both use the same centralized forecast-date contract and return a standard `ForecastResult`.

Component forecasts must have identical horizons and identical forecast dates before they can be combined.

---

# 29. Historical Fitted-Value Contract

Historical fitted values represent model-generated historical estimates.

For hybrid models they must not be reconstructed from the actual residual decomposition.

The correct contract is:

```text
historical hybrid fitted
        =
historical trend
        +
historical residual-model fitted
```

This applies to both:

```text
LinearTrendLSTM
KalmanLSTM
```

For an LSTM residual model with window length `w`, the first `w` fitted positions remain missing:

```text
Trend:
T0   T1   T2   T3   T4   T5   ...

Residual LSTM fitted:
NaN  NaN  NaN  NaN  R4   R5   ...

Hybrid fitted:
NaN  NaN  NaN  NaN  T4+R4 T5+R5 ...
```

This behavior is intentional.

Using:

```text
trend + actual residual
```

would simply reconstruct the observed measurement:

```text
trend + (measurement - trend)
        =
measurement
```

and would incorrectly produce a perfect historical fit.

`ForecastPlot` therefore uses genuine model-generated fitted values rather than the original decomposition residual.

---

# 30. Hybrid Diagnostics and Artifact Integration

Hybrid models share the diagnostic engine:

```text
HybridDiagnostics
```

The calculation lifecycle is:

```text
Hybrid Forecaster
       │
       ▼
HybridDiagnostics
       │
       ▼
diagnostic calculations
       │
       ▼
HybridDiagnosticsResult
```

`HybridDiagnostics` is the calculation engine.

`HybridDiagnosticsResult` provides the stable result representation used by reporting and artifact serialization.

## 30.1 Diagnostic Measures

The current hybrid diagnostic contract includes:

```text
decomposition verification

trend variance

residual variance

variance explained

residual mean

residual standard deviation

residual RMSE

lag-1 residual autocorrelation

Durbin-Watson statistic

Ljung-Box statistic

Ljung-Box p-value

quality score

quality grade

recommendations
```

The quality score is bounded by:

```text
0 <= quality_score <= 100
```

Diagnostic quality categories provide a compact interpretation of the residual behavior while the underlying numerical diagnostics remain available.

## 30.2 Diagnostic Persistence

Hybrid diagnostics are stored inside the existing standardized artifacts rather than creating a separate diagnostics file.

The standard physical result contract remains:

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

For hybrid runs:

```text
forecast.json
    └── complete diagnostics result

experiment.json
    └── concise diagnostic summary

report.txt
    └── human-readable diagnostic interpretation

experiment.log
    └── quality score and grade
```

This preserves compatibility with `ExperimentResult` and `ResultLoader`.

## 30.3 Shared Hybrid Artifact Contract

The standardized artifact regression contract is shared by:

```text
linear_trend_lstm
kalman_lstm
```

Both architectures are validated against the same requirements for:

```text
physical artifact structure

forecast schema

diagnostic schema

evaluation schema

experiment metadata

report contents

log contents

ResultLoader compatibility
```

Model-specific artifact parsing is therefore not required.

---

# 31. Standardized Model Matrix

The standardized forecasting architecture currently covers:

| Model           | Processed Data | Holdout Evaluation | Final Full-Data Fit | Standard Artifacts | Historical Model Fit | Hybrid Diagnostics |
| --------------- | -------------- | ------------------ | ------------------- | ------------------ | -------------------- | ------------------ |
| ARIMA           | ✓              | ✓                  | ✓                   | ✓                  | ✓                    | —                  |
| Kalman          | ✓              | ✓                  | ✓                   | ✓                  | ✓                    | —                  |
| LSTM            | ✓              | ✓                  | ✓                   | ✓                  | ✓                    | —                  |
| GRU             | ✓              | ✓                  | ✓                   | ✓                  | ✓                    | —                  |
| LinearTrendLSTM | ✓              | ✓                  | ✓                   | ✓                  | ✓                    | ✓                  |
| KalmanLSTM      | ✓              | ✓                  | ✓                   | ✓                  | ✓                    | ✓                  |

All standardized models participate in the common evaluation and artifact architecture.

The principal distinction is model-specific learning behavior, not evaluation infrastructure.

---

# 32. Standardized Demonstration Architecture

Standardized demonstrations use processed datasets and the common evaluation pipeline.

The general demonstration workflow is:

```text
BatteryDataset.from_processed_csv()
              │
              ├──────────────────────────┐
              │                          │
              ▼                          ▼
      ForecastEvaluator          TemporalPreprocessor
              │                          │
       evaluation model               full data
              │                          │
              ▼                          ▼
      EvaluationResult             fresh model
                                         │
                                         ▼
                                      fit()
                                         │
                                         ▼
                                   ForecastResult
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
               ForecastPlot          ReportWriter        Artifacts
```

Hybrid demonstrations additionally execute:

```text
model.diagnostics()
        │
        ▼
HybridDiagnosticsResult
        │
        ├── report
        ├── forecast artifact
        ├── experiment metadata
        └── experiment log
```

Legacy raw-data neural demonstrations remain available separately.

They are not treated as equivalent controlled benchmarks because their historical configurations may differ from the standardized demonstrations.

---

# 33. Sprint 12 — Multi-Model Evaluation Standardization

Sprint 12 established the common evaluation foundation.

Its principal contributions were:

* canonical processed-dataset consumption;
* standardized `EvaluationResult`;
* standardized `ForecastEvaluator`;
* temporal holdout splitting;
* leakage-safe training-only temporal preprocessing;
* centralized forecast-date generation;
* ARIMA standardized evaluation;
* Kalman standardized evaluation;
* standardized evaluation artifacts;
* `ResultLoader` compatibility;
* `ForecastComparison` compatibility.

Sprint 12 established the question:

> How should all forecasting models be evaluated under the same data and holdout protocol?

That architecture became the foundation for neural and hybrid integration in Sprint 13.

---

# 34. Sprint 13 — Neural & Hybrid Evaluation Standardization

Sprint 13 extended the standardized Sprint 12 architecture to neural and hybrid forecasting models.

Its principal objectives were:

* harden shared neural infrastructure;
* standardize LSTM evaluation;
* standardize GRU evaluation;
* harden the hybrid forecasting contract;
* integrate LinearTrendLSTM with `ForecastEvaluator`;
* integrate KalmanLSTM with `ForecastEvaluator`;
* establish genuine historical hybrid fitted-value semantics;
* integrate hybrid diagnostics with reports and artifacts;
* create standardized processed-data neural demonstrations;
* create standardized processed-data hybrid demonstrations;
* establish shared hybrid artifact regression coverage;
* preserve compatibility with legacy neural demonstrations.

## 34.1 Sprint 13 Milestones

```text
13.1
    ✓ Neural and hybrid contract audit

13.2
    ✓ Neural infrastructure hardening

13.3
    ✓ LSTM standard evaluation integration

13.4
    ✓ GRU standard evaluation integration

13.5
    ✓ Hybrid contract hardening

13.6
    ✓ Hybrid evaluation integration

13.7A
    ✓ Standardized LSTM demonstration

13.7B
    ✓ Standardized GRU demonstration

13.7C
    ✓ Standardized LinearTrendLSTM demonstration
    ✓ historical fitted-value contract
    ✓ diagnostics integration
    ✓ artifact regression

13.7D
    ✓ Standardized KalmanLSTM demonstration
    ✓ historical fitted-value contract
    ✓ diagnostics integration
    ✓ shared artifact regression

13.7E
    ✓ Architecture and regression review
```

## 34.2 Sprint 13 Regression Baseline

The Sprint 13 closure regression baseline is:

```text
517 tests passing
```

This baseline covers classical, neural, hybrid, evaluation, diagnostics, artifact, loading, reporting, and compatibility behavior.

## 34.3 Sprint 13 Architectural Outcome

At Sprint 13 closure:

```text
ARIMA
Kalman
LSTM
GRU
LinearTrendLSTM
KalmanLSTM
```

all participate in the standardized processed-data forecasting architecture.

The common contract is:

```text
canonical processed data
        ↓
leakage-safe evaluation
        ↓
standard EvaluationResult
        ↓
independent final forecasting
        ↓
standard ForecastResult
        ↓
standard artifacts
        ↓
ResultLoader / comparison / reporting
```

Hybrid models extend this contract with standardized decomposition diagnostics without changing the underlying artifact structure.

---

# 35. Future Work

Model optimization is intentionally separate from the evaluation-standardization work completed in Sprints 12 and 13.

Future work may include:

* ARIMA order optimization;
* Kalman Q/R tuning;
* LSTM hyperparameter optimization;
* GRU hyperparameter optimization;
* hybrid residual-model optimization;
* neural window-size search;
* unit-size search;
* repeated neural runs;
* forecast stability analysis;
* rolling-origin validation;
* hyperparameter ranking;
* controlled raw-versus-processed experiments using identical model configurations;
* standardized multi-model benchmark reports;
* experiment reproducibility manifests;
* release and package hardening.

Optimization should build on the established evaluation contract rather than introduce a separate evaluation path.

---

## 35.1 Sprint 14.9 — Dataset 410 Transition Diagnostics

Sprint 14.9 adds evidence-only diagnostics for the transition between the frozen measured-holdout benchmark and a separately refitted operational extrapolation for battery `732B-5610410`.

The two stages have distinct scientific meanings:

| Stage | Training data boundary | Forecast dates | Interpretation |
| --- | --- | --- | --- |
| Frozen benchmark | Through 2023-12-01 | 2024-03-01 to 2025-06-01 (6 quarters) | Predictions evaluated against reserved measured observations |
| Operational extrapolation | Through 2025-06-01 | 2025-09-01 to 2026-12-01 (6 quarters) | Future forecasts without observed targets; **unvalidated** |

The operational stage uses the same frozen model configurations, refitted through the later cutoff without reoptimization. Operational forecasts are **not** holdout benchmark scores, independent validation results, or evidence for model selection.

The complete runner is invoked from the repository root:

```text
python -m examples.optimization.run_410_complete
```

The runner calls the benchmark and operational execution functions once each, reuses their in-memory evidence, and generates transition figures without further fitting or prediction. Relevant outputs include:

```text
results/732B-5610410/optimized_benchmark/
results/732B-5610410/operational_extrapolation/
results/732B-5610410/transition_diagnostics/benchmark_to_operational_transition.png
```

The transition figure distinguishes measured history, frozen holdout predictions, unvalidated operational forecasts, and the two training boundaries (2023-12-01 and 2025-06-01). It is a presentation and diagnostic artifact, not a new experiment.

The dataset-410 complete runner and the full test suite were reported successful at the end of the transition-diagnostics implementation. The frozen benchmark predictions, metrics, model rankings, configurations, and seed policy were not revised for visualization.


## Sprint 14.9G — Dataset 110 transferred benchmark and operational diagnostics

Battery `732B-5610110` uses a canonical quarterly `QS-MAR` timeline from 1998-03-01 through 2025-03-01: 109 periods, 103 measured, and six unobserved quarters (2002-09-01; 2010-03-01; 2010-06-01; 2024-06-01; 2024-09-01; 2024-12-01). `is_observed` is preserved; imputed training values are never measured holdout targets.

**Measured-holdout benchmark:** fit only through 2022-12-01; evaluate exactly five measured quarters 2023-03-01 through 2024-03-01. The evaluator's `evaluation_end` filter is applied to the actual split data, not merely its length check. Holdout targets are `[167.17, 167.73, 168.30, 168.86, 169.90]` µV. Six frozen model specifications (ARIMA, Kalman, LSTM, GRU, LinearTrendLSTM, KalmanLSTM) are transferred from battery 410, not optimized for battery 110. Neural benchmark seed is 42; no holdout-driven parameter or seed selection is permitted.

**Operational extrapolation:** independently refit the same frozen specifications through 2025-03-01 (last measured 171.60 µV). Forecast six quarters 2025-06-01 through 2026-09-01. These are unvalidated predictions, without future measured actuals, forecast errors, or accuracy metrics. Historical missing quarters remain flagged as unobserved, even when interpolation is used inside the training partition.

**Evidence boundary:** `results/732B-5610110/transferred_benchmark/` holds historical evaluation; `operational_extrapolation/` holds future predictions; `transition_diagnostics/` holds a visualization reconstructed from stored full-precision benchmark predictions and operational JSON. Transition visualization performs no fitting or prediction. Two-decimal presentation CSVs must not be used for numerical validation or downstream scientific computation. Benchmark model rankings and stored metrics remain immutable during presentation refresh.


## 35.2 Deferred Model Optimization

A preliminary source-level review of neural forecasting was conducted following observed increases in LSTM/GRU holdout errors. This review did **not** establish a training defect and did **not** change the frozen benchmark.

Optimization of ARIMA, Kalman, LSTM, GRU, and hybrid models remains future work. It should use development-only tuning and leakage-safe validation (including rolling-origin validation where appropriate), preserve the existing standardized evaluation contract, and distinguish exploratory results from the frozen benchmark.

---

# 36. Milestone 16 — Validation, Optimization and Evidence Architecture

Milestone 16 extends the forecasting framework with explicit temporal-validation evidence, traceable optimization, and cross-battery numerical reporting. These additions preserve the existing forecasting interfaces and distinguish measured evidence from model-generated estimates.

## 36.1 Forecast alignment and evaluation

- Quarterly forecast timestamps are checked against the validation partition; alignment evidence records status, validation dates, forecast dates, horizon, and timestamp matching.
- Expanding-window optimization retains per-fold alignment evidence alongside RMSE, MAE, and MAPE metrics in candidate metadata.
- Strict alignment enforcement remains **opt-in** (`require_verified_alignment=False` by default) to preserve evaluator compatibility. Enabling it requires verified alignment rather than silently accepting partial evidence.

## 36.2 Neural optimization and reproducibility

- LSTM and GRU optimization record training diagnostics, events, search parameters, training budgets, candidate reporting, and provenance evidence.
- Dataset and training-partition fingerprints, environment provenance, and experiment manifests support reproducibility and traceability.
- Forecast artifacts and verification helpers connect neural optimization results with the evidence used to evaluate them.

## 36.3 Cross-battery interpretation and reporting

- Battery 410 and battery 110 workflows distinguish historical fitted values, measured external holdouts, and unvalidated operational forecasts.
- Historical-fit and external-accuracy explorers provide numerical views with explicit dates and provenance; dashboard and Excel exports expose comparison results without redefining their validation status.
- Numerical reconciliation checks presented tables against underlying saved evidence, including date-aligned measured-value enrichment for historical fits.
- Operational projections are **not** treated as externally validated accuracy measurements.

## 36.4 Validation boundary

The Milestone 16 local regression run reported **1,891 passed, 161 warnings, zero failures**. This verifies the tested implementation at that checkpoint; it does not assert that numerical-library warnings are eliminated or that future operational projections have been validated against observations.

# 37. Core Architectural Principle

ZenerEstimation distinguishes four stages:

```text
1. Structural Processing

raw measurements
        ↓
canonical timeline
        ↓
observation provenance
```

```text
2. Evaluation-Safe Preparation

canonical timeline
        ↓
holdout split
        ↓
training-only temporal transformations
```

```text
3. Model-Specific Learning

prepared training data
        ↓
model-specific transformations
        ↓
forecasting model
        ↓
ForecastResult
```

```text
4. Standardized Interpretation

ForecastResult
        +
EvaluationResult
        +
optional diagnostics
        ↓
reports / plots / artifacts
        ↓
ResultLoader
        ↓
comparison and prognostics
```

The governing principle is:

> Structural preparation defines what the data are. Evaluation-safe preparation defines what information the model is allowed to see. Model-specific learning defines how forecasts are generated. Standardized interpretation defines how those forecasts are evaluated, stored, compared, and communicated.

This separation is the foundation for reproducible, leakage-safe, and scientifically comparable battery degradation forecasting in ZenerEstimation.

---

# 38. Milestone 19 — B2-I2 Scientific Eligibility Evidence Architecture (Accepted)

**Acceptance date:** 18 November 2026. **Status:** Accepted at the software-contract validation level; not independently scientifically certified.

M19-B2-I2 extends the M18 forecast-plan and execution-evidence boundary with a typed, auditable scientific-eligibility assessment path:

```text
M18 verified forecast plan + execution artifact
                │
                ├──────────────► Temporal eligibility (I2-A)
                │
Assessment declaration + target evidence
                │
                └──────────────► Target observability (I2-B)
                                      │
Verified M18 forecast ────────────────┤
                                      ▼
                         Verified alignment (I2-C1)
                                      │
                                      ▼
                         Observed-only metrics (I2-C2)
                                      │
                  Temporal evidence + observability
                                      │
                                      ▼
                         Evidence integration (I2-C3)
                                      │
                                      ▼
                       Cross-contract checks (I2-D)
```

- `zenerestimation/assessment/temporal.py`: preserves verified, failed and unknown findings; verifies plan/execution and temporal boundaries.
- `zenerestimation/assessment/observability.py`: distinguishes verified measured targets from missing, interpolated, forecast-derived or unverified targets.
- `zenerestimation/assessment/observed_scoring.py`: aligns verified M18 forecasts with the declared calendar without dropping unobserved quarters.
- `zenerestimation/assessment/observed_metrics.py`: calculates RMSE/MAE on eligible observed targets; MAPE excludes zero-valued actuals from its denominator.
- `zenerestimation/assessment/evidence_integration.py`: checks assessment identity, digests, calendars, coverage and metrics, and produces detached evidence snapshots.

**Validation:** 136/136 I2 tests passed in one combined run (I2-A 19, I2-B 30, I2-C1 19, I2-C2 11, I2-C3 50, I2-D 7). Separately, 89/89 targeted M18 plan/execution contract tests passed. The two runs together report 225 passing test cases; they are not one combined 225-test run.

**Trust boundary:** Consistent typed evidence does not independently authenticate measurement provenance or chronology. `verified_findings` is not scientific certification; metric availability and full target coverage do not override FAILED or UNKNOWN temporal findings. Operational forecasts are not converted into measured validation evidence.

**Acceptance reference:** `M19_B2_I2_Final_Acceptance_Record.md`. No framework release, Git commit, or GitHub synchronization is asserted by this entry.

---

---

# 39. Milestone 19 — D1-B1/B2 Internal Generalization and Protected-Holdout Boundary (Accepted)

**Acceptance evidence:** 20 November 2026. **Status:** M19-D1-B1/B2 accepted at the tested software-contract level. This is not independent scientific certification, protected-holdout authorization, or a Milestone 19 release.

## 39.1 Scope and evidence flow

M19-D1-B1 re-verifies M18 internal model-selection artifacts and reconstructs M17 internal expanding-window accuracy evidence. M19-D1-B2 establishes a read-only, fail-closed boundary between that internal evidence and protected-holdout authorization. The M19-C2 independent benchmark authorization contract remains a separate authority; the D1 reconciliation layer does not replace it.

```text
Trusted M18 selection artifact + results + validation context
                            |
                            v
          D1-B1: verify internal selection and fold evidence
                            |
                            v
            INTERNAL_EVIDENCE_VERIFIED (internal only)
                            |
                            v
             D1-B2-B1: holdout boundary assessment
                            |
                            v
                 AUTHORIZATION_REQUIRED
                            |
           M19-C2 independent benchmark authorization inputs
                            |
                            v
             D1-B2-B2: read-only reconciliation
                            |
                            v
       AUTHORIZATION_REQUIRED or AUTHORIZATION_REJECTED
       No protected-holdout permissions are granted
```

## 39.2 Implemented contracts

- `zenerestimation/comparison/generalization_internal_evidence.py` (`verify_internal_generalization_evidence`): verifies the internal selection decision and reconstructs internal fold/pooled accuracy evidence. Ranking RMSE (mean of fold RMSE values) is distinct from pooled RMSE (all eligible observations combined). The function does not assess protected-holdout targets or calculate an external generalization gap.
- `zenerestimation/comparison/generalization_holdout_boundary.py` (`assess_generalization_holdout_boundary`): re-verifies D1-B1 evidence and returns `AUTHORIZATION_REQUIRED`; protected-holdout assessment, comparison, ranking, and disclosure permissions remain `False`.
- `zenerestimation/comparison/generalization_authorization_integration.py` (`reconcile_generalization_authorization`): recomputes the M19-C2 authorization assessment, checks its decision digest and fail-closed contract, and rejects inconsistent supplied decisions. It cannot return an authorized state.
- D1-B2-B3 regression tests verify separation from the existing single-model holdout eligibility, disclosure, scoring, and evaluation-evidence workflow; that workflow's production APIs were not modified by D1-B2.

For D1-B1/B2, `holdout_assessed` is `False` and `generalization_gap` is `None`. Verified internal evidence is **not** proof of external generalization. A consistent digest is **not** independent authentication or prospective authorization.

## 39.3 Acceptance and limitations

- D1-B2-B2 focused tests: **12 passed**.
- D1-B2-B3 downstream separation: **10 passed**.
- Combined D1 and existing single-model B3/B4 regression: **78 passed**.
- Full repository regression: **3,445 passed, 161 warnings, 1,520.87 seconds** (25:20); prior baseline was 3,413 passed with 161 warnings.
- Warnings originated from statistical diagnostics/ARIMA fitting and Keras layer configuration; no new warning-count increase was reported.

**Limits:** These tests validate the examined interfaces, not repository-wide access controls or independent provenance. The focused B2-B2 tests used a controlled stub for the positive-structure-but-insufficient M19-C2 decision; a real invalid C2 input path was rejected. No protected-holdout permission or measured external generalization gap is established.

**Checkpoint:** `docs/milestones/M19_D1_B1_B2_ACCEPTANCE_CHECKPOINT.md` (proposed repository destination). Local Git changes and extensive untracked artifacts still require selective review; no Git commit, GitHub synchronization, framework release, or complete D1 roadmap closure is asserted.

---

---

# 40. Milestone 19 — D1-C1/D Downstream Generalization Boundary (Accepted)

**Final D1 software-contract acceptance:** 23 November 2026. This section extends the accepted D1-B1/B2 boundary described in §39; it does not authorize protected-holdout access or numerical generalization-gap calculation.

## 40.1 D1-C1 fail-closed facade

`zenerestimation/comparison/generalization_gap_assessment.py` exposes `assess_generalization_gap()`, which delegates to read-only D1-B2-B2 reconciliation and accepts only the fail-closed `AUTHORIZATION_REQUIRED` / `AUTHORIZATION_REJECTED` states. It preserves verified internal-evidence identity while reporting `comparability_verified = False`, `holdout_assessed = False`, `generalization_gap = None`, and `generalization_gap_metric = None`; assessment, comparison, ranking and disclosure permissions are all `False`. This facade neither computes a numerical gap nor reads protected-holdout observations.

## 40.2 D1-D downstream separation and adversarial evidence

Existing `holdout_error_scoring.py` scores previously disclosed, declaration-bound single-model observations; `holdout_evaluation_evidence.py` reconstructs B4 evidence. Their software-verification digests do not establish external authentication, independent time provenance, or cross-model authorization. D1-D adversarial tests reject B1/B2/B3/B4 artifact promotion, permission/data injection, and unsupported benchmark binding. Test-fixture setup and `mappingproxy` deep-copy issues were corrected without production changes.

## 40.3 Verification and explicit limits

- D1-C1: **17 isolated contract tests** and **5 real-chain integration tests** passed.
- D1-D2: **14 adversarial tests** passed.
- Eight-file D1 combined regression: **86 passed**, including the above relevant tests.
- Full repository regression (2026-11-23): **3,481 passed, 161 warnings, 1,616.84 seconds (26:56), exit code 0**. Previous full-suite baseline: 3,445 passed, 161 warnings.
- No independent scientific authorization, protected-holdout permission, trusted provenance/chronology, participant-to-benchmark binding, or numerical generalization-gap value is established. The D1-C2 numerical pathway remains deferred pending separate authorization and comparability contracts.

**Final record:** `docs/milestones/M19_D1_Final_Acceptance_Record.md`. This is acceptance of tested D1 software contracts, not complete M19 release approval or GitHub publication.
