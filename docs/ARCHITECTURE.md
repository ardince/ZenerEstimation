# ZenerEstimation Architecture

**Document** : ARCHITECTURE.md  
**Framework Version** : 0.7.0  
**Document Version** : 0.7.0  
**Status** : Active  
**Last Updated** : July 2026

---

# 1. Project Status

ZenerEstimation is an open-source Python framework for battery
voltage forecasting and Remaining Useful Life (RUL) estimation.

Current implementation includes both forecasting and prognostics
under a common modular architecture.

## Current Status

| Item | Status |
|------|:------:|
| Framework Version | **0.7.0** |
| Development Stage | Active |
| Forecasting Models | ARIMA, Adaptive Kalman |
| Prognostics | Threshold + Monte Carlo RUL |
| Unit Tests | **62 Passing** |

---

# 2. Project Vision

The primary objective of ZenerEstimation is to provide a modular,
extensible and reproducible framework for battery voltage prediction
and prognostics.

The framework has been designed so that new forecasting algorithms,
visualization tools and prognostic models can be integrated without
modifying the existing architecture.

---

# 3. Overall Architecture

                          ZenerEstimation

                                 │
                                 ▼
                        Smart Dataset Loader
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
                ▼                                 ▼
          Forecasting Layer                Prognostics Layer
                │                                 │
     ┌──────────┼──────────┐            ┌─────────┼─────────┐
     │          │          │            │         │         │
   ARIMA      Kalman      LSTM        Threshold MonteCarlo RUL
     ✓          ✓          □             ✓         ✓        ✓
                │                                 │
                └──────────────┬──────────────────┘
                               ▼
                         Visualization
                               │
               Forecast Plot • Reports • Metadata
                               │
                               ▼
                      Experiment Registry

Legend

✓ Implemented

▶ Next Sprint

⏳ Planned

○ Long-Term Vision

---

# 4. Layered Architecture

## Data Layer

Responsible for

- Smart dataset loading
- Validation
- Missing period reconstruction
- Frequency detection
- Standardized BatteryDataset objects

Purpose

Every forecasting algorithm receives identical prepared datasets.

---

## Forecasting Layer

Current

- ARIMAForecaster
- KalmanForecaster

Future

- LSTMForecaster
- GRUForecaster
- HybridForecaster

Purpose

Every forecasting algorithm returns a common ForecastResult.

---

## Prognostics Layer

Current

- ThresholdEstimator
- MonteCarloRUL
- RULAnalyzer
- PrognosticResult

Purpose

Separate Remaining Useful Life estimation from forecasting.

---

## Visualization Layer

Current

- ForecastPlot

Purpose

Generate publication-quality forecast figures.

---

## Reporting Layer

Current

- ReportWriter
- Metadata Export
- JSON Metadata

Purpose

Generate reproducible experiment outputs.

---

## Experiment Layer

Current

- Experiment
- ExperimentRegistry

Purpose

Track every executed experiment together with associated artifacts.

---

# 5. Current Module Status

| Module | Status |
|---------|:------:|
| BatteryDataset | ✅ |
| SmartDatasetLoader | ✅ |
| ForecastResult | ✅ |
| PrognosticResult | ✅ |
| ARIMAForecaster | ✅ |
| KalmanForecaster | ✅ |
| ThresholdEstimator | ✅ |
| MonteCarloRUL | ✅ |
| RULAnalyzer | ✅ |
| ForecastPlot | ✅ |
| ReportWriter | ✅ |
| ExperimentRegistry | ✅ |
| Metadata Export | ✅ |
| Example Programs | ✅ |
| Unit Tests (62) | ✅ |

---

# 6. Sprint Roadmap

| Sprint | Theme | Status |
|---------|----------------------------|:------:|
| Sprint 1 | Project Foundation | ✅ |
| Sprint 2 | Dataset Infrastructure | ✅ |
| Sprint 3 | Forecast Framework | ✅ |
| Sprint 4 | Visualization | ✅ |
| Sprint 5 | ARIMA Integration | ✅ |
| Sprint 6 | Adaptive Kalman | ✅ |
| Sprint 7 | Prognostics Framework | ✅ |
| Sprint 8 | Documentation & Visualization | ▶ |
| Sprint 9 | Deep Learning Integration | ⏳ |
| Sprint 10 | Stable Framework | ○ |

---

# 7. Framework Maturity

| Version | Major Milestone | Status |
|----------|-------------------------------|:------:|
| v0.1 | Initial Prototype | ✅ |
| v0.2 | Dataset Infrastructure | ✅ |
| v0.3 | Forecast Framework | ✅ |
| v0.4 | Visualization Layer | ✅ |
| v0.5 | ARIMA Forecasting | ✅ |
| v0.6 | Adaptive Kalman Forecasting | ✅ |
| **v0.7** | **Generic Prognostics Framework** | ✅ |
| v0.8 | Documentation & Visualization | ▶ |
| v0.9 | Deep Learning Models | ⏳ |
| v1.0 | Stable Extensible Framework | ○ |

---

# 8. Design Principles

The architecture follows several core principles.

- Modular components
- Separation of concerns
- Model-independent interfaces
- Reproducible experiments
- Extensible architecture
- Test-driven development
- Human-readable reports

Forecasting and prognostics are intentionally separated so that
any forecasting model can later provide Remaining Useful Life
estimation without changing the surrounding framework.

---

# 9. Repository Layout

```
ZenerEstimation/

docs/
    ARCHITECTURE.md
    DEVELOPMENT_HISTORY.md
    RELEASE_NOTES.md

examples/

tests/

zenerestimation/

    data/

    forecasting/

    prognostics/

    visualization/

    reporting/

    experiments/
```

---

# Related Documentation

| Document | Purpose |
|----------|---------|
| ARCHITECTURE.md | Current system architecture |
| DEVELOPMENT_HISTORY.md | Evolution of the framework |
| RELEASE_NOTES.md | Version-by-version changes |

---

> ZenerEstimation is designed as a modular forecasting and
> prognostics framework in which forecasting algorithms,
> Remaining Useful Life estimation, visualization,
> reporting and experiment management remain independent,
> reusable and interoperable components.