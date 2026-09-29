"""
Real-data LinearTrendLSTM optimization demonstration.

Battery:
    732B-5610110

Workflow:
    processed dataset
        -> final benchmark boundary
        -> development-only temporal optimization
        -> untouched final benchmark evaluation
"""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=(
        "Non-invertible starting MA "
        "parameters found.*"
    ),
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=(
        "Non-stationary starting "
        "autoregressive parameters found.*"
    ),
    category=UserWarning,
)

from pathlib import Path

from statsmodels.tools.sm_exceptions import ConvergenceWarning

from zenerestimation.data.dataset import BatteryDataset

from zenerestimation.forecasting.neural.lstm import LSTMForecaster
from zenerestimation.forecasting.hybrid import LinearTrendLSTMForecaster

from zenerestimation.data.temporal.preprocessor import (
    TemporalPreprocessor,
)
from zenerestimation.evaluation.evaluator import (
    ForecastEvaluator,
)

from zenerestimation.optimization import (
    HybridNeuralParameterSpace,
    BenchmarkBoundary,
    ExpandingWindowSplitter,
    GenericOptimizer,
    OptimizationEvaluator,
    OptimizationArtifact,
)

from zenerestimation.utils.results import (
    create_result_files,
    save_metadata,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BATTERY = "732B-5610110"

DATA_PATH = Path(
    "datasets/processed/732B-5610110.csv"
)

EVALUATION_STEPS = 5
EVALUATION_END = "2024-03-01"

INTERNAL_FOLDS = 3
INTERNAL_VALIDATION_STEPS = 5

SELECTION_METRIC = "rmse"

HYBRID_EPOCHS = 100
HYBRID_BATCH_SIZE = 8
HYBRID_SEED = 42


# ---------------------------------------------------------
# Load processed dataset
# ---------------------------------------------------------

dataset = BatteryDataset.from_processed_csv(
    DATA_PATH,
    battery=BATTERY,
)

print()
print("=" * 70)
print("LinearTrendLSTM OPTIMIZATION — 732B-5610110")
print("=" * 70)

print()
print("Complete processed dataset")
print("-" * 70)

print(
    f"Rows             : {len(dataset)}"
)

print(
    f"Start            : "
    f"{dataset.data['ds'].iloc[0].date()}"
)

print(
    f"End              : "
    f"{dataset.data['ds'].iloc[-1].date()}"
)


# ---------------------------------------------------------
# LSTM search space
# ---------------------------------------------------------

PARAMETER_SPACE = HybridNeuralParameterSpace(
    window=(4, 6, 8, 10, 12),
    units=(24, 32),
)


def model_factory(**params):
    """Create a fresh LinearTrendLSTM candidate."""

    window = params["window"]
    units = params["units"]

    lstm_model = LSTMForecaster(
        window=window,
        units=units,
        epochs=HYBRID_EPOCHS,
        batch_size=HYBRID_BATCH_SIZE,
        seed=HYBRID_SEED,
    )

    return LinearTrendLSTMForecaster(
        lstm_model=lstm_model,
        window=window,
    )


# ---------------------------------------------------------
# Final benchmark boundary
# ---------------------------------------------------------

boundary = BenchmarkBoundary(
    evaluation_steps=EVALUATION_STEPS,
    evaluation_end=EVALUATION_END,
)

development = (
    boundary.development_dataset(
        dataset
    )
)

benchmark = (
    boundary.benchmark_data(
        dataset
    )
)

print()
print("Benchmark boundary")
print("-" * 70)

print(
    f"Development rows : {len(development)}"
)

print(
    f"Development end  : "
    f"{development.data['ds'].iloc[-1].date()}"
)

print(
    f"Benchmark rows   : {len(benchmark)}"
)

print(
    f"Benchmark start  : "
    f"{benchmark['ds'].iloc[0].date()}"
)

print(
    f"Benchmark end    : "
    f"{benchmark['ds'].iloc[-1].date()}"
)


# ---------------------------------------------------------
# Leakage-safe training preprocessing
# ---------------------------------------------------------

preprocessor = TemporalPreprocessor(
    method="linear",
    fill_edges=True,
)


# ---------------------------------------------------------
# Internal temporal validation
# ---------------------------------------------------------

splitter = ExpandingWindowSplitter(
    folds=INTERNAL_FOLDS,
    validation_steps=INTERNAL_VALIDATION_STEPS,
)

# ---------------------------------------------------------
# Optimize on Real-Dataset
# ---------------------------------------------------------

with warnings.catch_warnings():

    optimization_evaluator = OptimizationEvaluator(
        splitter,
        metric="rmse",
        preprocessor=TemporalPreprocessor(
            method="linear",
            fill_edges=True,
        ),
    )

    optimizer = GenericOptimizer(
        optimization_evaluator,
        model="LinearTrendLSTM",
    )

    optimization = optimizer.optimize(
        development,
        model_factory,
        PARAMETER_SPACE.candidates(),
    )


print()
print("Optimization configuration")
print("-" * 70)

print(
    f"Candidates       : "
    f"{PARAMETER_SPACE.candidate_count}"
)

print(
    f"Internal folds   : "
    f"{INTERNAL_FOLDS}"
)

print(
    f"Validation steps : "
    f"{INTERNAL_VALIDATION_STEPS}"
)

print(
    f"Selection metric : "
    f"{SELECTION_METRIC.upper()}"
)


# ---------------------------------------------------------
# Candidate ranking
# ---------------------------------------------------------

best_window = optimization.best_params["window"]
best_units = optimization.best_params["units"]

selected_lstm = LSTMForecaster(
    window=best_window,
    units=best_units,
    epochs=HYBRID_EPOCHS,
    batch_size=HYBRID_BATCH_SIZE,
    seed=HYBRID_SEED,
)

selected_model = LinearTrendLSTMForecaster(
    lstm_model=selected_lstm,
    window=best_window,
)

ranked = sorted(
    optimization.candidates,
    key=lambda candidate: (
        candidate.score
    ),
)

print()
print("Top LinearTrendLSTM candidates")
print("-" * 100)

print(
    f"{'Rank':<6}"
    f"{'Window':>10}"
    f"{'Units':>10}"
    f"{'RMSE':>14}"
    f"{'MAE':>14}"
    f"{'MAPE':>14}"
)

for rank, candidate in enumerate(
    ranked[:10],
    start=1,
):
    params = candidate.params

    print(
        f"{rank:<6}"
        f"{candidate.params['window']:>10}"
        f"{candidate.params['units']:>10}"
        f"{candidate.metrics['rmse']:>14.6f}"
        f"{candidate.metrics['mae']:>12.6f}"
        f"{candidate.metrics['mape']:>12.6f}"
    )

print()
print("Candidate fold RMSE")
print("-" * 82)

print(
    f"{'Rank':<6}"
    f"{'Window':>10}"
    f"{'Units':>10}"
    f"{'Fold 1':>12}"
    f"{'Fold 2':>12}"
    f"{'Fold 3':>12}"
    f"{'Mean':>12}"
)

for rank, candidate in enumerate(
    ranked[:10],
    start=1,
):
    fold_metrics = candidate.metadata[
        "fold_metrics"
    ]

    fold_rmse = [
        item["rmse"]
        for item in fold_metrics
    ]

    print(
        f"{rank:<6}"
        f"{candidate.params['window']:>10}"
        f"{candidate.params['units']:>10}"
        f"{fold_rmse[0]:>12.6f}"
        f"{fold_rmse[1]:>12.6f}"
        f"{fold_rmse[2]:>12.6f}"
        f"{candidate.metrics['rmse']:>12.6f}"
    )

print()
print("Selected LinearTrendLSTM configuration")
print("-" * 70)
print(
    f"Best parameters  : "
    f"{optimization.best_params}"
)
print(
    f"Internal RMSE    : "
    f"{optimization.best_score:.6f}"
)


# ---------------------------------------------------------
# Final benchmark evaluation
# ---------------------------------------------------------

benchmark_evaluator = ForecastEvaluator(
    evaluation_steps=EVALUATION_STEPS,
    preprocessor=preprocessor,
    evaluation_end=EVALUATION_END,
)

with warnings.catch_warnings():

    warnings.simplefilter(
        "ignore",
        ConvergenceWarning,
    )

    evaluation = benchmark_evaluator.evaluate(
        dataset=dataset,
        model=selected_model,
    )


print()
print("Final untouched benchmark")
print("-" * 70)

print(
    f"RMSE             : "
    f"{evaluation.rmse:.6f}"
)

print(
    f"MAE              : "
    f"{evaluation.mae:.6f}"
)

print(
    f"MAPE             : "
    f"{evaluation.mape:.6f}%"
)


print()
print("Benchmark predictions")
print("-" * 70)

print(
    f"{'Date':<14}"
    f"{'Actual':>12}"
    f"{'Predicted':>14}"
    f"{'Error':>14}"
)

for (
    date,
    actual,
    predicted,
) in zip(
    evaluation.dates,
    evaluation.actual,
    evaluation.predicted,
):

    error = (
        predicted
        - actual
    )

    print(
        f"{str(date)[:10]:<14}"
        f"{actual:>12.6f}"
        f"{predicted:>14.6f}"
        f"{error:>14.6f}"
    )

print()
print("=" * 70)
print("OPTIMIZATION COMPLETE")
print("=" * 70)

optimization_artifact = OptimizationArtifact(
    optimization,
    benchmark_steps=EVALUATION_STEPS,
)

# ---------------------------------------------------------
# Standardized result files
# ---------------------------------------------------------

result_files = create_result_files(
    battery=BATTERY,
    model="linear_trend_lstm",
    # include any other arguments required by the
    # EXISTING create_result_files() contract
)


# ---------------------------------------------------------
# Write optimization artifact
# ---------------------------------------------------------

optimization_path = (
    result_files.directory
    / "optimization.json"
)

save_metadata(
    optimization_path,
    optimization_artifact.to_dict(),
)

print(
    f"Optimization artifact: "
    f"{optimization_path}"
)