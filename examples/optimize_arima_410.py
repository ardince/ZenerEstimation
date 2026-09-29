"""
Real-data ARIMA optimization demonstration.

Battery:
    732B-5610410

Workflow:
    processed dataset
        -> final benchmark boundary
        -> development-only temporal optimization
        -> selected ARIMA order
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
from zenerestimation.data.temporal.preprocessor import (
    TemporalPreprocessor,
)
from zenerestimation.evaluation.evaluator import (
    ForecastEvaluator,
)
from zenerestimation.forecasting.arima import (
    ARIMAForecaster,
)
from zenerestimation.optimization import (
    ARIMAParameterSpace,
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

BATTERY = "732B-5610410"

DATA_PATH = Path(
    "datasets/processed/732B-5610410.csv"
)

EVALUATION_STEPS = 6

#INTERNAL_FOLDS = 3
#INTERNAL_VALIDATION_STEPS = 4

INTERNAL_FOLDS = 3
INTERNAL_VALIDATION_STEPS = 6

SELECTION_METRIC = "rmse"


# ---------------------------------------------------------
# Load processed dataset
# ---------------------------------------------------------

dataset = BatteryDataset.from_processed_csv(
    DATA_PATH,
    battery=BATTERY,
)

print()
print("=" * 70)
print("ARIMA OPTIMIZATION — 732B-5610410")
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
# Final benchmark boundary
# ---------------------------------------------------------

boundary = BenchmarkBoundary(
    evaluation_steps=EVALUATION_STEPS,
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

optimization_evaluator = (
    OptimizationEvaluator(
        splitter,
        metric=SELECTION_METRIC,
        preprocessor=preprocessor,
    )
)


# ---------------------------------------------------------
# ARIMA search space
# ---------------------------------------------------------

space = ARIMAParameterSpace(
    p=(
        0,
        1,
        2,
        3,
        4,
        5,
        8,
        12,
    ),
    d=(
        0,
        1,
    ),
    q=(
        0,
        1,
        2,
        3,
    ),
)

print()
print("Optimization configuration")
print("-" * 70)

print(
    f"Candidates       : "
    f"{space.candidate_count}"
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
# Optimize on development data ONLY
# ---------------------------------------------------------

optimizer = GenericOptimizer(
    optimization_evaluator,
    model="ARIMA",
)

with warnings.catch_warnings():

    warnings.simplefilter(
        "ignore",
        ConvergenceWarning,
    )

    optimization = optimizer.optimize(
        dataset=development,
        model_factory=ARIMAForecaster,
        candidates=space.candidates(),
    )

    optimization_artifact = OptimizationArtifact(
        optimization,
        benchmark_steps=EVALUATION_STEPS,
    )


# ---------------------------------------------------------
# Candidate ranking
# ---------------------------------------------------------

ranked = sorted(
    optimization.candidates,
    key=lambda candidate: (
        candidate.score
    ),
)

print()
print("Top ARIMA candidates")
print("-" * 70)

print(
    f"{'Rank':<6}"
    f"{'Order':<18}"
    f"{'RMSE':>12}"
    f"{'MAE':>12}"
    f"{'MAPE':>12}"
)

for rank, candidate in enumerate(
    ranked[:10],
    start=1,
):

    order = candidate.params[
        "order"
    ]

    print(
        f"{rank:<6}"
        f"{str(order):<18}"
        f"{candidate.metrics['rmse']:>12.6f}"
        f"{candidate.metrics['mae']:>12.6f}"
        f"{candidate.metrics['mape']:>12.6f}"
    )


print()
print("Selected ARIMA configuration")
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

selected_model = ARIMAForecaster(
    **optimization.best_params
)

benchmark_evaluator = ForecastEvaluator(
    evaluation_steps=EVALUATION_STEPS,
    preprocessor=preprocessor,
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


REFERENCE_RMSE = 0.407101
REFERENCE_MAE = 0.326640
REFERENCE_MAPE = 1.035697

print()
print("Reference comparison")
print("-" * 70)

print(
    f"{'Metric':<10}"
    f"{'Reference':>14}"
    f"{'Optimized':>14}"
    f"{'Difference':>14}"
)

comparisons = (
    (
        "RMSE",
        REFERENCE_RMSE,
        evaluation.rmse,
    ),
    (
        "MAE",
        REFERENCE_MAE,
        evaluation.mae,
    ),
    (
        "MAPE",
        REFERENCE_MAPE,
        evaluation.mape,
    ),
)

for (
    name,
    reference,
    optimized,
) in comparisons:

    print(
        f"{name:<10}"
        f"{reference:>14.6f}"
        f"{optimized:>14.6f}"
        f"{optimized - reference:>14.6f}"
    )


expected_dates = [
    "2024-03-01",
    "2024-06-01",
    "2024-09-01",
    "2024-12-01",
    "2025-03-01",
    "2025-06-01",
]

expected_values = [
    29.900,
    30.506,
    30.707,
    31.319,
    31.932,
    32.000,
]

actual_dates = (
    benchmark["ds"]
    .dt.strftime("%Y-%m-%d")
    .tolist()
)

actual_values = (
    benchmark["microVolt"]
    .astype(float)
    .tolist()
)

assert actual_dates == (
    expected_dates
)

for actual, expected in zip(
    actual_values,
    expected_values,
):
    assert abs(
        actual - expected
    ) < 1e-9


print()
print("Selected candidate fold metrics")
print("-" * 70)

selected_candidate = next(
    candidate
    for candidate in optimization.candidates
    if candidate.params == optimization.best_params
)

for index, metrics in enumerate(
    selected_candidate.metadata["fold_metrics"],
    start=1,
):
    print(
        f"Fold {index}: "
        f"RMSE={metrics['rmse']:.6f}  "
        f"MAE={metrics['mae']:.6f}  "
        f"MAPE={metrics['mape']:.6f}%"
    )


print()
print("Top candidate fold RMSE")
print("-" * 90)

print(
    f"{'Rank':<6}"
    f"{'Order':<18}"
    f"{'Fold 1':>12}"
    f"{'Fold 2':>12}"
    f"{'Fold 3':>12}"
    f"{'Mean':>12}"
)

for rank, candidate in enumerate(
    ranked[:10],
    start=1,
):
    fold_rmse = [
        fold["rmse"]
        for fold
        in candidate.metadata["fold_metrics"]
    ]

    print(
        f"{rank:<6}"
        f"{str(candidate.params['order']):<18}"
        f"{fold_rmse[0]:>12.6f}"
        f"{fold_rmse[1]:>12.6f}"
        f"{fold_rmse[2]:>12.6f}"
        f"{candidate.score:>12.6f}"
    )


# ---------------------------------------------------------
# Standardized result files
# ---------------------------------------------------------

result_files = create_result_files(
    battery=BATTERY,
    model="arima",
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