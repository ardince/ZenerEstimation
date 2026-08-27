"""
============================================================

ZenerEstimation
Official Forecast Comparison Demonstration

Compares previously stored forecasting experiments.

No model is retrained.

============================================================
"""

from __future__ import annotations

import argparse

from zenerestimation.comparison import ForecastComparison
from zenerestimation.utils.console import Console
from zenerestimation.utils.result_loader import ResultLoader


# ============================================================
# Configuration
# ============================================================

FRAMEWORK_VERSION = "0.10.0"

DEFAULT_MODELS = [

    "arima",

    "kalman",

    "lstm",

    "gru",

    "kalman_lstm",

]


# ============================================================
# Arguments
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(

        description=(
            "Compare stored ZenerEstimation forecasts."
        )

    )

    parser.add_argument(

        "--battery",

        required=True,

        help="Battery identifier.",

    )

    parser.add_argument(

        "--models",

        nargs="+",

        default=DEFAULT_MODELS,

        help=(
            "Models to compare. "
            "Default: arima kalman lstm gru kalman_lstm"
        ),

    )

    return parser.parse_args()


# ============================================================
# Formatting
# ============================================================


def display_model_name(
    model,
):

    names = {

        "arima":
            "ARIMA",

        "kalman":
            "Adaptive Kalman",

        "lstm":
            "LSTM",

        "gru":
            "GRU",

        "kalman_lstm":
            "Kalman-LSTM",

    }

    return names.get(
        model,
        model,
    )


# ============================================================
# Ranking Output
# ============================================================


def print_ranking(
    comparison,
    metric,
):

    ranking = comparison.rank(
        metric
    )

    print(
        f"{metric.upper()} Ranking"
    )

    print(
        "-" * 40
    )

    if not ranking:

        print(
            "No evaluation data available."
        )

        print()

        return

    for item in ranking:

        name = display_model_name(
            item["model"]
        )

        print(

            f"{item['rank']:>2}. "

            f"{name:<20} "

            f"{item['value']:.6f}"

        )

    print()


# ============================================================
# Main
# ============================================================


def main():

    args = parse_args()

    battery = args.battery

    requested_models = [

        model.lower()

        for model in args.models

    ]


    # ========================================================
    # Start
    # ========================================================

    Console.header(

        "Official Forecast Comparison Demonstration"

    )


    # ========================================================
    # Load Stored Results
    # ========================================================

    Console.section(

        "Loading Stored Forecast Results"

    )

    loader = ResultLoader()

    runs = []

    missing_models = []

    for model in requested_models:

        run = loader.latest(

            battery=battery,

            model=model,

        )

        if run is None:

            missing_models.append(
                model
            )

            continue

        runs.append(
            run
        )


    # ========================================================
    # Validation
    # ========================================================

    if not runs:

        raise RuntimeError(

            "No stored forecasting experiments "
            f"were found for battery {battery}."

        )

    Console.success(

        f"{len(runs)} model result(s) loaded."

    )

    print()

    print(
        f"Battery           : {battery}"
    )

    print(
        f"Models requested  : "
        f"{len(requested_models)}"
    )

    print(
        f"Models available  : "
        f"{len(runs)}"
    )

    print()


    # ========================================================
    # Included Models
    # ========================================================

    Console.section(
        "Models Included"
    )

    for run in runs:

        print(

            f"{display_model_name(run.model):<20} "

            f"Run #{run.run_number:<4} "

            f"{run.timestamp}"

        )

    print()


    # ========================================================
    # Missing Models
    # ========================================================

    if missing_models:

        Console.section(
            "Models Without Stored Results"
        )

        for model in missing_models:

            print(
                display_model_name(
                    model
                )
            )

        print()


    # ========================================================
    # Comparison
    # ========================================================

    Console.section(
        "Cross-Model Comparison"
    )

    comparison = ForecastComparison(
        runs
    )

    result = comparison.compare()

    Console.success(
        "Comparison completed."
    )

    print()


    # ========================================================
    # Metric Table
    # ========================================================

    Console.section(
        "Metric Summary"
    )

    metrics = result.metrics

    print(

        f"{'Model':<20}"
        f"{'RMSE':>12}"
        f"{'MAE':>12}"
        f"{'MAPE':>12}"

    )

    print(
        "-" * 56
    )

    for model, values in metrics.items():

        name = display_model_name(
            model
        )

        rmse = values.get(
            "rmse"
        )

        mae = values.get(
            "mae"
        )

        mape = values.get(
            "mape"
        )

        rmse_text = (
            f"{rmse:.6f}"
            if rmse is not None
            else "N/A"
        )

        mae_text = (
            f"{mae:.6f}"
            if mae is not None
            else "N/A"
        )

        mape_text = (
            f"{mape:.6f}"
            if mape is not None
            else "N/A"
        )

        print(

            f"{name:<20}"
            f"{rmse_text:>12}"
            f"{mae_text:>12}"
            f"{mape_text:>12}"

        )

    print()


    # ========================================================
    # Rankings
    # ========================================================

    Console.section(
        "Model Rankings"
    )

    for metric in (

        "rmse",

        "mae",

        "mape",

    ):

        print_ranking(

            comparison,

            metric,

        )


    # ========================================================
    # Best Models
    # ========================================================

    Console.section(
        "Best Models"
    )

    for metric in (

        "rmse",

        "mae",

        "mape",

    ):

        model = result.best_models.get(
            metric
        )

        if model is None:

            name = "N/A"

        else:

            name = display_model_name(
                model
            )

        print(

            f"{metric.upper():<8} : "
            f"{name}"

        )

    print()


    # ========================================================
    # Finished
    # ========================================================

    Console.header(

        "Comparison Completed Successfully"

    )

    print(

        f"Framework Version : "
        f"{FRAMEWORK_VERSION}"

    )

    print(

        f"Battery           : "
        f"{battery}"

    )

    print(

        f"Models Compared   : "
        f"{len(result.models)}"

    )


# ============================================================
# Entry Point
# ============================================================


if __name__ == "__main__":

    main()