"""
Artifact regression tests for the standardized
LinearTrendLSTM demonstration.

These tests validate the persisted artifact contract
without executing neural-network training.
"""

import json
import pytest

from zenerestimation.utils.result_loader import (
    ResultLoader,
)

from zenerestimation.utils.results import (
    create_result_files,
    save_metadata,
)


# ============================================================
# Constants
# ============================================================

BATTERY = "BAT001"
EXPERIMENT_ID = 101


HYBRID_MODELS = [
    pytest.param(
        "linear_trend_lstm",
        "LinearTrendLSTM",
        id="linear-trend-lstm",
    ),
    pytest.param(
        "kalman_lstm",
        "KalmanLSTM",
        id="kalman-lstm",
    ),
]


# ============================================================
# Helpers
# ============================================================

def make_diagnostics():

    return {
        "summary": {
            "family": "Hybrid",
            "decomposition_ok": True,
            "trend_variance": 12.3,
            "residual_variance": 1.2,
            "variance_explained": 0.91,
            "max_error": 0.0,
            "mean_error": 0.0,
            "rmse_error": 0.0,
            "residual_mean": 0.01,
            "residual_std": 0.52,
            "residual_rmse": 0.52,
            "lag1_autocorrelation": -0.05,
            "durbin_watson": 2.03,
            "ljung_box_statistic": 4.25,
            "ljung_box_pvalue": 0.83,
            "quality_score": 94.5,
            "quality_grade": "Very Good",
            "recommendations": [
                (
                    "Trend captures most of "
                    "the degradation."
                ),
                (
                    "Residuals resemble "
                    "white noise."
                ),
            ],
        },
        "metadata": {
            "family": "Hybrid",
        },
    }


def create_hybrid_run(
    root,
    model_slug,
    model_name,
):

    paths = create_result_files(
        battery=BATTERY,
        model=model_slug,
        root=root,
    )

    diagnostics = make_diagnostics()

    forecast = {
        "battery": BATTERY,
        "model": model_slug,
        "experiment_id": EXPERIMENT_ID,
        "horizon": 3,
        "dates": [
            "2025-06-01 00:00:00",
            "2025-09-01 00:00:00",
            "2025-12-01 00:00:00",
        ],
        "forecast": [
            170.1,
            170.5,
            170.9,
        ],
        "metadata": {
            "model": model_name,
            "horizon": 3,
        },
        "diagnostics": diagnostics,
    }

    evaluation = {
        "model": model_slug,
        "evaluation_steps": 3,
        "rmse": 0.25,
        "mae": 0.20,
        "mape": 0.12,
        "actual": [
            169.0,
            169.5,
            170.0,
        ],
        "predicted": [
            168.9,
            169.4,
            169.8,
        ],
        "dates": [
            "2024-09-01 00:00:00",
            "2024-12-01 00:00:00",
            "2025-03-01 00:00:00",
        ],
        "metadata": {
            "training_points": 100,
            "validation_points": 3,
            "preprocessing": {
                "name": "TemporalPreprocessor",
            },
        },
        "battery": BATTERY,
        "experiment_id": EXPERIMENT_ID,
    }

    experiment = {
        "experiment": {
            "id": EXPERIMENT_ID,
            "battery": BATTERY,
            "model": model_name,
            "version": "0.12.0",
            "horizon": 3,
            "metadata": {
                "workflow": (
                    "standardized_processed"
                ),
                "diagnostics": {
                    "quality_score": 94.5,
                    "quality_grade": (
                        "Very Good"
                    ),
                    "variance_explained": 0.91,
                    "residual_rmse": 0.52,
                    "lag1_autocorrelation": -0.05,
                    "durbin_watson": 2.03,
                    "ljung_box_pvalue": 0.83,
                },
            },
        },
        "dataset": {
            "rows": 103,
            "frequency": "Quarterly",
        },
    }

    save_metadata(
        paths.forecast,
        forecast,
    )

    save_metadata(
        paths.evaluation,
        evaluation,
    )

    save_metadata(
        paths.experiment,
        experiment,
    )

    paths.report.write_text(
        (
            "ZenerEstimation Forecast Report\n"
            "Evaluation\n"
            "RMSE: 0.25\n"
            "Hybrid Diagnostics\n"
            "Quality Score: 94.5\n"
            "Quality Grade: Very Good\n"
            "Recommendations\n"
        ),
        encoding="utf-8",
    )

    paths.log.write_text(
        (
            "ZenerEstimation Experiment Log\n"
            "Evaluation RMSE: 0.250000\n"
            "Evaluation MAE: 0.200000\n"
            "Evaluation MAPE: 0.120000%\n"
            "Hybrid Quality Score: 94.50\n"
            "Hybrid Quality Grade: Very Good\n"
            "Status: completed\n"
        ),
        encoding="utf-8",
    )

    paths.figure.write_bytes(
        b"synthetic forecast figure"
    )

    return paths


# ============================================================
# Physical Artifact Contract
# ============================================================

@pytest.mark.parametrize(
    "model_slug,model_name",
    HYBRID_MODELS,
)

def test_linear_trend_lstm_standard_artifacts_exist(
    tmp_path,
    model_slug,
    model_name,
):

    paths = create_hybrid_run(
        tmp_path,
        model_slug,
        model_name,
    )

    assert paths.forecast.exists()
    assert paths.evaluation.exists()
    assert paths.experiment.exists()
    assert paths.report.exists()
    assert paths.log.exists()
    assert paths.figure.exists()


@pytest.mark.parametrize(
    "model_slug,model_name",
    HYBRID_MODELS,
)

def test_linear_trend_lstm_artifact_names(
    tmp_path,
    model_slug,
    model_name,
):

    paths = create_hybrid_run(
        tmp_path,
        model_slug,
        model_name,
    )

    assert paths.forecast.name == (
        "forecast.json"
    )

    assert paths.evaluation.name == (
        "evaluation.json"
    )

    assert paths.experiment.name == (
        "experiment.json"
    )

    assert paths.report.name == (
        "report.txt"
    )

    assert paths.log.name == (
        "experiment.log"
    )

    assert paths.figure.name == (
        "forecast.png"
    )


# ============================================================
# Forecast Artifact Contract
# ============================================================

@pytest.mark.parametrize(
    "model_slug,model_name",
    HYBRID_MODELS,
)

def test_forecast_contains_standard_fields(
    tmp_path,
    model_slug,
    model_name,
):

    paths = create_hybrid_run(
        tmp_path,
        model_slug,
        model_name,
    )

    forecast = json.loads(
        paths.forecast.read_text(
            encoding="utf-8"
        )
    )

    assert forecast["battery"] == BATTERY
    assert forecast["model"] == model_slug

    assert forecast["experiment_id"] == (
        EXPERIMENT_ID
    )

    assert forecast["horizon"] == 3

    assert len(forecast["dates"]) == 3
    assert len(forecast["forecast"]) == 3

    assert "metadata" in forecast


@pytest.mark.parametrize(
    "model_slug,model_name",
    HYBRID_MODELS,
)

def test_forecast_contains_hybrid_diagnostics(
    tmp_path,
    model_slug,
    model_name,
):

    paths = create_hybrid_run(
        tmp_path,
        model_slug,
        model_name,
    )

    forecast = json.loads(
        paths.forecast.read_text(
            encoding="utf-8"
        )
    )

    assert "diagnostics" in forecast

    diagnostics = forecast["diagnostics"]

    assert "summary" in diagnostics
    assert "metadata" in diagnostics

    assert (
        diagnostics["metadata"]["family"]
        == "Hybrid"
    )


@pytest.mark.parametrize(
    "model_slug,model_name",
    HYBRID_MODELS,
)

def test_forecast_diagnostics_summary_contract(
    tmp_path,
    model_slug,
    model_name,
):

    paths = create_hybrid_run(
        tmp_path,
        model_slug,
        model_name,
    )

    forecast = json.loads(
        paths.forecast.read_text(
            encoding="utf-8"
        )
    )

    summary = (
        forecast["diagnostics"]["summary"]
    )

    required = {
        "family",
        "decomposition_ok",
        "trend_variance",
        "residual_variance",
        "variance_explained",
        "max_error",
        "mean_error",
        "rmse_error",
        "residual_mean",
        "residual_std",
        "residual_rmse",
        "lag1_autocorrelation",
        "durbin_watson",
        "ljung_box_statistic",
        "ljung_box_pvalue",
        "quality_score",
        "quality_grade",
        "recommendations",
    }

    assert required.issubset(
        summary.keys()
    )


@pytest.mark.parametrize(
    "model_slug,model_name",
    HYBRID_MODELS,
)

def test_forecast_diagnostics_semantics(
    tmp_path,
    model_slug,
    model_name,
):

    paths = create_hybrid_run(
        tmp_path,
        model_slug,
        model_name,
    )

    forecast = json.loads(
        paths.forecast.read_text(
            encoding="utf-8"
        )
    )

    summary = (
        forecast["diagnostics"]["summary"]
    )

    assert (
        summary["decomposition_ok"]
        is True
    )

    assert (
        0.0
        <= summary["variance_explained"]
        <= 1.0
    )

    assert (
        0.0
        <= summary["quality_score"]
        <= 100.0
    )

    assert summary["quality_grade"] in {
        "Excellent",
        "Very Good",
        "Good",
        "Fair",
        "Poor",
    }

    assert isinstance(
        summary["recommendations"],
        list,
    )


# ============================================================
# Evaluation Artifact Contract
# ============================================================

@pytest.mark.parametrize(
    "model_slug,model_name",
    HYBRID_MODELS,
)

def test_evaluation_contains_standard_metrics(
    tmp_path,
    model_slug,
    model_name,
):

    paths = create_hybrid_run(
        tmp_path,
        model_slug,
        model_name,
    )

    evaluation = json.loads(
        paths.evaluation.read_text(
            encoding="utf-8"
        )
    )

    assert evaluation["battery"] == BATTERY

    assert evaluation["experiment_id"] == (
        EXPERIMENT_ID
    )

    assert evaluation["evaluation_steps"] == 3

    assert "rmse" in evaluation
    assert "mae" in evaluation
    assert "mape" in evaluation

    assert evaluation["rmse"] >= 0.0
    assert evaluation["mae"] >= 0.0
    assert evaluation["mape"] >= 0.0


@pytest.mark.parametrize(
    "model_slug,model_name",
    HYBRID_MODELS,
)

def test_evaluation_preserves_holdout_metadata(
    tmp_path,
    model_slug,
    model_name,
):

    paths = create_hybrid_run(
        tmp_path,
        model_slug,
        model_name,
    )

    evaluation = json.loads(
        paths.evaluation.read_text(
            encoding="utf-8"
        )
    )

    metadata = evaluation["metadata"]

    assert metadata["training_points"] == 100
    assert metadata["validation_points"] == 3

    assert (
        metadata["preprocessing"]["name"]
        == "TemporalPreprocessor"
    )


# ============================================================
# Experiment Artifact Contract
# ============================================================

@pytest.mark.parametrize(
    "model_slug,model_name",
    HYBRID_MODELS,
)

def test_experiment_contains_standardized_workflow(
    tmp_path,
    model_slug,
    model_name,
):

    paths = create_hybrid_run(
        tmp_path,
        model_slug,
        model_name,
    )

    artifact = json.loads(
        paths.experiment.read_text(
            encoding="utf-8"
        )
    )

    experiment = artifact["experiment"]

    assert experiment["battery"] == BATTERY

    assert experiment["model"] == model_name

    assert (
        experiment["metadata"]["workflow"]
        == "standardized_processed"
    )


@pytest.mark.parametrize(
    "model_slug,model_name",
    HYBRID_MODELS,
)

def test_experiment_contains_diagnostic_summary(
    tmp_path,
    model_slug,
    model_name,
):

    paths = create_hybrid_run(
        tmp_path,
        model_slug,
        model_name,
    )

    artifact = json.loads(
        paths.experiment.read_text(
            encoding="utf-8"
        )
    )

    diagnostics = (
        artifact["experiment"]
        ["metadata"]
        ["diagnostics"]
    )

    required = {
        "quality_score",
        "quality_grade",
        "variance_explained",
        "residual_rmse",
        "lag1_autocorrelation",
        "durbin_watson",
        "ljung_box_pvalue",
    }

    assert required.issubset(
        diagnostics.keys()
    )


# ============================================================
# Human-Readable Artifacts
# ============================================================

@pytest.mark.parametrize(
    "model_slug,model_name",
    HYBRID_MODELS,
)

def test_report_contains_hybrid_information(
    tmp_path,
    model_slug,
    model_name,
):

    paths = create_hybrid_run(
        tmp_path,
        model_slug,
        model_name,
    )

    report = paths.report.read_text(
        encoding="utf-8"
    )

    assert "Evaluation" in report
    assert "Hybrid Diagnostics" in report
    assert "Quality Score" in report
    assert "Quality Grade" in report
    assert "Recommendations" in report


@pytest.mark.parametrize(
    "model_slug,model_name",
    HYBRID_MODELS,
)

def test_log_contains_evaluation_and_diagnostics(
    tmp_path,
    model_slug,
    model_name,
):

    paths = create_hybrid_run(
        tmp_path,
        model_slug,
        model_name,
    )

    log = paths.log.read_text(
        encoding="utf-8"
    )

    assert "Evaluation RMSE" in log
    assert "Evaluation MAE" in log
    assert "Evaluation MAPE" in log

    assert "Hybrid Quality Score" in log
    assert "Hybrid Quality Grade" in log

    assert "Status: completed" in log


# ============================================================
# ResultLoader Compatibility
# ============================================================

@pytest.mark.parametrize(
    "model_slug,model_name",
    HYBRID_MODELS,
)

def test_result_loader_loads_hybrid_package(
    tmp_path,
    model_slug,
    model_name,
):

    paths = create_hybrid_run(
        tmp_path,
        model_slug,
        model_name,
    )

    loader = ResultLoader(
        root=tmp_path
    )

    package = loader.load(
        paths.directory
    )

    assert package.battery == BATTERY
    assert package.model == model_slug

    assert package.forecast is not None
    assert package.evaluation is not None
    assert package.experiment is not None

    assert package.report is not None
    assert package.figure is not None
    assert package.log is not None


@pytest.mark.parametrize(
    "model_slug,model_name",
    HYBRID_MODELS,
)

def test_result_loader_preserves_diagnostics(
    tmp_path,
    model_slug,
    model_name,
):

    paths = create_hybrid_run(
        tmp_path,
        model_slug,
        model_name,
    )

    loader = ResultLoader(
        root=tmp_path
    )

    package = loader.load(
        paths.directory
    )

    diagnostics = (
        package.forecast["diagnostics"]
    )

    assert (
        diagnostics["metadata"]["family"]
        == "Hybrid"
    )

    assert (
        diagnostics["summary"]
        ["quality_grade"]
        == "Very Good"
    )