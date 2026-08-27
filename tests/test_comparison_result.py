"""
Tests for ComparisonResult.
"""

from zenerestimation.comparison import (
    ComparisonResult,
)


# ============================================================
# Helpers
# ============================================================


def make_result():

    return ComparisonResult(

        battery="BAT001",

        metrics={

            "arima": {
                "rmse": 1.8,
                "mae": 1.4,
                "mape": 4.8,
            },

            "kalman": {
                "rmse": 1.5,
                "mae": 1.2,
                "mape": 4.1,
            },

        },

        rankings={

            "rmse": [

                {
                    "rank": 1,
                    "model": "kalman",
                    "value": 1.5,
                },

                {
                    "rank": 2,
                    "model": "arima",
                    "value": 1.8,
                },

            ],

            "mae": [],

            "mape": [],

        },

        best_models={

            "rmse": "kalman",

            "mae": "kalman",

            "mape": "kalman",

        },

        metadata={

            "comparison_metrics": [
                "rmse",
                "mae",
                "mape",
            ],

            "lower_is_better": True,

        },

    )


# ============================================================
# Constructor
# ============================================================


def test_constructor():

    result = make_result()

    assert result.battery == "BAT001"


# ============================================================
# Models
# ============================================================


def test_models():

    result = make_result()

    assert result.models == [
        "arima",
        "kalman",
    ]


# ============================================================
# Metrics
# ============================================================


def test_metrics():

    result = make_result()

    metrics = result.metrics

    assert metrics["arima"]["rmse"] == 1.8

    assert metrics["kalman"]["mae"] == 1.2


# ============================================================
# Rankings
# ============================================================


def test_rankings():

    result = make_result()

    rankings = result.rankings

    assert rankings["rmse"][0]["model"] == (
        "kalman"
    )

    assert rankings["rmse"][0]["rank"] == 1


# ============================================================
# Best Models
# ============================================================


def test_best_models():

    result = make_result()

    best = result.best_models

    assert best["rmse"] == "kalman"

    assert best["mae"] == "kalman"

    assert best["mape"] == "kalman"


# ============================================================
# Metadata
# ============================================================


def test_metadata():

    result = make_result()

    metadata = result.metadata

    assert metadata["lower_is_better"] is True


# ============================================================
# Summary
# ============================================================


def test_summary():

    result = make_result()

    summary = result.summary()

    assert summary["battery"] == "BAT001"

    assert "metrics" in summary

    assert "rankings" in summary

    assert "best_models" in summary

    assert "metadata" in summary


# ============================================================
# Serialization
# ============================================================


def test_to_dict():

    result = make_result()

    data = result.to_dict()

    assert data == result.summary()


# ============================================================
# repr
# ============================================================


def test_repr():

    result = make_result()

    text = repr(result)

    assert "ComparisonResult" in text

    assert "BAT001" in text

    assert "models=2" in text


# ============================================================
# Deep-Copy Protection
# ============================================================


def test_metrics_are_copy():

    result = make_result()

    metrics = result.metrics

    metrics["arima"]["rmse"] = 999

    assert (
        result.metrics["arima"]["rmse"]
        == 1.8
    )


def test_rankings_are_copy():

    result = make_result()

    rankings = result.rankings

    rankings["rmse"][0]["rank"] = 99

    assert (
        result.rankings["rmse"][0]["rank"]
        == 1
    )


def test_best_models_are_copy():

    result = make_result()

    best = result.best_models

    best["rmse"] = "changed"

    assert (
        result.best_models["rmse"]
        == "kalman"
    )


def test_metadata_is_copy():

    result = make_result()

    metadata = result.metadata

    metadata["lower_is_better"] = False

    assert (
        result.metadata["lower_is_better"]
        is True
    )