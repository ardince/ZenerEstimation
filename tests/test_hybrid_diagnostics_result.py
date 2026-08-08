"""
Tests for HybridDiagnosticsResult.
"""

from zenerestimation.diagnostics import HybridDiagnosticsResult


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def make_summary():

    return {

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

        "ljung_box_pvalue": 0.83,

        "quality_score": 94.5,

        "quality_grade": "Very Good",

        "recommendations": [

            "Trend captures most of the degradation.",

            "Residuals resemble white noise.",

        ],

    }


# ---------------------------------------------------------
# Constructor
# ---------------------------------------------------------

def test_constructor():

    summary = make_summary()

    result = HybridDiagnosticsResult(summary)

    assert result is not None


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

def test_summary():

    summary = make_summary()

    result = HybridDiagnosticsResult(summary)

    assert result.summary() == summary


# ---------------------------------------------------------
# Properties
# ---------------------------------------------------------

def test_quality_score():

    result = HybridDiagnosticsResult(make_summary())

    assert result.quality_score == 94.5


def test_quality_grade():

    result = HybridDiagnosticsResult(make_summary())

    assert result.quality_grade == "Very Good"


def test_recommendations():

    result = HybridDiagnosticsResult(make_summary())

    rec = result.recommendations

    assert isinstance(rec, list)

    assert len(rec) == 2


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------

def test_metadata():

    metadata = {

        "battery": "732B-5610110",

        "version": "0.10.0",

    }

    result = HybridDiagnosticsResult(

        make_summary(),

        metadata=metadata,

    )

    assert result.metadata == metadata


# ---------------------------------------------------------
# Serialization
# ---------------------------------------------------------

def test_to_dict():

    metadata = {

        "battery": "732B-5610110",

    }

    result = HybridDiagnosticsResult(

        make_summary(),

        metadata=metadata,

    )

    d = result.to_dict()

    assert "summary" in d

    assert "metadata" in d

    assert d["metadata"]["battery"] == "732B-5610110"


# ---------------------------------------------------------
# repr()
# ---------------------------------------------------------

def test_repr():

    result = HybridDiagnosticsResult(

        make_summary()

    )

    text = repr(result)

    assert "HybridDiagnosticsResult" in text

    assert "94.5" in text

    assert "Very Good" in text


# ---------------------------------------------------------
# Deep-copy protection
# ---------------------------------------------------------

def test_summary_is_copy():

    result = HybridDiagnosticsResult(

        make_summary()

    )

    s = result.summary()

    s["quality_score"] = 0

    assert result.quality_score == 94.5


def test_metadata_is_copy():

    result = HybridDiagnosticsResult(

        make_summary(),

        metadata={

            "battery": "732B",

        },

    )

    meta = result.metadata

    meta["battery"] = "XXXX"

    assert result.metadata["battery"] == "732B"