
"""M17-B2-G6-A: external accuracy profile reporting contract."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from zenerestimation.evaluation.accuracy_diagnostics import (
    build_accuracy_diagnostics,
)
from zenerestimation.evaluation.accuracy_profile import (
    build_accuracy_profile,
)
from zenerestimation.visualization.optimized_neural_comparison import (
    external_accuracy_profile_table,
)


COLUMNS = [
    "Model",
    "External RMSE (µV)",
    "External MAE (µV)",
    "External bias (µV)",
    "First-step signed error (µV)",
    "First-step absolute error (µV)",
    "Alignment",
    "Evaluation scope",
    "Stability interpretation",
]

SINGLE_SEQUENCE = (
    "Single external sequence; not cross-fold stability"
)


def make_profile(
    model="LSTM",
    *,
    actual=None,
    predicted=None,
    alignment="verified",
    scope="external_holdout",
):
    if actual is None:
        actual = np.array([10.0, 11.0, 12.0])
    if predicted is None:
        predicted = np.array([11.0, 10.0, 14.0])

    dates = pd.date_range(
        "2024-03-01", periods=len(actual), freq="QS-MAR"
    )

    diagnostics = build_accuracy_diagnostics(
        [
            {
                "fold_index": 0,
                "dates": dates,
                "actual": np.asarray(actual, dtype=float),
                "predicted": np.asarray(predicted, dtype=float),
                "alignment_status": alignment,
            }
        ],
        dataset="732B-5610410",
        model=model,
        scope=scope,
    )

    return build_accuracy_profile(diagnostics)


@pytest.fixture
def profiles():
    return {
        "LSTM": make_profile("LSTM"),
        "GRU": make_profile(
            "GRU",
            predicted=np.array([9.0, 12.0, 13.0]),
        ),
        "KalmanLSTM": make_profile(
            "KalmanLSTM",
            predicted=np.array([10.0, 11.0, 12.0]),
        ),
    }


def test_g6a_returns_expected_dataframe_schema(profiles):
    table = external_accuracy_profile_table(
        profiles, ["LSTM", "GRU"]
    )

    assert isinstance(table, pd.DataFrame)
    assert table.columns.tolist() == COLUMNS
    assert len(table) == 2


def test_g6a_preserves_requested_model_order(profiles):
    table = external_accuracy_profile_table(
        profiles, ["KalmanLSTM", "LSTM", "GRU"]
    )

    assert table["Model"].tolist() == [
        "KalmanLSTM", "LSTM", "GRU"
    ]


def test_g6a_numerical_values_match_profile(profiles):
    table = external_accuracy_profile_table(
        profiles, ["LSTM"]
    )

    row = table.iloc[0]
    profile = profiles["LSTM"]

    assert row["External RMSE (µV)"] == pytest.approx(
        profile["overall"]["rmse_microVolt"]
    )
    assert row["External MAE (µV)"] == pytest.approx(
        profile["overall"]["mae_microVolt"]
    )
    assert row["External bias (µV)"] == pytest.approx(
        profile["overall"]["bias_microVolt"]
    )
    assert row["First-step signed error (µV)"] == pytest.approx(
        profile["first_step"]["mean_error_microVolt"]
    )
    assert row["First-step absolute error (µV)"] == pytest.approx(
        profile["first_step"]["mae_microVolt"]
    )


def test_g6a_preserves_signed_bias(profiles):
    table = external_accuracy_profile_table(
        profiles, ["LSTM"]
    )

    # Errors: +1, -1, +2; mean signed bias = +2/3.
    assert table.iloc[0]["External bias (µV)"] == pytest.approx(
        2.0 / 3.0
    )


def test_g6a_reports_verified_external_scope(profiles):
    table = external_accuracy_profile_table(
        profiles, ["LSTM"]
    )

    row = table.iloc[0]

    assert row["Alignment"] == "verified"
    assert row["Evaluation scope"] == "external_holdout"


def test_g6a_labels_single_sequence_stability(profiles):
    table = external_accuracy_profile_table(
        profiles, ["LSTM"]
    )

    assert table.iloc[0]["Stability interpretation"] == (
        SINGLE_SEQUENCE
    )


def test_g6a_rejects_internal_validation_profile(profiles):
    corrupted = deepcopy(profiles)
    corrupted["LSTM"] = make_profile(
        "LSTM", scope="internal_expanding_window"
    )

    with pytest.raises(ValueError):
        external_accuracy_profile_table(
            corrupted, ["LSTM"]
        )


def test_g6a_rejects_positional_only_alignment(profiles):
    corrupted = deepcopy(profiles)
    corrupted["LSTM"] = make_profile(
        "LSTM", alignment="positional_only"
    )

    with pytest.raises(ValueError):
        external_accuracy_profile_table(
            corrupted, ["LSTM"]
        )


def test_g6a_rejects_missing_requested_model(profiles):
    with pytest.raises(ValueError):
        external_accuracy_profile_table(
            profiles, ["LinearTrendLSTM"]
        )


def test_g6a_rejects_model_identity_mismatch(profiles):
    corrupted = deepcopy(profiles)
    corrupted["LSTM"]["provenance"]["model"] = "GRU"

    with pytest.raises(ValueError):
        external_accuracy_profile_table(
            corrupted, ["LSTM"]
        )


def test_g6a_rejects_nonfinite_metric(profiles):
    corrupted = deepcopy(profiles)
    corrupted["LSTM"]["overall"]["rmse_microVolt"] = np.nan

    with pytest.raises(ValueError):
        external_accuracy_profile_table(
            corrupted, ["LSTM"]
        )


def test_g6a_rejects_incomplete_profile(profiles):
    corrupted = deepcopy(profiles)
    del corrupted["LSTM"]["first_step"]["mae_microVolt"]

    with pytest.raises(ValueError):
        external_accuracy_profile_table(
            corrupted, ["LSTM"]
        )


def test_g6a_rejects_duplicate_model_selection(profiles):
    with pytest.raises(ValueError):
        external_accuracy_profile_table(
            profiles, ["LSTM", "LSTM"]
        )


def test_g6a_rejects_unknown_profile_version(profiles):
    corrupted = deepcopy(profiles)
    corrupted["LSTM"]["profile_version"] = "unknown"

    with pytest.raises(ValueError):
        external_accuracy_profile_table(
            corrupted, ["LSTM"]
        )


def test_g6a_does_not_mutate_profiles(profiles):
    original = deepcopy(profiles)

    external_accuracy_profile_table(
        profiles, ["GRU", "LSTM"]
    )

    assert profiles == original


def test_g6a_does_not_add_operational_accuracy(profiles):
    table = external_accuracy_profile_table(
        profiles, ["LSTM", "GRU"]
    )

    assert set(table["Evaluation scope"]) == {
        "external_holdout"
    }
    assert not any(
        "operational" in column.lower()
        for column in table.columns
    )
