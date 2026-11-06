
"""M17-B2-G6-B: read-only Streamlit reporting integration."""

from contextlib import nullcontext
from copy import deepcopy
import sys
import types

import pandas as pd
import pytest

from zenerestimation.visualization.optimized_neural_comparison import (
    HOLDOUT,
    MODELS,
    render_external_accuracy_profile_section,
)


def make_evidence():
    holdout = pd.DataFrame(
        [
            {
                "date": date.strftime("%Y-%m-%d"),
                "model": model,
                "actual_microvolt": 30.0 + i,
                "predicted_microvolt": 30.0 + i + j * 0.1,
                "error_pred_minus_actual": j * 0.1,
            }
            for j, model in enumerate(MODELS)
            for i, date in enumerate(HOLDOUT)
        ]
    )

    manifest = {
        "battery": "732B-5610410",
        "evaluation_status":
            "post_optimization_external_holdout_evaluation",
        "holdout_dates": HOLDOUT.strftime("%Y-%m-%d").tolist(),
        "selected": {
            model: {"internal_rmse": 0.3}
            for model in MODELS
        },
        "external_holdout_metrics": {
            model: {
                "rmse": j * 0.1,
                "mae": j * 0.1,
                "first_step_signed_error_pred_minus_actual": j * 0.1,
                "first_step_absolute_error": j * 0.1,
            }
            for j, model in enumerate(MODELS)
        },
    }

    return manifest, holdout


class FakeStreamlit:
    def __init__(self):
        self.expanders = []
        self.tables = []
        self.captions = []
        self.warnings = []

    def expander(self, label):
        self.expanders.append(label)
        return nullcontext()

    def dataframe(self, frame, **kwargs):
        self.tables.append((frame.copy(deep=True), kwargs))

    def caption(self, message):
        self.captions.append(str(message))

    def warning(self, message):
        self.warnings.append(str(message))


@pytest.fixture
def fake_st(monkeypatch):
    fake = FakeStreamlit()
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        fake,
    )
    return fake


def test_g6b_renders_one_expander_and_table(fake_st):
    manifest, holdout = make_evidence()

    render_external_accuracy_profile_section(
        manifest, holdout, ["LSTM", "GRU"]
    )

    assert len(fake_st.expanders) == 1
    assert len(fake_st.tables) == 1
    assert "external holdout" in fake_st.expanders[0].lower()


def test_g6b_respects_selected_model_order(fake_st):
    manifest, holdout = make_evidence()

    render_external_accuracy_profile_section(
        manifest, holdout, ["GRU", "LSTM"]
    )

    table = fake_st.tables[0][0]

    assert table["Model"].tolist() == ["GRU", "LSTM"]


def test_g6b_preserves_reporting_schema(fake_st):
    manifest, holdout = make_evidence()

    render_external_accuracy_profile_section(
        manifest, holdout, ["LSTM"]
    )

    table = fake_st.tables[0][0]

    assert "External RMSE (µV)" in table.columns
    assert "External bias (µV)" in table.columns
    assert "First-step signed error (µV)" in table.columns
    assert "Evaluation scope" in table.columns


def test_g6b_displays_scientific_boundary_caption(fake_st):
    manifest, holdout = make_evidence()

    render_external_accuracy_profile_section(
        manifest, holdout, ["LSTM"]
    )

    captions = " ".join(fake_st.captions).lower()

    assert "post-selection" in captions
    assert "future accuracy" in captions
    assert "cross-fold" in captions


def test_g6b_does_not_mutate_source_evidence(fake_st):
    manifest, holdout = make_evidence()

    original_manifest = deepcopy(manifest)
    original_holdout = holdout.copy(deep=True)

    render_external_accuracy_profile_section(
        manifest, holdout, ["LSTM", "GRU"]
    )

    assert manifest == original_manifest
    pd.testing.assert_frame_equal(holdout, original_holdout)


def test_g6b_invalid_evidence_warns_without_table(fake_st):
    manifest, holdout = make_evidence()

    corrupted = holdout.copy(deep=True)
    corrupted.loc[0, "actual_microvolt"] = float("nan")

    render_external_accuracy_profile_section(
        manifest, corrupted, ["LSTM"]
    )

    assert len(fake_st.warnings) == 1
    assert len(fake_st.tables) == 0


def test_g6b_does_not_expose_operational_accuracy(fake_st):
    manifest, holdout = make_evidence()

    render_external_accuracy_profile_section(
        manifest, holdout, ["LSTM"]
    )

    table = fake_st.tables[0][0]

    assert set(table["Evaluation scope"]) == {
        "external_holdout"
    }
    assert not any(
        "operational" in column.lower()
        for column in table.columns
    )


def test_g6b_empty_selection_does_not_render(fake_st):
    manifest, holdout = make_evidence()

    render_external_accuracy_profile_section(
        manifest, holdout, []
    )

    assert not fake_st.tables
    assert not fake_st.expanders
