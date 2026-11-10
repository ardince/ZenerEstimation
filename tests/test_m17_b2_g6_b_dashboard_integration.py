
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
    """Minimal Streamlit test double for B2 and C4 reporting."""

    def __init__(self):
        self.expanders = []
        self.tables = []
        self.captions = []
        self.warnings = []
        self.errors = []
        self.infos = []
        self.markdowns = []
        self.session_state = {}

        # Simulated widget values.
        self.threshold_choice = "0.50 µV (default)"
        self.custom_threshold = 0.5
        self.apply_clicked = False

    def expander(self, label):
        self.expanders.append(label)
        return nullcontext()

    def dataframe(self, frame, **kwargs):
        self.tables.append((frame.copy(deep=True), kwargs))

    def caption(self, message):
        self.captions.append(str(message))

    def warning(self, message):
        self.warnings.append(str(message))

    def error(self, message):
        self.errors.append(str(message))

    def info(self, message):
        self.infos.append(str(message))

    def markdown(self, message):
        self.markdowns.append(str(message))

    def divider(self):
        pass

    def selectbox(self, label, options, index=0, **kwargs):
        assert self.threshold_choice in options
        return self.threshold_choice

    def number_input(self, label, **kwargs):
        return self.custom_threshold

    def button(self, label, **kwargs):
        if kwargs.get("disabled", False):
            return False

        clicked = self.apply_clicked
        self.apply_clicked = False
        return clicked



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



def test_c4_acceptance_is_opt_in(fake_st):
    manifest, holdout = make_evidence()

    render_external_accuracy_profile_section(
        manifest, holdout, ["LSTM", "GRU"]
    )

    # Original B2 table only.
    assert len(fake_st.tables) == 1
    assert not fake_st.errors


def test_c4_apply_displays_acceptance_table(fake_st):
    manifest, holdout = make_evidence()
    fake_st.apply_clicked = True

    render_external_accuracy_profile_section(
        manifest, holdout, ["LSTM", "GRU"]
    )

    assert len(fake_st.tables) == 2

    acceptance = fake_st.tables[1][0]

    assert acceptance["model"].tolist() == ["LSTM", "GRU"]
    assert acceptance["status"].tolist() == ["pass", "pass"]
    assert set(acceptance["threshold_microVolt"]) == {0.5}
    assert set(acceptance["comparison_status"]) == {
        "directly_comparable"
    }


def test_c4_acceptance_persists_on_unchanged_rerun(fake_st):
    manifest, holdout = make_evidence()
    fake_st.apply_clicked = True

    render_external_accuracy_profile_section(
        manifest, holdout, ["LSTM", "GRU"]
    )

    # Simulate Streamlit rerun without clicking Apply again.
    fake_st.tables.clear()

    render_external_accuracy_profile_section(
        manifest, holdout, ["LSTM", "GRU"]
    )

    assert len(fake_st.tables) == 2


def test_c4_threshold_change_hides_stale_acceptance(fake_st):
    manifest, holdout = make_evidence()
    fake_st.apply_clicked = True

    render_external_accuracy_profile_section(
        manifest, holdout, ["LSTM", "GRU"]
    )

    fake_st.tables.clear()
    fake_st.threshold_choice = "0.25 µV"

    render_external_accuracy_profile_section(
        manifest, holdout, ["LSTM", "GRU"]
    )

    assert len(fake_st.tables) == 1
    assert fake_st.infos


def test_c4_model_change_hides_stale_acceptance(fake_st):
    manifest, holdout = make_evidence()
    fake_st.apply_clicked = True

    render_external_accuracy_profile_section(
        manifest, holdout, ["LSTM", "GRU"]
    )

    fake_st.tables.clear()

    render_external_accuracy_profile_section(
        manifest, holdout, ["GRU"]
    )

    assert len(fake_st.tables) == 1
    assert fake_st.infos


def test_c4_custom_threshold_changes_acceptance(fake_st):
    manifest, holdout = make_evidence()
    fake_st.threshold_choice = "Custom"
    fake_st.custom_threshold = 0.05
    fake_st.apply_clicked = True

    render_external_accuracy_profile_section(
        manifest, holdout, ["LSTM", "GRU"]
    )

    assert len(fake_st.tables) == 2

    acceptance = fake_st.tables[1][0]

    # LSTM has zero error; GRU has 0.1 µV error.
    assert acceptance["status"].tolist() == ["pass", "fail"]
    assert set(acceptance["threshold_microVolt"]) == {0.05}


def test_c4_does_not_mutate_source_evidence_on_apply(fake_st):
    manifest, holdout = make_evidence()

    original_manifest = deepcopy(manifest)
    original_holdout = holdout.copy(deep=True)

    fake_st.apply_clicked = True

    render_external_accuracy_profile_section(
        manifest, holdout, ["LSTM", "GRU"]
    )

    assert manifest == original_manifest
    pd.testing.assert_frame_equal(
        holdout, original_holdout
    )
