
"""M17-B2-G6-B-G5: dashboard runtime continuity."""

from contextlib import nullcontext
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

from zenerestimation.visualization import (
    optimized_neural_comparison as module,
)


class FakeStreamlit:
    def __init__(self):
        self.events = []
        self.session_state = {}

    def selectbox(self, label, options, **kwargs):
        self.events.append("run_selection")
        return options[0]

    def multiselect(self, label, options, **kwargs):
        self.events.append("model_selection")
        return ["LSTM"]

    def caption(self, *args, **kwargs):
        pass

    def subheader(self, *args, **kwargs):
        pass

    def dataframe(self, *args, **kwargs):
        self.events.append("dataframe")

    def plotly_chart(self, figure, **kwargs):
        self.events.append(kwargs.get("key", "chart"))

    def expander(self, *args, **kwargs):
        return nullcontext()

    def warning(self, message):
        self.events.append("warning")
        self.events.append(str(message))

    def info(self, message):
        self.events.append("info")

    def error(self, message):
        self.events.append("error")

    def download_button(self, *args, **kwargs):
        self.events.append("excel_export")


@pytest.fixture
def dashboard_fixture(monkeypatch, tmp_path):
    fake = FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake)

    run_dir = (
        tmp_path
        / "results"
        / "732B-5610410"
        / "optimized_neural_forecasts_16_4"
        / "run_test"
    )
    run_dir.mkdir(parents=True)

    manifest = {
        "selected": {
            model: {"internal_rmse": 0.2}
            for model in module.MODELS
        },
        "external_holdout_metrics": {
            model: {
                "rmse": 0.3,
                "mae": 0.2,
                "first_step_absolute_error": 0.1,
            }
            for model in module.MODELS
        },
    }

    holdout = pd.DataFrame({"placeholder": [1]})
    future = pd.DataFrame({"placeholder": [1]})

    monkeypatch.setattr(
        module,
        "load_optimized_neural_forecasts",
        lambda root, run: (manifest, holdout, future),
    )

    monkeypatch.setattr(
        module,
        "load_historical_fitted",
        lambda *args: None,
    )

    monkeypatch.setattr(
        module,
        "make_holdout_figure",
        lambda *args: object(),
    )

    monkeypatch.setattr(
        module,
        "make_operational_figure",
        lambda *args: object(),
    )

    # Force invalid external accuracy evidence while leaving
    # the existing dashboard inputs available.
    def invalid_profiles(*args):
        raise ValueError("Synthetic invalid external profile")

    monkeypatch.setattr(
        module,
        "build_external_holdout_accuracy_profiles",
        invalid_profiles,
    )

    from zenerestimation.visualization import (
        optimized_neural_numerical_explorer as explorer,
    )

    monkeypatch.setattr(
        explorer,
        "render_optimized_numerical_explorer",
        lambda *args: fake.events.append("numerical_explorer"),
    )

    from zenerestimation.visualization import (
        cross_battery_dashboard_sources as sources,
    )

    monkeypatch.setattr(
        sources,
        "load_410_operational_forecasts",
        lambda *args: pd.DataFrame(
            {
                "date": module.FUTURE,
                **{
                    model: [30.0] * len(module.FUTURE)
                    for model in module.MODELS
                },
            }
        ),
    )

    from zenerestimation.visualization import (
        combined_excel_comparison as excel,
    )

    monkeypatch.setattr(
        excel,
        "build_comparison_sheets",
        lambda *args: {"overview": pd.DataFrame()},
    )

    monkeypatch.setattr(
        excel,
        "excel_bytes",
        lambda *args, **kwargs: b"synthetic-workbook",
    )

    return fake, tmp_path


def test_g6b_invalid_profile_does_not_stop_dashboard(
    dashboard_fixture,
):
    fake, root = dashboard_fixture

    module.render_optimized_neural_comparison(root)

    assert any(
        "Synthetic invalid external profile" in str(event)
        for event in fake.events
    )

    assert "optimized_neural_holdout_values_chart" in fake.events
    assert "optimized_neural_operational_comparison" in fake.events
    assert "numerical_explorer" in fake.events
    assert "excel_export" in fake.events


def test_g6b_dashboard_reporting_order(dashboard_fixture):
    fake, root = dashboard_fixture

    module.render_optimized_neural_comparison(root)

    events = fake.events

    selection = events.index("model_selection")
    warning = events.index("warning")
    holdout = events.index(
        "optimized_neural_holdout_values_chart"
    )
    operational = events.index(
        "optimized_neural_operational_comparison"
    )
    explorer = events.index("numerical_explorer")
    export = events.index("excel_export")

    assert (
        selection
        < warning
        < holdout
        < operational
        < explorer
        < export
    )


def test_g6b_invalid_profile_does_not_raise_dashboard_error(
    dashboard_fixture,
):
    fake, root = dashboard_fixture

    module.render_optimized_neural_comparison(root)

    assert "error" not in fake.events
    assert "excel_export" in fake.events
