
"""M17-B2-G6-B-G4: dashboard wiring and reporting continuity."""

import inspect
from pathlib import Path

from zenerestimation.visualization import (
    optimized_neural_comparison as module,
)


def test_g6b_helper_is_called_by_main_dashboard():
    source = inspect.getsource(
        module.render_optimized_neural_comparison
    )

    assert (
        "render_external_accuracy_profile_section("
        in source
    )


def test_g6b_helper_follows_model_selection():
    source = inspect.getsource(
        module.render_optimized_neural_comparison
    )

    selection = source.index(
        "models = st.multiselect("
    )
    helper = source.index(
        "render_external_accuracy_profile_section("
    )
    historical = source.index(
        "load_historical_fitted("
    )

    assert selection < helper < historical


def test_g6b_existing_holdout_and_operational_charts_remain():
    source = inspect.getsource(
        module.render_optimized_neural_comparison
    )

    assert "make_holdout_figure(" in source
    assert "make_operational_figure(" in source

    assert (
        "render_optimized_numerical_explorer("
        in source
    )


def test_g6b_existing_excel_export_remains():
    source = inspect.getsource(
        module.render_optimized_neural_comparison
    )

    assert "build_comparison_sheets(" in source
    assert "excel_bytes(" in source
    assert "st.download_button(" in source


def test_g6b_dashboard_still_loads_saved_artifacts():
    source = inspect.getsource(
        module.render_optimized_neural_comparison
    )

    assert (
        "load_optimized_neural_forecasts(root, run)"
        in source
    )


def test_g6b_helper_does_not_perform_model_training():
    source = inspect.getsource(
        module.render_external_accuracy_profile_section
    )

    assert ".fit(" not in source
    assert ".predict(" not in source
    assert ".to_csv(" not in source
    assert ".to_excel(" not in source
    assert "write_text(" not in source
    assert "write_bytes(" not in source


def test_g6b_helper_uses_existing_profile_pipeline():
    source = inspect.getsource(
        module.render_external_accuracy_profile_section
    )

    assert (
        "build_external_holdout_accuracy_profiles("
        in source
    )
    assert "external_accuracy_profile_table(" in source
