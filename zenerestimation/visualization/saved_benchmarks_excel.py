"""Read-only seven-sheet Excel export of the selected battery's saved benchmarks.

No forecast reconstruction, interpolation, training, or writes to source artifacts.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from zenerestimation.visualization.cross_battery_dashboard_numerical_ui import (
    BATTERY_110, BATTERY_410, MODELS, numerical_result, orient_numerical_table,
)
from zenerestimation.visualization.combined_excel_comparison import SAVED_SHEETS as SHEETS, excel_bytes


def build_saved_benchmark_sheets(comparison, repository_root: str | Path, *,
                                 battery_id: str, models=None) -> dict[str, pd.DataFrame]:
    """Create detached source-precision tables for one battery and selected models."""
    if battery_id not in (BATTERY_410, BATTERY_110):
        raise ValueError('Unknown battery')
    selected = tuple(MODELS if models is None else models)
    if not selected or len(set(selected)) != len(selected) or any(m not in MODELS for m in selected):
        raise ValueError('Select one or more unique supported models')

    def get(kind):
        raw, status = numerical_result(comparison, repository_root, battery_id=battery_id,
                                       kind=kind, models=selected)
        wide = orient_numerical_table(raw, 'Dates as rows')
        if wide.empty:
            # Preserve a meaningful schema without inventing values or dates.
            wide = pd.DataFrame(columns=['Quarter', *selected])
        if 'Quarter' in wide and not wide.empty:
            dates = pd.to_datetime(wide['Quarter'], errors='raise')
            if dates.isna().any() or dates.duplicated().any():
                raise ValueError(f'{kind}: invalid or duplicate quarters')
            wide = wide.sort_values('Quarter').reset_index(drop=True)
        return wide.copy(deep=True), status

    metrics, metrics_status = get('metrics')
    fits, fits_status = get('historical_fits')
    historical_errors, historical_errors_status = get('historical_errors')
    holdout, holdout_status = get('holdout')
    holdout_errors, holdout_errors_status = get('holdout_errors')
    operational, operational_status = get('operational')

    def primary_measure(frame, suffix):
        """Keep one model column per selected model, as in the optimized export."""
        if frame.empty or 'Quarter' not in frame:
            return pd.DataFrame(columns=['Quarter', *selected])
        columns = ['Quarter'] + (['Measured (µV)'] if 'Measured (µV)' in frame else [])
        output = frame[columns].copy(deep=True)
        for model in selected:
            name = model if model in frame else f'{model} · {suffix}'
            if name in frame:
                output[model] = frame[name]
        return output

    fits = primary_measure(fits, 'fitted')
    historical_errors = primary_measure(historical_errors, 'fitted − measured')
    holdout = primary_measure(holdout, 'prediction')
    holdout_errors = primary_measure(holdout_errors, 'prediction − measured')

    # The Overview sheet includes available benchmark metrics without claiming
    # cross-battery comparability or inventing missing source values.
    overview_rows = [
        ('Battery', battery_id),
        ('Selected models', ', '.join(selected)),
        ('Scope', 'Saved benchmarks only; no optimized neural results'),
        ('Benchmark metrics provenance', metrics_status),
        ('Historical fitted values', fits_status),
        ('Historical signed errors', historical_errors_status),
        ('External holdout', holdout_status),
        ('Holdout signed errors', holdout_errors_status),
        ('Operational forecasts', operational_status),
        ('Historical evidence', 'Regenerated diagnostics; in-sample, not independent validation'),
        ('Holdout evidence', 'Provenance differs by battery; consult status above'),
        ('Operational evidence', 'Unvalidated extrapolations, not measured outcomes'),
        ('Signed errors', 'Prediction or fitted value minus physically measured value (µV)'),
        ('Missing measurements', 'Blank; never interpolated or replaced by zero'),
        ('Precision', 'Source numeric values retained; Excel displays up to 15 significant digits'),
    ]
    required_metrics = ('model', 'rmse', 'mae')
    if not metrics.empty and not set(required_metrics).issubset(metrics.columns):
        raise ValueError('Saved benchmark metrics require model, rmse and mae columns')
    metric_table = metrics.reindex(columns=[
        'model', 'rmse', 'mae', 'mape', 'rmse_rank'
    ]).copy(deep=True)
    metric_table.columns = ['Model', 'RMSE (µV)', 'MAE (µV)', 'MAPE (%)', 'RMSE rank']
    overview_table = pd.DataFrame(overview_rows, columns=['Property', 'Value'])
    return {
        SHEETS[0]: overview_table,
        SHEETS[1]: fits,
        SHEETS[2]: historical_errors,
        SHEETS[3]: holdout,
        SHEETS[4]: holdout_errors,
        SHEETS[5]: operational,
        SHEETS[6]: metric_table,
    }


def saved_benchmark_excel_bytes(sheets: dict[str, pd.DataFrame], *, display_precision="Full") -> bytes:
    """Use the same formatting/serialization as the optimized seven-sheet workbook."""
    return excel_bytes(sheets, display_precision=display_precision)
