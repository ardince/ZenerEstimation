"""Regression checks for measured observations embedded in historical fits."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _load():
    path = Path(__file__).resolve().parents[1] / 'zenerestimation' / 'visualization' / 'cross_battery_dashboard_numerical_ui.py'
    spec = importlib.util.spec_from_file_location('numerical_ui_measured_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_historical_fits_include_only_real_measurements(monkeypatch, tmp_path):
    ui = _load()
    from zenerestimation.visualization import cross_battery_dashboard_numerical as source
    raw = pd.DataFrame({
        'battery_id': [ui.BATTERY_410] * 4,
        'date': ['2023-03-01', '2023-03-01', '2023-06-01', '2023-06-01'],
        'model': ['LSTM', 'GRU', 'LSTM', 'GRU'],
        'fitted_microvolt': [1.1, 1.2, 2.1, 2.2],
        'origin': ['regenerated_diagnostic_not_frozen'] * 4,
    })
    comparison = SimpleNamespace(historical_fits=raw)
    monkeypatch.setattr(ui, 'load_410_observed_history', lambda root: pd.DataFrame({
        'date': pd.to_datetime(['2023-03-01']), 'measured_microvolt': [1.0]
    }))
    result, status = ui.numerical_result(comparison, tmp_path, battery_id=ui.BATTERY_410,
                                         kind='historical_fits', models=['LSTM', 'GRU'])
    wide = ui.orient_numerical_table(result, 'Dates as rows')
    assert wide.columns.tolist() == ['Quarter', 'Measured (µV)', 'LSTM', 'GRU']
    assert wide['Measured (µV)'].iloc[0] == 1.0
    assert pd.isna(wide['Measured (µV)'].iloc[1])
    assert wide['LSTM'].tolist() == [1.1, 2.1]
    assert 'physical observations' in status
    assert 'Measured historical observations' not in ui.KINDS
    assert raw.columns.tolist() == ['battery_id', 'date', 'model', 'fitted_microvolt', 'origin']
    transposed = ui.orient_numerical_table(result, 'Dates as columns')
    assert 'Measured (µV)' in transposed['Model / value'].tolist()
