from io import BytesIO
from types import SimpleNamespace
import pandas as pd
import pytest
from openpyxl import load_workbook

from zenerestimation.visualization import saved_benchmarks_excel as mod


def test_saved_workbook_six_sheets_and_missing_actuals(monkeypatch, tmp_path):
    def fake_result(comparison, root, *, battery_id, kind, models):
        if kind == 'metrics':
            return pd.DataFrame({'model': ['LSTM'], 'rmse': [0.5], 'mae': [0.4]}), 'Saved metrics'
        if kind == 'operational':
            return pd.DataFrame({'date': ['2025-09-01'], 'LSTM': [3.0]}), 'Unvalidated'
        column = {'historical_fits': 'fitted_microvolt',
                  'historical_errors': 'error_fitted_minus_measured',
                  'holdout': 'predicted_microvolt',
                  'holdout_errors': 'error_microvolt'}[kind]
        data = {'date': ['2023-03-01', '2023-06-01'], 'model': ['LSTM', 'LSTM'],
                column: [1.1, 2.2]}
        if kind in ('historical_fits', 'holdout'):
            data['measured_microvolt'] = [1.0, float('nan')]
        return pd.DataFrame(data), 'Evidence provenance'
    monkeypatch.setattr(mod, 'numerical_result', fake_result)
    sheets = mod.build_saved_benchmark_sheets(None, tmp_path, battery_id=mod.BATTERY_410, models=['LSTM'])
    assert tuple(sheets) == mod.SHEETS
    assert sheets[mod.SHEETS[1]]['Measured (µV)'].iloc[0] == 1.0
    assert pd.isna(sheets[mod.SHEETS[1]]['Measured (µV)'].iloc[1])
    assert sheets[mod.SHEETS[4]]['LSTM'].tolist() == [1.1, 2.2]
    blob = mod.saved_benchmark_excel_bytes(sheets)
    wb = load_workbook(BytesIO(blob), data_only=True)
    assert tuple(wb.sheetnames) == mod.SHEETS
    ws = wb[mod.SHEETS[1]]
    assert ws['B2'].value == 1.0
    assert ws['B3'].value is None
    assert ws.freeze_panes == 'A2'


def test_saved_workbook_rejects_unknown_battery(tmp_path):
    with pytest.raises(ValueError, match='battery'):
        mod.build_saved_benchmark_sheets(None, tmp_path, battery_id='unknown', models=['LSTM'])
