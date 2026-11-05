"""Focused tests for the seven-sheet workbook layout.

Run from the repository root after applying the patch.
"""
from io import BytesIO
from types import SimpleNamespace
import pandas as pd
import pytest
from openpyxl import load_workbook

from zenerestimation.visualization import combined_excel_comparison as combined
from zenerestimation.visualization import saved_benchmarks_excel as saved


def test_seven_sheet_serialization_and_error_statistics():
    frames = {name: pd.DataFrame({'Property': ['Battery'], 'Value': ['410']})
              if name == combined.SHEETS[0]
              else pd.DataFrame({'Quarter': ['2024-03-01', '2024-06-01'], 'LSTM': [-0.5, 0.25]})
              for name in combined.SHEETS[1:]}
    frames = {combined.SHEETS[0]: pd.DataFrame({'Property': ['Battery'], 'Value': ['410']}), **frames}
    frames[combined.SHEETS[6]] = pd.DataFrame({'Model': ['LSTM'], 'RMSE (µV)': [0.123456]})
    blob = combined.excel_bytes(frames, display_precision=4)
    wb = load_workbook(BytesIO(blob))
    assert wb.sheetnames == list(combined.SHEETS)
    assert wb[combined.SHEETS[0]].max_column == 2
    assert wb[combined.SHEETS[6]]['B2'].value == 0.123456
    assert wb[combined.SHEETS[6]]['B2'].number_format == '0.0000'
    assert wb[combined.SHEETS[2]]['B6'].value.startswith('=IF(COUNT(')


def test_saved_builder_moves_metrics_out_of_overview(monkeypatch, tmp_path):
    def fake_result(comparison, root, *, battery_id, kind, models):
        if kind == 'metrics':
            return pd.DataFrame({'model': ['LSTM'], 'rmse': [0.5], 'mae': [0.4]}), 'Saved metrics'
        if kind == 'operational':
            return pd.DataFrame({'Quarter': ['2025-09-01'], 'LSTM': [3.0]}), 'Unvalidated'
        return pd.DataFrame({'Quarter': ['2023-03-01'], 'LSTM': [1.1]}), 'Evidence'
    monkeypatch.setattr(saved, 'numerical_result', fake_result)
    monkeypatch.setattr(saved, 'orient_numerical_table', lambda frame, _: frame.copy())
    sheets = saved.build_saved_benchmark_sheets(None, tmp_path, battery_id=saved.BATTERY_410, models=['LSTM'])
    assert list(sheets) == list(combined.SAVED_SHEETS)
    assert sheets[combined.SHEETS[0]].columns.tolist() == ['Property', 'Value']
    assert not sheets[combined.SHEETS[0]]['Property'].str.contains('LSTM ·').any()
    assert sheets[combined.SAVED_SHEETS[6]].loc[0, 'RMSE (µV)'] == 0.5
    assert pd.isna(sheets[combined.SAVED_SHEETS[6]].loc[0, 'MAPE (%)'])


def test_missing_required_saved_metrics_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(saved, 'numerical_result', lambda *args, **kwargs:
                        (pd.DataFrame({'model': ['LSTM'], 'rmse': [0.5]}), 'Evidence'))
    monkeypatch.setattr(saved, 'orient_numerical_table', lambda frame, _: frame.copy())
    with pytest.raises(ValueError, match='model, rmse and mae'):
        saved.build_saved_benchmark_sheets(None, tmp_path, battery_id=saved.BATTERY_410, models=['LSTM'])
