from types import SimpleNamespace
import pandas as pd
import pytest

from zenerestimation.visualization.combined_excel_comparison import (
    _combine, _dated, SHEETS, excel_bytes)


def test_exact_date_outer_join_and_missing_values():
    saved = pd.DataFrame({'Quarter': ['2024-03-01', '2024-06-01'],
                          'Measured (µV)': [1.0, 2.0], 'LSTM': [1.2, 2.3]})
    optimized = pd.DataFrame({'Quarter': ['2024-06-01', '2024-09-01'],
                              'Measured (µV)': [2.0, 3.0], 'LSTM': [2.1, 3.2]})
    result = _combine(saved, optimized, ('LSTM',), measured=True)
    assert result['Quarter'].tolist() == ['2024-03-01', '2024-06-01', '2024-09-01']
    assert result.loc[1, 'LSTM · saved'] == 2.3
    assert result.loc[1, 'LSTM · optimized'] == 2.1
    assert pd.isna(result.loc[0, 'LSTM · optimized'])
    assert pd.isna(result.loc[2, 'LSTM · saved'])


def test_disagreeing_measured_values_are_rejected():
    a = pd.DataFrame({'Quarter': ['2024-03-01'], 'Measured (µV)': [1.0], 'LSTM': [1.1]})
    b = pd.DataFrame({'Quarter': ['2024-03-01'], 'Measured (µV)': [2.0], 'LSTM': [2.1]})
    with pytest.raises(ValueError, match='disagree'):
        _combine(a, b, ('LSTM',), measured=True)


def test_operational_delta_is_prediction_difference():
    a = pd.DataFrame({'Quarter': ['2025-09-01'], 'GRU': [5.0]})
    b = pd.DataFrame({'Quarter': ['2025-09-01'], 'GRU': [5.75]})
    result = _combine(a, b, ('GRU',), operational=True)
    assert result['GRU · optimized − saved'].iloc[0] == .75


def test_duplicate_quarters_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        _dated(pd.DataFrame({'Quarter': ['2024-03-01'] * 2}), 'test')


def test_six_sheet_excel_file_has_expected_sheets(tmp_path):
    pytest.importorskip('openpyxl')
    import zipfile
    import io
    sheets = {name: pd.DataFrame({'Quarter': ['2024-03-01'], 'LSTM': [1.25]})
              for name in SHEETS}
    payload = excel_bytes(sheets)
    assert payload[:2] == b'PK'
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        xml = z.read('xl/workbook.xml').decode('utf-8')
    assert all(name in xml for name in SHEETS)
