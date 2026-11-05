"""Both exports use the same writer; verify statistical footer boundaries."""
from io import BytesIO
import pandas as pd
from openpyxl import load_workbook
from zenerestimation.visualization.combined_excel_comparison import SHEETS, excel_bytes
from zenerestimation.visualization.saved_benchmarks_excel import saved_benchmark_excel_bytes


def _sheets():
    return {name: pd.DataFrame({'Quarter': ['2024-03-01', '2024-06-01', '2024-09-01'],
                                'LSTM': [-0.5, None, 0.25], 'GRU': [1.0, None, None]})
            for name in SHEETS}


def test_both_workbooks_append_live_error_statistics_only_to_error_sheets():
    for writer in (excel_bytes, saved_benchmark_excel_bytes):
        wb = load_workbook(BytesIO(writer(_sheets())))
        for name in (SHEETS[2], SHEETS[4]):
            ws = wb[name]
            assert ws['A6'].value == 'Count'
            assert ws['B6'].value == '=COUNT(B2:B4)'
            assert ws['B7'].value == '=IF(COUNT(B2:B4)>0,AVERAGE(B2:B4),"")'
            assert ws['B8'].value == '=IF(COUNT(B2:B4)>1,STDEV(B2:B4),"")'
            assert ws['C8'].value == '=IF(COUNT(C2:C4)>1,STDEV(C2:C4),"")'
            assert ws.auto_filter.ref == 'A1:C4'
        assert wb[SHEETS[5]]['A6'].value is None
        assert wb[SHEETS[1]]['A6'].value is None
