"""Displayed and full-precision CSV exports use separate detached data."""
import pandas as pd
from zenerestimation.visualization.cross_battery_dashboard_numerical_ui import csv_bytes, display_table


def test_csv_display_precision_is_independent_of_source():
    original = pd.DataFrame({'Quarter': ['2024-03-01'], 'LSTM': [1.2345678912345]})
    displayed = display_table(original, decimals=2)
    assert displayed.loc[0, 'LSTM'] == 1.23
    assert b'1.23' in csv_bytes(displayed)
    assert b'1.2345678912345' in csv_bytes(original)
    assert original.loc[0, 'LSTM'] == 1.2345678912345


def test_csv_full_precision_display_is_unchanged():
    original = pd.DataFrame({'Quarter': ['2024-03-01'], 'GRU': [-0.123456789]})
    assert csv_bytes(display_table(original, decimals=None)) == csv_bytes(original)
