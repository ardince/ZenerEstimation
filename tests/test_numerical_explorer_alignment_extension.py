import importlib.util
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1] / 'zenerestimation' / 'visualization'

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_four_battery_choices():
    module = load('optimized_neural_numerical_explorer')
    assert len(module.BATTERY_OPTIONS) == 4
    assert module.resolve_batteries(module.BATTERY_OPTIONS[3], ['732B-5610110']) == ('732B-5610110',)
    assert module.resolve_batteries(module.BATTERY_OPTIONS[2], []) == ('732B-5610410', '732B-5610110')
    with pytest.raises(ValueError):
        module.resolve_batteries('invalid', [])


def test_benchmark_date_orientation_preserves_signed_errors():
    module = load('cross_battery_dashboard_numerical_ui')
    raw = pd.DataFrame({'date': ['2024-03-01', '2024-06-01'], 'model': ['LSTM', 'LSTM'],
                        'error_microvolt': [-0.5, 0.25], 'origin': ['source', 'source']})
    wide = module.orient_numerical_table(raw, 'Dates as columns')
    assert wide.loc[0, '2024-03-01'] == -0.5
    assert wide.loc[0, '2024-06-01'] == 0.25
    assert raw.columns.tolist() == ['date', 'model', 'error_microvolt', 'origin']

    rows = module.orient_numerical_table(raw, 'Dates as rows')

    assert rows.columns.tolist() == ['Quarter', 'LSTM']
    assert rows['Quarter'].tolist() == [
        '2024-03-01',
        '2024-06-01',
    ]
    assert rows['LSTM'].tolist() == [-0.5, 0.25]

    # The original source DataFrame must remain unchanged.
    assert raw.columns.tolist() == [
        'date', 'model', 'error_microvolt', 'origin'
    ]


def test_benchmark_orientation_rejects_duplicate_model_date():
    module = load('cross_battery_dashboard_numerical_ui')
    raw = pd.DataFrame({'date': ['2024-03-01'] * 2, 'model': ['LSTM'] * 2,
                        'error_microvolt': [0.1, 0.2]})
    with pytest.raises(ValueError, match='Duplicate'):
        module.orient_numerical_table(raw, 'Dates as columns')


def test_signed_error_options_exposed():
    module = load('cross_battery_dashboard_numerical_ui')
    assert 'holdout_errors' in module.KINDS.values()
    assert 'historical_errors' in module.KINDS.values()
