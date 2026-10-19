"""Milestone 16.1 Section C regression tests for ForecastResult."""
import json

import pandas as pd
import pytest

from zenerestimation.forecasting.result import ForecastResult


def make_result(index=None, fitted=None, metadata=None):
    dates = pd.to_datetime(['2024-03-01', '2024-06-01', '2024-09-01'])
    values = pd.Series([30.1, 30.5, 30.9], index=index)
    return ForecastResult(
        model='Example', forecast=values, horizon=3,
        dates=pd.DatetimeIndex(dates), fitted=fitted,
        metadata={} if metadata is None else metadata,
    )


def test_dataframe_preserves_positional_values_with_nondefault_index():
    result = make_result(index=[10, 20, 30])
    frame = result.to_dataframe()
    assert list(frame.columns) == ['ds', 'forecast']
    assert frame.shape == (3, 2)
    assert frame['ds'].tolist() == list(result.dates)
    assert frame['forecast'].tolist() == [30.1, 30.5, 30.9]
    assert not frame.isna().any().any()


def test_dataframe_works_with_datetime_series_index():
    result = make_result(index=pd.date_range('2000-01-01', periods=3))
    assert result.to_dataframe()['forecast'].tolist() == [30.1, 30.5, 30.9]


def test_dataframe_rejects_date_value_length_mismatch():
    result = make_result()
    result.dates = result.dates[:2]
    with pytest.raises(ValueError, match='equal lengths'):
        result.to_dataframe()


def test_to_dict_preserves_dates_values_and_metadata():
    result = make_result(index=[10, 20, 30], metadata={'source': 'test', 'fold': 2})
    payload = result.to_dict()
    assert payload == {
        'model': 'Example', 'horizon': 3,
        'dates': ['2024-03-01', '2024-06-01', '2024-09-01'],
        'forecast': [30.1, 30.5, 30.9],
        'metadata': {'source': 'test', 'fold': 2},
    }
    assert json.loads(json.dumps(payload)) == payload


def test_save_json_roundtrip(tmp_path):
    result = make_result(metadata={'label': 'µV'})
    output = tmp_path / 'forecast.json'
    result.save_json(output)
    with output.open(encoding='utf-8') as handle:
        restored = json.load(handle)
    assert restored == result.to_dict()


def test_optional_fitted_does_not_change_export_schema():
    fitted = pd.Series([28.5, 29.0])
    without_fit = make_result()
    with_fit = make_result(fitted=fitted)
    assert without_fit.to_dict() == with_fit.to_dict()
    assert without_fit.to_dataframe().equals(with_fit.to_dataframe())
    assert without_fit.summary()['has_fit'] is False
    assert with_fit.summary()['has_fit'] is True
