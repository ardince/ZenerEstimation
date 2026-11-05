import numpy as np
import pandas as pd
import pytest
from zenerestimation.visualization.optimized_neural_numerical_explorer import OPTIONS, numerical_result, orient_table

MODELS = ('LSTM', 'GRU', 'LinearTrendLSTM', 'KalmanLSTM')
DATES = pd.date_range('2024-03-01', periods=6, freq='QS-MAR')

def sample():
    holdout = pd.DataFrame([{'date': date, 'model': model, 'actual_microvolt': 10.,
                             'predicted_microvolt': 11. + i,
                             'error_pred_minus_actual': 1. + i}
                            for model in MODELS for i, date in enumerate(DATES)])
    future = holdout[['date','model','predicted_microvolt']].copy()
    return holdout, future

def test_holdout_dynamic_model_columns():
    holdout, future = sample()
    table, status = numerical_result('.', {}, holdout, future, None, None,
                                      kind=OPTIONS[3], models=['GRU', 'LSTM'])
    assert list(table.columns) == ['Quarter', 'Measured (µV)', 'GRU', 'LSTM']
    assert len(table) == 6
    assert table['Measured (µV)'].eq(10).all()

def test_holdout_errors_and_orientation():
    holdout, future = sample()
    table, _ = numerical_result('.', {}, holdout, future, None, None,
                                kind=OPTIONS[4], models=['LSTM'])
    assert table['LSTM'].iloc[0] == 1.
    transposed = orient_table(table, 'Dates as columns')
    assert transposed.columns[1] == '2024-03-01'
    assert transposed['Series'].tolist() == ['LSTM']

def test_operational_difference_is_not_accuracy():
    holdout, future = sample()
    saved = pd.DataFrame({'date': DATES, 'LSTM': [10.] * 6})
    table, status = numerical_result('.', {}, holdout, future, None, saved,
                                      kind=OPTIONS[6], models=['LSTM'])
    assert table['LSTM'].iloc[0] == 1.
    assert 'NOT errors' in status

def test_historical_missing_values():
    holdout, future = sample()
    historical = pd.DataFrame([{'date':date,'model':'LSTM',
                                'measured_microvolt': 10. if i else np.nan,
                                'fitted_microvolt': 11. if i else np.nan,
                                'error_fitted_minus_measured': 1. if i else np.nan}
                               for i,date in enumerate(DATES)])
    table, _ = numerical_result('.', {}, holdout, future, historical, None,
                                kind=OPTIONS[1], models=['LSTM'])
    assert pd.isna(table['LSTM'].iloc[0])
    assert pd.isna(table['Measured (µV)'].iloc[0])

def test_invalid_selection_rejected():
    holdout, future = sample()
    with pytest.raises(ValueError, match='Unknown'):
        numerical_result('.', {}, holdout, future, None, None,
                         kind=OPTIONS[3], models=['BAD'])
