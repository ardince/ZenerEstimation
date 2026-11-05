import pandas as pd
import pytest
from zenerestimation.visualization.cross_battery_dashboard_numerical_ui import orient_numerical_table


def test_date_rows_compare_models_without_repeated_ids():
    t = pd.DataFrame({'battery_id':['410']*4,'date':['2024-03-01','2024-06-01']*2,
                      'model':['LSTM']*2+['GRU']*2,'predicted_microvolt':[1.,2.,3.,4.],
                      'measured_microvolt':[1.5,2.5,1.5,2.5], 'origin':['diagnostic']*4})
    result = orient_numerical_table(t,'Dates as rows')
    assert list(result) == ['Quarter','Measured (µV)','LSTM','GRU']
    assert result['GRU'].tolist() == [3.,4.]
    assert 'battery_id' not in result
    assert 'model' not in result
    transposed = orient_numerical_table(t,'Dates as columns')
    assert '2024-03-01' in transposed
    assert 'LSTM' in transposed['Model / value'].tolist()


def test_signed_error_view():
    t = pd.DataFrame({'date':['2024-03-01']*2,'model':['LSTM','GRU'],
                      'error_microvolt':[-0.2,0.3]})
    result = orient_numerical_table(t,'Dates as rows')
    assert result.loc[0,'LSTM'] == pytest.approx(-0.2)
    assert result.loc[0,'GRU'] == pytest.approx(0.3)


def test_reject_conflicting_measurements():
    t = pd.DataFrame({'date':['2024-03-01']*2,'model':['LSTM','GRU'],
                      'measured_microvolt':[1.,2.], 'predicted_microvolt':[1.,2.]})
    with pytest.raises(ValueError,match='Inconsistent measured'):
        orient_numerical_table(t,'Dates as rows')


def test_operational_and_observed():
    operational = pd.DataFrame({'date':['2025-09-01'],'LSTM':[3.], 'GRU':[4.]})
    assert list(orient_numerical_table(operational,'Dates as rows')) == ['Quarter','LSTM','GRU']
    assert '2025-09-01' in orient_numerical_table(operational,'Dates as columns')
    observed = pd.DataFrame({'date':['2023-12-01'],'measured_microvolt':[5.]})
    assert list(orient_numerical_table(observed,'Dates as rows')) == ['Quarter','measured_microvolt']
