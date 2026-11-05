import pandas as pd
import pytest
from zenerestimation.visualization.optimized_neural_comparison import (
    HOLDOUT, MODELS, holdout_date_columns_table,
)


def sample():
    return pd.DataFrame([
        {'date': date, 'model': model, 'actual_microvolt': 30.0 + i,
         'predicted_microvolt': 30.0 + i + offset,
         'error_pred_minus_actual': offset}
        for model, offset in zip(MODELS, (0.1, -0.2, 0.3, -0.4))
        for i, date in enumerate(HOLDOUT)
    ])


def test_predictions_and_errors_share_date_columns():
    table = holdout_date_columns_table(sample(), MODELS)
    assert table.columns.tolist() == ['Model', *HOLDOUT.strftime('%Y-%m-%d')]
    assert len(table) == 1 + 2 * len(MODELS)
    assert table['Model'].tolist()[:5] == ['Measured', *MODELS]
    assert table.loc[table['Model'] == 'GRU · prediction − measured', '2024-03-01'].iloc[0] == pytest.approx(-0.2)
    assert table.loc[table['Model'] == 'LinearTrendLSTM · prediction − measured', '2024-06-01'].iloc[0] == pytest.approx(0.3)


def test_rejects_inconsistent_saved_error():
    data = sample()
    data.loc[0, 'error_pred_minus_actual'] = 9.0
    with pytest.raises(ValueError, match='signed'):
        holdout_date_columns_table(data, MODELS)


def test_selected_models_only():
    table = holdout_date_columns_table(sample(), ['LSTM'])
    assert table['Model'].tolist() == ['Measured', 'LSTM', 'LSTM · prediction − measured']
