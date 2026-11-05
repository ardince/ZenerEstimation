import pandas as pd
import pytest
from zenerestimation.visualization.optimized_neural_comparison import (
    holdout_date_columns_table, operational_date_columns_table,
    HOLDOUT, FUTURE, MODELS,
)


def test_holdout_dates_are_columns():
    rows = [{'date': d, 'model': m, 'actual_microvolt': 10.0,
             'predicted_microvolt': float(i)}
            for m in MODELS for i, d in enumerate(HOLDOUT)]
    result = holdout_date_columns_table(pd.DataFrame(rows), MODELS)
    assert result.columns.tolist() == ['Model', *HOLDOUT.strftime('%Y-%m-%d')]

    expected_rows = (
        ['Measured']
        + list(MODELS)
        + [
            f'{model} · prediction − measured'
            for model in MODELS
        ]
    )

    assert result['Model'].tolist() == expected_rows

    for model in MODELS:
        prediction = result.loc[
            result['Model'] == model
        ].iloc[0, 1:].to_numpy(dtype=float)

        error = result.loc[
            result['Model'] == f'{model} · prediction − measured'
        ].iloc[0, 1:].to_numpy(dtype=float)

        import numpy as np

        np.testing.assert_allclose(
            error,
            prediction - 10.0,
            rtol=0,
            atol=1e-12,
        )

    assert result.iloc[0, 1] == 10.0


def test_operational_dates_are_columns_and_saved_rows():
    rows = [{'date': d, 'model': m, 'predicted_microvolt': float(i)}
            for m in MODELS for i, d in enumerate(FUTURE)]
    saved = pd.DataFrame({'date': FUTURE, **{m: [10.0] * 6 for m in MODELS}})
    result = operational_date_columns_table(pd.DataFrame(rows), MODELS, saved)
    assert result.columns.tolist() == ['Model / source', *FUTURE.strftime('%Y-%m-%d')]
    assert len(result) == 8
    assert result.iloc[1, 1] == 10.0


def test_rejects_misaligned_saved_dates():
    rows = [{'date': d, 'model': m, 'predicted_microvolt': float(i)}
            for m in MODELS for i, d in enumerate(FUTURE)]
    saved = pd.DataFrame({'date': FUTURE + pd.DateOffset(months=3),
                          **{m: [10.0] * 6 for m in MODELS}})
    with pytest.raises(ValueError, match='dates'):
        operational_date_columns_table(pd.DataFrame(rows), MODELS, saved)
