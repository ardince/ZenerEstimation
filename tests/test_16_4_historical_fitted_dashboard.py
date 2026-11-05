import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

MODULE = Path('zenerestimation/visualization/optimized_neural_comparison.py')
spec = importlib.util.spec_from_file_location('neural_comparison_fitted', MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture(tmp_path):
    folder = tmp_path / 'results' / '732B-5610410' / 'optimized_neural_forecasts_16_4' / 'sample'
    folder.mkdir(parents=True)
    dates = pd.date_range('1998-03-01', '2023-12-01', freq='QS-MAR')
    rows = []
    for model in mod.MODELS:
        warmup = 12 if model == 'KalmanLSTM' else 4
        for i, date in enumerate(dates):
            observed = i != 20
            fitted = float(i + 0.1) if i >= warmup else np.nan
            rows.append(dict(date=date.strftime('%Y-%m-%d'), model=model,
                             measured_microvolt=float(i) if observed else np.nan,
                             is_observed=observed, was_interpolated=not observed,
                             is_warmup=i < warmup, fitted_microvolt=fitted,
                             error_fitted_minus_measured=(0.1 if observed and i >= warmup else np.nan),
                             fit_scope='development_in_sample'))
    pd.DataFrame(rows).to_csv(folder / 'historical_fitted_values.csv', index=False)
    manifest = dict(development_cutoff='2023-12-01',
                    files=dict(holdout='holdout_predictions.csv', operational='operational_predictions.csv', historical_fitted='historical_fitted_values.csv'),
                    selected={model:dict(params=dict(window=12 if model=='KalmanLSTM' else 4)) for model in mod.MODELS},
                    historical_fitted=dict(status='development_in_sample_diagnostic_not_external_validation',
                                           cutoff='2023-12-01', models=list(mod.MODELS),
                                           rows_per_model=104, warmup={model:12 if model=='KalmanLSTM' else 4 for model in mod.MODELS}))
    return folder, manifest


def test_historical_load_and_date_columns(tmp_path):
    _, manifest = fixture(tmp_path)
    frame = mod.load_historical_fitted(tmp_path, 'sample', manifest)
    assert frame.shape == (416, 9)
    table = mod.historical_fitted_date_columns_table(frame, mod.MODELS)
    assert table.shape == (9, 105)
    assert table.columns[1] == '1998-03-01'
    assert table.columns[-1] == '2023-12-01'
    assert pd.isna(table.loc[0, '2003-03-01'])
    assert table.loc[2, '2023-12-01'] == pytest.approx(0.1)


def test_historical_bad_signed_error_rejected(tmp_path):
    folder, manifest = fixture(tmp_path)
    path = folder / 'historical_fitted_values.csv'
    frame = pd.read_csv(path)
    frame.loc[frame.error_fitted_minus_measured.notna(), 'error_fitted_minus_measured'] = 999.0
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='signed errors mismatch'):
        mod.load_historical_fitted(tmp_path, 'sample', manifest)


def test_historical_bad_date_rejected(tmp_path):
    folder, manifest = fixture(tmp_path)
    path = folder / 'historical_fitted_values.csv'
    frame = pd.read_csv(path)
    frame.loc[0, 'date'] = '1998-04-01'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Historical dates mismatch'):
        mod.load_historical_fitted(tmp_path, 'sample', manifest)


def test_legacy_manifest_does_not_require_historical(tmp_path):
    _, manifest = fixture(tmp_path)
    manifest.pop('historical_fitted')
    manifest['files'].pop('historical_fitted')
    assert mod.load_historical_fitted(tmp_path, 'sample', manifest) is None
