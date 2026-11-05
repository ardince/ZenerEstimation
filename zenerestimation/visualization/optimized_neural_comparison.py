"""Strict read-only validation of Milestone 16.4 optimized-neural forecast artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = ('LSTM', 'GRU', 'LinearTrendLSTM', 'KalmanLSTM')
HOLDOUT = pd.date_range('2024-03-01', periods=6, freq='QS-MAR')
FUTURE = pd.date_range('2025-09-01', periods=6, freq='QS-MAR')


def _validate_predictions(frame, expected, columns):
    if list(frame.columns) != columns or len(frame) != 24:
        raise ValueError('Prediction CSV schema or row count mismatch')
    if frame[['date', 'model']].isna().any().any():
        raise ValueError('Missing date/model')
    frame = frame.copy()
    frame['date'] = pd.to_datetime(frame['date'], errors='raise')
    if frame.duplicated(['date', 'model']).any():
        raise ValueError('Duplicate date/model predictions')
    if set(frame['model']) != set(MODELS):
        raise ValueError('Unexpected model list')
    for model in MODELS:
        if not frame.loc[frame.model.eq(model), 'date'].reset_index(drop=True).equals(pd.Series(expected)):
            raise ValueError(f'Unexpected or unordered forecast dates: {model}')
    for column in columns[2:]:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f'Nonfinite {column}')
    return frame


def load_optimized_neural_forecasts(root: str | Path, run_id: str):
    if not run_id or Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('Invalid run ID')
    folder = Path(root) / 'results' / '732B-5610410' / 'optimized_neural_forecasts_16_4' / run_id
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'),
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    expected = {'schema_version': 'optimized-neural-forecasts-16-4-v1',
                'battery': '732B-5610410', 'seed': 42,
                'development_cutoff': '2023-12-01', 'operational_cutoff': '2025-06-01',
                'horizon': 6,
                'evaluation_status': 'post_optimization_external_holdout_evaluation',
                'operational_status': 'unvalidated_future_extrapolation_no_actuals_or_accuracy_metrics',
                }
    required_files = {'holdout': 'holdout_predictions.csv', 'operational': 'operational_predictions.csv'}
    if not isinstance(manifest.get('files'), dict) or any(manifest['files'].get(k) != v for k, v in required_files.items()):
        raise ValueError('Invalid manifest forecast files')
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f'Invalid manifest field: {key}')
    if manifest.get('holdout_dates') != HOLDOUT.strftime('%Y-%m-%d').tolist() or manifest.get('operational_dates') != FUTURE.strftime('%Y-%m-%d').tolist():
        raise ValueError('Manifest dates mismatch')
    if set(manifest.get('selected', {})) != set(MODELS) or set(manifest.get('external_holdout_metrics', {})) != set(MODELS):
        raise ValueError('Manifest model set mismatch')
    holdout = _validate_predictions(pd.read_csv(folder / 'holdout_predictions.csv'), HOLDOUT,
        ['date', 'model', 'actual_microvolt', 'predicted_microvolt', 'error_pred_minus_actual'])
    future = _validate_predictions(pd.read_csv(folder / 'operational_predictions.csv'), FUTURE,
        ['date', 'model', 'predicted_microvolt'])
    if not np.allclose(holdout.predicted_microvolt - holdout.actual_microvolt,
                       holdout.error_pred_minus_actual, atol=1e-9, rtol=0):
        raise ValueError('Signed holdout errors inconsistent')
    for model in MODELS:
        h = holdout.loc[holdout.model.eq(model)]
        errors = h.error_pred_minus_actual.to_numpy(dtype=float)
        metrics = manifest['external_holdout_metrics'][model]
        checks = {'rmse': float(np.sqrt(np.mean(errors ** 2))),
                  'mae': float(np.mean(np.abs(errors))),
                  'first_step_signed_error_pred_minus_actual': float(errors[0]),
                  'first_step_absolute_error': float(abs(errors[0]))}
        for name, actual in checks.items():
            if not np.isclose(float(metrics[name]), actual, rtol=0, atol=1e-8):
                raise ValueError(f'Holdout metric mismatch: {model}/{name}')
        if not np.isfinite(float(manifest['selected'][model]['internal_rmse'])):
            raise ValueError('Nonfinite internal RMSE')
    return manifest, holdout, future


def holdout_values_table(holdout: pd.DataFrame, models) -> pd.DataFrame:
    """Return measured and predicted values with the stored signed error."""
    selected = holdout.loc[holdout['model'].isin(models),
                           ['date', 'model', 'actual_microvolt',
                            'predicted_microvolt', 'error_pred_minus_actual']].copy()
    return selected.rename(columns={
        'date': 'Quarter', 'model': 'Model',
        'actual_microvolt': 'Measured (µV)',
        'predicted_microvolt': 'New prediction (µV)',
        'error_pred_minus_actual': 'Prediction − measured (µV)',
    }).reset_index(drop=True)


def operational_values_table(future: pd.DataFrame, models, saved=None) -> pd.DataFrame:
    """Keep all future predictions; attach saved forecasts only by exact date."""
    selected = future.loc[future['model'].isin(models),
                          ['date', 'model', 'predicted_microvolt']].copy()
    if saved is not None:
        if 'date' not in saved or saved['date'].duplicated().any():
            raise ValueError('Saved operational dates missing or duplicated')
        if not pd.DatetimeIndex(pd.to_datetime(saved['date'])).equals(FUTURE):
            raise ValueError('Saved operational forecast dates do not match new run')
        saved_long = saved.melt(id_vars='date', value_vars=list(models),
                                var_name='model', value_name='saved_prediction')
        saved_long['date'] = pd.to_datetime(saved_long['date'])
        selected = selected.merge(saved_long, on=['date', 'model'],
                                  how='left', validate='one_to_one')
        if selected['saved_prediction'].isna().any():
            raise ValueError('Missing saved operational predictions')
        selected['new_minus_saved'] = selected['predicted_microvolt'] - selected['saved_prediction']
    return selected.rename(columns={
        'date': 'Quarter', 'model': 'Model',
        'predicted_microvolt': 'New prediction (µV)',
        'saved_prediction': 'Saved prediction (µV)',
        'new_minus_saved': 'New − saved (µV)',
    }).reset_index(drop=True)


def holdout_date_columns_table(holdout: pd.DataFrame, models) -> pd.DataFrame:
    """Measured, model predictions and signed errors; quarterly dates are columns.

    Error convention: prediction minus measured, in microvolts. Keep the
    existing model prediction rows and append matching error rows so the
    previous date-column table remains backwards-compatible.
    """
    selected = holdout.loc[holdout['model'].isin(models)].copy()
    columns = HOLDOUT.strftime('%Y-%m-%d').tolist()
    selected['Quarter'] = pd.to_datetime(selected['date']).dt.strftime('%Y-%m-%d')
    if selected.duplicated(['model', 'Quarter']).any():
        raise ValueError('Duplicate holdout model/quarter')
    actual_check = selected.groupby('Quarter')['actual_microvolt'].nunique(dropna=False)
    if not actual_check.reindex(columns).eq(1).all():
        raise ValueError('Inconsistent or missing holdout measurements')
    actual = selected.drop_duplicates('Quarter').set_index('Quarter')['actual_microvolt'].reindex(columns)
    predicted = selected.pivot(index='model', columns='Quarter', values='predicted_microvolt')
    predicted = predicted.reindex(index=list(models), columns=columns)
    if actual.isna().any() or predicted.isna().any().any():
        raise ValueError('Incomplete holdout date-column comparison')
    if not np.isfinite(actual.to_numpy(dtype=float)).all() or not np.isfinite(predicted.to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite holdout values')
    if 'error_pred_minus_actual' in selected:
        expected = selected['predicted_microvolt'] - selected['actual_microvolt']
        errors = pd.to_numeric(selected['error_pred_minus_actual'], errors='raise')
        if not np.allclose(expected, errors, rtol=0, atol=1e-9):
            raise ValueError('Inconsistent saved signed holdout errors')
    rows = [{'Model': 'Measured', **actual.to_dict()}]
    rows.extend({'Model': model, **predicted.loc[model].to_dict()} for model in models)
    rows.extend({'Model': f'{model} · prediction − measured',
                 **(predicted.loc[model] - actual).to_dict()} for model in models)
    return pd.DataFrame(rows, columns=['Model', *columns])


def operational_date_columns_table(future: pd.DataFrame, models, saved=None) -> pd.DataFrame:
    """Optimized and optionally saved future predictions, date columns."""
    selected = future.loc[future['model'].isin(models)].copy()
    selected['Quarter'] = pd.to_datetime(selected['date']).dt.strftime('%Y-%m-%d')
    columns = FUTURE.strftime('%Y-%m-%d')
    pivot = selected.pivot(index='model', columns='Quarter', values='predicted_microvolt')
    pivot = pivot.reindex(index=list(models), columns=columns)
    if pivot.isna().any().any():
        raise ValueError('Incomplete optimized future comparison')
    rows = []
    for model in models:
        rows.append({'Model / source': f'{model} · optimized', **pivot.loc[model].to_dict()})
        if saved is not None:
            if 'date' not in saved or model not in saved:
                raise ValueError('Missing saved operational dates or model')
            dates = pd.DatetimeIndex(pd.to_datetime(saved['date']))
            if not dates.equals(FUTURE):
                raise ValueError('Saved operational dates do not match new run')
            values = pd.to_numeric(saved[model], errors='raise').to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite saved forecast')
            rows.append({'Model / source': f'{model} · saved', **dict(zip(columns, values))})
    return pd.DataFrame(rows, columns=['Model / source', *columns])


def make_holdout_figure(holdout: pd.DataFrame, models):
    import plotly.graph_objects as go
    fig = go.Figure()
    actual = holdout.loc[holdout.model.eq(MODELS[0])]
    fig.add_scatter(x=actual.date, y=actual.actual_microvolt,
                    mode='lines+markers', name='Measured',
                    line={'color': 'black', 'width': 3})
    for model in models:
        rows = holdout.loc[holdout.model.eq(model)]
        fig.add_scatter(x=rows.date, y=rows.predicted_microvolt,
                        mode='lines+markers', name=f'{model} · optimized')
    fig.update_layout(title='Historical external holdout · predictions versus measurements',
                      xaxis_title='Quarter', yaxis_title='µV', hovermode='x unified')
    return fig


def make_operational_figure(future: pd.DataFrame, models, saved=None):
    import plotly.graph_objects as go
    fig = go.Figure()
    for model in models:
        rows = future.loc[future.model.eq(model)]
        fig.add_scatter(x=rows.date, y=rows.predicted_microvolt,
                        mode='lines+markers', name=f'{model} · optimized')
        if saved is not None:
            fig.add_scatter(x=saved.date, y=saved[model], mode='lines',
                            line={'dash': 'dash'}, name=f'{model} · saved')
    fig.update_layout(title='Operational forecasts · optimized versus saved (unvalidated)',
                      xaxis_title='Quarter', yaxis_title='µV', hovermode='x unified')
    return fig




def load_historical_fitted(root: str | Path, run_id: str, manifest: dict) -> pd.DataFrame | None:
    """Validate model-native in-sample fitted evidence; None for older runs."""
    info = manifest.get('historical_fitted')
    filename = manifest.get('files', {}).get('historical_fitted')
    if info is None and filename is None:
        return None
    if not isinstance(info, dict) or filename != 'historical_fitted_values.csv':
        raise ValueError('Historical fitted manifest incomplete')
    if (info.get('status') != 'development_in_sample_diagnostic_not_external_validation'
            or info.get('cutoff') != manifest['development_cutoff']
            or info.get('models') != list(MODELS)):
        raise ValueError('Historical fitted provenance mismatch')
    expected_dates = pd.date_range('1998-03-01', '2023-12-01', freq='QS-MAR')
    if info.get('rows_per_model') != len(expected_dates):
        raise ValueError('Historical fitted quarter count mismatch')
    folder = Path(root) / 'results' / '732B-5610410' / 'optimized_neural_forecasts_16_4' / run_id
    frame = pd.read_csv(folder / filename)
    columns = ['date', 'model', 'measured_microvolt', 'is_observed',
               'was_interpolated', 'is_warmup', 'fitted_microvolt',
               'error_fitted_minus_measured', 'fit_scope']
    if list(frame.columns) != columns or len(frame) != len(expected_dates) * len(MODELS):
        raise ValueError('Historical fitted CSV schema/count mismatch')
    frame['date'] = pd.to_datetime(frame['date'], errors='raise')
    if frame[['date', 'model']].isna().any().any() or frame.duplicated(['date', 'model']).any():
        raise ValueError('Missing or duplicate historical model/quarter')
    if set(frame['model']) != set(MODELS) or not frame['fit_scope'].eq('development_in_sample').all():
        raise ValueError('Historical fitted model/scope mismatch')
    for field in ('is_observed', 'was_interpolated', 'is_warmup'):
        if frame[field].dtype != bool:
            raise ValueError(f'Invalid historical boolean field: {field}')
    for field in ('measured_microvolt', 'fitted_microvolt', 'error_fitted_minus_measured'):
        frame[field] = pd.to_numeric(frame[field], errors='raise')
        values = frame[field].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite historical {field}')
    reference = None
    for model in MODELS:
        part = frame.loc[frame.model.eq(model)].reset_index(drop=True)
        if not pd.DatetimeIndex(part.date).equals(expected_dates):
            raise ValueError(f'Historical dates mismatch: {model}')
        window = info.get('warmup', {}).get(model)
        if type(window) is not int or window != manifest['selected'][model]['params']['window']:
            raise ValueError(f'Historical warm-up metadata mismatch: {model}')
        warmup = np.arange(len(part)) < window
        if not np.array_equal(part.is_warmup.to_numpy(), warmup):
            raise ValueError(f'Historical warm-up flags mismatch: {model}')
        observed = part.is_observed.to_numpy()
        if not np.array_equal(part.was_interpolated.to_numpy(), ~observed):
            raise ValueError(f'Historical observation flags mismatch: {model}')
        if not np.array_equal(part.measured_microvolt.notna().to_numpy(), observed):
            raise ValueError(f'Historical observed measurements mismatch: {model}')
        if not np.array_equal(part.fitted_microvolt.notna().to_numpy(), ~warmup):
            raise ValueError(f'Historical fitted warm-up nulls mismatch: {model}')
        has_error = observed & ~warmup
        if not np.array_equal(part.error_fitted_minus_measured.notna().to_numpy(), has_error):
            raise ValueError(f'Historical signed-error nulls mismatch: {model}')
        actual_errors = part.loc[has_error, 'fitted_microvolt'].to_numpy() - part.loc[has_error, 'measured_microvolt'].to_numpy()
        if not np.allclose(actual_errors, part.loc[has_error, 'error_fitted_minus_measured'], rtol=0, atol=1e-9):
            raise ValueError(f'Historical signed errors mismatch: {model}')
        obs_signature = part[['date', 'is_observed', 'measured_microvolt']].copy()
        if reference is not None and not obs_signature.equals(reference):
            raise ValueError('Historical measurements differ between models')
        reference = obs_signature
    return frame


def historical_fitted_date_columns_table(frame: pd.DataFrame, models) -> pd.DataFrame:
    """104 quarterly columns, original measurements, fitted values, and signed errors."""
    dates = pd.date_range('1998-03-01', '2023-12-01', freq='QS-MAR')
    columns = dates.strftime('%Y-%m-%d').tolist()
    first = frame.loc[frame.model.eq(MODELS[0])].set_index('date')
    actual = first['measured_microvolt'].reindex(dates)
    rows = [{'Model / diagnostic': 'Measured (observed only)', **dict(zip(columns, actual))}]
    for model in models:
        part = frame.loc[frame.model.eq(model)].set_index('date')
        for label, field in (('fitted', 'fitted_microvolt'),
                             ('fitted − measured', 'error_fitted_minus_measured')):
            values = part[field].reindex(dates)
            rows.append({'Model / diagnostic': f'{model} · {label}', **dict(zip(columns, values))})
    return pd.DataFrame(rows, columns=['Model / diagnostic', *columns])


def make_historical_fitted_figure(frame: pd.DataFrame, models):
    import plotly.graph_objects as go
    fig = go.Figure()
    first = frame.loc[frame.model.eq(MODELS[0])]
    fig.add_scatter(x=first.date, y=first.measured_microvolt,
                    mode='lines+markers', name='Measured (observed)',
                    line={'color': 'black', 'width': 2})
    for model in models:
        part = frame.loc[frame.model.eq(model)]
        fig.add_scatter(x=part.date, y=part.fitted_microvolt,
                        mode='lines', name=f'{model} · fitted')
    fig.update_layout(title='Development-period in-sample fitted values (not external forecasts)',
                      xaxis_title='Quarter', yaxis_title='µV', hovermode='x unified')
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def historical_fitted_metrics(frame: pd.DataFrame, models) -> pd.DataFrame:
    rows = []
    for model in models:
        part = frame.loc[frame.model.eq(model)]
        errors = part['error_fitted_minus_measured'].dropna().to_numpy(dtype=float)
        rows.append({'Model': model, 'Observed fitted quarters': len(errors),
                     'In-sample fitted RMSE (µV)': float(np.sqrt(np.mean(errors**2))),
                     'In-sample fitted MAE (µV)': float(np.mean(np.abs(errors)))})
    return pd.DataFrame(rows)

def render_optimized_neural_comparison(root: str | Path):
    """Render saved neural values and charts; never fit or rewrite a model."""
    import streamlit as st
    base = Path(root) / 'results' / '732B-5610410' / 'optimized_neural_forecasts_16_4'
    runs = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    if not runs:
        st.info('No optimized neural forecast runs available yet.')
        return
    run = st.selectbox('Optimized neural forecast run', runs, key='optimized_neural_forecast_run')
    try:
        manifest, holdout, future = load_optimized_neural_forecasts(root, run)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f'Optimized neural forecast artifact invalid: {exc}')
        return

    st.caption('Post-optimization external holdout: 2024-03 to 2025-06 · '
               'operational: 2025-09 to 2026-12 · future accuracy unknown')
    rows = [{'Model': model,
             'Internal RMSE (µV)': manifest['selected'][model]['internal_rmse'],
             'New external RMSE (µV)': manifest['external_holdout_metrics'][model]['rmse'],
             'External MAE (µV)': manifest['external_holdout_metrics'][model]['mae'],
             'First-step absolute error (µV)': manifest['external_holdout_metrics'][model]['first_step_absolute_error']}
            for model in MODELS]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption('Internal validation and external holdout use different partitions and horizons. '
               'Do not interpret their difference as an accuracy gain.')
    models = st.multiselect('Neural models shown', MODELS, default=list(MODELS),
                            key='optimized_neural_compare_models')
    if not models:
        st.info('Select at least one model to display predictions.')
        return

    # Only newer runs have historical fitted evidence; legacy runs stay usable.
    try:
        historical = load_historical_fitted(root, run, manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.warning(f'Historical fitted evidence unavailable or invalid: {exc}')
        historical = None
    if historical is not None:
        st.subheader('Historical fitted values · development period (1998–2023)')
        st.caption('In-sample fitted diagnostics, not independent forecasts. '
                   'Blank cells indicate model warm-up or missing original measurements. '
                   'Use the chart range slider to inspect individual years.')
        st.plotly_chart(make_historical_fitted_figure(historical, models),
                        width="stretch", key='optimized_neural_historical_fitted_chart')
        with st.expander('Historical fitted diagnostic metrics (in-sample only)'):
            st.dataframe(historical_fitted_metrics(historical, models),
                         hide_index=True, width="stretch")

    st.subheader('Historical holdout · actual numerical predictions')
    st.caption('Measured values and model predictions, six quarters. '
               'Signed error = prediction minus measurement (µV).')
    st.plotly_chart(make_holdout_figure(holdout, models), width="stretch",
                    key='optimized_neural_holdout_values_chart')

    st.subheader('Future operational forecasts · actual numerical predictions')
    st.caption('Six unvalidated future quarters. Differences between saved and optimized '
               'forecasts are not prediction errors.')
    saved = None
    try:
        from zenerestimation.visualization.cross_battery_dashboard_sources import load_410_operational_forecasts
        candidate = load_410_operational_forecasts(root)
        # Fail safely: never compare rows at different timestamps.
        if not pd.DatetimeIndex(pd.to_datetime(candidate['date'])).equals(FUTURE):
            raise ValueError('Saved operational dates do not match new run')
        for model in models:
            if model not in candidate or not np.isfinite(pd.to_numeric(candidate[model], errors='raise')).all():
                raise ValueError(f'Invalid saved operational values for {model}')
        saved = candidate
    except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
        st.warning(f'Saved operational comparison unavailable: {exc}. '
                   'Showing optimized neural forecasts alone.')
    st.plotly_chart(make_operational_figure(future, models, saved),
                    width="stretch",
                    key='optimized_neural_operational_comparison')
    from zenerestimation.visualization.optimized_neural_numerical_explorer import render_optimized_numerical_explorer
    render_optimized_numerical_explorer(root, manifest, holdout, future, historical, saved, models)

    # One six-sheet workbook, aligned by exact quarter; source artifacts untouched.
    st.subheader('Combined Excel evidence comparison')
    st.caption('Six worksheets: overview, historical fitted values and signed errors, '
               'external holdout values and signed errors, and operational forecasts. '
               'Saved and optimized values are adjacent by model and quarter; '
               'missing evidence stays blank.')
    try:
        from zenerestimation.visualization.combined_excel_comparison import (
            build_comparison_sheets, excel_bytes)
        workbook_sheets = build_comparison_sheets(
            root, run, manifest, holdout, future, historical, saved, models)
        workbook_data = excel_bytes(
            workbook_sheets,
            display_precision=st.session_state.get('optimized_numerical_precision', 'Full'),
        )
    except ImportError as exc:
        st.warning('Excel export requires artifact_tool in the zener environment: '
                   'python -m pip install artifact_tool. Details: ' + str(exc))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.warning('Combined Excel evidence unavailable: ' + str(exc))
    else:
        st.download_button('Download combined six-sheet Excel workbook (.xlsx)',
                           data=workbook_data,
                           file_name=f'ZenerEstimation_{run}_combined_comparison.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                           key='optimized_neural_combined_excel_export')

    st.caption('Source: read-only optimized neural forecast run ' + run +
               '. Frozen benchmark and original operational artifacts are unchanged.')
