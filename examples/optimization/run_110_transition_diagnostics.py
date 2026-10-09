"""Sprint 14.9G.5: render dataset-110 transition from *stored* evidence only.

Run from the repository root:
    python -m examples.optimization.run_110_transition_diagnostics

No forecasting model is imported, fitted, or asked to predict. Never use rounded
presentation CSVs as the source of benchmark or operational forecasts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BATTERY = '732B-5610110'
MODELS = ('ARIMA', 'Kalman', 'LSTM', 'GRU', 'LinearTrendLSTM', 'KalmanLSTM')
ROOT = Path('results') / BATTERY
BENCHMARK_DIR = ROOT / 'transferred_benchmark'
OPERATIONAL_DIR = ROOT / 'operational_extrapolation'
OUTPUT_DIR = ROOT / 'transition_diagnostics'
BENCHMARK_DATES = pd.date_range('2023-03-01', periods=5, freq='QS-MAR')
OPERATIONAL_DATES = pd.date_range('2025-06-01', periods=6, freq='QS-MAR')
MEASURED_HOLDOUT = np.array([167.17, 167.73, 168.30, 168.86, 169.90])
CUTOFF = pd.Timestamp('2025-03-01')
DEVELOPMENT_END = pd.Timestamp('2022-12-01')
UNOBSERVED = pd.DatetimeIndex(pd.to_datetime([
    '2002-09-01', '2010-03-01', '2010-06-01',
    '2024-06-01', '2024-09-01', '2024-12-01',
]))


def _dates(values, expected, label):
    dates = pd.DatetimeIndex(pd.to_datetime(values, format='%Y-%m-%d', errors='raise'))
    if not dates.equals(expected):
        raise ValueError(f'{label}: dates differ from frozen timeline')
    return dates


def _finite(values, expected_count, label):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if len(arr) != expected_count or not np.isfinite(arr).all():
        raise ValueError(f'{label}: expected {expected_count} finite values')
    return arr


def load_evidence(*, dataset_path=None, benchmark_dir=None, operational_dir=None):
    """Read original observations, full-precision benchmark, and operational JSON."""
    dataset_path = Path(dataset_path or Path('datasets/processed') / f'{BATTERY}.csv')
    benchmark_dir = Path(benchmark_dir or BENCHMARK_DIR)
    operational_dir = Path(operational_dir or OPERATIONAL_DIR)
    history = pd.read_csv(dataset_path)
    if not {'ds', 'microVolt', 'is_observed'}.issubset(history.columns):
        raise ValueError('Processed dataset missing required observation columns')
    history['ds'] = pd.to_datetime(history['ds'], errors='raise')
    if history['ds'].duplicated().any() or not history['ds'].is_monotonic_increasing:
        raise ValueError('Processed history dates must be unique and ordered')
    history = history.loc[history['ds'] <= CUTOFF].copy()
    expected_history = pd.date_range('1998-03-01', CUTOFF, freq='QS-MAR')
    if not pd.DatetimeIndex(history['ds']).equals(expected_history):
        raise ValueError('Processed historical timeline differs from approved dataset 110')
    flags = history['is_observed'].map({'True': True, 'False': False, True: True, False: False})
    if flags.isna().any():
        raise ValueError('Invalid observed flags')
    history['is_observed'] = flags.astype(bool)
    values = pd.to_numeric(history['microVolt'], errors='coerce')
    if not np.array_equal(history['is_observed'].to_numpy(), np.isfinite(values.to_numpy(dtype=float))):
        raise ValueError('Observation provenance does not match missing values')
    missing = pd.DatetimeIndex(history.loc[~history['is_observed'], 'ds'])
    if not missing.equals(UNOBSERVED):
        raise ValueError('Unobserved quarter set changed')
    if not np.isclose(float(values.iloc[-1]), 171.60, atol=1e-6, rtol=0):
        raise ValueError('Last measured value is not 171.60 µV')

    benchmark_path = benchmark_dir / 'holdout_predictions_full_precision.csv'
    if not benchmark_path.is_file():
        raise FileNotFoundError(
            f'Full-precision benchmark file required: {benchmark_path}. '
            'Do not use the rounded presentation CSV.'
        )
    benchmark = pd.read_csv(benchmark_path, dtype={'date': str})
    if list(benchmark.columns) != ['date', 'measured_microVolt', *MODELS]:
        raise ValueError('Unexpected benchmark prediction columns')
    _dates(benchmark['date'], BENCHMARK_DATES, 'benchmark')
    np.testing.assert_allclose(
        _finite(benchmark['measured_microVolt'], 5, 'holdout actuals'),
        MEASURED_HOLDOUT, atol=1e-6, rtol=0,
        err_msg='Frozen measured holdout values changed',
    )
    for model in MODELS:
        _finite(benchmark[model], 5, f'{model} benchmark')
    history_holdout = history.set_index('ds')['microVolt'].reindex(BENCHMARK_DATES)
    np.testing.assert_allclose(
        history_holdout.to_numpy(dtype=float), MEASURED_HOLDOUT,
        atol=1e-6, rtol=0,
        err_msg='Processed measurements disagree with benchmark actuals',
    )

    op_path = operational_dir / 'future_extrapolation.json'
    operational = json.loads(op_path.read_text(encoding='utf-8'))
    if (operational.get('battery') != BATTERY
            or operational.get('kind') != 'operational_extrapolation_not_holdout_evaluation'
            or operational.get('scientific_status') != 'unvalidated_future_operational_extrapolation'
            or operational.get('training_cutoff') != '2025-03-01'
            or operational.get('optimized_for_dataset_110') is not False
            or operational.get('models') != list(MODELS)
            or operational.get('measured_actuals') is not None
            or operational.get('evaluation_metrics') is not None):
        raise ValueError('Operational scientific provenance does not match approved contract')
    _dates(operational['forecast_dates'], OPERATIONAL_DATES, 'operational')
    if set(operational['predictions']) != set(MODELS):
        raise ValueError('Operational model set differs from frozen benchmark')
    predictions = {
        name: _finite(operational['predictions'][name], 6, f'{name} operational')
        for name in MODELS
    }
    return history, benchmark, predictions


def render_transition(history, benchmark, predictions, *, output_dir=OUTPUT_DIR):
    """Render two distinct periods, preserving measured and unobserved provenance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax, zoom) = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                                  gridspec_kw={'height_ratios': [1.15, 1]})
    measured = history.loc[history['is_observed']]
    recent = measured.loc[measured['ds'] >= pd.Timestamp('2022-03-01')]
    for panel in (ax, zoom):
        panel.plot(recent['ds'], recent['microVolt'], 'ko-', lw=1.8,
                   ms=4, label='Measured observations', zorder=5)
        panel.axvline(DEVELOPMENT_END, color='gray', linestyle=':', lw=1.5,
                      label='Benchmark training end · 2022-12')
        panel.axvline(CUTOFF, color='black', linestyle=':', lw=1.5,
                      label='Operational refit cutoff · 2025-03')
        panel.axvspan(OPERATIONAL_DATES[0], OPERATIONAL_DATES[-1],
                      color='gray', alpha=0.06)
        panel.grid(alpha=0.22)
        panel.set_ylabel('microVolt (µV)')
    # Historical gaps have no measured values: draw markers on the top panel's
    # baseline only, rather than inventing interpolated observation values.
    for date in UNOBSERVED[UNOBSERVED >= pd.Timestamp('2022-03-01')]:
        ax.axvline(date, color='red', alpha=0.25, linestyle='--', lw=0.9)
    colors = plt.get_cmap('tab10')
    for i, name in enumerate(MODELS):
        color = colors(i)
        bench_values = benchmark[name].to_numpy(dtype=float)
        ax.plot(BENCHMARK_DATES, bench_values, '--o', color=color, ms=4,
                lw=1.5, label=f'{name} · measured holdout forecast')
        zoom.plot(OPERATIONAL_DATES, predictions[name], '-.s', color=color,
                  ms=4, lw=1.7, label=f'{name} · operational, unvalidated')
    ax.set_title(f'Battery {BATTERY} — Frozen 5-quarter measured-holdout benchmark')
    zoom.set_title('Separate six-quarter operational refit · unvalidated future forecasts')
    zoom.set_xlabel('Date')
    ax.legend(ncol=2, fontsize=8, loc='upper left')
    zoom.legend(ncol=2, fontsize=8, loc='upper left')
    ax.set_xlim(pd.Timestamp('2022-02-01'), pd.Timestamp('2026-11-01'))
    fig.autofmt_xdate()
    fig.tight_layout()
    image_path = output_dir / 'benchmark_to_operational_transition.png'
    fig.savefig(image_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return image_path


def run_transition(*, dataset_path=None, benchmark_dir=None, operational_dir=None,
                   output_dir=OUTPUT_DIR):
    history, benchmark, predictions = load_evidence(
        dataset_path=dataset_path,
        benchmark_dir=benchmark_dir,
        operational_dir=operational_dir,
    )
    return render_transition(history, benchmark, predictions, output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-path', type=Path)
    parser.add_argument('--benchmark-dir', type=Path)
    parser.add_argument('--operational-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    path = run_transition(**vars(args))
    print(f'Dataset-110 transition diagnostics saved: {path}')
    print('No model fit or predict calls were performed.')


if __name__ == '__main__':
    main()
