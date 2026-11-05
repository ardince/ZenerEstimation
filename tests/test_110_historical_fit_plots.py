"""Read-only plotting checks; never fit forecasting models."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from examples.optimization import run_110_historical_fit_plots as plots


def test_frozen_ranking_uses_saved_predictions():
    frozen = pd.DataFrame({'measured_microVolt': [10., 11.],
        **{name: [10.1, 11.1] for name in plots.MODEL_NAMES}})
    frozen['LSTM'] = [10.0, 11.0]
    frozen['KalmanLSTM'] = [20.0, 21.0]
    best, worst, metrics = plots.frozen_ranking(frozen)
    assert best == 'LSTM'
    assert worst == 'KalmanLSTM'
    assert metrics['LSTM'] == 0


def test_read_evidence_keeps_unobserved_quarter_missing(tmp_path, monkeypatch):
    dev = tmp_path / 'diagnostic'
    bench = tmp_path / 'benchmark'
    dev.mkdir(); bench.mkdir()
    dates = pd.date_range('1998-03-01', periods=100, freq='QS-MAR')
    fitted = pd.DataFrame({'date': dates.strftime('%Y-%m-%d'),
        **{name: np.arange(100, dtype=float) for name in plots.MODEL_NAMES}})
    fitted.to_csv(dev / 'historical_fitted_values.csv', index=False)
    holdout_dates = pd.to_datetime(['2023-03-01', '2023-06-01', '2023-09-01',
                                     '2023-12-01', '2024-03-01'])
    actual = np.asarray(plots.diagnostic.transferred.HOLDOUT_VALUES)
    frozen = pd.DataFrame({'date': holdout_dates.strftime('%Y-%m-%d'),
        'measured_microVolt': actual,
        **{name: actual + 0.1 for name in plots.MODEL_NAMES}})
    frozen.to_csv(bench / plots.diagnostic.PREDICTION_FILE, index=False)
    provenance = {'status': 'regenerated_historical_fit_diagnostic_not_frozen_benchmark',
        'benchmark_predictions_sha256': plots.diagnostic.sha256(bench / plots.diagnostic.PREDICTION_FILE),
        'development_quarters': 100, 'development_end': '2022-12-01',
        'models': list(plots.MODEL_NAMES)}
    (dev / 'diagnostic_provenance.json').write_text(json.dumps(provenance))
    raw = pd.DataFrame({'ds': dates.append(pd.DatetimeIndex(holdout_dates)),
        'microVolt': [float(i) for i in range(100)] + list(actual),
        'is_observed': [True] * 105})
    raw.loc[2, 'microVolt'] = np.nan
    raw.loc[2, 'is_observed'] = False
    history, result, frozen_read, _ = plots.read_evidence(
        diagnostic_dir=dev, benchmark_dir=bench, dataset=SimpleNamespace(data=raw))
    assert pd.isna(history.loc[history['ds'] == dates[2], 'measured']).all()
    assert len(result) == 100
    assert len(frozen_read) == 5


def test_frozen_hash_mismatch_is_rejected(tmp_path):
    dev = tmp_path / 'diagnostic'; bench = tmp_path / 'benchmark'
    dev.mkdir(); bench.mkdir()
    (dev / 'historical_fitted_values.csv').write_text('date\n')
    (dev / 'diagnostic_provenance.json').write_text(json.dumps({
        'status': 'regenerated_historical_fit_diagnostic_not_frozen_benchmark',
        'benchmark_predictions_sha256': 'wrong'}))
    (bench / plots.diagnostic.PREDICTION_FILE).write_text('tampered')
    with pytest.raises(ValueError, match='hash'):
        plots.read_evidence(diagnostic_dir=dev, benchmark_dir=bench,
            dataset=SimpleNamespace(data=pd.DataFrame({'ds': ['2022-12-01'],
                'microVolt': [1.], 'is_observed': [True]})))


def test_history_plot_has_external_legend_and_holdout_detail(tmp_path, monkeypatch):
    """Verify both panels reuse saved evidence without fitting a model."""
    dates = pd.date_range('1998-03-01', periods=100, freq='QS-MAR')
    history_dates = dates.append(pd.date_range('2023-03-01', periods=5, freq='QS-MAR'))
    history = pd.DataFrame({'ds': history_dates,
                            'measured': np.linspace(69, 170, len(history_dates))})
    fitted = pd.DataFrame({'date': dates,
                           **{name: np.linspace(69, 167, 100)
                              for name in plots.MODEL_NAMES}})
    frozen = pd.DataFrame({'date': history_dates[-5:],
                           'measured_microVolt': history['measured'].iloc[-5:].to_numpy(),
                           **{name: np.linspace(167, 172, 5)
                              for name in plots.MODEL_NAMES}})
    # Inspect the figure immediately before saving; the function closes it.
    captured = {}
    from matplotlib.figure import Figure
    original_savefig = Figure.savefig

    def inspect(self, *args, **kwargs):
        captured['axes'] = len(self.axes)
        captured['legends'] = len(self.legends)
        captured['zoom_xlim'] = self.axes[1].get_xlim()
        captured['full_xlim'] = self.axes[0].get_xlim()
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', inspect)
    output = tmp_path / 'chart.png'
    plots.draw_history(history, fitted, frozen, plots.MODEL_NAMES,
                       title='Diagnostic', destination=output)
    assert output.is_file()
    assert captured['axes'] == 2
    assert captured['legends'] == 1
    assert captured['zoom_xlim'][1] - captured['zoom_xlim'][0] < 1000
    assert captured['full_xlim'][1] - captured['full_xlim'][0] > 8000
