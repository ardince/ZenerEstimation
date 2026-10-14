"""Sprint 15.4.2 deterministic synthetic regression tests."""
from dataclasses import replace
import pandas as pd
import pytest

from zenerestimation.visualization.cross_battery_artifacts import ArtifactContractError, MODELS
from zenerestimation.visualization.cross_battery_comparison import CrossBatteryComparison
from zenerestimation.visualization.cross_battery_generalization import analyze_generalization

B410, B110 = '732B-5610410', '732B-5610110'


def fixture_comparison():
    summaries, metrics, holdout = [], [], []
    for battery, protocol, steps, first, start, origin, order in (
        (B410, 'optimized', 6, 10.0, '2024-03-01', 'regenerated_diagnostic_not_frozen', list(MODELS)),
        (B110, 'transferred', 5, 100.0, '2023-03-01', 'frozen_transferred_benchmark', list(reversed(MODELS))),
    ):
        dates = pd.date_range(start, periods=steps, freq='QS-MAR').strftime('%Y-%m-%d').tolist()
        summaries.append(dict(battery_id=battery, benchmark_type=protocol, holdout_steps=steps,
                              holdout_start=dates[0], holdout_end=dates[-1]))
        for model in MODELS:
            rmse = float(order.index(model) + 1)
            metrics.append(dict(battery_id=battery, benchmark_type=protocol, model=model,
                                rmse=rmse, mae=rmse/2, mape=rmse/10, rmse_rank=order.index(model)+1))
        for date in dates:
            for model in MODELS:
                holdout.append(dict(battery_id=battery, benchmark_type=protocol, date=date,
                                    model=model, measured_microvolt=first,
                                    predicted_microvolt=first+0.1, origin=origin))
    return CrossBatteryComparison(summary=pd.DataFrame(summaries), model_metrics=pd.DataFrame(metrics),
                                  holdout_predictions=pd.DataFrame(holdout), evidence=pd.DataFrame(),
                                  historical_fits=pd.DataFrame())


def test_scales_normalized_metrics_and_ratio():
    result = analyze_generalization(fixture_comparison())
    assert result.battery_scales.mean_absolute_measured_microvolt.tolist() == [10., 100.]
    assert len(result.normalized_metrics) == 12
    assert len(result.ranking_comparison) == 6
    arima = result.ranking_comparison.query("model == 'ARIMA'").iloc[0]
    assert arima.nrmse_410_percent == pytest.approx(10.)
    assert arima.nrmse_110_percent == pytest.approx(6.)
    assert arima.nrmse_ratio_110_over_410 == pytest.approx(.6)
    assert arima.rank_change_110_minus_410 == 5
    assert result.spearman_rho == pytest.approx(-1.)


def test_input_frames_not_mutated():
    comparison = fixture_comparison()
    original = [f.copy(deep=True) for f in (comparison.summary, comparison.model_metrics, comparison.holdout_predictions)]
    analyze_generalization(comparison)
    for after, before in zip((comparison.summary, comparison.model_metrics, comparison.holdout_predictions), original):
        pd.testing.assert_frame_equal(after, before)


@pytest.mark.parametrize('battery', [B410, B110])
def test_missing_holdout_rejected(battery):
    comparison = fixture_comparison()
    bad = comparison.holdout_predictions.loc[~((comparison.holdout_predictions.battery_id == battery) &
                                               (comparison.holdout_predictions.model == 'ARIMA'))]
    with pytest.raises(ArtifactContractError, match='holdout'):
        analyze_generalization(replace(comparison, holdout_predictions=bad))


@pytest.mark.parametrize('battery', [B410, B110])
def test_bad_provenance_rejected(battery):
    comparison = fixture_comparison()
    bad = comparison.holdout_predictions.copy()
    bad.loc[bad.battery_id.eq(battery), 'origin'] = 'unknown'
    with pytest.raises(ArtifactContractError, match='provenance'):
        analyze_generalization(replace(comparison, holdout_predictions=bad))


def test_conflicting_observed_values_rejected():
    comparison = fixture_comparison()
    bad = comparison.holdout_predictions.copy()
    ix = bad.index[bad.battery_id.eq(B410)][0]
    bad.loc[ix, 'measured_microvolt'] = 123
    with pytest.raises(ArtifactContractError, match='Inconsistent measured'):
        analyze_generalization(replace(comparison, holdout_predictions=bad))


def test_zero_scale_rejected():
    comparison = fixture_comparison()
    bad = comparison.holdout_predictions.copy()
    bad.loc[bad.battery_id.eq(B410), 'measured_microvolt'] = 0
    with pytest.raises(ArtifactContractError, match='normalization scale'):
        analyze_generalization(replace(comparison, holdout_predictions=bad))


def test_missing_model_metrics_rejected():
    comparison = fixture_comparison()
    with pytest.raises(ArtifactContractError, match='twelve'):
        analyze_generalization(replace(comparison, model_metrics=comparison.model_metrics.iloc[:-1]))


def test_unchanged_saved_metrics_are_authoritative():
    comparison = fixture_comparison()
    bad = comparison.holdout_predictions.copy()
    bad['predicted_microvolt'] = 999999.0  # deliberately incompatible with RMSE
    result = analyze_generalization(replace(comparison, holdout_predictions=bad))
    assert result.normalized_metrics.query("battery_id == @B410 and model == 'ARIMA'").iloc[0].rmse_microvolt == 1.0
