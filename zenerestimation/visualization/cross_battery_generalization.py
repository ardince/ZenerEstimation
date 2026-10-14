"""Sprint 15.4.2: deterministic, read-only cross-battery holdout comparison.

Uses saved benchmark metrics as authoritative numerators and observed holdout
values solely for scale denominators. Regenerated 410 predictions are *not*
used to recalculate frozen metrics. This is descriptive, not a causal transfer
experiment or an uncertainty estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from zenerestimation.visualization.cross_battery_artifacts import ArtifactContractError, MODELS
from zenerestimation.visualization.cross_battery_comparison import CrossBatteryComparison, load_comparison

BATTERIES = ('732B-5610410', '732B-5610110')
EXPECTED_ORIGIN = {
    BATTERIES[0]: 'regenerated_diagnostic_not_frozen',
    BATTERIES[1]: 'frozen_transferred_benchmark',
}
EXPECTED_BENCHMARK = {BATTERIES[0]: 'optimized', BATTERIES[1]: 'transferred'}
DENOMINATOR_FLOOR = 1e-9  # microvolt; guard zero / nearly-zero scales


@dataclass(frozen=True)
class GeneralizationAnalysis:
    normalized_metrics: pd.DataFrame
    ranking_comparison: pd.DataFrame
    battery_scales: pd.DataFrame
    spearman_rho: float
    interpretation: str


def _reject(message: str) -> None:
    raise ArtifactContractError(message)


def analyze_generalization(comparison: CrossBatteryComparison) -> GeneralizationAnalysis:
    """Analyze the full six-model battery pair without mutating any input frames."""
    summary = comparison.summary
    metrics = comparison.model_metrics
    holdout = comparison.holdout_predictions
    for frame, fields, label in (
        (summary, {'battery_id', 'benchmark_type', 'holdout_steps', 'holdout_start', 'holdout_end'}, 'summary'),
        (metrics, {'battery_id', 'benchmark_type', 'model', 'rmse', 'mae', 'mape', 'rmse_rank'}, 'metrics'),
        (holdout, {'battery_id', 'benchmark_type', 'date', 'model', 'measured_microvolt', 'predicted_microvolt', 'origin'}, 'holdout'),
    ):
        if not isinstance(frame, pd.DataFrame) or not fields.issubset(frame.columns):
            _reject(f'Invalid {label} schema')
    if len(summary) != 2 or set(summary.battery_id) != set(BATTERIES) or summary.battery_id.duplicated().any():
        _reject('Exactly one summary per battery is required')
    if len(metrics) != 12 or metrics[['battery_id', 'model']].duplicated().any():
        _reject('Exactly twelve unique model metric rows required')
    if set(metrics.battery_id) != set(BATTERIES):
        _reject('Metric battery identities mismatch')
    if set(holdout.battery_id) != set(BATTERIES):
        _reject('Both holdout datasets required for normalization')

    scales = []
    normalized = []
    for battery in BATTERIES:
        info = summary.loc[summary.battery_id.eq(battery)].iloc[0]
        if info.benchmark_type != EXPECTED_BENCHMARK[battery]:
            _reject(f'Unexpected benchmark protocol: {battery}')
        steps = int(info.holdout_steps)
        if steps != (6 if battery == BATTERIES[0] else 5):
            _reject(f'Unexpected holdout horizon: {battery}')
        part = holdout.loc[holdout.battery_id.eq(battery)].copy()
        if len(part) != steps * len(MODELS) or part[['date', 'model']].duplicated().any():
            _reject(f'Incomplete or duplicate holdout: {battery}')
        if not part.origin.eq(EXPECTED_ORIGIN[battery]).all():
            _reject(f'Unexpected holdout provenance: {battery}')
        if not part.benchmark_type.eq(EXPECTED_BENCHMARK[battery]).all():
            _reject(f'Holdout benchmark protocol mismatch: {battery}')
        dates = sorted(part.date.unique().tolist())
        if len(dates) != steps or dates[0] != str(info.holdout_start) or dates[-1] != str(info.holdout_end):
            _reject(f'Holdout dates inconsistent with summary: {battery}')
        if any(set(part.loc[part.date.eq(date), 'model']) != set(MODELS) for date in dates):
            _reject(f'Missing holdout models: {battery}')
        observed = []
        for date in dates:
            vals = pd.to_numeric(part.loc[part.date.eq(date), 'measured_microvolt'], errors='coerce').to_numpy(dtype=float)
            if not all(math.isfinite(v) for v in vals) or not all(v == vals[0] for v in vals):
                _reject(f'Inconsistent measured holdout values: {battery} {date}')
            observed.append(float(vals[0]))
        scale = sum(abs(v) for v in observed) / steps
        if not math.isfinite(scale) or scale <= DENOMINATOR_FLOOR:
            _reject(f'Invalid holdout normalization scale: {battery}')
        scales.append({'battery_id': battery, 'benchmark_type': info.benchmark_type,
                       'holdout_start': dates[0], 'holdout_end': dates[-1],
                       'holdout_steps': steps, 'mean_absolute_measured_microvolt': scale,
                       'holdout_origin': EXPECTED_ORIGIN[battery]})
        sub = metrics.loc[metrics.battery_id.eq(battery)]
        if set(sub.model) != set(MODELS) or not sub.benchmark_type.eq(EXPECTED_BENCHMARK[battery]).all():
            _reject(f'Invalid metric model coverage/protocol: {battery}')
        for model in MODELS:
            entry = sub.loc[sub.model.eq(model)].iloc[0]
            values = [float(entry[key]) for key in ('rmse', 'mae', 'mape')]
            if not all(math.isfinite(v) and v >= 0 for v in values):
                _reject(f'Invalid saved benchmark metrics: {battery} {model}')
            rank = int(entry.rmse_rank)
            if rank < 1 or rank > len(MODELS) or rank != entry.rmse_rank:
                _reject(f'Invalid saved RMSE rank: {battery} {model}')
            normalized.append({'battery_id': battery, 'benchmark_type': info.benchmark_type,
                               'model': model, 'rmse_microvolt': values[0], 'mae_microvolt': values[1],
                               'mape_percent_saved': values[2], 'rmse_rank_saved': rank,
                               'mean_absolute_measured_microvolt': scale,
                               'nrmse_percent': 100 * values[0] / scale,
                               'nmae_percent': 100 * values[1] / scale,
                               'holdout_origin': EXPECTED_ORIGIN[battery]})
        if set(sub.rmse_rank.astype(int)) != set(range(1, len(MODELS) + 1)):
            _reject(f'Non-permutation RMSE ranking: {battery}')
    frame = pd.DataFrame(normalized)
    rows = []
    ranks_410, ranks_110 = [], []
    for model in MODELS:
        a = frame.loc[frame.battery_id.eq(BATTERIES[0]) & frame.model.eq(model)].iloc[0]
        b = frame.loc[frame.battery_id.eq(BATTERIES[1]) & frame.model.eq(model)].iloc[0]
        if a.nrmse_percent <= 0:
            _reject(f'Undefined NRMSE ratio for zero 410 RMSE: {model}')
        ranks_410.append(int(a.rmse_rank_saved))
        ranks_110.append(int(b.rmse_rank_saved))
        rows.append({'model': model, 'rank_410': ranks_410[-1], 'rank_110': ranks_110[-1],
                     'rank_change_110_minus_410': ranks_110[-1] - ranks_410[-1],
                     'nrmse_410_percent': float(a.nrmse_percent),
                     'nrmse_110_percent': float(b.nrmse_percent),
                     'nrmse_ratio_110_over_410': float(b.nrmse_percent / a.nrmse_percent)})
    # Both rank vectors are complete permutations of 1..6 (no ties).
    n = len(MODELS)
    rho = 1 - 6 * sum((a-b)**2 for a, b in zip(ranks_410, ranks_110)) / (n*(n*n-1))
    return GeneralizationAnalysis(
        normalized_metrics=frame, ranking_comparison=pd.DataFrame(rows),
        battery_scales=pd.DataFrame(scales), spearman_rho=float(rho),
        interpretation=('Descriptive cross-battery comparison only: 410 is optimized, '
                        '110 is transferred. Different holdout dates and battery scales '
                        'preclude interpreting ratios as causal transfer penalties. '
                        '410 holdout predictions are regenerated diagnostics; '
                        'saved benchmark metrics remain authoritative.'),
    )


def load_generalization(repository_root: str = '.') -> GeneralizationAnalysis:
    """Read-only convenience entry point; no model fitting or artifact writes."""
    return analyze_generalization(load_comparison(repository_root))
