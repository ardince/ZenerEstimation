"""Read-only dashboard tables from validated cross-battery benchmark records.

No fitting, forecast generation, normalization by unverified scales, or disk writes.
Comparisons retain their original benchmark protocols and units.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import math
import pandas as pd

from zenerestimation.visualization.cross_battery_artifacts import (
    BatteryBenchmarkRecord, CrossBatteryArtifactLoader, ArtifactContractError,
    MODELS, METRICS,
)


@dataclass(frozen=True)
class CrossBatteryComparison:
    summary: pd.DataFrame
    model_metrics: pd.DataFrame
    evidence: pd.DataFrame
    holdout_predictions: pd.DataFrame
    historical_fits: pd.DataFrame


def build_comparison(records: Mapping[str, BatteryBenchmarkRecord]) -> CrossBatteryComparison:
    """Transform already validated records into independent dashboard tables.

    Metrics are absolute microvolt errors (MAPE is percentage). Neither RMSE
    nor rankings are interpreted as an apples-to-apples transfer experiment.
    """
    if set(records) != {'732B-5610410', '732B-5610110'}:
        raise ArtifactContractError('Comparison requires exactly batteries 410 and 110')
    summaries, metrics_rows, evidence_rows, prediction_rows, fit_rows = [], [], [], [], []
    for battery in ('732B-5610410', '732B-5610110'):
        record = records[battery]
        if record.battery_id != battery:
            raise ArtifactContractError('Record battery identifier mismatch')
        if list(record.metrics.index) != list(MODELS) or not set(METRICS).issubset(record.metrics.columns):
            raise ArtifactContractError('Unexpected model metrics schema')
        best = min(MODELS, key=lambda model: (float(record.metrics.loc[model, 'rmse']), model))
        summaries.append(dict(battery_id=battery, benchmark_type=record.benchmark_type,
                              development_end=record.development_end, holdout_start=record.holdout_start,
                              holdout_end=record.holdout_end, holdout_steps=record.holdout_steps,
                              best_rmse_model=best, best_rmse_microvolt=float(record.metrics.loc[best, 'rmse'])))
        for model in MODELS:
            values = {m: float(record.metrics.loc[model, m]) for m in METRICS}
            if not all(math.isfinite(v) and v >= 0 for v in values.values()):
                raise ArtifactContractError('Invalid metric in comparison')
            metrics_rows.append(dict(battery_id=battery, benchmark_type=record.benchmark_type,
                                     model=model, **values,
                                     rmse_rank=next(x['rank'] for x in record.rankings['rmse'] if x['model'] == model)))
        for key, item in sorted(record.evidence.items()):
            evidence_rows.append(dict(battery_id=battery, benchmark_type=record.benchmark_type,
                                      evidence_type=key, structured=item.structured, visual=item.visual,
                                      origin=item.origin, source=item.source))


        if record.holdout_predictions is not None:
            holdout_evidence = record.evidence.get('holdout_predictions')

            if holdout_evidence is not None:
                holdout_origin = holdout_evidence.origin
            elif battery == '732B-5610110':
                holdout_origin = 'frozen_transferred_benchmark'
            else:
                holdout_origin = 'regenerated_410_diagnostic_not_frozen_benchmark'

            for _, row in record.holdout_predictions.iterrows():
                for model in MODELS:
                    prediction_rows.append(dict(battery_id=battery, benchmark_type=record.benchmark_type,
                                                date=str(row['date']), model=model,
                                                measured_microvolt=float(row['measured_microVolt']),
                                                predicted_microvolt=float(row[model]),
                                                origin=holdout_origin))

        if record.historical_fits is not None:
            for _, row in record.historical_fits.iterrows():
                for model in MODELS:
                    value = row[model]
                    fit_rows.append(dict(battery_id=battery, date=str(row['date']), model=model,
                                         fitted_microvolt=float(value) if pd.notna(value) else float('nan'),
                                         origin='regenerated_diagnostic_not_frozen'))
    return CrossBatteryComparison(
        summary=pd.DataFrame(summaries), model_metrics=pd.DataFrame(metrics_rows),
        evidence=pd.DataFrame(evidence_rows),
        holdout_predictions=pd.DataFrame(prediction_rows, columns=[
            'battery_id','benchmark_type','date','model','measured_microvolt','predicted_microvolt','origin']),
        historical_fits=pd.DataFrame(fit_rows, columns=[
            'battery_id','date','model','fitted_microvolt','origin']),
    )


def load_comparison(repository_root: str = '.') -> CrossBatteryComparison:
    """Convenience read-only entry point for Sprint 15.3."""
    return build_comparison(CrossBatteryArtifactLoader(repository_root).load_both())
