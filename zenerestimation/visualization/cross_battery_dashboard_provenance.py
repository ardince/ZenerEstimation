"""Sprint 15.3.4: read-only scientific provenance for cross-battery evidence.

This adapter does not fit models, change artifacts, or equate file hashes with
scientific validation. Operational metadata are accepted only after the existing
forecast loaders have validated their corresponding saved predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd

from zenerestimation.visualization.cross_battery_dashboard_sources import (
    BATTERY as BATTERY_110,
    BATTERY_410,
    MODELS,
    STATUS,
    load_110_observed_history,
    load_410_observed_history,
    load_110_operational_forecasts,
    load_410_operational_forecasts,
)


@dataclass(frozen=True)
class ProvenanceView:
    protocols: pd.DataFrame
    evidence: pd.DataFrame
    sources: pd.DataFrame


def _json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f'Invalid JSON constant: {value}')
    value = json.loads(path.read_text(encoding='utf-8'), parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_provenance(repository_root: str | Path, comparison: Any) -> ProvenanceView:
    """Validate saved operational series and build display-only provenance tables.

    `comparison` must be the result of `load_comparison(repository_root)`.
    The existing artifact loader remains responsible for benchmark validation.
    """
    root = Path(repository_root)
    summary = comparison.summary.copy(deep=True)
    evidence = comparison.evidence.copy(deep=True)
    if set(summary['battery_id']) != {BATTERY_410, BATTERY_110} or len(summary) != 2:
        raise ValueError('Expected one benchmark summary per battery')
    if not {'battery_id', 'benchmark_type', 'development_end', 'holdout_start',
            'holdout_end', 'holdout_steps'}.issubset(summary.columns):
        raise ValueError('Incomplete benchmark summary')
    if not {'battery_id', 'evidence_type', 'structured', 'visual', 'origin', 'source'}.issubset(evidence.columns):
        raise ValueError('Incomplete evidence availability table')

    protocols: list[dict[str, Any]] = []
    source_rows: list[dict[str, str]] = []
    for battery, history_loader, forecast_loader in (
        (BATTERY_410, load_410_observed_history, load_410_operational_forecasts),
        (BATTERY_110, load_110_observed_history, load_110_operational_forecasts),
    ):
        benchmark = summary.loc[summary.battery_id.eq(battery)].iloc[0]
        observed = history_loader(root)
        forecasts = forecast_loader(root)  # Enforce existing scientific/data contract.
        folder = root / 'results' / battery / 'operational_extrapolation'
        metadata_path = folder / 'future_extrapolation.json'
        metadata = _json(metadata_path)
        if metadata.get('scientific_status') != STATUS or metadata.get('measured_actuals') is not None or metadata.get('evaluation_metrics') is not None:
            raise ValueError(f'Operational status inconsistent for {battery}')
        if metadata.get('models') != list(MODELS):
            raise ValueError(f'Unexpected operational models for {battery}')
        cutoff = pd.Timestamp(metadata['training_cutoff'])
        if observed.empty or observed.date.max() != cutoff:
            raise ValueError(f'Last measured date and operational cutoff disagree for {battery}')
        if forecasts.date.min() <= cutoff:
            raise ValueError(f'Operational forecasts overlap training for {battery}')
        if battery == BATTERY_110:
            if metadata.get('optimized_for_dataset_110') is not False or metadata.get('configuration_source_battery') != BATTERY_410:
                raise ValueError('Dataset 110 transfer provenance mismatch')
            config_origin = 'Frozen configurations transferred from dataset 410; no 110 optimization'
            benchmark_origin = 'frozen_cross_battery_parameter_transfer'
            forecast_csv = folder / 'future_extrapolation_full_precision.csv'
        else:
            if metadata.get('configuration_source') != 'frozen_410_optimized_benchmark':
                raise ValueError('Dataset 410 configuration provenance mismatch')
            config_origin = 'Frozen dataset 410 optimized benchmark configurations'
            benchmark_origin = 'optimized_benchmark'
            forecast_csv = folder / 'future_extrapolation.csv'
        protocols.append({
            'battery_id': battery,
            'benchmark_type': benchmark['benchmark_type'],
            'benchmark_origin': benchmark_origin,
            'configuration_source': config_origin,
            'development_end': benchmark['development_end'],
            'holdout_start': benchmark['holdout_start'],
            'holdout_end': benchmark['holdout_end'],
            'holdout_steps': int(benchmark['holdout_steps']),
            'operational_training_cutoff': cutoff.strftime('%Y-%m-%d'),
            'last_observed_microvolt': float(observed.measured_microvolt.iloc[-1]),
            'operational_start': forecasts.date.min().strftime('%Y-%m-%d'),
            'operational_end': forecasts.date.max().strftime('%Y-%m-%d'),
            'operational_quarters': len(forecasts),
            'operational_status': STATUS,
            'operational_update_policy': metadata.get('model_update_policy'),
            'operational_evaluated': False,
        })
        paths = [root / 'datasets' / 'processed' / f'{battery}.csv', metadata_path, forecast_csv]
        if battery == BATTERY_410:
            paths += [root / 'results' / battery / 'optimized_benchmark' / 'benchmark.json',
                      root / 'results' / battery / 'optimized_benchmark' / 'provenance.json']
        else:
            paths += [root / 'results' / battery / 'transferred_benchmark' / 'benchmark.json',
                      root / 'results' / battery / 'transferred_benchmark' / 'transfer_provenance_validated.json']
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f'Provenance source unavailable: {path}')
            source_rows.append({'battery_id': battery,
                                'source': path.relative_to(root).as_posix(),
                                'sha256': _hash(path),
                                'meaning': 'File identity only; not scientific validation'})
    return ProvenanceView(protocols=pd.DataFrame(protocols), evidence=evidence,
                          sources=pd.DataFrame(source_rows))


def render_provenance_section(view: ProvenanceView) -> None:
    """Render only previously validated provenance; Streamlit is optional."""
    import streamlit as st
    with st.expander('Scientific provenance and reproducibility', expanded=False):
        st.warning('Operational forecasts are unvalidated. Benchmark protocols differ: 410 optimized, 110 frozen transfer. Cross-battery metric differences are descriptive, not a controlled generalization test.')
        st.dataframe(view.protocols, hide_index=True, width="stretch")
        st.caption('Evidence availability and origin (unavailable structured series are not reconstructed).')
        st.dataframe(view.evidence, hide_index=True, width="stretch")
        st.caption('SHA-256 values identify saved files; they do not establish forecast accuracy or scientific validity.')
        st.dataframe(view.sources, hide_index=True, width="stretch")
