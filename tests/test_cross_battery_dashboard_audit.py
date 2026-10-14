"""Focused audit tests using isolated, intentionally minimal fake sources."""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from zenerestimation.visualization import cross_battery_dashboard_audit as audit


def _series():
    history = pd.DataFrame({'date': pd.to_datetime(['2025-03-01']), 'measured_microvolt': [10.]})
    future = pd.DataFrame({'date': pd.to_datetime(['2025-06-01']), 'ARIMA': [11.]})
    return history, future


def _patch_sources(monkeypatch, tmp_path, *, broken_forecast=False, bad_boundary=False,
                   broken_provenance=False, missing_source=False):
    history, future = _series()
    if bad_boundary:
        future.loc[0, 'date'] = pd.Timestamp('2025-09-01')

    for name in ('load_410_observed_history', 'load_110_observed_history'):
        monkeypatch.setattr(audit, name, lambda root: history.copy())
    def forecast(root):
        if broken_forecast:
            raise ValueError('CSV/JSON disagreement')
        return future.copy()
    for name in ('load_410_operational_forecasts', 'load_110_operational_forecasts'):
        monkeypatch.setattr(audit, name, forecast)

    source = tmp_path / 'sample.csv'
    if not missing_source:
        source.write_bytes(b'saved source')
    digest = audit._sha256(source) if source.exists() else '0' * 64

    def provenance(root, comparison):
        if broken_provenance:
            raise ValueError('Scientific status mismatch')
        return SimpleNamespace(
            protocols=pd.DataFrame({'battery_id': list(audit.BATTERIES),
                                    'operational_evaluated': [False, False]}),
            sources=pd.DataFrame([{'battery_id': audit.BATTERY_410, 'source': 'sample.csv',
                                   'sha256': digest, 'meaning': 'File identity only'}]),
        )
    monkeypatch.setattr(audit, 'build_provenance', provenance)


def test_clean_sources_pass(monkeypatch, tmp_path):
    _patch_sources(monkeypatch, tmp_path)
    report = audit.audit_dashboard_sources(tmp_path, object())
    assert report.passed
    assert report.failures.empty
    assert report.checks['check'].str.startswith('source_sha256:').any()


def test_csv_json_disagreement_is_reported(monkeypatch, tmp_path):
    _patch_sources(monkeypatch, tmp_path, broken_forecast=True)
    report = audit.audit_dashboard_sources(tmp_path, object())
    assert not report.passed
    assert (report.failures['check'] == 'operational_csv_json_and_status').sum() == 2


def test_temporal_gap_is_reported(monkeypatch, tmp_path):
    _patch_sources(monkeypatch, tmp_path, bad_boundary=True)
    report = audit.audit_dashboard_sources(tmp_path, object())
    assert (report.failures['check'] == 'operational_temporal_boundary').sum() == 2


def test_provenance_failure_is_reported(monkeypatch, tmp_path):
    _patch_sources(monkeypatch, tmp_path, broken_provenance=True)
    report = audit.audit_dashboard_sources(tmp_path, object())
    assert not report.passed
    assert 'scientific_provenance_contract' in report.failures['check'].tolist()


def test_missing_source_is_reported_when_manifest_lists_it(monkeypatch, tmp_path):
    _patch_sources(monkeypatch, tmp_path, missing_source=True)
    report = audit.audit_dashboard_sources(tmp_path, object())
    assert not report.passed
    assert report.failures['check'].str.startswith('source_sha256:').any()


def test_report_does_not_modify_source(monkeypatch, tmp_path):
    _patch_sources(monkeypatch, tmp_path)
    path = tmp_path / 'sample.csv'
    original = path.read_bytes()
    audit.audit_dashboard_sources(tmp_path, object())
    assert path.read_bytes() == original
