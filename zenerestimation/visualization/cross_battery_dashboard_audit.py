"""Sprint 15.3.4.2: read-only source-integrity and consistency audit.

Run against the already loaded frozen benchmark comparison. A failed check is
reported explicitly; this module never repairs, regenerates, or writes evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from zenerestimation.visualization.cross_battery_dashboard_sources import (
    BATTERY as BATTERY_110, BATTERY_410,
    load_110_observed_history, load_410_observed_history,
    load_110_operational_forecasts, load_410_operational_forecasts,
)
from zenerestimation.visualization.cross_battery_dashboard_provenance import (
    build_provenance,
)

BATTERIES = (BATTERY_410, BATTERY_110)


@dataclass(frozen=True)
class DashboardAudit:
    checks: pd.DataFrame
    sources: pd.DataFrame

    @property
    def passed(self) -> bool:
        return bool(len(self.checks) and self.checks['passed'].all())

    @property
    def failures(self) -> pd.DataFrame:
        return self.checks.loc[~self.checks['passed']].copy(deep=True)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit_dashboard_sources(repository_root: str | Path, comparison: Any) -> DashboardAudit:
    """Audit saved evidence, recording failures without modifying source files.

    Note: recomputing a hash verifies the displayed fingerprint, not whether
    a source was altered relative to a separately trusted historical manifest.
    """
    root = Path(repository_root).resolve()
    checks: list[dict[str, Any]] = []
    sources = pd.DataFrame(columns=['battery_id', 'source', 'sha256', 'meaning'])

    def check(battery: str, name: str, operation: Callable[[], Any]) -> bool:
        try:
            operation()
        except Exception as exc:
            checks.append({'battery_id': battery, 'check': name, 'passed': False,
                           'detail': f'{type(exc).__name__}: {exc}'})
            return False
        checks.append({'battery_id': battery, 'check': name, 'passed': True,
                       'detail': 'Verified against current saved artifacts'})
        return True

    # Independent loader checks run even if provenance aggregation fails.
    for battery, history_loader, forecast_loader in (
        (BATTERY_410, load_410_observed_history, load_410_operational_forecasts),
        (BATTERY_110, load_110_observed_history, load_110_operational_forecasts),
    ):
        check(battery, 'observed_history_schema', lambda loader=history_loader: loader(root))
        check(battery, 'operational_csv_json_and_status', lambda loader=forecast_loader: loader(root))

        def temporal(loader_h=history_loader, loader_f=forecast_loader) -> None:
            history = loader_h(root)
            forecast = loader_f(root)
            if history.empty or forecast.empty:
                raise ValueError('Empty observed history or operational forecast')
            if forecast.date.min() != history.date.max() + pd.DateOffset(months=3):
                raise ValueError('Forecast does not begin one quarter after last observation')
            if (forecast.date <= history.date.max()).any():
                raise ValueError('Operational forecast overlaps observed history')
        check(battery, 'operational_temporal_boundary', temporal)

    def provenance() -> None:
        nonlocal sources
        view = build_provenance(root, comparison)
        sources = view.sources.copy(deep=True)
        if set(view.protocols.battery_id) != set(BATTERIES):
            raise ValueError('Provenance protocols do not cover both batteries')
        if not view.protocols.operational_evaluated.eq(False).all():
            raise ValueError('Operational forecast mislabeled as evaluated')
    check('both', 'scientific_provenance_contract', provenance)

    # Hash checks are separate from the provenance contract, so a missing or
    # unreadable source is clearly identified by its path in the report.
    if not sources.empty:
        for item in sources.itertuples(index=False):
            battery, relative, expected = item.battery_id, item.source, item.sha256

            def verify_file(relative=relative, expected=expected) -> None:
                path = root / relative
                if not path.is_file():
                    raise FileNotFoundError(path)
                if _sha256(path) != expected:
                    raise ValueError(f'Fingerprint mismatch: {relative}')
            check(battery, f'source_sha256:{relative}', verify_file)
    return DashboardAudit(checks=pd.DataFrame(checks), sources=sources)
