"""Sprint 15.3.5.3: read-only reconciliation of numerical explorer tables.

Checks the numerical UI against the already validated source adapters. It does
not claim external tamper detection or independently reproduce model forecasts.
"""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from zenerestimation.visualization.cross_battery_dashboard_numerical_ui import (
    BATTERY_110, BATTERY_410, csv_bytes, display_table, numerical_result,
)
from zenerestimation.visualization.cross_battery_dashboard_sources import (
    load_110_observed_history, load_410_observed_history,
    load_110_operational_forecasts, load_410_operational_forecasts,
)

KINDS = ("metrics", "holdout", "historical_fits", "observed", "operational")


@dataclass(frozen=True)
class NumericalReconciliation:
    checks: pd.DataFrame

    @property
    def passed(self) -> bool:
        return bool(self.checks["passed"].all())

    @property
    def failures(self) -> pd.DataFrame:
        return self.checks.loc[~self.checks["passed"]].copy(deep=True)


def _equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Compare columns, row order, values, NaNs and datetime values."""
    pd.testing.assert_frame_equal(actual.reset_index(drop=True),
                                  expected.reset_index(drop=True),
                                  check_exact=True, check_dtype=True)


def _export_roundtrip(source: pd.DataFrame) -> None:
    """CSV uses 17 significant digits, sufficient for float64 round-trip."""
    exported = pd.read_csv(BytesIO(csv_bytes(source)), encoding="utf-8-sig")
    assert list(exported) == list(source)
    assert len(exported) == len(source)
    for col in source:
        if pd.api.types.is_float_dtype(source[col]):
            original = source[col].to_numpy(dtype="float64")
            loaded = pd.to_numeric(exported[col], errors="raise").to_numpy(dtype="float64")
            if not np.array_equal(original, loaded, equal_nan=True):
                # pandas CSV parsing can introduce a final ULP difference;
                # compare the decimal CSV representation instead.
                from csv import reader
                rows = list(reader(BytesIO(csv_bytes(source)).read().decode("utf-8-sig").splitlines()))
                j = rows[0].index(col)
                for i, value in enumerate(original):
                    cell = rows[i + 1][j]
                    if np.isnan(value):
                        assert cell == ""
                    else:
                        assert float(cell) == value
        elif pd.api.types.is_datetime64_any_dtype(source[col]):
            expected = source[col].dt.strftime("%Y-%m-%d")
            assert exported[col].astype(str).tolist() == expected.tolist()
        else:
            # For integers and strings, pandas should retain scalar content.
            assert exported[col].astype(str).tolist() == source[col].astype(str).tolist()


def reconcile_numerical_results(comparison, repository_root: str | Path) -> NumericalReconciliation:
    """Audit all batteries and kinds without mutating artifacts or inputs."""
    root = Path(repository_root)
    rows = []
    observed = {BATTERY_110: load_110_observed_history,
                BATTERY_410: load_410_observed_history}
    operational = {BATTERY_110: load_110_operational_forecasts,
                   BATTERY_410: load_410_operational_forecasts}
    attrs = {"metrics": "model_metrics", "holdout": "holdout_predictions",
             "historical_fits": "historical_fits"}
    for battery in (BATTERY_410, BATTERY_110):
        for kind in KINDS:
            try:
                table, status = numerical_result(
                    comparison, root, battery_id=battery, kind=kind, models=None)
                assert isinstance(status, str) and status.strip()
                if kind in attrs:
                    original = getattr(comparison, attrs[kind])
                    expected = original.loc[original.battery_id.eq(battery)].copy()
                    if kind == "holdout" and not expected.empty:
                        expected["error_microvolt"] = (expected.predicted_microvolt
                                                       - expected.measured_microvolt)
                        expected["absolute_error_microvolt"] = expected.error_microvolt.abs()
                    sort_cols = [c for c in ("date", "rmse_rank", "model") if c in expected]
                    if sort_cols:
                        expected = expected.sort_values(sort_cols, kind="stable")
                    expected = expected.reset_index(drop=True)
                    _equal(table, expected)
                    if battery == BATTERY_410 and kind in ("holdout", "historical_fits"):
                        assert table.empty and "unavailable" in status.lower()
                elif kind == "observed":
                    _equal(table, observed[battery](root))
                else:
                    _equal(table, operational[battery](root))
                    assert "unvalidated" in status.lower()
                _export_roundtrip(table)
                shown = display_table(table, decimals=2)
                assert list(shown) == list(table) and len(shown) == len(table)
                # Presentation rounding must not affect the source export.
                assert csv_bytes(table) == csv_bytes(table.copy(deep=True))
                rows.append({"battery_id": battery, "kind": kind,
                             "passed": True, "detail": f"{len(table)} rows reconciled"})
            except Exception as exc:
                rows.append({"battery_id": battery, "kind": kind,
                             "passed": False, "detail": f"{type(exc).__name__}: {exc}"})
    return NumericalReconciliation(pd.DataFrame(rows))
