"""Read-only numerical explorer adapter for CrossBatteryComparison."""
from dataclasses import dataclass
import pandas as pd

BATTERIES = ("732B-5610410", "732B-5610110")
SOURCES = {"metrics": "model_metrics", "holdout": "holdout_predictions",
           "historical_fits": "historical_fits"}
REQUIRED = {
    "metrics": {"battery_id", "model", "rmse", "mae", "mape", "rmse_rank"},
    "holdout": {"battery_id", "date", "model", "measured_microvolt",
                "predicted_microvolt", "origin"},
    "historical_fits": {"battery_id", "date", "model", "fitted_microvolt", "origin"},
}

@dataclass(frozen=True)
class NumericalTable:
    available: bool
    status: str
    data: pd.DataFrame

def numerical_table(comparison, *, battery_id, kind, models=None):
    """Return detached, unrounded source values; never reconstruct absent evidence."""
    if battery_id not in BATTERIES or kind not in SOURCES:
        raise ValueError("Unknown battery or numerical table kind")
    source = getattr(comparison, SOURCES[kind])
    if not isinstance(source, pd.DataFrame) or not REQUIRED[kind] <= set(source):
        raise ValueError("Numerical source schema mismatch")
    data = source.loc[source["battery_id"].eq(battery_id)].copy(deep=True)
    if models is not None:
        data = data.loc[data["model"].isin(tuple(models))].copy()
    if kind == "holdout" and not data.empty:
        expected_origin = ("regenerated_diagnostic_not_frozen"
                           if battery_id == BATTERIES[0] else "frozen_transferred_benchmark")
        if not data["origin"].eq(expected_origin).all():
            raise ValueError("Holdout predictions have unexpected provenance")
        data["error_microvolt"] = (data["predicted_microvolt"]
                                  - data["measured_microvolt"])
        data["absolute_error_microvolt"] = data["error_microvolt"].abs()
    if kind == "historical_fits" and not data.empty:
        if not data["origin"].eq("regenerated_diagnostic_not_frozen").all():
            raise ValueError("Historical fits have unexpected provenance")
    if data.empty:
        status = ("Structured evidence unavailable; do not reconstruct from images."
                  if battery_id == BATTERIES[0] and kind != "metrics"
                  else "No rows for this selection.")
        return NumericalTable(False, status, data.reset_index(drop=True))
    status = {"metrics": "Saved benchmark metrics; protocols differ by battery.",
              "holdout": ("Regenerated diagnostic predictions, NOT original frozen evidence; errors derived."
                          if battery_id == BATTERIES[0] else "Frozen transferred predictions; errors derived."),
              "historical_fits": "Regenerated diagnostics, not frozen evidence."}[kind]
    sort = [c for c in ("date", "rmse_rank", "model") if c in data]
    if sort:
        data = data.sort_values(sort, kind="stable")
    return NumericalTable(True, status, data.reset_index(drop=True))
