"""Sprint 15.3.5.2: read-only Streamlit numerical results explorer.

Streamlit is imported only inside render_numerical_explorer; pure helpers are
independently testable. CSV exports preserve source float precision.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd

from zenerestimation.visualization.cross_battery_dashboard_numerical import numerical_table
from zenerestimation.visualization.cross_battery_dashboard_sources import (
    BATTERY as BATTERY_110,
    BATTERY_410,
    MODELS,
    load_110_observed_history,
    load_410_observed_history,
    load_110_operational_forecasts,
    load_410_operational_forecasts,
)

KINDS = {
    "Benchmark metrics": "metrics",
    "Frozen holdout predictions and errors": "holdout",
    "Regenerated historical fitted values": "historical_fits",
    "Measured historical observations": "observed",
    "Unvalidated operational forecasts": "operational",
}


def numerical_result(comparison, repository_root: str | Path, *,
                     battery_id: str, kind: str, models=None):
    """Return (table, scientific status), without changing original sources."""
    if battery_id not in (BATTERY_410, BATTERY_110):
        raise ValueError(f"Unknown battery: {battery_id}")
    if kind not in KINDS.values():
        raise ValueError(f"Unknown numerical result: {kind}")
    if models is not None:
        selected = tuple(models)
        if any(model not in MODELS for model in selected):
            raise ValueError("Unknown model selection")
    else:
        selected = MODELS

    if kind in ("metrics", "holdout", "historical_fits"):
        result = numerical_table(comparison, battery_id=battery_id,
                                 kind=kind, models=selected)
        return result.data.copy(deep=True), result.status

    if kind == "observed":
        loader = (load_110_observed_history if battery_id == BATTERY_110
                  else load_410_observed_history)
        table = loader(repository_root).copy(deep=True)
        return table, "Measured observations only; unobserved quarters excluded."

    loader = (load_110_operational_forecasts if battery_id == BATTERY_110
              else load_410_operational_forecasts)
    table = loader(repository_root).copy(deep=True)
    table = table.loc[:, ["date", *selected]].copy()
    return table, ("Unvalidated future operational extrapolations; "
                   "not measured actuals or holdout performance.")


def csv_bytes(table: pd.DataFrame) -> bytes:
    """UTF-8 CSV of original numeric values, independent of display rounding."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError("Expected a DataFrame")
    stream = StringIO()
    table.to_csv(stream, index=False, float_format="%.17g", date_format="%Y-%m-%d")
    return stream.getvalue().encode("utf-8-sig")


def display_table(table: pd.DataFrame, *, decimals: int | None) -> pd.DataFrame:
    """Return a detached, optionally rounded presentation copy."""
    if decimals is not None and (not isinstance(decimals, int) or not 0 <= decimals <= 15):
        raise ValueError("Decimals must be None or an integer between 0 and 15")
    result = table.copy(deep=True)
    if decimals is not None:
        numeric = result.select_dtypes(include="number").columns
        # Do not round integral ranks or other integer identifiers.
        floating = [col for col in numeric if pd.api.types.is_float_dtype(result[col])]
        result.loc[:, floating] = result.loc[:, floating].round(decimals)
    return result


def render_numerical_explorer(comparison, repository_root: str | Path) -> None:
    """Interactive read-only tables and source-precision CSV export."""
    import streamlit as st

    st.header("Numerical Results Explorer")
    st.caption("Saved scientific evidence only. Display rounding never changes CSV exports.")
    left, middle, right = st.columns([2, 3, 2])
    with left:
        battery = st.selectbox("Battery", (BATTERY_410, BATTERY_110),
                               key="numerical_battery")
    with middle:
        label = st.selectbox("Numerical result", tuple(KINDS),
                             key="numerical_kind")
    with right:
        precision = st.selectbox("Display precision", ("Full", 2, 4, 6, 10),
                                 key="numerical_precision")
    kind = KINDS[label]
    selected = list(MODELS)
    if kind != "observed":
        selected = st.multiselect("Models", list(MODELS), default=list(MODELS),
                                  key="numerical_models")
    table, status = numerical_result(comparison, repository_root,
                                     battery_id=battery, kind=kind, models=selected)
    st.info(status)
    if table.empty:
        st.warning("No structured numerical rows are available for this selection.")
        return
    shown = display_table(table, decimals=None if precision == "Full" else precision)
    st.dataframe(shown, hide_index=True, width="stretch")
    st.caption(f"{len(table)} rows · Source values retained for export")
    st.download_button("Download source-precision CSV", data=csv_bytes(table),
                       file_name=f"{battery}_{kind}.csv", mime="text/csv",
                       key="numerical_export")
