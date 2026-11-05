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
    "Holdout signed errors (prediction − measured)": "holdout_errors",
    "Regenerated historical fitted values": "historical_fits",
    "Historical fitted signed errors (fitted − measured)": "historical_errors",
    "Unvalidated operational forecasts": "operational",
}


def numerical_result(comparison, repository_root: str | Path, *,
                     battery_id: str, kind: str, models=None):
    """Return (table, scientific status), without changing original sources."""
    if battery_id not in (BATTERY_410, BATTERY_110):
        raise ValueError(f"Unknown battery: {battery_id}")
    if kind not in (*KINDS.values(), "observed"):
        raise ValueError(f"Unknown numerical result: {kind}")
    if models is not None:
        selected = tuple(models)
        if any(model not in MODELS for model in selected):
            raise ValueError("Unknown model selection")
    else:
        selected = MODELS

    source_kind = {"holdout_errors": "holdout", "historical_errors": "historical_fits"}.get(kind, kind)
    if source_kind in ("metrics", "holdout", "historical_fits"):

        result = numerical_table(comparison, battery_id=battery_id,
                                 kind=source_kind, models=selected)
        table = result.data.copy(deep=True)
        if kind == "holdout_errors":
            table = table.loc[:, [c for c in ("battery_id", "date", "model", "error_microvolt", "absolute_error_microvolt", "origin") if c in table]]
            return table, result.status + " Signed error = prediction − measured (µV)."
        if kind == "historical_fits":
            # Attach physical measurements by exact quarter, not interpolated
            # training targets. Preserve missing observations as NaN.
            observed_loader = (load_110_observed_history if battery_id == BATTERY_110
                               else load_410_observed_history)
            observed = observed_loader(repository_root).copy(deep=True)
            if not table.empty:
                table["date"] = pd.to_datetime(table["date"], errors="raise")
                observed["date"] = pd.to_datetime(observed["date"], errors="raise")
                table = table.merge(observed, on="date", how="left", validate="many_to_one")
            return (table, result.status +
                    " Measured (µV) contains physical observations only; "
                    "unobserved quarters remain blank.")
        if kind == "historical_errors":
            observed = (load_110_observed_history if battery_id == BATTERY_110
                        else load_410_observed_history)(repository_root)
            if not table.empty:
                table["date"] = pd.to_datetime(table["date"], errors="raise")
                table = table.merge(observed, on="date", how="left", validate="many_to_one")
                table["error_fitted_minus_measured"] = (pd.to_numeric(table["fitted_microvolt"], errors="coerce")
                                                       - table["measured_microvolt"])
                table = table.loc[:, ["battery_id", "date", "model", "error_fitted_minus_measured", "origin"]]
            return table, result.status + " Signed error = fitted − observed measurement (µV); missing observations are not interpolated actuals."
        return table, result.status

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
    table.to_csv(stream, index=False, float_format=lambda value: repr(float(value)), date_format="%Y-%m-%d")
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


def orient_numerical_table(table: pd.DataFrame, orientation: str) -> pd.DataFrame:
    """Show dates once and models side-by-side, without changing source evidence.

    Metadata (including battery and provenance) remains in the source table and
    export's scientific status; this view is for a single selected battery.
    """
    if orientation not in ("Dates as rows", "Dates as columns"):
        raise ValueError("Unknown table orientation")
    if "date" not in table or table.empty:
        return table.copy(deep=True)

    t = table.copy(deep=True)
    t["date"] = pd.to_datetime(t["date"], errors="raise")
    if t["date"].isna().any():
        raise ValueError("Invalid numerical dates")
    values = [c for c in ("predicted_microvolt", "measured_microvolt", "error_microvolt",
                          "absolute_error_microvolt", "fitted_microvolt",
                          "error_fitted_minus_measured") if c in t]
    if not values:
        # Operational forecasts are already date-indexed, with one column per model.
        if "model" not in t:
            wide = t.drop(columns=["battery_id", "origin"], errors="ignore")
            if wide["date"].duplicated().any():
                raise ValueError("Duplicate numerical dates")
            wide = wide.sort_values("date").reset_index(drop=True)
            wide["date"] = wide["date"].dt.strftime("%Y-%m-%d")
            if orientation == "Dates as rows":
                return wide.rename(columns={"date": "Quarter"})
            return wide.set_index("date").T.rename_axis("Model").reset_index()
        return t

    if "model" not in t:
        if t["date"].duplicated().any():
            raise ValueError("Duplicate observed dates")
        wide = t[["date", *values]].sort_values("date").reset_index(drop=True)
        wide["date"] = wide["date"].dt.strftime("%Y-%m-%d")
        if orientation == "Dates as rows":
            return wide.rename(columns={"date": "Quarter"})
        return wide.set_index("date").T.rename_axis("Numerical value").reset_index()

    if t.duplicated(["date", "model"]).any():
        raise ValueError("Duplicate model/date values")
    if "battery_id" in t and t["battery_id"].nunique(dropna=False) > 1:
        raise ValueError("Cannot mix batteries in one numerical comparison")

    dates = sorted(t["date"].unique())
    # The source adapter sorts by date/model, which can put GRU before LSTM.
    # Use the dashboard's canonical model order for stable side-by-side comparisons.
    present_models = set(t["model"].tolist())
    models = [model for model in MODELS if model in present_models]
    if len(models) != len(present_models):
        raise ValueError("Unknown model in numerical table")
    output = pd.DataFrame({"Quarter": pd.DatetimeIndex(dates).strftime("%Y-%m-%d")})
    if "measured_microvolt" in values:
        counts = t.groupby("date")["measured_microvolt"].nunique(dropna=False)
        if not counts.eq(1).all():
            raise ValueError("Inconsistent measured values across models")
        observed = t.groupby("date")["measured_microvolt"].first()
        output["Measured (µV)"] = observed.reindex(dates).to_numpy()
    names = {
        "predicted_microvolt": "prediction",
        "fitted_microvolt": "fitted",
        "error_microvolt": "prediction − measured",
        "absolute_error_microvolt": "absolute error",
        "error_fitted_minus_measured": "fitted − measured",
    }
    model_values = [v for v in values if v != "measured_microvolt"]
    for value in model_values:
        pivot = t.pivot(index="date", columns="model", values=value).reindex(index=dates, columns=models)
        for model in models:
            label = model if len(model_values) == 1 else f"{model} · {names[value]}"
            output[label] = pivot[model].to_numpy()
    if orientation == "Dates as rows":
        return output
    return output.set_index("Quarter").T.rename_axis("Model / value").reset_index()


def render_numerical_explorer(comparison, repository_root: str | Path) -> None:
    """Interactive read-only tables and source-precision CSV export."""
    import streamlit as st

    st.header("Numerical Results Explorer")
    st.caption("Saved scientific evidence only. Use the six-sheet Excel workbook for export.")
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
    orientation = st.selectbox("Table orientation", ("Dates as rows", "Dates as columns"),
                               key="numerical_orientation")
    kind = KINDS[label]
    selected = list(MODELS)
    selected = st.multiselect("Models", list(MODELS), default=list(MODELS),
                              key="numerical_models")
    table, status = numerical_result(comparison, repository_root,
                                     battery_id=battery, kind=kind, models=selected)
    st.info(status)
    if table.empty:
        st.warning("No structured numerical rows are available for this selection.")
    else:
        table = orient_numerical_table(table, orientation)
        shown = display_table(table, decimals=None if precision == "Full" else precision)
        st.dataframe(shown, hide_index=True, width="stretch")
        st.caption(f"{len(table)} rows · Numerical values shown at selected display precision")
    st.subheader("Saved Benchmarks · Combined Excel evidence")
    st.caption("Six worksheets: overview with benchmark metrics, historical fitted "
               "values and signed errors, holdout predictions and signed errors, "
               "and unvalidated operational forecasts. Each quarter is one row; "
               "only physically measured values appear as actuals.")
    try:
        from zenerestimation.visualization.saved_benchmarks_excel import (
            build_saved_benchmark_sheets, saved_benchmark_excel_bytes,
        )
        sheets = build_saved_benchmark_sheets(
            comparison, repository_root, battery_id=battery, models=selected)
        workbook = saved_benchmark_excel_bytes(sheets, display_precision=precision)
    except ImportError as exc:
        st.warning("Excel export requires openpyxl: python -m pip install openpyxl. "
                   + str(exc))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.warning("Saved Benchmarks Excel export unavailable: " + str(exc))
    else:
        st.download_button(
            "Download Saved Benchmarks six-sheet Excel workbook (.xlsx)",
            data=workbook,
            file_name=f"ZenerEstimation_{battery}_saved_benchmarks.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="saved_benchmarks_combined_excel_export",
        )
