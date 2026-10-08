"""Six-quarter operational forecasts, separate from the frozen 410 benchmark.

Run from repository root:
    python -m examples.optimization.run_410_operational_extrapolation
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.data.temporal import TemporalPreprocessor
from examples.optimization.run_410_optimized_benchmark import (
    BATTERY,
    create_model_specs, load_dataset,
)

CUTOFF = pd.Timestamp('2025-06-01')
HORIZON = 6
OUTPUT_DIR = Path('results') / BATTERY / 'operational_extrapolation'
EXPECTED_DATES = pd.date_range('2025-09-01', periods=HORIZON, freq='QS-MAR')
EXPECTED_MODELS = ('ARIMA', 'Kalman', 'LSTM', 'GRU', 'LinearTrendLSTM', 'KalmanLSTM')


def prepare_operational_training(dataset: BatteryDataset) -> BatteryDataset:
    """Apply a strict date cutoff before any preprocessing or model fitting."""
    source = dataset.data.copy(deep=True)
    source['ds'] = pd.to_datetime(source['ds'])
    if source['ds'].isna().any() or source['ds'].duplicated().any():
        raise ValueError('Invalid or duplicate source dates')
    selected = source.loc[source['ds'] <= CUTOFF].sort_values('ds').copy()
    if selected.empty or pd.Timestamp(selected['ds'].iloc[-1]) != CUTOFF:
        raise ValueError('The declared 2025-06-01 observation is missing')
    if pd.isna(selected['microVolt'].iloc[-1]):
        raise ValueError('Cutoff measurement is missing')
    training = BatteryDataset(selected.reset_index(drop=True))
    # Preserve dataset identity and source provenance.
    for attr in ('_battery', '_source_type', '_source_path'):
        if hasattr(dataset, attr):
            setattr(training, attr, getattr(dataset, attr))
    training.metadata = dataset.metadata.copy()
    prepared = TemporalPreprocessor().fit_transform(training)
    prepared.prepare()
    dates = pd.DatetimeIndex(pd.to_datetime(prepared.data['ds']))
    if dates[-1] != CUTOFF or (dates > CUTOFF).any():
        raise ValueError('Training cutoff violation')
    return prepared


def compute_forecasts(training, specs):
    """Fit each precommitted configuration once; never use holdout scoring."""
    results = {}
    for spec in specs:
        model = spec.create_model()
        model.fit(training)
        result = model.predict(steps=HORIZON)
        dates = pd.DatetimeIndex(pd.to_datetime(result.dates))
        values = np.asarray(result.forecast, dtype=float).reshape(-1)
        if result.model != spec.name:
            raise ValueError(f'{spec.name}: model identity mismatch')
        if not dates.equals(EXPECTED_DATES):
            raise ValueError(f'{spec.name}: unexpected forecast dates: {list(dates)}')
        if len(values) != HORIZON or not np.isfinite(values).all():
            raise ValueError(f'{spec.name}: invalid forecast values')
        results[spec.name] = values.copy()
    return results


def build_forecast_table(forecasts):
    """Build the canonical table used by every saved operational artifact."""
    if not forecasts:
        raise ValueError("No operational forecasts were supplied")
    table = pd.DataFrame({"date": EXPECTED_DATES.strftime("%Y-%m-%d")})
    for name, raw_values in forecasts.items():
        values = np.asarray(raw_values, dtype=float).reshape(-1)
        if len(values) != HORIZON or not np.isfinite(values).all():
            raise ValueError(f"{name}: invalid forecast values")
        table[name] = values
    return table


def validate_operational_contract(table, specs, require_expected_models=False):
    """Validate dates/models without introducing evaluation evidence."""
    spec_names = tuple(spec.name for spec in specs)
    if len(spec_names) != len(set(spec_names)):
        raise ValueError("Duplicate model specifications are not allowed")
    if list(table.columns) != ["date", *spec_names]:
        raise ValueError("Forecast table columns do not match model specifications")
    if table["date"].tolist() != list(EXPECTED_DATES.strftime("%Y-%m-%d")):
        raise ValueError("Forecast table dates do not match the operational horizon")
    if require_expected_models and spec_names != EXPECTED_MODELS:
        raise ValueError("Operational run must use the six frozen 410 model configurations in benchmark order")
    values = table.drop(columns="date").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Operational forecast table contains non-finite values")
    if require_expected_models and values.size != HORIZON * len(EXPECTED_MODELS):
        raise ValueError("Operational run must contain exactly 36 predictions")


def _json_safe(value):
    """Convert common NumPy/pandas values into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    return value


def build_provenance(training, table, specs):
    """Build provenance for this unvalidated operational extrapolation."""
    configurations = {s.name: _json_safe(dict(s.params)) for s in specs}
    predictions = {
        name: [float(v) for v in table[name].to_numpy(dtype=float)]
        for name in table.columns if name != "date"
    }
    neural_seeds = {
        name: params["seed"] for name, params in configurations.items()
        if "seed" in params
    }
    return {
        "kind": "operational_extrapolation_not_holdout_evaluation",
        "battery": BATTERY,
        "scientific_status": "unvalidated_future_operational_extrapolation",
        "configuration_source": "frozen_410_optimized_benchmark",
        "model_update_policy": "refit_through_training_cutoff_without_reoptimization",
        "training_cutoff": CUTOFF.strftime("%Y-%m-%d"),
        "training_start": pd.Timestamp(training.data["ds"].iloc[0]).strftime("%Y-%m-%d"),
        "training_rows": int(len(training.data)),
        "forecast_start": EXPECTED_DATES[0].strftime("%Y-%m-%d"),
        "forecast_end": EXPECTED_DATES[-1].strftime("%Y-%m-%d"),
        "horizon_quarters": HORIZON,
        "forecast_dates": list(table["date"]),
        "models": [c for c in table.columns if c != "date"],
        "prediction_count": int(HORIZON * (len(table.columns) - 1)),
        "model_configurations": configurations,
        "neural_seeds": neural_seeds,
        "predictions": predictions,
        "measured_actuals": None,
        "forecast_errors": None,
        "evaluation_metrics": None,
        "uncertainty_intervals": None,
    }


def render_report(table, provenance):
    """Render the report solely from stored operational forecast evidence."""
    lines = [
        "OPERATIONAL EXTRAPOLATION — NOT A HOLDOUT BENCHMARK",
        f"Battery: {provenance['battery']}",
        f"Training cutoff: {provenance['training_cutoff']}",
        f"Forecast period: {provenance['forecast_start']} to {provenance['forecast_end']}",
        f"Forecast horizon: {provenance['horizon_quarters']} quarters",
        "Model configurations: inherited unchanged from frozen 410 benchmark.",
        "Model update policy: refitted through the training cutoff without reoptimization.",
        "Scientific status: unvalidated future operational extrapolation.",
        "No measured actuals, forecast errors, or accuracy metrics exist for these future dates.",
        "",
        table.to_string(index=False, float_format=lambda v: f"{v:.6f}"),
        "",
    ]
    return "\n".join(lines)


def save_outputs(training, forecasts, specs, output_dir=OUTPUT_DIR, require_expected_models=False):
    """Write one internally consistent set of operational artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_forecast_table(forecasts)
    validate_operational_contract(table, specs, require_expected_models=require_expected_models)
    provenance = build_provenance(training, table, specs)

    csv_path = output_dir / "future_extrapolation.csv"
    table.to_csv(csv_path, index=False, float_format="%.12g")
    json_path = output_dir / "future_extrapolation.json"
    json_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    text_path = output_dir / "future_extrapolation.txt"
    text_path.write_text(render_report(table, provenance), encoding="utf-8")

    history = training.data.copy()
    history["ds"] = pd.to_datetime(history["ds"])
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(history["ds"], history["microVolt"], color="black", linewidth=1.7,
            label="Measured / prepared training history")
    for name in table.columns:
        if name == "date":
            continue
        ax.plot(EXPECTED_DATES, table[name].to_numpy(dtype=float), marker="o",
                linestyle="--", linewidth=1.6, label=f"{name} extrapolation")
    ax.axvline(CUTOFF, color="gray", linestyle=":", label="Training cutoff")
    ax.set_xlim(pd.Timestamp("2022-01-01"), EXPECTED_DATES[-1] + pd.DateOffset(months=2))
    ax.set_title("410 — Six-quarter operational voltage extrapolation (unvalidated)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Voltage (µV)")
    ax.set_ylim(25, 35)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    plot_path = output_dir / "future_extrapolation.png"
    fig.savefig(plot_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return {"csv": csv_path, "json": json_path, "report": text_path, "figure": plot_path}


@dataclass(frozen=True)
class OperationalRun:
    training: object
    forecasts: dict
    specs: tuple
    artifact_paths: dict


def run_operational_extrapolation():
    """Execute operational refitting once and return existing evidence."""
    dataset = load_dataset()
    training = prepare_operational_training(dataset)
    specs = create_model_specs()
    forecasts = compute_forecasts(training, specs)
    paths = save_outputs(training, forecasts, specs, require_expected_models=True)
    print('\nOPERATIONAL FORECAST (NOT BENCHMARK RESULTS)')
    for name, path in paths.items():
        print(f'{name}: {path}')
    print((paths['report']).read_text(encoding='utf-8'))

    return OperationalRun(
        training=training,
        forecasts=forecasts,
        specs=tuple(specs),
        artifact_paths=paths,
    )


def main():
    """Preserve standalone CLI behavior."""
    run_operational_extrapolation()


if __name__ == '__main__':
    main()
