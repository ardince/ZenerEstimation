"""Generate six-model development-only historical reconstruction evidence.

Run from the repository root:
    python examples/diagnostics/run_16_4_c1_historical_fit.py

The canonical processed CSV preserves three missing quarters. TemporalPreprocessor
interpolates the development-only modeling copy; evaluation uses genuine measurements.
No holdout data are used for fitting or scoring.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.data.temporal import TemporalPreprocessor
from zenerestimation.visualization.historical_fit_evidence import (
    build_historical_fit_evidence,
    common_interval_metrics,
    write_historical_fit_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
CUTOFF = pd.Timestamp("2023-12-01")
BATTERY = "732B-5610410"


def load_specs(benchmark_file: Path):
    module_name = "zener_frozen_benchmark_specs"
    spec = importlib.util.spec_from_file_location(module_name, benchmark_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import benchmark: {benchmark_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Required by @dataclass during import.
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="results/732B-5610410/historical_fit_16_4_c1")
    args = parser.parse_args()

    benchmark_file = ROOT / "examples/optimization/run_410_optimized_benchmark.py"
    source_file = ROOT / "datasets/processed/732B-5610410.csv"
    if args.dry_run:
        print("Canonical processed dataset:", source_file)
        print("Frozen model specifications:", benchmark_file)
        print("Development cutoff:", CUTOFF.date())
        print("Models: ARIMA, Kalman, LSTM, GRU, LinearTrendLSTM, KalmanLSTM")
        print("Missing quarters: model-only temporal preprocessing; exclude from accuracy")
        print("No model fitted or benchmark output modified")
        return

    benchmark = load_specs(benchmark_file)
    # Uses the existing canonical processed dataset, not the raw CSV.
    source = benchmark.load_dataset()
    raw = source.data.copy()
    raw["ds"] = pd.to_datetime(raw["ds"])
    raw = raw.loc[raw["ds"] <= CUTOFF].sort_values("ds").reset_index(drop=True)
    if raw.empty or raw["ds"].iloc[-1] != CUTOFF:
        raise ValueError("Missing expected development cutoff")
    if raw["ds"].duplicated().any():
        raise ValueError("Duplicate development timestamps")

    observed = pd.Series(
        pd.to_numeric(raw["microVolt"], errors="raise").to_numpy(dtype=float),
        index=pd.DatetimeIndex(raw["ds"]), name="observed_microVolt",
    )
    if np.isinf(observed.to_numpy(dtype=float)).any():
        raise ValueError("Observed measurements contain infinity")
    missing_dates = observed.index[observed.isna()].strftime("%Y-%m-%d").tolist()
    print(f"Development quarters: {len(observed)}; missing measurements: {len(missing_dates)}")
    if missing_dates:
        print("Unmeasured quarters:", ", ".join(missing_dates))

    # Fit the repository's temporal preprocessor ONLY to the development partition.
    development_raw = BatteryDataset(raw)
    development = TemporalPreprocessor().fit_transform(development_raw)
    processed = development.data.copy()
    processed_dates = pd.DatetimeIndex(pd.to_datetime(processed["ds"]))
    if not processed_dates.equals(observed.index):
        raise ValueError("Preprocessing changed the development timeline")
    modeled = np.asarray(development.target, dtype=float)
    if len(modeled) != len(observed) or not np.isfinite(modeled).all():
        raise ValueError("Preprocessing did not produce a finite modeling series")
    # Measured values must be unchanged by interpolation.
    mask = observed.notna().to_numpy()
    if not np.allclose(modeled[mask], observed.to_numpy()[mask], rtol=0, atol=1e-10):
        raise ValueError("Preprocessing changed genuine measured values")

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    artifacts = []
    for model_spec in benchmark.create_model_specs():
        print("Fitting development-only:", model_spec.name, flush=True)
        model = model_spec.factory(**model_spec.params)
        model.fit(development)
        fitted = getattr(model, "fitted", None)
        if fitted is None and callable(getattr(model, "fitted_values", None)):
            fitted = model.fitted_values()
        if fitted is None:
            print("  No fitted values exposed; recording unavailable status")
            fitted = pd.Series(np.nan, index=observed.index)
        elif not isinstance(fitted, pd.Series):
            raise TypeError(f"{model_spec.name}: fitted output must be a pandas Series")
        elif isinstance(fitted.index, pd.RangeIndex):
            if not fitted.index.equals(development.data.index) or len(fitted) != len(observed):
                raise ValueError(f"{model_spec.name}: ambiguous row-indexed fitted output")
            fitted = pd.Series(fitted.to_numpy(dtype=float), index=observed.index)

        evidence = build_historical_fit_evidence(
            model_spec.name, observed, fitted,
            cutoff=str(CUTOFF.date()), configuration=dict(model_spec.params),
        )
        if model_spec.name == "ARIMA":
            # Diagnostic only: statsmodels exposes this on some ARIMA wrappers.
            result = getattr(model, "model_fit", None)
            if result is None:
                result = getattr(model, "fitted_model", None)
            mle = getattr(result, "mle_retvals", None)
            evidence["optimizer_convergence"] = (
                {"converged": bool(mle["converged"]), "source": "mle_retvals"}
                if isinstance(mle, dict) and "converged" in mle
                else {"converged": None, "source": "unavailable"}
            )
        evidence["preprocessing"] = {
            "source": "canonical_processed_csv",
            "modeling_transform": "TemporalPreprocessor.fit_transform(development_only)",
            "missing_measurement_dates": missing_dates,
            "interpolated_measurements_excluded_from_accuracy": True,
        }
        # Recompute checksum after adding provenance.
        import hashlib
        evidence.pop("sha256_payload", None)
        canonical = json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False)
        evidence["sha256_payload"] = hashlib.sha256(canonical.encode()).hexdigest()
        path = write_historical_fit_evidence(evidence, output / f"{model_spec.name}_historical_fit.json")
        artifacts.append(evidence)
        print(" ", path, evidence["metrics"])

    common = common_interval_metrics(artifacts)
    output.mkdir(parents=True, exist_ok=True)
    (output / "common_interval_metrics.json").write_text(
        json.dumps(common, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("Common fitted dates:", len(common["dates"]))
    print("Common comparison models:", ", ".join(common["included_models"]) or "none")
    for name, reason in common["excluded_models"].items():
        print(f"Excluded from common comparison: {name} ({reason})")
    print("IMPORTANT: historical reconstruction metrics are not external forecast accuracy.")


if __name__ == "__main__":
    main()
