"""Read-only, evidence-aware loader for frozen cross-battery benchmark artifacts.

No forecasting/model modules are imported. All paths are resolved under an
explicit repository root. Source artifacts are never written or regenerated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import math

import pandas as pd

MODELS = ("ARIMA", "Kalman", "LSTM", "GRU", "LinearTrendLSTM", "KalmanLSTM")
METRICS = ("rmse", "mae", "mape")


class ArtifactContractError(ValueError):
    """An artifact is missing, malformed, or contradicts frozen evidence."""


@dataclass(frozen=True)
class EvidenceAvailability:
    structured: bool
    visual: bool
    source: str | None
    origin: str


@dataclass(frozen=True)
class BatteryBenchmarkRecord:
    battery_id: str
    benchmark_type: str
    development_end: str
    holdout_start: str
    holdout_end: str
    holdout_steps: int
    metrics: pd.DataFrame
    rankings: dict[str, list[dict[str, Any]]]
    provenance: dict[str, Any]
    holdout_predictions: pd.DataFrame | None
    historical_fits: pd.DataFrame | None
    evidence: dict[str, EvidenceAvailability]
    source_hashes: dict[str, str]


def _strict_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactContractError(f"Missing artifact: {path}")
    def reject_constant(value: str) -> None:
        raise ArtifactContractError(f"Non-finite JSON token {value} in {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(f"Invalid JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ArtifactContractError(f"Expected JSON object: {path}")
    return data


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _date(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ArtifactContractError(f"Expected ISO date for {context}")
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ArtifactContractError(f"Invalid date for {context}: {value}") from exc
    if stamp.strftime("%Y-%m-%d") != value:
        raise ArtifactContractError(f"Noncanonical date for {context}: {value}")
    return value


def _positive_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArtifactContractError(f"Expected positive integer for {context}")
    return value


def _metrics_and_rankings(benchmark: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    raw = benchmark.get("metrics")
    ranks = benchmark.get("rankings")
    if not isinstance(raw, dict) or set(raw) != set(MODELS):
        raise ArtifactContractError("Benchmark metrics must contain exactly six known models")
    rows = []
    for model in MODELS:
        vals = raw[model]
        if not isinstance(vals, dict) or not set(METRICS).issubset(vals):
            raise ArtifactContractError(f"Missing metrics for {model}")
        row = {"model": model}
        for metric in METRICS:
            value = vals[metric]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ArtifactContractError(f"Invalid {metric} for {model}")
            row[metric] = float(value)
        rows.append(row)
    if not isinstance(ranks, dict):
        raise ArtifactContractError("Missing benchmark rankings")
    for metric in METRICS:
        entries = ranks.get(metric)
        if not isinstance(entries, list) or len(entries) != len(MODELS):
            raise ArtifactContractError(f"Invalid ranking length for {metric}")
        if {entry.get("model") for entry in entries if isinstance(entry, dict)} != set(MODELS):
            raise ArtifactContractError(f"Invalid ranked models for {metric}")
        if sorted(entry.get("rank") for entry in entries) != list(range(1, len(MODELS) + 1)):
            raise ArtifactContractError(f"Invalid ranks for {metric}")
        expected = sorted(MODELS, key=lambda m: (raw[m][metric], m))
        actual = [entry["model"] for entry in sorted(entries, key=lambda e: e["rank"])]
        # Permit tied scores in either order; check monotonic values and identity.
        values = [float(raw[m][metric]) for m in actual]
        if any(a > b + 1e-10 for a, b in zip(values, values[1:])):
            raise ArtifactContractError(f"Rank ordering contradicts metrics: {metric}")
        for entry in entries:
            val = entry.get("value")
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ArtifactContractError(f"Invalid ranked value for {metric}")
            if not math.isclose(float(val), float(raw[entry["model"]][metric]), rel_tol=1e-10, abs_tol=1e-10):
                raise ArtifactContractError(f"Ranking value contradicts metrics: {metric}")
    metadata = benchmark.get("metadata", {})
    if not isinstance(metadata, dict) or metadata.get("lower_is_better") is not True:
        raise ArtifactContractError("Expected lower_is_better=true")
    if set(metadata.get("comparison_metrics", [])) != set(METRICS):
        raise ArtifactContractError("Unexpected comparison metrics")
    return pd.DataFrame(rows).set_index("model"), ranks


def _read_table(path: Path, required: tuple[str, ...], *, expected_rows: int | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise ArtifactContractError(f"Missing artifact: {path}")
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ArtifactContractError(f"Unreadable CSV {path}: {exc}") from exc
    if tuple(df.columns) != required:
        raise ArtifactContractError(f"Unexpected columns in {path}: {list(df.columns)}")
    if expected_rows is not None and len(df) != expected_rows:
        raise ArtifactContractError(f"Unexpected row count in {path}: {len(df)}")
    dates = [_date(v, f"{path.name}.date") for v in df["date"]]
    if dates != sorted(set(dates)):
        raise ArtifactContractError(f"Dates must be unique and increasing: {path}")
    for col in required[1:]:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ArtifactContractError(f"Non-numeric column {col} in {path}") from exc
        if df[col].notna().any() and not df[col].dropna().map(math.isfinite).all():
            raise ArtifactContractError(f"Non-finite column {col} in {path}")
    return df


class CrossBatteryArtifactLoader:
    """Load frozen benchmark artifacts from an explicit repository root."""

    def __init__(self, repository_root: str | Path = ".") -> None:
        self.root = Path(repository_root).resolve()

    def load(self, battery_id: str) -> BatteryBenchmarkRecord:
        if battery_id not in ("732B-5610410", "732B-5610110"):
            raise ArtifactContractError(f"Unsupported battery: {battery_id}")
        is_110 = battery_id == "732B-5610110"
        kind = "transferred" if is_110 else "optimized"
        folder = "transferred_benchmark" if is_110 else "optimized_benchmark"
        base = self.root / "results" / battery_id / folder
        benchmark_path = base / "benchmark.json"
        benchmark = _strict_json(benchmark_path)
        if benchmark.get("battery") != battery_id:
            raise ArtifactContractError("Benchmark battery identity mismatch")
        metrics, rankings = _metrics_and_rankings(benchmark)
        hashes = {str(benchmark_path.relative_to(self.root)): _hash(benchmark_path)}
        metrics_csv = base / "benchmark.csv"
        if not metrics_csv.is_file():
            raise ArtifactContractError(f"Missing benchmark CSV: {metrics_csv}")
        try:
            csv = pd.read_csv(metrics_csv)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise ArtifactContractError(f"Invalid benchmark CSV: {exc}") from exc
        if list(csv.columns) != ["model", *METRICS] or set(csv["model"]) != set(MODELS) or len(csv) != len(MODELS):
            raise ArtifactContractError("Benchmark CSV model/column mismatch")
        for row in csv.itertuples(index=False):
            for metric in METRICS:
                if not math.isclose(float(getattr(row, metric)), metrics.loc[row.model, metric], rel_tol=1e-10, abs_tol=1e-10):
                    raise ArtifactContractError(f"Benchmark CSV/JSON mismatch: {row.model}/{metric}")
        hashes[str(metrics_csv.relative_to(self.root))] = _hash(metrics_csv)
        evidence: dict[str, EvidenceAvailability] = {
            "metrics": EvidenceAvailability(True, False, str(metrics_csv.relative_to(self.root)), "frozen_benchmark"),
            "rankings": EvidenceAvailability(True, False, str(benchmark_path.relative_to(self.root)), "frozen_benchmark"),
        }
        predictions = fits = None
        if is_110:
            original = base / "transfer_provenance.json"
            validated = base / "transfer_provenance_validated.json"
            # Original is intentionally malformed; the validated derivative is explicit.
            provenance = _strict_json(validated)
            if not original.is_file():
                raise ArtifactContractError("Missing original provenance source")
            if provenance.get("battery") != battery_id or provenance.get("experiment_type") != "frozen_cross_battery_parameter_transfer":
                raise ArtifactContractError("Unexpected transferred provenance")
            development_end = _date(provenance.get("training_end"), "training_end")
            dates = [_date(x, "holdout_dates") for x in provenance.get("holdout_dates", [])]
            steps = _positive_integer(provenance.get("holdout_steps"), "holdout_steps")
            if len(dates) != steps or dates != sorted(set(dates)) or dates[0] <= development_end:
                raise ArtifactContractError("Inconsistent transferred holdout protocol")
            pred_path = base / "holdout_predictions_full_precision.csv"
            predictions = _read_table(pred_path, ("date", "measured_microVolt", *MODELS), expected_rows=steps)
            if predictions["date"].tolist() != dates or predictions[list(MODELS)].isna().any().any() or predictions["measured_microVolt"].isna().any():
                raise ArtifactContractError("Transferred predictions mismatch holdout dates or contain gaps")
            diag_base = self.root / "results" / battery_id / "historical_fit_diagnostics"
            fits_path = diag_base / "historical_fitted_values.csv"
            fits = _read_table(fits_path, ("date", *MODELS))
            if len(fits) == 0 or fits["date"].iloc[-1] != development_end or fits["date"].iloc[0] > development_end:
                raise ArtifactContractError("Historical fitted range does not match development cutoff")
            diag_path = diag_base / "diagnostic_provenance.json"
            diagnostic = _strict_json(diag_path)
            if diagnostic.get("status") != "regenerated_historical_fit_diagnostic_not_frozen_benchmark":
                raise ArtifactContractError("Historical fit provenance must identify regenerated diagnostic")
            if diagnostic.get("benchmark_predictions_sha256") != _hash(pred_path):
                raise ArtifactContractError("Frozen predictions hash contradicts diagnostic provenance")
            for p in (original, validated, pred_path, fits_path, diag_path):
                hashes[str(p.relative_to(self.root))] = _hash(p)
            evidence["holdout_predictions"] = EvidenceAvailability(True, True, str(pred_path.relative_to(self.root)), "frozen_transferred_benchmark")
            evidence["historical_fits"] = EvidenceAvailability(True, True, str(fits_path.relative_to(self.root)), "regenerated_diagnostic_not_frozen")
            plot_base = diag_base
        else:
            prov_path = base / "provenance.json"
            provenance = _strict_json(prov_path)
            if provenance.get("battery") != battery_id:
                raise ArtifactContractError("Optimized provenance battery mismatch")
            protocol = provenance.get("metadata", {})
            if not isinstance(protocol, dict):
                raise ArtifactContractError("Missing optimized benchmark protocol")
            development_end = _date(protocol.get("development_end"), "development_end")
            dates = [_date(protocol.get("benchmark_start"), "benchmark_start"), _date(protocol.get("benchmark_end"), "benchmark_end")]
            steps = _positive_integer(protocol.get("benchmark_steps"), "benchmark_steps")
            if dates[0] <= development_end or dates[-1] < dates[0]:
                raise ArtifactContractError("Invalid optimized holdout window")
            hashes[str(prov_path.relative_to(self.root))] = _hash(prov_path)
            # The frozen 410 benchmark persisted plots and metrics, not numerical
            # trajectories. Optional later diagnostics are a separate evidence tier.
            diag_base = self.root / "results" / battery_id / "historical_fit_diagnostics"
            diag_path = diag_base / "diagnostic_provenance.json"
            if diag_path.is_file():
                diagnostic = _strict_json(diag_path)
                if diagnostic.get("status") != "regenerated_410_diagnostic_not_frozen_benchmark":
                    raise ArtifactContractError("Unexpected 410 diagnostic status")
                if diagnostic.get("battery") != battery_id or diagnostic.get("frozen_benchmark_modified") is not False:
                    raise ArtifactContractError("410 diagnostic identity or frozen-status mismatch")
                if _date(diagnostic.get("development_end"), "410 diagnostic development_end") != development_end:
                    raise ArtifactContractError("410 diagnostic development cutoff mismatch")
                if diagnostic.get("development_quarters") != 104:
                    raise ArtifactContractError("Unexpected 410 diagnostic development length")
                expected_dates = pd.date_range(dates[0], periods=steps, freq="QS-MAR").strftime("%Y-%m-%d").tolist()
                if expected_dates[-1] != dates[-1] or diagnostic.get("holdout_dates") != expected_dates:
                    raise ArtifactContractError("410 diagnostic holdout calendar mismatch")
                if diagnostic.get("models") != list(MODELS):
                    raise ArtifactContractError("410 diagnostic model identity mismatch")
                frozen_hashes = diagnostic.get("frozen_benchmark_sha256")
                if not isinstance(frozen_hashes, dict):
                    raise ArtifactContractError("Missing frozen benchmark hashes in 410 diagnostic")
                for filename in ("benchmark.csv", "benchmark.json", "provenance.json"):
                    if frozen_hashes.get(filename) != _hash(base / filename):
                        raise ArtifactContractError(f"410 diagnostic frozen source hash mismatch: {filename}")
                artifacts = diagnostic.get("artifacts")
                if not isinstance(artifacts, dict) or artifacts.get("historical_fitted_values") != "historical_fitted_values.csv" or artifacts.get("regenerated_holdout_predictions") != "regenerated_holdout_predictions.csv" or artifacts.get("metric_reconciliation") != "metric_reconciliation.csv":
                    raise ArtifactContractError("Unexpected 410 diagnostic artifact references")
                if diagnostic.get("all_metrics_within_tolerance") is not True:
                    raise ArtifactContractError("410 diagnostic reports failed metric reconciliation")
                pred_path = diag_base / "regenerated_holdout_predictions.csv"
                fit_path = diag_base / "historical_fitted_values.csv"
                predictions = _read_table(pred_path, ("date", "measured_microVolt", *MODELS), expected_rows=steps)
                fits = _read_table(fit_path, ("date", *MODELS), expected_rows=104)
                if predictions["date"].tolist() != expected_dates or predictions.iloc[:, 1:].isna().any().any():
                    raise ArtifactContractError("410 regenerated holdout dates or predictions invalid")
                expected_history = pd.date_range(end=development_end, periods=104, freq="QS-MAR").strftime("%Y-%m-%d").tolist()
                if fits["date"].tolist() != expected_history:
                    raise ArtifactContractError("410 regenerated historical dates invalid")
                # Validate predictions independently against authoritative frozen
                # aggregate metrics; do not trust the diagnostic's PASS flag alone.
                actual = predictions["measured_microVolt"].to_numpy(dtype=float)
                if (actual == 0).any():
                    raise ArtifactContractError("Zero measured value in 410 diagnostic")
                import numpy as np
                for model in MODELS:
                    error = predictions[model].to_numpy(dtype=float) - actual
                    derived = {
                        "rmse": float(np.sqrt(np.mean(error ** 2))),
                        "mae": float(np.mean(np.abs(error))),
                        "mape": float(np.mean(np.abs(error / actual)) * 100),
                    }
                    for metric in METRICS:
                        if not math.isclose(derived[metric], float(metrics.loc[model, metric]), rel_tol=0, abs_tol=1e-6):
                            raise ArtifactContractError(f"410 diagnostic metric mismatch: {model}/{metric}")
                for path in (diag_path, pred_path, fit_path):
                    hashes[str(path.relative_to(self.root))] = _hash(path)
                evidence["holdout_predictions"] = EvidenceAvailability(True, True, str(pred_path.relative_to(self.root)), "regenerated_diagnostic_not_frozen")
                evidence["historical_fits"] = EvidenceAvailability(True, True, str(fit_path.relative_to(self.root)), "regenerated_diagnostic_not_frozen")
            else:
                evidence["holdout_predictions"] = EvidenceAvailability(False, True, None, "frozen_benchmark_plot_only")
                evidence["historical_fits"] = EvidenceAvailability(False, True, None, "frozen_benchmark_plot_only")
            plot_base = base
        for name in ("benchmark_history.png", "benchmark_best_worst_history.png", "benchmark_forecast.png"):
            path = plot_base / name
            if name == "benchmark_forecast.png" and is_110:
                path = base / name
            if not path.is_file():
                raise ArtifactContractError(f"Missing expected plot: {path}")
            key = name.removesuffix(".png")
            evidence[key] = EvidenceAvailability(False, True, str(path.relative_to(self.root)), "diagnostic_visualization" if is_110 and name != "benchmark_forecast.png" else "frozen_benchmark_visualization")
        return BatteryBenchmarkRecord(
            battery_id=battery_id, benchmark_type=kind, development_end=development_end,
            holdout_start=dates[0], holdout_end=dates[-1], holdout_steps=steps,
            metrics=metrics, rankings=rankings, provenance=provenance,
            holdout_predictions=predictions, historical_fits=fits,
            evidence=evidence, source_hashes=hashes,
        )

    def load_both(self) -> dict[str, BatteryBenchmarkRecord]:
        return {battery: self.load(battery) for battery in ("732B-5610410", "732B-5610110")}
