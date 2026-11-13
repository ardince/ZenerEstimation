"""M18-C6-B3: independently checked, read-only forecast execution plans.

A PLANNED record is neither execution authorization nor evidence of training.
The caller must independently trust the supplied training_context and selection
inputs; content digests alone do not authenticate their origin.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.data.fingerprint import (
    fingerprint_dataset, verify_dataset_fingerprint,
)
from zenerestimation.optimization.result import OptimizationResult
from zenerestimation.optimization.internal_selection_adapter import prepare_selected_forecast

PLAN_VERSION = "m18_c6_forecast_plan_v1"
TRAINING_SCOPE = "final_training_processed"
_PLAN_FIELDS = frozenset({
    "plan_version", "status", "selected_model", "registry_model",
    "decision_digest", "model_params", "parameters_digest",
    "training_provenance", "forecast_horizon", "forecast_dates", "plan_digest",
})
_CONTEXT_FIELDS = frozenset({"scope", "dataset_fingerprint", "train_start", "train_end"})
_FINGERPRINT_FIELDS = frozenset({"algorithm", "digest", "canonicalization_version", "row_count"})


def _json_value(value: Any, path: str = "value") -> Any:
    """Convert conventional numeric containers into finite, detached JSON data."""
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not isfinite(number):
            raise ValueError(f"non-finite number at {path}")
        return number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, dict):
        if not all(type(key) is str for key in value):
            raise TypeError(f"non-string parameter key at {path}")
        return {key: _json_value(item, f"{path}.{key}") for key, item in value.items()}
    raise TypeError(f"unsupported JSON parameter at {path}: {type(value).__name__}")


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()


def _calendar_day(value: Any, name: str) -> pd.Timestamp:
    if type(value) is str:
        # Only canonical date-only strings are accepted, not datetime/offset strings.
        try:
            parsed = datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"{name} must be a canonical calendar date") from exc
        if parsed.strftime("%Y-%m-%d") != value:
            raise ValueError(f"{name} must use YYYY-MM-DD")
        return pd.Timestamp(parsed)
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
            raise ValueError(f"{name} must be a timezone-naive calendar date")
        return stamp
    raise TypeError(f"{name} must be a calendar date")


def _training_provenance(dataset: Any, context: Any) -> tuple[dict, pd.Timestamp]:
    if not isinstance(dataset, BatteryDataset) or not dataset.is_processed:
        raise TypeError("training_dataset must be a canonical processed BatteryDataset")
    if not isinstance(context, Mapping) or set(context) != _CONTEXT_FIELDS:
        raise ValueError("training_context must have the exact trusted context fields")
    if context["scope"] != TRAINING_SCOPE:
        raise ValueError("unsupported final training scope")
    expected = context["dataset_fingerprint"]
    if not isinstance(expected, Mapping) or set(expected) != _FINGERPRINT_FIELDS:
        raise ValueError("training_context requires a complete canonical fingerprint")

    frame = dataset.data.copy(deep=True)
    if frame.empty or "ds" not in frame.columns:
        raise ValueError("training dataset must contain dated observations")
    dates = [_calendar_day(value, "training ds") for value in frame["ds"]]
    if any(left >= right for left, right in zip(dates, dates[1:])):
        raise ValueError("training timestamps must be strictly increasing")
    start, end = dates[0], dates[-1]
    if (_calendar_day(context["train_start"], "train_start") != start or
            _calendar_day(context["train_end"], "train_end") != end):
        raise ValueError("training boundaries differ from trusted context")

    actual = fingerprint_dataset(dataset).to_dict()
    if not verify_dataset_fingerprint(dataset, expected) or dict(expected) != actual:
        raise ValueError("training dataset does not match trusted fingerprint")
    return {**actual, "train_start": start.strftime("%Y-%m-%d"),
            "train_end": end.strftime("%Y-%m-%d")}, end


def _forecast_calendar(values: Any, horizon: int, train_end: pd.Timestamp) -> list[str]:
    if not isinstance(values, (list, tuple, pd.DatetimeIndex)) or len(values) != horizon:
        raise ValueError("forecast_dates must contain exactly forecast_horizon dates")
    stamps = [_calendar_day(value, f"forecast_dates[{i}]") for i, value in enumerate(values)]
    previous = train_end
    for stamp in stamps:
        if stamp <= previous:
            raise ValueError("forecast dates must be strictly increasing after training")
        previous = stamp
    return [stamp.strftime("%Y-%m-%d") for stamp in stamps]


def build_internal_forecast_plan(
    artifact: Mapping, *, results: Mapping, validation_context: Mapping,
    registry: Any, training_dataset: BatteryDataset, training_context: Mapping,
    forecast_horizon: int, forecast_dates: Any,
) -> dict:
    """Reverify selection and supplied training evidence; never execute models."""
    prepared = prepare_selected_forecast(
        artifact, results=results, validation_context=validation_context,
        registry=registry,
    )
    if type(forecast_horizon) is not int or forecast_horizon < 1:
        raise ValueError("forecast_horizon must be a positive integer")
    selected = prepared["selected_model"]
    if not isinstance(results, Mapping) or selected not in results:
        raise ValueError("selected model missing from trusted optimization results")
    optimization = results[selected]
    if not isinstance(optimization, OptimizationResult) or optimization.model != selected:
        raise ValueError("selected optimization result identity mismatch")
    if optimization.metric != "rmse":
        raise ValueError("selected optimization metric must be rmse")
    matching = [candidate for candidate in optimization.candidates
                if candidate.status == "evaluated" and candidate.params == optimization.best_params
                and candidate.score is not None and
                isfinite(float(candidate.score)) and
                abs(float(candidate.score) - float(optimization.best_score)) <=
                1e-12 * max(1.0, abs(float(optimization.best_score)))]
    if len(matching) != 1:
        raise ValueError("verified winning candidate must resolve uniquely")
    params = _json_value(optimization.best_params, "model_params")
    if not isinstance(params, dict) or not params:
        raise ValueError("winning model parameters must be a nonempty mapping")
    provenance, train_end = _training_provenance(training_dataset, training_context)
    dates = _forecast_calendar(forecast_dates, forecast_horizon, train_end)
    plan = {
        "plan_version": PLAN_VERSION,
        "status": "PLANNED",
        "selected_model": selected,
        "registry_model": prepared["registry_model"],
        "decision_digest": prepared["decision_digest"],
        "model_params": params,
        "parameters_digest": _digest(params),
        "training_provenance": provenance,
        "forecast_horizon": forecast_horizon,
        "forecast_dates": dates,
    }
    plan["plan_digest"] = _digest(plan)
    assert set(plan) == _PLAN_FIELDS
    return deepcopy(plan)


def verify_internal_forecast_plan(
    plan: Mapping, *, artifact: Mapping, results: Mapping,
    validation_context: Mapping, registry: Any,
    training_dataset: BatteryDataset, training_context: Mapping,
    forecast_horizon: int, forecast_dates: Any,
) -> bool:
    """Independently rebuild against trusted inputs; malformed trust inputs raise."""
    expected = build_internal_forecast_plan(
        artifact, results=results, validation_context=validation_context,
        registry=registry, training_dataset=training_dataset,
        training_context=training_context, forecast_horizon=forecast_horizon,
        forecast_dates=forecast_dates,
    )
    return type(plan) is dict and set(plan) == _PLAN_FIELDS and plan == expected
