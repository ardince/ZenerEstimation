"""M18-C6-C3: opt-in, verified, in-memory final forecast execution.

The caller asserts execution authorization; this is not authentication of a
person. The registry resolves a class by name but does not pin its code.
No external holdout data is accepted by this API.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import pandas as pd

from zenerestimation.data.fingerprint import fingerprint_dataset
from zenerestimation.forecasting.base import BaseForecastModel
from zenerestimation.forecasting.models import ModelRegistry
from zenerestimation.forecasting.result import ForecastResult
from zenerestimation.optimization.internal_forecast_plan import (
    verify_internal_forecast_plan,
)

EXECUTION_VERSION = "m18_c6_execution_v1"


def _validate_result(result: object, plan: Mapping) -> None:
    """Check result values, identity, dates, and series index independently."""
    if not isinstance(result, ForecastResult):
        raise TypeError("predict must return ForecastResult")
    if type(result.model) is not str or result.model != plan["selected_model"]:
        raise ValueError("forecast result model differs from selected model")
    horizon = plan["forecast_horizon"]
    if type(result.horizon) is not int or result.horizon != horizon:
        raise ValueError("forecast result horizon differs from approved horizon")

    expected = pd.DatetimeIndex(plan["forecast_dates"])
    if not isinstance(result.dates, pd.DatetimeIndex) or not result.dates.equals(expected):
        raise ValueError("forecast result dates differ from approved calendar")
    if not isinstance(result.forecast, pd.Series) or len(result.forecast) != horizon:
        raise ValueError("forecast must be a Series of the approved length")
    if not isinstance(result.forecast.index, pd.DatetimeIndex) or not result.forecast.index.equals(expected):
        raise ValueError("forecast Series index differs from approved calendar")
    try:
        values = result.forecast.to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("forecast values must be numeric") from exc
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("forecast values must be finite")


def execute_internal_forecast_plan(
    plan: Mapping,
    *,
    artifact: Mapping,
    results: Mapping,
    validation_context: Mapping,
    registry: ModelRegistry,
    training_dataset,
    training_context: Mapping,
    forecast_horizon: int,
    forecast_dates,
    execution_authorized: bool = False,
) -> dict:
    """Execute only a reverified plan under explicit, separate opt-in.

    Model exceptions propagate. A successful response is returned only after
    prediction validation and training-mutation checks. No persistence occurs.
    """
    if execution_authorized is not True:
        raise PermissionError("explicit boolean execution authorization is required")

    verified = verify_internal_forecast_plan(
        plan,
        artifact=artifact,
        results=results,
        validation_context=validation_context,
        registry=registry,
        training_dataset=training_dataset,
        training_context=training_context,
        forecast_horizon=forecast_horizon,
        forecast_dates=forecast_dates,
    )
    if verified is not True:
        raise ValueError("forecast plan does not match independently trusted inputs")

    if not isinstance(registry, ModelRegistry):
        raise TypeError("registry must be a ModelRegistry")
    model_cls = registry.get(plan["registry_model"])
    if not isinstance(model_cls, type) or not issubclass(model_cls, BaseForecastModel):
        raise TypeError("selected registry entry must be a forecast model class")

    # Snapshot and detach both constructor arguments and final training inputs.
    parameters = deepcopy(plan["model_params"])
    training_copy = deepcopy(training_dataset)
    original_frame = training_copy.data.copy(deep=True)
    original_fingerprint = fingerprint_dataset(training_copy).to_dict()
    original_metadata = deepcopy(training_copy.metadata)
    original_battery = deepcopy(training_copy.battery)
    original_observed_mask = deepcopy(training_copy.observed_mask)

    model = model_cls(**parameters)
    model.fit(training_copy)

    # A fit-time mutation cannot be allowed to feed an unapproved prediction.
    if (fingerprint_dataset(training_copy).to_dict() != original_fingerprint
            or not training_copy.data.equals(original_frame)
            or training_copy.metadata != original_metadata
            or training_copy.battery != original_battery
            or not np.array_equal(training_copy.observed_mask, original_observed_mask)):
        raise ValueError("model fit modified approved final training data")

    result = model.predict(plan["forecast_horizon"])
    _validate_result(result, plan)

    # Return a detached result so the model's retained mutable Series cannot
    # silently modify the execution response after validation.
    result = deepcopy(result)
    return {
        "execution_version": EXECUTION_VERSION,
        "status": "EXECUTED",
        "selected_model": plan["selected_model"],
        "plan_digest": plan["plan_digest"],
        "decision_digest": plan["decision_digest"],
        "forecast_result": result,
        "execution_evidence": {
            "execution_authorized": True,
            "model_params_digest": plan["parameters_digest"],
            "training_fingerprint": deepcopy(training_context["dataset_fingerprint"]),
            "forecast_horizon": plan["forecast_horizon"],
            "forecast_dates": list(plan["forecast_dates"]),
            "result_validated": True,
        },
    }
