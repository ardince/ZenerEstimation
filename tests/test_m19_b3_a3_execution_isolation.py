"""M19-B3-A3: deterministic execution-seam isolation tests.

These tests exercise the real A2 partition/guard with a controlled model fixture.
They do not replace a genuine M18-C6 execution-artifact integration test.
"""
from copy import deepcopy

import pandas as pd
import pytest

from zenerestimation.assessment.eligibility import (
    AssessmentDeclaration, AssessmentTrack, CONTRACT_VERSION,
)
from zenerestimation.assessment.holdout_isolation import (
    isolate_declared_holdout, require_isolated_training_dataset,
)
from zenerestimation.data.dataset import BatteryDataset


@pytest.fixture
def declaration():
    return AssessmentDeclaration(
        contract_version=CONTRACT_VERSION,
        assessment_id="m19-b3-a3-fixture",
        assessment_track=AssessmentTrack.RETROSPECTIVE,
        dataset_id="410",
        development_cutoff="2023-12-01",
        forecast_dates=("2024-03-01", "2024-06-01"),
    )


@pytest.fixture
def full_frame():
    return pd.DataFrame({
        "ds": pd.to_datetime([
            "2023-03-01", "2023-06-01", "2023-09-01",
            "2023-12-01", "2024-03-01", "2024-06-01",
        ]),
        "microVolt": [27., 28., 29., 29.55, 29.9, 30.506],
    })


class DeterministicForecast:
    """Small controlled fixture, deliberately not an M18 production model."""

    def __init__(self, *, offset):
        self.offset = float(offset)
        self.fit_dates = None
        self.fit_values = None

    def fit(self, dataset):
        frame = dataset.data
        self.fit_dates = tuple(pd.to_datetime(frame["ds"]).dt.strftime("%Y-%m-%d"))
        self.fit_values = tuple(float(v) for v in frame["microVolt"])
        return self

    def predict(self, horizon):
        if self.fit_values is None:
            raise RuntimeError("model not fitted")
        return tuple(self.fit_values[-1] + self.offset * i for i in range(1, horizon + 1))


def _run(frame, declaration, *, params=None):
    isolated = isolate_declared_holdout(BatteryDataset(frame), declaration=declaration)
    # This guard is deliberately at the execution call site, immediately before fit.
    require_isolated_training_dataset(isolated.development, isolated=isolated)
    chosen = deepcopy(params if params is not None else {"offset": 0.25})
    model = DeterministicForecast(**chosen)
    model.fit(isolated.development)
    prediction = model.predict(len(declaration.forecast_dates))
    actual = tuple(float(x) for x in isolated.holdout["microVolt"])
    rmse = (sum((a - p) ** 2 for a, p in zip(actual, prediction)) / len(actual)) ** .5
    return isolated, model, chosen, prediction, rmse


def test_holdout_perturbation_does_not_change_training_or_predictions(full_frame, declaration):
    a = _run(full_frame, declaration)
    changed = full_frame.copy(deep=True)
    changed.loc[changed["ds"] > pd.Timestamp(declaration.development_cutoff), "microVolt"] = [100., -100.]
    b = _run(changed, declaration)
    pd.testing.assert_frame_equal(a[0].development.data, b[0].development.data)
    assert a[1].fit_dates == b[1].fit_dates
    assert a[1].fit_values == b[1].fit_values
    assert a[2] == b[2]
    assert a[3] == b[3]
    assert a[4] != b[4]


def test_training_guard_rejects_full_dataset_before_fit(full_frame, declaration):
    isolated = isolate_declared_holdout(BatteryDataset(full_frame), declaration=declaration)
    model = DeterministicForecast(offset=.25)
    with pytest.raises(ValueError, match="training input differs"):
        require_isolated_training_dataset(BatteryDataset(full_frame), isolated=isolated)
        model.fit(BatteryDataset(full_frame))
    assert model.fit_values is None


def test_training_guard_rejects_mutated_development_before_fit(full_frame, declaration):
    isolated = isolate_declared_holdout(BatteryDataset(full_frame), declaration=declaration)
    modified = isolated.development.data.copy(deep=True)
    modified.loc[0, "microVolt"] = 999.
    model = DeterministicForecast(offset=.25)
    with pytest.raises(ValueError, match="training input differs"):
        require_isolated_training_dataset(BatteryDataset(modified), isolated=isolated)
        model.fit(BatteryDataset(modified))
    assert model.fit_values is None


def test_training_fingerprint_is_invariant_to_holdout_values(
    full_frame, declaration
):
    import hashlib

    def development_digest(dataset):
        frame = dataset.data.copy(deep=True)

        # Deterministic representation of the isolated observations.
        payload = frame.to_csv(
            index=False,
            date_format="%Y-%m-%d",
            float_format="%.17g",
            lineterminator="\n",
        ).encode("utf-8")

        return hashlib.sha256(payload).hexdigest()

    a = isolate_declared_holdout(
        BatteryDataset(full_frame),
        declaration=declaration,
    )

    altered = full_frame.copy(deep=True)
    altered.loc[
        altered["ds"] > pd.Timestamp(declaration.development_cutoff),
        "microVolt",
    ] = [999.0, -999.0]

    b = isolate_declared_holdout(
        BatteryDataset(altered),
        declaration=declaration,
    )

    assert development_digest(a.development) == development_digest(
        b.development
    )

def test_extra_future_row_fails_before_fit(full_frame, declaration):
    extra = pd.DataFrame({"ds": [pd.Timestamp("2024-09-01")], "microVolt": [31.]})
    with pytest.raises(ValueError):
        _run(pd.concat([full_frame, extra], ignore_index=True), declaration)


def test_forecast_uses_exact_declared_horizon(full_frame, declaration):
    isolated, model, params, predictions, rmse = _run(full_frame, declaration)
    assert len(predictions) == len(declaration.forecast_dates)
    assert max(model.fit_dates) == declaration.development_cutoff
    assert min(declaration.forecast_dates) > max(model.fit_dates)
