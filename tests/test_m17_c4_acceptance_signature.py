
"""M17-C4-E: acceptance signature and stale-result safeguards."""

from copy import deepcopy

import pytest

from zenerestimation.visualization.m17_accuracy_acceptance_view import (
    build_acceptance_signature,
)
from tests.test_m17_c4_accuracy_acceptance_view import make_profiles


def signature(profiles, models, threshold=0.5):
    return build_acceptance_signature(
        profiles,
        models=models,
        threshold_microVolt=threshold,
    )


def test_signature_is_deterministic():
    profiles = make_profiles()
    assert signature(profiles, ["ModelA", "ModelB"]) == signature(
        deepcopy(profiles), ["ModelA", "ModelB"]
    )


def test_signature_changes_when_threshold_changes():
    profiles = make_profiles()
    assert signature(profiles, ["ModelA"], 0.5) != signature(
        profiles, ["ModelA"], 1.0
    )


def test_signature_changes_when_model_selection_changes():
    profiles = make_profiles()
    assert signature(profiles, ["ModelA"]) != signature(
        profiles, ["ModelB"]
    )


def test_signature_changes_when_selection_order_changes():
    profiles = make_profiles()
    assert signature(
        profiles, ["ModelA", "ModelB"]
    ) != signature(
        profiles, ["ModelB", "ModelA"]
    )


def test_signature_changes_when_evidence_changes():
    profiles = make_profiles()
    changed = deepcopy(profiles)

    changed["ModelA"]["by_date"][0]["predicted_microVolt"] += 0.01

    assert signature(profiles, ["ModelA"]) != signature(
        changed, ["ModelA"]
    )


def test_signature_does_not_mutate_profiles():
    profiles = make_profiles()
    original = deepcopy(profiles)

    signature(profiles, ["ModelA"])

    assert profiles == original


@pytest.mark.parametrize(
    "threshold",
    [-0.1, float("nan"), float("inf"), True, "0.5"],
)
def test_signature_rejects_invalid_threshold(threshold):
    with pytest.raises((TypeError, ValueError)):
        signature(make_profiles(), ["ModelA"], threshold)


def test_signature_rejects_duplicate_models():
    with pytest.raises(ValueError):
        signature(make_profiles(), ["ModelA", "ModelA"])


def test_signature_rejects_unknown_model():
    with pytest.raises(ValueError):
        signature(make_profiles(), ["Unknown"])
