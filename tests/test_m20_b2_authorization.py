from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import pytest

from zenerestimation.optimization.authorization import (
    LocalOptimizationCoordinator, OptimizationAuthorizationAuthority,
    OptimizationExecutionGrant, OptimizationExecutionRequest,
)


def request(**changes):
    defaults = dict(dataset_id="732B-5610410", development_fingerprint="sha256:" + "a" * 64,
                    development_cutoff="2023-12-01", model_family="LSTM",
                    strategy="grid", max_trials=5, max_epoch_folds=100,
                    validation_protocol_digest="sha256:" + "b" * 64)
    return OptimizationExecutionRequest(**(defaults | changes))


def test_request_digest_is_deterministic_and_scope_sensitive():
    assert request().digest == request().digest
    assert request().digest != request(strategy="random").digest
    assert request().to_dict()["scope"] == "internal_development_optimization"


@pytest.mark.parametrize("change", [
    {"dataset_id": ""}, {"development_fingerprint": "sha256:bad"},
    {"development_cutoff": "2023-02-30"}, {"model_family": "unknown"},
    {"strategy": "not-supported"}, {"max_trials": True}, {"max_trials": 0},
    {"max_epoch_folds": None}, {"max_epoch_folds": False},
    {"validation_protocol_digest": "bad"}, {"scope": "scientific_certification"},
    {"contract_version": "V0"},
])
def test_invalid_requests_fail_closed(change):
    with pytest.raises((ValueError, TypeError)):
        request(**change)


def test_non_neural_budget_optional():
    assert request(model_family="ARIMA", max_epoch_folds=None).max_epoch_folds is None


def test_issue_verify_consume_and_replay():
    coordinator = LocalOptimizationCoordinator()
    req = request()
    grant = coordinator.approve(req)
    assert coordinator.verify(grant, req) is True
    coordinator.consume(grant, req)
    with pytest.raises(PermissionError):
        coordinator.consume(grant, req)
    with pytest.raises(PermissionError):
        coordinator.verify(grant, req)


def test_mismatched_request_and_cross_authority_rejected():
    a, b = LocalOptimizationCoordinator(), LocalOptimizationCoordinator()
    req = request()
    grant = a.approve(req)
    with pytest.raises(PermissionError):
        a.consume(grant, replace(req, max_trials=6))
    with pytest.raises(PermissionError):
        b.consume(grant, req)
    assert a.verify(grant, req)


def test_fabricated_and_reconstructed_grants_rejected():
    coordinator = LocalOptimizationCoordinator()
    req = request()
    original = coordinator.approve(req)
    forged = OptimizationExecutionGrant(original.grant_id, original.authority_id,
                                        original.request_digest, original.expires_at, "fake")
    with pytest.raises(PermissionError):
        coordinator.consume(forged, req)
    assert "_proof" not in original.to_dict()
    assert coordinator.verify(original, req)


def test_revoke_prevents_use():
    coordinator = LocalOptimizationCoordinator()
    req = request()
    grant = coordinator.approve(req)
    coordinator.revoke(grant)
    with pytest.raises(PermissionError):
        coordinator.consume(grant, req)


def test_expiration_with_injected_clock():
    now = [datetime(2026, 11, 25, tzinfo=timezone.utc)]
    coordinator = LocalOptimizationCoordinator(clock=lambda: now[0])
    req = request()
    grant = coordinator.approve(req, expires_at=now[0] + timedelta(seconds=1))
    now[0] += timedelta(seconds=1)
    with pytest.raises(PermissionError):
        coordinator.consume(grant, req)
    with pytest.raises(ValueError):
        coordinator.approve(req, expires_at=now[0])


def test_concurrent_consume_allows_exactly_one():
    coordinator = LocalOptimizationCoordinator()
    req = request()
    grant = coordinator.approve(req)
    def attempt(_):
        try:
            coordinator.consume(grant, req)
            return True
        except PermissionError:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(attempt, range(16)))
    assert outcomes.count(True) == 1


def test_authority_rejects_issuance_without_coordinator_capability():
    authority = OptimizationAuthorizationAuthority()
    with pytest.raises(PermissionError):
        authority._issue(request(), capability=object())


def test_explicit_approval_does_not_execute_models():
    coordinator = LocalOptimizationCoordinator()
    grant = coordinator.approve(request())
    assert grant.request_digest == request().digest
