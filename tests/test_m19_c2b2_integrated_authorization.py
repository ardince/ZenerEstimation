"""M19-C2-B2-C1 integration decision tests.

These are isolated orchestration tests: monkeypatched stage results do not
replace the accepted C1/B2 real-chain tests or independent provenance proof.
"""
from copy import deepcopy
import pytest

from zenerestimation.comparison import benchmark_authorization_integration as m


def _stages(monkeypatch, overrides=None):
    calls = []
    data = {
        "benchmark_scope": {"status": "AUTHORIZATION_REQUIRED", "scope_digest": "scope.1", "scientific_alignment_status": "SCIENTIFICALLY_ALIGNED"},
        "evidence_structure": {"status": "AUTHORIZATION_REQUIRED"},
        "signature_verification": {"status": "AUTHORIZATION_REQUIRED", "signature_verified": True, "issuer_trusted": True},
        "prospective_binding": {"status": "AUTHORIZATION_REQUIRED", "commitment_digest": "commitment.1", "commitment_binding_verified": True},
        "signed_chronology": {"status": "AUTHORIZATION_REQUIRED", "signed_events_verified": True, "trusted_checkpoint_verified": True},
    }
    data.update(overrides or {})
    def patch(fn_name, stage):
        def fake(**kwargs):
            calls.append((stage, kwargs))
            result = data[stage]
            if isinstance(result, Exception):
                raise result
            return deepcopy(result)
        monkeypatch.setattr(m, fn_name, fake)
    patch("assess_benchmark_scope", "benchmark_scope")
    patch("assess_authorization_evidence_structure", "evidence_structure")
    patch("verify_authorization_signature", "signature_verification")
    patch("assess_prospective_commitment_binding", "prospective_binding")
    patch("verify_trusted_authorization_chronology", "signed_chronology")
    return calls


def _args():
    return dict(inputs=["mock.input.1", "mock.input.2"], scope_declaration={"scope": 1},
                authorization_evidence={"evidence": 1}, prospective_commitment={"commitment": 1},
                authorization_registration={"registration": 1},
                protected_holdout_events=[{"access": 1}],
                signature_trust_registry={"signature_registry": 1},
                chronology_trust_registry={"chronology_registry": 1},
                checkpoint={"checkpoint": 1})


def test_all_positive_stages_still_fail_closed(monkeypatch):
    calls = _stages(monkeypatch)
    result = m.assess_independent_benchmark_authorization(**_args())
    assert result["status"] == "AUTHORIZATION_REQUIRED"
    assert result["scientific_alignment_verified"]
    assert result["signature_verified"] and result["commitment_binding_verified"]
    assert result["signed_events_verified"] and result["trusted_checkpoint_verified"]
    assert result["reasons"] == ["INDEPENDENT_TIME_AND_ACCESS_COMPLETENESS_NOT_ESTABLISHED"]
    assert not result["independent_time_provenance_verified"]
    assert not result["access_completeness_verified"]
    assert not result["prospective_authorization_verified"]
    assert not result["comparison_permitted"] and not result["ranking_permitted"]
    assert [name for name, _ in calls] == list(m._STAGE_NAMES)


@pytest.mark.parametrize("stage", list(m._STAGE_NAMES))
def test_any_rejected_stage_rejects(monkeypatch, stage):
    _stages(monkeypatch, {stage: {"status": "AUTHORIZATION_REJECTED"}})
    result = m.assess_independent_benchmark_authorization(**_args())
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert not result["comparison_permitted"]


@pytest.mark.parametrize("stage", list(m._STAGE_NAMES))
def test_any_unexpected_status_rejects(monkeypatch, stage):
    _stages(monkeypatch, {stage: {"status": "AUTHORIZED"}})
    result = m.assess_independent_benchmark_authorization(**_args())
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert result["authorization"] == "not_established"


@pytest.mark.parametrize("stage", list(m._STAGE_NAMES))
def test_stage_validation_exception_rejects(monkeypatch, stage):
    _stages(monkeypatch, {stage: ValueError("invalid")})
    result = m.assess_independent_benchmark_authorization(**_args())
    assert result["status"] == "AUTHORIZATION_REJECTED"


def test_missing_evidence_remains_required(monkeypatch):
    calls = _stages(monkeypatch)
    args = _args()
    args["authorization_evidence"] = None
    args["signature_trust_registry"] = None
    result = m.assess_independent_benchmark_authorization(**args)
    assert result["status"] == "AUTHORIZATION_REQUIRED"
    assert "EXTERNAL_EVIDENCE_MISSING" in result["reasons"]
    assert "signature_verification" not in [stage for stage, _ in calls]


def test_missing_chronology_remains_required(monkeypatch):
    _stages(monkeypatch)
    args = _args()
    args["authorization_registration"] = None
    args["protected_holdout_events"] = None
    args["checkpoint"] = None
    args["chronology_trust_registry"] = None
    result = m.assess_independent_benchmark_authorization(**args)
    assert result["status"] == "AUTHORIZATION_REQUIRED"


def test_present_chronology_without_registry_remains_required(monkeypatch):
    calls = _stages(monkeypatch)
    args = _args()
    args["chronology_trust_registry"] = None
    result = m.assess_independent_benchmark_authorization(**args)
    assert result["status"] == "AUTHORIZATION_REQUIRED"
    assert "CHRONOLOGY_TRUST_REGISTRY_MISSING" in result["reasons"]
    assert "signed_chronology" not in [stage for stage, _ in calls]


def test_signature_and_chronology_use_distinct_registries(monkeypatch):
    calls = _stages(monkeypatch)
    args = _args()
    m.assess_independent_benchmark_authorization(**args)
    records = dict(calls)
    assert records["signature_verification"]["trust_registry"] is args["signature_trust_registry"]
    assert records["signed_chronology"]["trust_registry"] is args["chronology_trust_registry"]


def test_signature_registry_alias(monkeypatch):
    calls = _stages(monkeypatch)
    args = _args()
    registry = args.pop("signature_trust_registry")
    args["trust_registry"] = registry
    m.assess_independent_benchmark_authorization(**args)
    assert dict(calls)["signature_verification"]["trust_registry"] is registry


def test_conflicting_signature_registries_rejected(monkeypatch):
    calls = _stages(monkeypatch)
    args = _args()
    args["trust_registry"] = {"another": "registry"}
    result = m.assess_independent_benchmark_authorization(**args)
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert not calls


def test_deterministic_read_only_and_digest(monkeypatch):
    _stages(monkeypatch)
    args = _args()
    original = deepcopy(args)
    first = m.assess_independent_benchmark_authorization(**args)
    second = m.assess_independent_benchmark_authorization(**args)
    assert first == second
    assert first["decision_digest"] == m._digest({k: v for k, v in first.items() if k != "decision_digest"})
    assert args == original


def test_mismatched_prospective_commitment_rejected(monkeypatch):
    _stages(monkeypatch, {"prospective_binding": {"status": "AUTHORIZATION_REJECTED", "reasons": ["PARTICIPANT_MISMATCH"]}})
    assert m.assess_independent_benchmark_authorization(**_args())["status"] == "AUTHORIZATION_REJECTED"


def test_retrospective_approval_rejected(monkeypatch):
    _stages(monkeypatch, {"signed_chronology": {"status": "AUTHORIZATION_REJECTED", "reasons": ["NON_PROSPECTIVE_APPROVAL"]}})
    assert m.assess_independent_benchmark_authorization(**_args())["status"] == "AUTHORIZATION_REJECTED"


def test_no_model_execution_in_integration(monkeypatch):
    calls = _stages(monkeypatch)
    m.assess_independent_benchmark_authorization(**_args())
    assert len(calls) == 5
