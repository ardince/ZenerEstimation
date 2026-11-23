"""M19-B3-B4 deterministic evaluation evidence focused contracts."""
from copy import deepcopy
import json
import pytest

from zenerestimation.assessment.holdout_error_scoring import score_disclosed_holdout
from zenerestimation.assessment.holdout_target_disclosure import _digest
from zenerestimation.assessment.holdout_evaluation_evidence import (
    EVIDENCE_VERSION, build_holdout_evaluation_evidence, verify_holdout_evaluation_evidence,
)
from tests.test_m19_b3_b3_holdout_error_scoring import case  # noqa: F401


@pytest.fixture
def chain(case):
    score = score_disclosed_holdout(**case)
    return {**case, "scoring_snapshot": score}


def test_build_and_verify(chain):
    result = build_holdout_evaluation_evidence(**chain)
    assert result["evidence_version"] == EVIDENCE_VERSION
    assert result["status"] == "HOLDOUT_EVALUATION_EVIDENCE_VERIFIED"
    assert result["scientific_independence"] == "not_established"
    assert result["authentication"] == "not_established"
    assert result["metrics"] == chain["scoring_snapshot"]["metrics"]
    assert result["by_date"] == chain["scoring_snapshot"]["by_date"]
    assert result["evaluation_evidence_digest"] == _digest({k: v for k, v in result.items() if k != "evaluation_evidence_digest"})
    assert verify_holdout_evaluation_evidence(result, **chain)


def test_deterministic_and_detached(chain):
    first = build_holdout_evaluation_evidence(**chain)
    second = build_holdout_evaluation_evidence(**chain)
    assert first == second
    json.dumps(first, allow_nan=False)
    first["by_date"][0]["signed_error_microVolt"] = 999
    assert second["by_date"][0]["signed_error_microVolt"] != 999
    assert chain["scoring_snapshot"]["by_date"][0]["signed_error_microVolt"] != 999


@pytest.mark.parametrize("path,value", [
    (("metrics", "rmse_microVolt"), 999.0),
    (("metrics", "mae_microVolt"), 999.0),
    (("metrics", "bias_microVolt"), 999.0),
    (("denominators", "rmse"), 1),
    (("by_date", 0, "signed_error_microVolt"), 999.0),
    (("by_date", 0, "target_date"), "2024-09-01"),
    (("forecast_dates", 0), "2024-09-01"),
    (("execution_artifact_digest",), "other"),
    (("mape_zero_policy",), "wrong"),
    (("scientific_independence",), "established"),
])
def test_redigested_scoring_tamper_rejected(chain, path, value):
    target = chain["scoring_snapshot"]
    cursor = target
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    target["scoring_digest"] = _digest({k: v for k, v in target.items() if k != "scoring_digest"})
    with pytest.raises(ValueError, match="reconstructed"):
        build_holdout_evaluation_evidence(**chain)


def test_scoring_digest_tamper_rejected(chain):
    chain["scoring_snapshot"]["scoring_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="digest"):
        build_holdout_evaluation_evidence(**chain)


def test_extra_scoring_key_rejected(chain):
    chain["scoring_snapshot"]["unexpected"] = True
    with pytest.raises(ValueError, match="schema"):
        build_holdout_evaluation_evidence(**chain)


def test_nonfinite_scoring_rejected(chain):
    chain["scoring_snapshot"]["metrics"]["rmse_microVolt"] = float("nan")
    with pytest.raises(ValueError, match="canonical JSON"):
        build_holdout_evaluation_evidence(**chain)


def test_evidence_tamper_rejected_even_with_new_digest(chain):
    evidence = build_holdout_evaluation_evidence(**chain)
    evidence["metrics"]["rmse_microVolt"] = 999
    evidence["evaluation_evidence_digest"] = _digest({k: v for k, v in evidence.items() if k != "evaluation_evidence_digest"})
    assert not verify_holdout_evaluation_evidence(evidence, **chain)


def test_eligibility_tamper_rejected(chain):
    chain["eligibility_snapshot"]["predicted_microVolt"][0] = 99
    with pytest.raises(ValueError, match="digest"):
        build_holdout_evaluation_evidence(**chain)


def test_disclosure_tamper_rejected(chain):
    chain["disclosure_snapshot"]["observed_microVolt"][0] = 99
    with pytest.raises(ValueError, match="digest"):
        build_holdout_evaluation_evidence(**chain)


def test_verifier_rejects_extra_evidence_field(chain):
    evidence = build_holdout_evaluation_evidence(**chain)
    evidence["unexpected"] = "value"
    assert not verify_holdout_evaluation_evidence(evidence, **chain)


def test_verifier_rejects_non_mapping(chain):
    assert not verify_holdout_evaluation_evidence(None, **chain)
