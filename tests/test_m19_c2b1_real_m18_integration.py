"""M19-C2-B1 integration: genuine M18 GRU chain plus independently issued LSTM fixture.

The LSTM is synthetic B1-B4 evidence, NOT an independently executed M18 model.
Neither matching scope nor genuine M18 provenance establishes external approval.
"""
from copy import deepcopy
import json
import re

import pytest

from tests.test_m18_forecast_execution_contract import ProbeModel
from tests.test_m18_forecast_plan_contract import training as processed_training  # noqa: F401
from tests.test_m19_b3_b1_m18_real_integration import declaration, full_frame  # noqa: F401
from tests.test_m19_c1_real_m18_integration import genuine_chain  # noqa: F401
from tests.test_m19_c2a_independent_chain_alignment import _issue_chain

from zenerestimation.assessment.holdout_target_disclosure import _digest
from zenerestimation.comparison.benchmark_contract import build_benchmark_entry
from zenerestimation.comparison.benchmark_comparability import (
    VerifiedBenchmarkInput, assess_scientific_alignment,
)
from zenerestimation.comparison.benchmark_authorization import (
    build_benchmark_scope_declaration, verify_benchmark_scope_declaration,
    assess_benchmark_scope,
)


def _genuine_member(context):
    entry = build_benchmark_entry(
        run_id="410.GRU.real-m18.c2b1", model_family="GRU", **context,
    )
    return VerifiedBenchmarkInput(entry=entry, **context)


def _mixed_pair(context):
    genuine = _genuine_member(context)
    entry = genuine.entry
    dates = tuple(entry["forecast_dates"])
    actuals = tuple(row["observed_microVolt"] for row in entry["by_date"])
    # This synthetic chain is separately constructed before B2/B3/B4/C1 issuance.
    # The real M18 GRU is NOT rebranded as an independently executed LSTM.
    synthetic = _issue_chain(
        run_id="410.LSTM.synthetic.c2b1",
        dataset_id=entry["dataset_id"], dates=dates, actuals=actuals,
        predictions=tuple(v + 0.1 for v in actuals),
        model="LSTM", family="LSTM",
    )
    return [genuine, synthetic]


def _scope(pair, **overrides):
    first = pair[0].entry
    participants = [
        {key: member.entry[key] for key in (
            "run_id", "assessment_id", "model_family", "benchmark_entry_digest"
        )}
        for member in pair
    ]
    args = dict(
        benchmark_id="bench.410.m18.c2b1",
        dataset_id=first["dataset_id"],
        forecast_dates=list(first["forecast_dates"]),
        participants=participants,
    )
    args.update(overrides)
    return build_benchmark_scope_declaration(**args)


@pytest.fixture
def mixed_chain(genuine_chain):
    context, events = genuine_chain
    pair = _mixed_pair(context)
    assert assess_scientific_alignment(pair)["status"] == "SCIENTIFICALLY_ALIGNED"
    assert ProbeModel.events == events
    return pair, events


def _no_execution(events):
    assert ProbeModel.events == events


def test_genuine_m18_chain_enters_b1_without_new_execution(mixed_chain):
    pair, events = mixed_chain
    result = assess_benchmark_scope(inputs=pair, declaration=_scope(pair))
    assert result["status"] == "AUTHORIZATION_REQUIRED"
    assert result["reasons"] == []
    assert result["scientific_alignment_status"] == "SCIENTIFICALLY_ALIGNED"
    _no_execution(events)


def test_matching_scope_never_grants_approval_or_ranking(mixed_chain):
    pair, events = mixed_chain
    result = assess_benchmark_scope(inputs=pair, declaration=_scope(pair))
    assert result["authorization"] == "not_established"
    assert result["authentication"] == "not_established"
    assert result["comparison_permitted"] is False
    assert result["ranking_permitted"] is False
    assert result["status"] != "AUTHORIZED"
    _no_execution(events)


def test_decision_deterministic_order_independent_read_only(mixed_chain):
    pair, events = mixed_chain
    before = [deepcopy(member.entry) for member in pair]
    declaration = _scope(pair)
    original_declaration = deepcopy(declaration)
    first = assess_benchmark_scope(inputs=pair, declaration=declaration)
    assert first == assess_benchmark_scope(inputs=pair, declaration=declaration)
    assert first == assess_benchmark_scope(inputs=pair[::-1], declaration=declaration)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", first["decision_digest"])
    json.dumps(first, allow_nan=False)
    assert [member.entry for member in pair] == before
    assert declaration == original_declaration
    _no_execution(events)


@pytest.mark.parametrize("field,reason", [
    ("dataset_id", "DATASET_SCOPE_MISMATCH"),
    ("forecast_dates", "TIMESTAMP_SCOPE_MISMATCH"),
    ("participants", "PARTICIPANT_SCOPE_MISMATCH"),
])
def test_valid_but_mismatched_scope_rejected(mixed_chain, field, reason):
    pair, events = mixed_chain
    scope = _scope(pair)
    args = {
        "dataset_id": scope["dataset_id"],
        "forecast_dates": scope["forecast_dates"],
        "participants": scope["participants"],
    }
    if field == "dataset_id":
        args[field] = "110" if args[field] != "110" else "other-dataset"
    elif field == "forecast_dates":
        from datetime import date, timedelta
        args[field] = [
            (date.fromisoformat(d) + timedelta(days=1)).isoformat()
            for d in args[field]
        ]

    else:
        args[field] = deepcopy(args[field])

        replacement_digest = "sha256:" + "f" * 64

        if replacement_digest == pair[0].entry["benchmark_entry_digest"]:
            replacement_digest = "sha256:" + "e" * 64

        args[field][0]["benchmark_entry_digest"] = replacement_digest

    altered = _scope(pair, **args)
    assert verify_benchmark_scope_declaration(altered)
    result = assess_benchmark_scope(inputs=pair, declaration=altered)
    assert result["status"] == "AUTHORIZATION_REJECTED"
    assert reason in result["reasons"]
    assert result["comparison_permitted"] is False
    _no_execution(events)


def test_redigested_false_approval_claim_rejected(mixed_chain):
    pair, events = mixed_chain
    scope = _scope(pair)
    scope["scope_status"] = "AUTHORIZED"
    scope["scope_digest"] = _digest({k: v for k, v in scope.items() if k != "scope_digest"})
    assert not verify_benchmark_scope_declaration(scope)
    with pytest.raises(ValueError, match="scope declaration"):
        assess_benchmark_scope(inputs=pair, declaration=scope)
    _no_execution(events)


def test_cross_run_entry_substitution_fails_c1(mixed_chain):
    pair, events = mixed_chain
    altered = deepcopy(pair[1].entry)
    altered["metrics"]["rmse_microVolt"] = 999.0
    substituted = VerifiedBenchmarkInput(
        entry=altered, declaration=pair[1].declaration,
        eligibility_snapshot=pair[1].eligibility_snapshot,
        disclosure_snapshot=pair[1].disclosure_snapshot,
        scoring_snapshot=pair[1].scoring_snapshot,
        evaluation_evidence=pair[1].evaluation_evidence,
    )
    with pytest.raises(ValueError, match="C1 upstream verification"):
        assess_benchmark_scope(inputs=[pair[0], substituted], declaration=_scope(pair))
    _no_execution(events)


def test_redigested_upstream_substitution_fails_c1(mixed_chain):
    pair, events = mixed_chain
    changed = deepcopy(pair[1].evaluation_evidence)
    changed["metrics"]["rmse_microVolt"] = 999.0
    changed["evaluation_evidence_digest"] = _digest({
        k: v for k, v in changed.items() if k != "evaluation_evidence_digest"
    })
    substituted = VerifiedBenchmarkInput(
        entry=pair[1].entry, declaration=pair[1].declaration,
        eligibility_snapshot=pair[1].eligibility_snapshot,
        disclosure_snapshot=pair[1].disclosure_snapshot,
        scoring_snapshot=pair[1].scoring_snapshot,
        evaluation_evidence=changed,
    )
    with pytest.raises(ValueError, match="C1 upstream verification"):
        assess_benchmark_scope(inputs=[pair[0], substituted], declaration=_scope(pair))
    _no_execution(events)


def test_observation_identity_preserved_through_b1(mixed_chain):
    pair, events = mixed_chain
    aligned = assess_scientific_alignment(pair)
    result = assess_benchmark_scope(inputs=pair, declaration=_scope(pair))
    assert result["observation_digest"] == aligned["observation_digest"]
    assert result["scope_digest"] == _scope(pair)["scope_digest"]
    _no_execution(events)
