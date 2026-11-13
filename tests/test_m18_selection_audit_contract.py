"""M18-C2-B red-phase contract tests for auditable selection decisions.

Expected first failure: internal_selection_audit does not exist yet.
Fixtures use canonical M17 accuracy profiles and M18 provenance, not mocks.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from zenerestimation.data.dataset import BatteryDataset
from zenerestimation.evaluation.accuracy_diagnostics import build_accuracy_diagnostics
from zenerestimation.evaluation.accuracy_profile import build_accuracy_profile
from zenerestimation.optimization.internal_provenance import build_internal_validation_context
from zenerestimation.optimization.validation import ExpandingWindowSplitter
from zenerestimation.optimization.candidate import CandidateResult
from zenerestimation.optimization.result import OptimizationResult
from zenerestimation.optimization.internal_selection import select_internal_model
from zenerestimation.optimization.internal_selection_audit import (
    build_selection_decision_record,
    verify_selection_decision_record,
)

DATES = (
    ("2022-03-01", "2022-06-01"),
    ("2022-09-01", "2022-12-01"),
)


def context(*, dataset_id="synthetic-dev-v1", dates=DATES):
    """Trusted context supplied independently of candidate metadata."""
    development = BatteryDataset(pd.DataFrame({
        "ds": pd.date_range("2020-03-01", periods=12, freq="QS-MAR"),
        "microVolt": np.arange(1, 13, dtype=float),
    }))
    trusted = build_internal_validation_context(
        development,
        ExpandingWindowSplitter(folds=2, validation_steps=2),
        dataset_id=dataset_id,
    )
    # Preserve the original B1 test helper's ability to supply deliberately
    # inconsistent validation dates for negative contract tests.
    if dates != DATES:
        trusted["fold_validation_dates"] = [list(fold) for fold in dates]
    return trusted


def candidate(
    rmse=(0.2, 0.4),
    *,
    dates=DATES,
    statuses=("verified", "verified"),
    selection_metric="rmse",
    score=None,
    scope="internal_expanding_window",
    model="LSTM",
    dataset_id="synthetic-dev-v1",
):
    mean_rmse = sum(rmse) / len(rmse)

    if score is None:
        score = mean_rmse

    measured_folds = []
    for index, (fold_dates, fold_rmse) in enumerate(zip(dates, rmse)):
        actual = np.full(len(fold_dates), 10.0)
        measured_folds.append({
            "fold_index": index,
            "dates": pd.DatetimeIndex(fold_dates),
            "actual": actual,
            "predicted": actual + float(fold_rmse),
            "alignment_status": "verified",
        })
    diagnostics = build_accuracy_diagnostics(
        measured_folds, dataset=dataset_id, model=model,
        scope=scope,
    )
    profile = build_accuracy_profile(diagnostics)

    return CandidateResult(
        params={"window": 4},
        score=score,
        metrics={"rmse": mean_rmse},
        metadata={
            "validation": "expanding_window",
            "folds": len(rmse),
            "validation_steps": 2,
            "selection_metric": selection_metric,
            "fold_metrics": [
                {"rmse": value}
                for value in rmse
            ],
            "fold_alignment": [
                {
                    "fold_index": index,
                    "status": status,
                    "validation_steps": len(fold_dates),
                    "forecast_horizon_provided": True,
                    "forecast_dates_provided": status == "verified",
                    "validation_dates": list(fold_dates),
                    "forecast_dates": (
                        list(fold_dates)
                        if status == "verified"
                        else None
                    ),
                    "timestamp_match": (
                        True if status == "verified" else None
                    ),
                }
                for index, (fold_dates, status)
                in enumerate(zip(dates, statuses))
            ],
            "internal_provenance": deepcopy(context(dataset_id=dataset_id)),
            "accuracy_profile": profile,
        },
    )


def result(model, *, rmse=(0.2, 0.4), **candidate_overrides):
    selected = candidate(rmse, model=model, **candidate_overrides)

    return OptimizationResult(
        model=model,
        metric="rmse",
        best_params=selected.params,
        best_score=selected.score,
        candidates=(selected,),
        metadata={"validation": "internal_temporal"},
    )



def unique_results():
    return {
        "LSTM": result("LSTM", rmse=(0.1, 0.6)),
        "GRU": result("GRU", rmse=(0.2, 0.4)),
    }



CONTEXT_FIELDS = {
    "protocol_version", "dataset_id", "dataset_fingerprint", "scope",
    "fold_validation_dates", "fold_training_end_dates", "folds", "validation_steps",
}
RECORD_FIELDS = {
    "decision_record_version", "decision", "validation_provenance", "decision_digest",
}


def build(results=None, *, trusted=None, authorized=False):
    return build_selection_decision_record(
        unique_results() if results is None else results,
        validation_context=context() if trusted is None else trusted,
        authorized=authorized,
    )


def verify(record, results=None, *, trusted=None):
    return verify_selection_decision_record(
        record,
        results=unique_results() if results is None else results,
        validation_context=context() if trusted is None else trusted,
    )


def canonical_payload(record):
    payload = {key: value for key, value in record.items() if key != "decision_digest"}
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def recompute_digest(record):
    record["decision_digest"] = "sha256:" + hashlib.sha256(
        canonical_payload(record)
    ).hexdigest()
    return record


def assert_verified(verification):
    assert verification == {
        "status": "verified",
        "decision_record_version": "m18_c2_decision_v1",
        "decision_digest_match": True,
        "ranking_match": True,
        "provenance_match": True,
        "selection_match": True,
    }


@pytest.mark.parametrize("authorized,expected_status,selected", [
    (False, "PENDING_AUTHORIZATION", None),
    (True, "SELECTED", "GRU"),
])
def test_unique_winner_audited_with_original_c1_decision(authorized, expected_status, selected):
    results = unique_results()
    trusted = context()
    record = build(results, trusted=trusted, authorized=authorized)
    assert set(record) == RECORD_FIELDS
    assert record["decision_record_version"] == "m18_c2_decision_v1"
    assert record["decision"] == select_internal_model(
        results, validation_context=trusted, authorized=authorized,
    )
    assert record["decision"]["status"] == expected_status
    assert record["decision"]["selected_model"] == selected
    assert_verified(verify(record, results, trusted=trusted))


@pytest.mark.parametrize("authorized", [False, True])
def test_exact_tie_is_auditable_but_never_selected(authorized):
    results = {"LSTM": result("LSTM"), "GRU": result("GRU")}
    record = build(results, authorized=authorized)
    assert record["decision"]["status"] == "TIE_UNRESOLVED"
    assert record["decision"]["selected_model"] is None
    assert record["decision"]["recommended_model"] is None
    assert record["decision"]["selection_score"] is None
    assert_verified(verify(record, results))


def test_provenance_contains_exact_complete_trusted_context():
    trusted = context()
    record = build(trusted=trusted)
    assert set(record["validation_provenance"]) == CONTEXT_FIELDS
    assert record["validation_provenance"] == trusted
    assert record["validation_provenance"] is not trusted
    assert record["validation_provenance"]["protocol_version"] == "m18_internal_provenance_v1"
    assert record["validation_provenance"]["scope"] == "internal_expanding_window"
    assert record["validation_provenance"]["dataset_fingerprint"].startswith("sha256:")


def test_digest_is_sha256_of_canonical_utf8_payload_without_digest_field():
    record = build()
    assert record["decision_digest"] == (
        "sha256:" + hashlib.sha256(canonical_payload(record)).hexdigest()
    )
    assert len(record["decision_digest"]) == 71
    json.dumps(record, sort_keys=True, allow_nan=False)


def test_repeat_and_model_input_order_produce_identical_record_and_json():
    results = unique_results()
    a = build(results)
    b = build(results)
    c = build(dict(reversed(list(results.items()))))
    assert a == b == c
    assert canonical_payload(a) == canonical_payload(b) == canonical_payload(c)


def test_authorization_changes_decision_and_digest_not_ranking_or_provenance():
    pending = build(authorized=False)
    selected = build(authorized=True)
    assert pending["decision"]["ranking"] == selected["decision"]["ranking"]
    assert pending["validation_provenance"] == selected["validation_provenance"]
    assert pending["decision_digest"] != selected["decision_digest"]


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", [], {}])
def test_builder_rejects_non_boolean_authorization(invalid):
    with pytest.raises((TypeError, ValueError)):
        build(authorized=invalid)


def test_builder_rejects_external_validation_scope():
    trusted = context()
    trusted["scope"] = "external_holdout"
    with pytest.raises(ValueError):
        build(trusted=trusted)


def test_builder_rejects_empty_results():
    with pytest.raises(ValueError):
        build({})


def test_digest_corruption_rejected():
    record = build()
    record["decision_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        verify(record)


@pytest.mark.parametrize("field,value", [
    ("dataset_id", "another-dataset"),
    ("dataset_fingerprint", "sha256:" + "a" * 64),
    ("scope", "external_holdout"),
    ("protocol_version", "other-version"),
    ("folds", 3),
    ("validation_steps", 3),
    ("fold_validation_dates", [["2023-03-01", "2023-06-01"]]),
    ("fold_training_end_dates", ["2020-03-01", "2021-03-01"]),
])
def test_provenance_tampering_rejected_even_with_recomputed_digest(field, value):
    record = build()
    record["validation_provenance"][field] = value
    recompute_digest(record)
    with pytest.raises(ValueError):
        verify(record)


@pytest.mark.parametrize("change", [
    lambda d: d.update(status="SELECTED"),
    lambda d: d.update(authorized=True),
    lambda d: d.update(selected_model="LSTM"),
    lambda d: d.update(recommended_model="LSTM"),
    lambda d: d.update(selection_score=0.9),
    lambda d: d.update(selection_metric="mae"),
    lambda d: d.update(selection_version="other"),
    lambda d: d.update(ranking_version="other"),
    lambda d: d["ranking"][0].update(model="LSTM"),
    lambda d: d["ranking"][0].update(score=0.9),
    lambda d: d["ranking"][0].update(rank=2),
])
def test_decision_tampering_rejected_even_with_recomputed_digest(change):
    record = build()
    change(record["decision"])
    recompute_digest(record)
    with pytest.raises(ValueError):
        verify(record)


@pytest.mark.parametrize("change", [
    lambda r: r.pop("decision"),
    lambda r: r.update(unexpected="not audited"),
    lambda r: r.update(decision_record_version="other-version"),
    lambda r: r["validation_provenance"].pop("folds"),
    lambda r: r["validation_provenance"].update(unexpected="extra"),
    lambda r: r["decision"].pop("ranking"),
    lambda r: r["decision"].update(unexpected="extra"),
])
def test_missing_extra_or_unsupported_schema_rejected(change):
    record = build()
    change(record)
    recompute_digest(record)
    with pytest.raises(ValueError):
        verify(record)


@pytest.mark.parametrize("field,value", [
    ("folds", True),
    ("validation_steps", 2.0),
    ("dataset_id", 123),
])
def test_wrong_type_provenance_rejected_even_if_python_equality_would_pass(field, value):
    record = build()
    record["validation_provenance"][field] = value
    recompute_digest(record)
    with pytest.raises(ValueError):
        verify(record)


def test_wrong_type_decision_rejected_even_if_python_equality_would_pass():
    record = build()
    record["decision"]["authorized"] = 0  # 0 == False in Python
    recompute_digest(record)
    with pytest.raises(ValueError):
        verify(record)


def test_verification_rejects_changed_independent_context():
    record = build()
    trusted = context(dataset_id="different-dev")
    with pytest.raises(ValueError):
        verify(record, trusted=trusted)


def test_verification_rejects_changed_independent_results():
    record = build()
    changed = {"LSTM": result("LSTM", rmse=(0.1, 0.6)),
               "GRU": result("GRU", rmse=(0.3, 0.5))}
    with pytest.raises(ValueError):
        verify(record, changed)


def test_new_dataset_identifier_supported_without_allowlist():
    dataset_id = "new-instrument-family-987"
    trusted = context(dataset_id=dataset_id)
    results = {
        "GRU": result("GRU", rmse=(0.2, 0.4), dataset_id=dataset_id),
        "LSTM": result("LSTM", rmse=(0.1, 0.6), dataset_id=dataset_id),
    }
    record = build(results, trusted=trusted, authorized=True)
    assert record["validation_provenance"]["dataset_id"] == dataset_id
    assert_verified(verify(record, results, trusted=trusted))


def test_builder_and_verifier_do_not_mutate_inputs_or_record():
    results = unique_results()
    trusted = context()
    results_before = {key: value.to_dict() for key, value in results.items()}
    trusted_before = deepcopy(trusted)
    record = build(results, trusted=trusted, authorized=True)
    record_before = deepcopy(record)
    assert_verified(verify(record, results, trusted=trusted))
    assert record == record_before
    assert trusted == trusted_before
    assert {key: value.to_dict() for key, value in results.items()} == results_before


def test_roundtrip_through_json_preserves_verification():
    record = build(authorized=True)
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded == record
    assert_verified(verify(decoded))
