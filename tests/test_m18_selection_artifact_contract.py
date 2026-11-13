"""M18-C3-B: contract-first tests for standalone internal selection artifacts.

RED PHASE: production module internal_selection_artifact is deliberately absent.
Uses real M18-C2 test fixtures: canonical M17 accuracy profiles, verified
fold alignment, M18 internal provenance, and OptimizationResult objects.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tests.test_m18_selection_audit_contract import (
    context,
    result,
    unique_results,
)
from zenerestimation.optimization.internal_selection_audit import (
    build_selection_decision_record,
    verify_selection_decision_record,
)
from zenerestimation.optimization.internal_selection_artifact import (
    build_internal_selection_artifact,
    save_internal_selection_artifact,
    verify_internal_selection_artifact,
)

ARTIFACT_VERSION = "m18_c3_selection_artifact_v1"
ARTIFACT_TYPE = "internal_model_selection"
ARTIFACT_FIELDS = {
    "artifact_version", "artifact_type", "selection_decision_record",
}


def build(*, results=None, trusted=None, authorized=False):
    return build_internal_selection_artifact(
        unique_results() if results is None else results,
        validation_context=context() if trusted is None else trusted,
        authorized=authorized,
    )


def verify(artifact, *, results=None, trusted=None):
    return verify_internal_selection_artifact(
        artifact,
        results=unique_results() if results is None else results,
        validation_context=context() if trusted is None else trusted,
    )


def digest_payload(record):
    return json.dumps(
        {k: v for k, v in record.items() if k != "decision_digest"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def recompute_digest(record):
    record["decision_digest"] = (
        "sha256:" + hashlib.sha256(digest_payload(record)).hexdigest()
    )
    return record


@pytest.mark.parametrize("authorized,status,selected", [
    (False, "PENDING_AUTHORIZATION", None),
    (True, "SELECTED", "GRU"),
])
def test_wrapper_and_complete_c2_record(authorized, status, selected):
    results, trusted = unique_results(), context()
    artifact = build(results=results, trusted=trusted, authorized=authorized)
    assert set(artifact) == ARTIFACT_FIELDS
    assert artifact["artifact_version"] == ARTIFACT_VERSION
    assert artifact["artifact_type"] == ARTIFACT_TYPE
    record = artifact["selection_decision_record"]
    assert record == build_selection_decision_record(
        results, validation_context=trusted, authorized=authorized,
    )
    assert record["decision"]["status"] == status
    assert record["decision"]["selected_model"] == selected
    assert record["decision_digest"] == (
        "sha256:" + hashlib.sha256(digest_payload(record)).hexdigest()
    )
    assert verify(artifact, results=results, trusted=trusted)["status"] == "verified"


@pytest.mark.parametrize("authorized", [False, True])
def test_exact_tie_is_persistable_without_forced_selection(authorized):
    results = {
        "LSTM": result("LSTM", rmse=(0.2, 0.4)),
        "GRU": result("GRU", rmse=(0.2, 0.4)),
    }
    artifact = build(results=results, authorized=authorized)
    decision = artifact["selection_decision_record"]["decision"]
    assert decision["status"] == "TIE_UNRESOLVED"
    assert decision["selected_model"] is None
    assert verify(artifact, results=results)["status"] == "verified"


def test_build_is_deterministic_independent_of_model_mapping_order():
    results = unique_results()
    first = build(results=results)
    second = build(results=dict(reversed(list(results.items()))))
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_build_and_verify_do_not_mutate_inputs_or_artifact():
    results, trusted = unique_results(), context()
    before_results = {k: v.to_dict() for k, v in results.items()}
    before_context = deepcopy(trusted)
    artifact = build(results=results, trusted=trusted)
    before_artifact = deepcopy(artifact)
    verify(artifact, results=results, trusted=trusted)
    assert artifact == before_artifact
    assert trusted == before_context
    assert {k: v.to_dict() for k, v in results.items()} == before_results
    artifact["selection_decision_record"]["decision"]["ranking"][0]["score"] = -999
    assert build(results=results, trusted=trusted) == before_artifact


def test_authorization_changes_only_embedded_decision_not_provenance():
    pending, selected = build(), build(authorized=True)
    p = pending["selection_decision_record"]
    s = selected["selection_decision_record"]
    assert p["validation_provenance"] == s["validation_provenance"]
    assert p["decision"]["ranking"] == s["decision"]["ranking"]
    assert p["decision_digest"] != s["decision_digest"]


@pytest.mark.parametrize("authorized", [None, 0, 1, "true", [], {}])
def test_build_rejects_non_boolean_authorization(authorized):
    with pytest.raises((TypeError, ValueError)):
        build(authorized=authorized)


def test_build_rejects_external_scope():
    trusted = context()
    trusted["scope"] = "external_holdout"
    with pytest.raises((TypeError, ValueError)):
        build(trusted=trusted)


def test_build_rejects_empty_results():
    with pytest.raises((TypeError, ValueError)):
        build(results={})


def test_save_is_deterministic_utf8_and_roundtrips(tmp_path):
    artifact = build()
    path = tmp_path / "internal_selection.json"
    returned = save_internal_selection_artifact(artifact, path)
    assert returned == path
    contents = path.read_bytes()
    assert contents == (
        json.dumps(
            artifact, sort_keys=True, indent=2,
            ensure_ascii=False, allow_nan=False,
        ) + "\n"
    ).encode("utf-8")
    loaded = json.loads(contents)
    assert loaded == artifact
    assert verify(loaded)["status"] == "verified"
    assert loaded["selection_decision_record"]["decision_digest"] == (
        artifact["selection_decision_record"]["decision_digest"]
    )
    save_internal_selection_artifact(artifact, path)
    assert path.read_bytes() == contents


def test_save_creates_nested_directories(tmp_path):
    destination = tmp_path / "experiment" / "selection" / "internal_selection.json"
    assert save_internal_selection_artifact(build(), destination) == destination
    assert destination.is_file()


@pytest.mark.parametrize("bad_path", [
    "internal_selection.txt", "internal_selection", "internal_selection.csv",
])
def test_save_rejects_non_json_extension_before_writing(tmp_path, bad_path):
    path = tmp_path / bad_path
    with pytest.raises((TypeError, ValueError)):
        save_internal_selection_artifact(build(), path)
    assert not path.exists()


def test_save_accepts_string_path(tmp_path):
    path = tmp_path / "internal_selection.json"
    assert save_internal_selection_artifact(build(), str(path)) == path
    assert path.is_file()


@pytest.mark.parametrize("mutator", [
    lambda a: a.pop("artifact_version"),
    lambda a: a.update(extra="unexpected"),
    lambda a: a.update(artifact_version="unsupported"),
    lambda a: a.update(artifact_version=1),
    lambda a: a.update(artifact_type="benchmark"),
    lambda a: a.update(artifact_type=None),
    lambda a: a.pop("selection_decision_record"),
    lambda a: a.update(selection_decision_record=[]),
])
def test_invalid_wrapper_rejected_by_save_and_verify(tmp_path, mutator):
    artifact = build()
    mutator(artifact)
    path = tmp_path / "nested" / "internal_selection.json"
    with pytest.raises((TypeError, ValueError)):
        save_internal_selection_artifact(artifact, path)
    assert not path.exists()
    assert not path.parent.exists()
    with pytest.raises((TypeError, ValueError)):
        verify(artifact)


@pytest.mark.parametrize("mutator", [
    lambda r: r.update(decision_digest="sha256:" + "0" * 64),
    lambda r: r["decision"].update(selected_model="LSTM"),
    lambda r: r["validation_provenance"].update(dataset_id="altered"),
    lambda r: r.update(extra="unexpected"),
])
def test_save_rejects_invalid_embedded_record_before_touching_files(tmp_path, mutator):
    artifact = build()
    mutator(artifact["selection_decision_record"])
    path = tmp_path / "internal_selection.json"
    path.write_text("pre-existing evidence", encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        save_internal_selection_artifact(artifact, path)
    assert path.read_text(encoding="utf-8") == "pre-existing evidence"


def test_save_rejects_nonfinite_value_even_with_recomputed_digest(tmp_path):
    artifact = build()
    record = artifact["selection_decision_record"]
    record["decision"]["ranking"][0]["score"] = float("nan")
    # Digest recomputation itself must reject non-finite values.
    with pytest.raises(ValueError):
        recompute_digest(record)
    path = tmp_path / "internal_selection.json"
    with pytest.raises((TypeError, ValueError)):
        save_internal_selection_artifact(artifact, path)
    assert not path.exists()


def test_save_does_not_mutate_artifact(tmp_path):
    artifact = build()
    snapshot = deepcopy(artifact)
    save_internal_selection_artifact(artifact, tmp_path / "internal_selection.json")
    assert artifact == snapshot


def test_save_does_not_touch_other_artifacts(tmp_path):
    other_names = [
        "optimization.json", "evaluation.json", "benchmark.json",
        "neural_training.json",
    ]
    for name in other_names:
        (tmp_path / name).write_bytes(b"existing independent artifact\n")
    before = {name: (tmp_path / name).read_bytes() for name in other_names}
    save_internal_selection_artifact(build(), tmp_path / "internal_selection.json")
    assert {name: (tmp_path / name).read_bytes() for name in other_names} == before


def test_save_rejects_file_as_parent_without_modifying_it(tmp_path):
    parent = tmp_path / "not_a_directory"
    parent.write_text("untouched", encoding="utf-8")
    with pytest.raises((OSError, ValueError)):
        save_internal_selection_artifact(build(), parent / "internal_selection.json")
    assert parent.read_text(encoding="utf-8") == "untouched"


def test_verify_delegates_to_independent_c2_replay():
    results, trusted = unique_results(), context()
    artifact = build(results=results, trusted=trusted)
    expected = verify_selection_decision_record(
        artifact["selection_decision_record"],
        results=results, validation_context=trusted,
    )
    assert verify(artifact, results=results, trusted=trusted) == expected


@pytest.mark.parametrize("changed", ["results", "context"])
def test_verify_rejects_changed_independent_evidence(changed):
    artifact = build()
    if changed == "results":
        results = {
            "LSTM": result("LSTM", rmse=(0.05, 0.10)),
            "GRU": result("GRU", rmse=(0.2, 0.4)),
        }
        with pytest.raises((TypeError, ValueError)):
            verify(artifact, results=results)
    else:
        trusted = context(dataset_id="another-development-dataset")
        with pytest.raises((TypeError, ValueError)):
            verify(artifact, trusted=trusted)


def test_verify_rejects_tampering_even_if_digest_recomputed():
    artifact = build()
    record = artifact["selection_decision_record"]
    record["decision"]["selected_model"] = "LSTM"
    recompute_digest(record)
    with pytest.raises((TypeError, ValueError)):
        verify(artifact)


def test_verify_rejects_provenance_tampering_even_if_digest_recomputed():
    artifact = build()
    record = artifact["selection_decision_record"]
    record["validation_provenance"]["dataset_id"] = "other-dataset"
    recompute_digest(record)
    with pytest.raises((TypeError, ValueError)):
        verify(artifact)


def test_save_does_not_claim_independent_scientific_verification(tmp_path):
    # A recomputed digest can be locally self-consistent. Save has no independently
    # trusted results/context and therefore cannot authenticate a forged decision.
    artifact = build()
    record = artifact["selection_decision_record"]
    record["decision"]["selected_model"] = "LSTM"
    recompute_digest(record)
    path = tmp_path / "internal_selection.json"
    assert save_internal_selection_artifact(artifact, path) == path
    assert path.is_file()
    with pytest.raises((TypeError, ValueError)):
        verify(json.loads(path.read_text(encoding="utf-8")))


def test_no_external_holdout_or_benchmark_payload_fields():
    artifact = build()
    assert set(artifact) == ARTIFACT_FIELDS
    serialized = json.dumps(artifact, sort_keys=True)
    for token in (
        '"actual"', '"predicted"', '"predictions"', '"errors"',
        '"benchmark_rmse"', '"external_holdout"', '"evaluation"',
    ):
        assert token not in serialized
