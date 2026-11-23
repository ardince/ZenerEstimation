"""M19-C2-A: strict scientific alignment, never authorization."""
from copy import deepcopy
import json

import pytest

from tests.test_m19_b3_b3_holdout_error_scoring import case  # noqa: F401
from tests.test_m19_b3_b4_evaluation_evidence import chain  # noqa: F401
from zenerestimation.assessment.holdout_evaluation_evidence import build_holdout_evaluation_evidence
from zenerestimation.comparison.benchmark_contract import build_benchmark_entry
from zenerestimation.comparison.benchmark_comparability import (
    VerifiedBenchmarkInput, assess_scientific_alignment,
)


@pytest.fixture
def pair(chain):
    evidence = build_holdout_evaluation_evidence(**chain)
    context = {**chain, "evaluation_evidence": evidence}
    return [VerifiedBenchmarkInput(
        entry=build_benchmark_entry(run_id=f"410.GRU.{i}", model_family="GRU", **context),
        **context,
    ) for i in range(2)]


def test_identical_verified_entries_align_without_authorization(pair):
    result = assess_scientific_alignment(pair)
    assert result["status"] == "SCIENTIFICALLY_ALIGNED"
    assert result["authorization"] == "not_established"
    assert result["ranking_permitted"] is False
    assert result["comparison_permitted"] is False
    assert result["shared_horizon"] == 2
    assert result["observation_digest"].startswith("sha256:")
    json.dumps(result, allow_nan=False)


def test_order_independent_and_read_only(pair):
    before = [deepcopy(item.entry) for item in pair]
    assert assess_scientific_alignment(pair) == assess_scientific_alignment(pair[::-1])
    assert [item.entry for item in pair] == before


@pytest.mark.parametrize("path,value", [
    (("dataset_id",), "110"),
    (("forecast_horizon",), 3),
    (("forecast_dates", 0), "2024-09-01"),
    (("by_date", 0, "observed_microVolt"), 123.0),
    (("by_date", 0, "target_date"), "2024-09-01"),
    (("metrics", "rmse_microVolt"), 123.0),
    (("authentication",), "established"),
])
def test_tampered_entry_fails_c1_before_alignment(pair, path, value):
    entry = pair[1].entry
    node = entry
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match="C1 upstream verification"):
        assess_scientific_alignment(pair)


def test_upstream_evidence_substitution_fails(pair):
    pair[1].evaluation_evidence["metrics"]["rmse_microVolt"] = 999.0
    with pytest.raises(ValueError, match="C1 upstream verification"):
        assess_scientific_alignment(pair)


def test_duplicate_run_id_fails(pair):
    with pytest.raises(ValueError, match="duplicate run_id"):
        assess_scientific_alignment([pair[0], pair[0]])


@pytest.mark.parametrize("bad", [[], [None], "invalid"])
def test_invalid_input_fails(pair, bad):
    with pytest.raises((ValueError, TypeError)):
        assess_scientific_alignment(bad)


def test_invalid_member_fails(pair):
    with pytest.raises(TypeError, match="VerifiedBenchmarkInput"):
        assess_scientific_alignment([pair[0], {}])
