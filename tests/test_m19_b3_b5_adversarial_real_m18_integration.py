"""M19-B3-B5 adversarial genuine M18 -> B1 -> B2 -> B3 -> B4 tests.

Uses the accepted M18 ProbeModel fixture and real upstream contracts, not
mocked verifiers. Digest equality is not authentication or independence proof.
"""
from copy import deepcopy

import pandas as pd
import pytest

from tests.test_m18_forecast_plan_contract import training as processed_training
from tests.test_m18_forecast_execution_contract import ProbeModel
from tests.test_m19_b3_b1_m18_real_integration import declaration, full_frame, _reset_probe, _real_case
from tests.test_m19_b3_b2_m18_real_integration import _case, _report
from tests.test_m19_b3_b4_m18_real_integration import _complete
from zenerestimation.assessment.eligibility import AssessmentDeclaration, AssessmentTrack, CONTRACT_VERSION
from zenerestimation.assessment.holdout_evaluation_eligibility import establish_holdout_evaluation_eligibility
from zenerestimation.assessment.holdout_target_disclosure import disclose_holdout_targets, _digest
from zenerestimation.assessment.holdout_error_scoring import score_disclosed_holdout
from zenerestimation.assessment.holdout_evaluation_evidence import (
    build_holdout_evaluation_evidence, verify_holdout_evaluation_evidence,
)
from zenerestimation.assessment.holdout_isolation import isolate_declared_holdout, require_isolated_training_dataset
from zenerestimation.data.dataset import BatteryDataset


def _redigest(record, field):
    record[field] = _digest({k: v for k, v in record.items() if k != field})
    return record


def _reject_chain(declaration, eligibility, disclosure, scoring, evidence):
    with pytest.raises((ValueError, TypeError, KeyError, OverflowError)):
        build_holdout_evaluation_evidence(
            declaration=declaration, eligibility_snapshot=eligibility,
            disclosure_snapshot=disclosure, scoring_snapshot=scoring,
        )
    assert not verify_holdout_evaluation_evidence(
        evidence, declaration=declaration, eligibility_snapshot=eligibility,
        disclosure_snapshot=disclosure, scoring_snapshot=scoring,
    )


@pytest.mark.parametrize('field,value', [
    ('assessment_id', 'other-assessment'),
    ('dataset_id', '110'),
])
def test_cross_assessment_declaration_substitution_fails_closed(
    processed_training, full_frame, declaration, field, value,
):
    _reset_probe()
    eligible, disclosed, scored, evidence = _complete(processed_training, full_frame, declaration)
    events = deepcopy(ProbeModel.events)
    params = dict(
        contract_version=CONTRACT_VERSION, assessment_id=declaration.assessment_id,
        assessment_track=AssessmentTrack.RETROSPECTIVE, dataset_id=declaration.dataset_id,
        development_cutoff=declaration.development_cutoff,
        forecast_dates=declaration.forecast_dates,
    )
    params[field] = value
    other = AssessmentDeclaration(**params)
    _reject_chain(other, eligible, disclosed, scored, evidence)
    assert ProbeModel.events == events


@pytest.mark.parametrize('field,value', [
    ('selected_model', 'LSTM'),
    ('decision_digest', 'sha256:other'),
    ('plan_digest', 'sha256:other'),
    ('execution_artifact_digest', 'sha256:other'),
    ('predicted_microVolt', [100.0, 101.0]),
])
def test_redigested_b1_forecast_or_provenance_substitution_fails_closed(
    processed_training, full_frame, declaration, field, value,
):
    _reset_probe()
    eligible, disclosed, scored, evidence = _complete(processed_training, full_frame, declaration)
    events = deepcopy(ProbeModel.events)
    forged = deepcopy(eligible)
    forged[field] = value
    _redigest(forged, 'eligibility_digest')
    _reject_chain(declaration, forged, disclosed, scored, evidence)
    assert ProbeModel.events == events


@pytest.mark.parametrize('field,value', [
    ('eligibility_digest', 'sha256:other'),
    ('execution_artifact_digest', 'sha256:other'),
    ('forecast_dates', ['2024-06-01', '2024-09-01']),
    ('observed_microVolt', [100.0, 101.0]),
])
def test_redigested_b2_disclosure_substitution_fails_closed(
    processed_training, full_frame, declaration, field, value,
):
    _reset_probe()
    eligible, disclosed, scored, evidence = _complete(processed_training, full_frame, declaration)
    events = deepcopy(ProbeModel.events)
    forged = deepcopy(disclosed)
    forged[field] = value
    _redigest(forged, 'disclosure_digest')
    _reject_chain(declaration, eligible, forged, scored, evidence)
    assert ProbeModel.events == events


@pytest.mark.parametrize('path,value', [
    (('metrics', 'rmse_microVolt'), 999.0),
    (('metrics', 'bias_microVolt'), 999.0),
    (('denominators', 'rmse'), 1),
    (('by_date', 0, 'signed_error_microVolt'), 999.0),
    (('by_date', 1, 'squared_error_microVolt2'), 999.0),
    (('execution_artifact_digest',), 'sha256:other'),
])
def test_redigested_b3_metric_and_chain_substitution_fails_closed(
    processed_training, full_frame, declaration, path, value,
):
    _reset_probe()
    eligible, disclosed, scored, evidence = _complete(processed_training, full_frame, declaration)
    events = deepcopy(ProbeModel.events)
    forged = deepcopy(scored)
    node = forged
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value
    _redigest(forged, 'scoring_digest')
    _reject_chain(declaration, eligible, disclosed, forged, evidence)
    assert ProbeModel.events == events


@pytest.mark.parametrize('path,value', [
    (('metrics', 'mae_microVolt'), 999.0),
    (('denominators', 'bias'), 1),
    (('by_date', 0, 'absolute_error_microVolt'), 999.0),
    (('scoring_digest',), 'sha256:other'),
    (('authentication',), 'established'),
    (('scientific_independence',), 'established'),
])
def test_redigested_b4_evidence_substitution_fails_closed(
    processed_training, full_frame, declaration, path, value,
):
    _reset_probe()
    eligible, disclosed, scored, evidence = _complete(processed_training, full_frame, declaration)
    events = deepcopy(ProbeModel.events)
    forged = deepcopy(evidence)
    node = forged
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value
    _redigest(forged, 'evaluation_evidence_digest')
    assert not verify_holdout_evaluation_evidence(
        forged, declaration=declaration, eligibility_snapshot=eligible,
        disclosure_snapshot=disclosed, scoring_snapshot=scored,
    )
    assert ProbeModel.events == events


def test_isolation_rejects_extra_future_dates_before_model_execution(
    processed_training, full_frame, declaration,
):
    _reset_probe()
    extra = pd.DataFrame({'ds': pd.to_datetime(['2024-09-01']), 'microVolt': [999.0]})
    contaminated = pd.concat([full_frame, extra], ignore_index=True)
    with pytest.raises(ValueError, match='holdout calendar'):
        isolate_declared_holdout(BatteryDataset(contaminated), declaration=declaration)
    assert ProbeModel.events == []


def test_isolation_rejects_full_dataset_as_training_input(
    processed_training, full_frame, declaration,
):
    _reset_probe()
    full = BatteryDataset(full_frame)
    isolated = isolate_declared_holdout(full, declaration=declaration)
    with pytest.raises(ValueError, match='training input differs'):
        require_isolated_training_dataset(full, isolated=isolated)
    assert ProbeModel.events == []


def test_unverified_observation_cannot_be_disclosed_or_scored(
    processed_training, full_frame, declaration,
):
    _reset_probe()
    isolated, eligible = _case(processed_training, full_frame, declaration)
    events = deepcopy(ProbeModel.events)
    with pytest.raises(ValueError, match='coverage'):
        disclose_holdout_targets(
            declaration=declaration, eligibility_snapshot=eligible,
            isolated=isolated, observability_report=_report(declaration, isolated, verified=False),
        )
    assert ProbeModel.events == events


def test_cross_run_valid_artifact_substitution_rejected(
    processed_training, full_frame, declaration,
):
    _reset_probe()
    eligible_a, disclosed_a, scored_a, evidence_a = _complete(processed_training, full_frame, declaration)
    changed = full_frame.copy(deep=True)
    changed.loc[changed['ds'] > pd.Timestamp(declaration.development_cutoff), 'microVolt'] = [31.9, 32.506]
    _reset_probe()
    eligible_b, disclosed_b, scored_b, evidence_b = _complete(processed_training, changed, declaration)
    assert eligible_a == eligible_b
    events = deepcopy(ProbeModel.events)
    _reject_chain(declaration, eligible_a, disclosed_b, scored_a, evidence_a)
    assert not verify_holdout_evaluation_evidence(
        evidence_a, declaration=declaration, eligibility_snapshot=eligible_b,
        disclosure_snapshot=disclosed_b, scoring_snapshot=scored_b,
    )
    assert verify_holdout_evaluation_evidence(
        evidence_b, declaration=declaration, eligibility_snapshot=eligible_b,
        disclosure_snapshot=disclosed_b, scoring_snapshot=scored_b,
    )
    assert ProbeModel.events == events
