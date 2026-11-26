"""M20-B5-C: process-local, single-use authorized optimization execution.

Approval is an explicit trusted workflow operation, NOT authentication of a
human or protection against arbitrary code running in this Python process.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from threading import RLock
from typing import Any, Callable

from .authorization import (LocalOptimizationCoordinator,
                            OptimizationExecutionGrant, OptimizationExecutionRequest)
from .development_binding import (build_validation_protocol_digest,
                                  verify_authorized_development_dataset)
from .evaluator import ForecastAlignmentError, OptimizationEvaluator
from .candidate import CandidateResult
from .result import OptimizationResult
from .search_space_m20 import FiniteSearchSpace, candidate_digest
from .search_strategy_m20 import BoundedSearchEngine
from .internal_provenance import build_internal_validation_context

PLAN_VERSION = 'm20_b5_execution_plan_v1'
OUTCOME_VERSION = 'm20_b5_execution_outcome_v1'
_METRICS = frozenset({'rmse', 'mae', 'mape'})


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    return 'sha256:' + sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ApprovedExecutionPlan:
    request_digest: str
    model_family: str
    strategy: str
    selection_metric: str
    seed: int
    search_space_digest: str
    validation_protocol_digest: str
    model_factory_id: str
    preprocessing_policy: str = 'none'
    require_verified_alignment: bool = True
    record_internal_provenance: bool = True
    plan_version: str = PLAN_VERSION

    def __post_init__(self):
        if self.plan_version != PLAN_VERSION:
            raise ValueError('unsupported plan version')
        if type(self.model_family) is not str or not self.model_family:
            raise ValueError('model_family required')
        if type(self.strategy) is not str or not self.strategy:
            raise ValueError('strategy required')
        if type(self.selection_metric) is not str or self.selection_metric not in _METRICS:
            raise ValueError('unsupported selection metric')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if self.require_verified_alignment is not True or self.record_internal_provenance is not True:
            raise ValueError('strict alignment and provenance are mandatory')
        if type(self.model_factory_id) is not str or not self.model_factory_id.strip():
            raise ValueError('model_factory_id required')
        if type(self.preprocessing_policy) is not str or not self.preprocessing_policy.strip():
            raise ValueError('preprocessing_policy required')
        for name in ('request_digest', 'search_space_digest', 'validation_protocol_digest'):
            value = getattr(self, name)
            if type(value) is not str or len(value) != 71 or not value.startswith('sha256:') or any(c not in '0123456789abcdef' for c in value[7:]):
                raise ValueError(f'invalid {name}')

    @property
    def digest(self) -> str:
        return _digest({name: getattr(self, name) for name in self.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class AuthorizedOptimizationOutcome:
    status: str
    request_digest: str
    execution_plan_digest: str
    grant_audit: dict[str, Any]
    development_binding_evidence: dict[str, Any] | None
    search_snapshot: dict[str, Any] | None
    candidate_results: tuple[CandidateResult, ...]
    optimization_result: OptimizationResult | None
    failure: dict[str, str] | None
    version: str = OUTCOME_VERSION

    def __post_init__(self):
        if type(self.status) is not str or self.status not in {'COMPLETED', 'ALL_FAILED', 'ABORTED'}:
            raise ValueError('invalid outcome status')
        if type(self.candidate_results) is not tuple or any(type(c) is not CandidateResult for c in self.candidate_results):
            raise TypeError('candidate_results must be a tuple of CandidateResult')
        if self.status == 'COMPLETED' and (self.optimization_result is None or self.failure is not None):
            raise ValueError('completed outcome requires optimization result and no failure')
        if self.status == 'ALL_FAILED' and (self.optimization_result is not None or self.failure is not None):
            raise ValueError('all-failed outcome cannot contain optimization result or failure')
        if self.status == 'ABORTED' and (self.optimization_result is not None or self.failure is None):
            raise ValueError('aborted outcome requires failure and no optimization result')
        for name in ('grant_audit', 'development_binding_evidence', 'search_snapshot', 'failure'):
            object.__setattr__(self, name, deepcopy(getattr(self, name)))
        # CandidateResult has frozen fields but mutable nested dictionaries.
        object.__setattr__(self, 'candidate_results', deepcopy(self.candidate_results))
        object.__setattr__(self, 'optimization_result', deepcopy(self.optimization_result))

    def to_dict(self) -> dict[str, Any]:
        return deepcopy({
            'version': self.version, 'status': self.status,
            'request_digest': self.request_digest,
            'execution_plan_digest': self.execution_plan_digest,
            'grant_audit': self.grant_audit,
            'development_binding_evidence': self.development_binding_evidence,
            'search_snapshot': self.search_snapshot,
            'candidate_results': [c.to_dict() for c in self.candidate_results],
            'optimization_result': (self.optimization_result.to_dict()
                                    if self.optimization_result is not None else None),
            'failure': self.failure,
        })


class CandidateModelError(Exception):
    """Exception originating in approved model construction, fit, or prediction."""

    def __init__(self, original: Exception):
        super().__init__(str(original))
        self.original = original


def _guarded_factory(factory):
    """Tag model-originated errors without catching evaluator contract failures."""
    class GuardedModel:
        def __init__(self, model):
            self._model = model

        def fit(self, dataset):
            try:
                return self._model.fit(dataset)
            except Exception as exc:
                raise CandidateModelError(exc) from exc

        def predict(self, steps):
            try:
                return self._model.predict(steps)
            except Exception as exc:
                raise CandidateModelError(exc) from exc

        def __getattr__(self, name):
            return getattr(self._model, name)

    def create(**params):
        try:
            return GuardedModel(factory(**params))
        except Exception as exc:
            raise CandidateModelError(exc) from exc
    return create


class AuthorizedOptimizationRunner:
    """Trusted local plan approval + execution; one plan binding per grant.

    The caller must restrict access to approve_plan and the model factory.
    This does not provide an adversarial in-process security boundary.
    """

    def __init__(self, coordinator: LocalOptimizationCoordinator):
        if type(coordinator) is not LocalOptimizationCoordinator:
            raise TypeError('coordinator must be LocalOptimizationCoordinator')
        self._coordinator = coordinator
        self._lock = RLock()
        self._approved: dict[int, tuple[object, ...]] = {}

    def approve_plan(self, *, grant: OptimizationExecutionGrant,
                     request: OptimizationExecutionRequest,
                     plan: ApprovedExecutionPlan, parameter_space: FiniteSearchSpace,
                     model_factory: Callable[..., Any], preprocessor=None) -> None:
        """Explicit trusted approval of the exact callable and configuration."""
        with self._lock:
            self._coordinator.verify(grant, request)
            self._check_plan(request, plan, parameter_space, model_factory, preprocessor)
            key = id(grant)
            if key in self._approved:
                raise PermissionError('grant already has an approved execution plan')
            # Freeze mutable approved configuration at the approval boundary.
            # Callable code/closure identity remains a trusted-process assumption.
            self._approved[key] = (grant, request.digest, plan.digest, deepcopy(parameter_space),
                                   model_factory, deepcopy(preprocessor))

    @staticmethod
    def _check_plan(request, plan, space, factory, preprocessor):
        if type(request) is not OptimizationExecutionRequest or type(plan) is not ApprovedExecutionPlan:
            raise TypeError('invalid request or plan')
        if type(space) is not FiniteSearchSpace:
            raise TypeError('parameter_space must be FiniteSearchSpace')
        if not callable(factory):
            raise TypeError('model_factory must be callable')
        if (plan.request_digest != request.digest or
            plan.model_family != request.model_family or
            plan.strategy != request.strategy or
            plan.search_space_digest != space.digest or
            plan.validation_protocol_digest != request.validation_protocol_digest or
            space.model_family != request.model_family):
            raise ValueError('execution plan / request / search space mismatch')
        if plan.preprocessing_policy == 'none' and preprocessor is not None:
            raise ValueError('unexpected preprocessor')
        if plan.preprocessing_policy != 'none':
            if preprocessor is None or type(preprocessor).__name__ != plan.preprocessing_policy:
                raise ValueError('preprocessing policy mismatch')
            if not callable(getattr(preprocessor, 'fit_transform', None)):
                raise TypeError('preprocessor must support fit_transform')

    def run(self, *, grant: OptimizationExecutionGrant,
            request: OptimizationExecutionRequest, plan: ApprovedExecutionPlan,
            declaration, source_dataset, training_dataset, splitter) -> AuthorizedOptimizationOutcome:
        """Preflight before consumption; after consumption return audited outcomes."""
        with self._lock:
            self._coordinator.verify(grant, request)
            approved = self._approved.get(id(grant))
            if approved is None or approved[0] is not grant or approved[1] != request.digest or approved[2] != plan.digest:
                raise PermissionError('no trusted approval for this grant and plan')
            _, _, _, space, factory, preprocessor = approved
            # An approved preprocessor is a detached snapshot, not the mutable
            # object originally supplied to approve_plan().
            self._check_plan(request, plan, space, factory, preprocessor)
            binding = verify_authorized_development_dataset(
                request, declaration=declaration, source_dataset=source_dataset,
                training_dataset=training_dataset, splitter=splitter)
            # Construct a fresh evaluator from the approved settings, not a caller-supplied evaluator.
            verified_development = deepcopy(binding.development)
            verified_splitter = deepcopy(splitter)
            context = build_internal_validation_context(
                verified_development, verified_splitter, dataset_id=request.dataset_id)
            if (context['dataset_fingerprint'] != request.development_fingerprint or
                build_validation_protocol_digest(context) != request.validation_protocol_digest):
                raise ValueError('development context changed before execution')
            folds = verified_splitter.split(verified_development)
            if len(folds) != context['folds']:
                raise ValueError('verified fold count mismatch')
            engine = BoundedSearchEngine(request=request, parameter_space=space,
                                         seed=plan.seed, fold_count=len(folds))
            evaluator = OptimizationEvaluator(
                verified_splitter, metric=plan.selection_metric,
                preprocessor=deepcopy(preprocessor), require_verified_alignment=True,
                record_internal_provenance=True, accuracy_dataset_id=request.dataset_id)
            # This atomic transition, not verify(), is the execution permission.
            self._coordinator.consume(grant, request)
            del self._approved[id(grant)]

        results: list[CandidateResult] = []
        failure = None
        status = 'ABORTED'
        try:
            while True:
                proposal = engine.propose()
                if proposal is None:
                    break
                try:
                    result = evaluator.evaluate(
                        dataset=deepcopy(verified_development), model_factory=_guarded_factory(factory),
                        params=deepcopy(proposal.params))
                except CandidateModelError as exc:
                    result = CandidateResult.failed(params=proposal.params,
                                                    error=exc.original,
                                                    metadata={'validation': 'internal_temporal'})
                if type(result) is not CandidateResult or candidate_digest(result.params) != proposal.candidate_id:
                    raise ValueError('candidate result identity mismatch')
                if result.status == 'evaluated':
                    provenance = result.metadata.get('internal_provenance')
                    if provenance != context:
                        raise ValueError('candidate provenance mismatch')
                    alignments = result.metadata.get('fold_alignment')
                    if (type(alignments) is not list or len(alignments) != len(folds) or
                        any(type(a) is not dict or a.get('status') != 'verified' or
                            a.get('fold_index') != i for i, a in enumerate(alignments))):
                        raise ValueError('candidate alignment evidence missing or unverified')
                    if result.metadata.get('selection_metric') != plan.selection_metric:
                        raise ValueError('selection metric mismatch')
                    if result.score != result.metrics.get(plan.selection_metric):
                        raise ValueError('candidate score / metric mismatch')
                if result.status == 'failed' and (result.score is not None or result.metrics):
                    raise ValueError('failed candidate carries score or metrics')
                engine.record_result(proposal, status=result.status, score=result.score)
                results.append(deepcopy(result))
            successful = [r for r in results if r.status == 'evaluated']
            if successful:
                best = min(successful, key=lambda r: r.score)
                optimized = OptimizationResult(
                    model=request.model_family, metric=plan.selection_metric,
                    best_params=best.params, best_score=best.score,
                    candidates=tuple(deepcopy(results)),
                    metadata={'request_digest': request.digest,
                              'execution_plan_digest': plan.digest,
                              'search': request.strategy, 'objective': 'minimize'})
                status = 'COMPLETED'
            else:
                optimized = None
                status = 'ALL_FAILED'
        except Exception as exc:
            optimized = None
            failure = {'category': 'CONTRACT_VIOLATION' if isinstance(exc, (ForecastAlignmentError, ValueError, PermissionError, AssertionError)) else 'EXECUTION_ABORTED',
                       'error_type': type(exc).__name__, 'error_message': str(exc)}
            status = 'ABORTED'
        return AuthorizedOptimizationOutcome(
            status=status, request_digest=request.digest,
            execution_plan_digest=plan.digest, grant_audit=grant.to_dict(),
            development_binding_evidence=binding.evidence,
            search_snapshot=engine.snapshot(), candidate_results=tuple(results),
            optimization_result=optimized, failure=failure)
