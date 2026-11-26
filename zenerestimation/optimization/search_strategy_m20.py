"""M20-B4: bounded deterministic proposals and auditable feedback.

No dataset, model factory, grant, or optimizer is accepted. The B5 runner must
consume authorization and verify development data before calling propose().
A proposal reserves one trial and its planned epoch-fold cost; failures do not
refund either. This is local workflow accounting, not a security boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from hashlib import sha256
import json
import math
import random
from typing import Any

import numpy as np

from .authorization import OptimizationExecutionRequest
from .search_space_m20 import FiniteSearchSpace, NEURAL_FAMILIES, candidate_digest

STRATEGY_VERSION = 'm20_b4_strategy_v1'
BAYES_POOL_LIMIT = 4096  # bound candidate matrix work and memory
BAYES_TRIAL_LIMIT = 128  # bound cubic GP solve and quadratic Gram memory


@dataclass(frozen=True, slots=True)
class SearchProposal:
    sequence: int
    index: int
    candidate_id: str
    params: dict[str, Any]
    planned_epoch_folds: int

    def __post_init__(self):
        from copy import deepcopy
        object.__setattr__(self, 'params', deepcopy(self.params))


@dataclass(frozen=True, slots=True)
class SearchFeedback:
    sequence: int
    candidate_id: str
    status: str
    score: float | None


class BoundedSearchEngine:
    """Single-outstanding-proposal search; not thread-safe or a run authority.

    Bayesian mode uses an RBF-kernel Gaussian-process posterior with fixed
    unit length-scale on normalized ordinal dimension positions, noise 1e-6,
    and UCB exploration coefficient 1.0 for minimization (mean - std).
    Initial proposal is seeded random; ties resolve by candidate index.
    The finite candidate pool is capped at BAYES_POOL_LIMIT.
    """

    def __init__(self, *, request: OptimizationExecutionRequest,
                 parameter_space: FiniteSearchSpace, seed: int, fold_count: int):
        if type(request) is not OptimizationExecutionRequest:
            raise TypeError('request must be OptimizationExecutionRequest')
        if type(parameter_space) is not FiniteSearchSpace or parameter_space.model_family != request.model_family:
            raise ValueError('parameter space / request family mismatch')
        if type(seed) is not int or seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if type(fold_count) is not int or fold_count < 1:
            raise ValueError('fold_count must be a positive integer')
        if request.model_family in NEURAL_FAMILIES and 'epochs' not in parameter_space.names:
            raise ValueError('neural space requires epochs')
        if request.strategy == 'bayesian' and parameter_space.candidate_count > BAYES_POOL_LIMIT:
            raise ValueError('Bayesian candidate pool exceeds bounded limit')
        if request.strategy == 'bayesian' and request.max_trials > BAYES_TRIAL_LIMIT:
            raise ValueError('Bayesian trial limit exceeded')
        self.request = request
        self.space = parameter_space
        self.seed = seed
        self.fold_count = fold_count
        self._rng = random.Random(seed)
        self._used: set[int] = set()
        self._pending: SearchProposal | None = None
        self._feedback: list[SearchFeedback] = []
        self._proposals: list[SearchProposal] = []  # private, independent audit copies
        self._reserved_cost = 0
        self._stop_reason: str | None = None
        self._random_order = (iter(self._rng.sample(range(parameter_space.candidate_count),
                                      min(request.max_trials, parameter_space.candidate_count)))
                              if request.strategy == 'random' else None)

    def _cost(self, params: dict[str, Any]) -> int:
        if self.request.model_family not in NEURAL_FAMILIES:
            return 0
        epochs = params.get('epochs')
        if type(epochs) is not int or epochs < 1:
            raise ValueError('neural candidate requires positive epochs')
        return epochs * self.fold_count

    def _bayesian_index(self) -> int:
        available = [i for i in range(self.space.candidate_count) if i not in self._used]
        observations = [(p.index, f.score) for p, f in zip(self._proposals, self._feedback)
                        if f.status == 'evaluated']
        if not observations:
            return available[self._rng.randrange(len(available))]
        def features(index: int) -> np.ndarray:
            positions = []
            for dim in reversed(self.space.dimensions):
                index, pos = divmod(index, len(dim))
                positions.append(pos / max(1, len(dim) - 1))
            return np.asarray(list(reversed(positions)), dtype=float)
        x = np.stack([features(i) for i, _ in observations])
        # Scale BEFORE any matrix operation to avoid overflow in the GP posterior.
        # Preserve relative scores; do not silently clip extreme observations.
        scores = [float(s) for _, s in observations]
        scale = max(abs(score) for score in scores)
        y = np.asarray([score / scale for score in scores] if scale else scores,
                       dtype=float)
        xa = np.stack([features(i) for i in available])
        def kernel(a, b):
            return np.exp(-0.5 * np.sum((a[:, None, :] - b[None, :, :])**2, axis=2))
        gram = kernel(x, x) + 1e-6 * np.eye(len(x))
        cross = kernel(xa, x)
        try:
            alpha = np.linalg.solve(gram, y)
            solved = np.linalg.solve(gram, cross.T)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError('Bayesian surrogate solve failed') from exc
        mean = cross @ alpha
        variance = np.maximum(0.0, 1.0 - np.sum(cross * solved.T, axis=1))
        acquisition = mean - np.sqrt(variance)
        if not (np.isfinite(mean).all() and np.isfinite(variance).all()
                and np.isfinite(acquisition).all()):
            raise RuntimeError('Bayesian surrogate produced nonfinite acquisition')
        return available[int(np.argmin(acquisition))]

    def propose(self) -> SearchProposal | None:
        if self._pending is not None:
            raise RuntimeError('record feedback before requesting another proposal')
        if self._stop_reason is not None:
            return None
        if len(self._proposals) >= self.request.max_trials:
            self._stop_reason = 'max_trials'
            return None
        if len(self._used) == self.space.candidate_count:
            self._stop_reason = 'space_exhausted'
            return None
        if self.request.strategy == 'grid':
            index = len(self._proposals)
        elif self.request.strategy == 'random':
            index = next(self._random_order)
        else:
            index = self._bayesian_index()
        params = self.space.candidate_at(index)
        cost = self._cost(params)
        limit = self.request.max_epoch_folds
        if limit is not None and self._reserved_cost + cost > limit:
            self._stop_reason = 'max_epoch_folds'
            return None
        proposal = SearchProposal(len(self._proposals) + 1, index,
                                  candidate_digest(params), params, cost)
        self._used.add(index)
        self._proposals.append(deepcopy(proposal))
        self._reserved_cost += cost
        self._pending = proposal
        return proposal

    def record_result(self, proposal: SearchProposal, *, status: str,
                      score: float | None = None) -> None:
        if self._pending is None or proposal is not self._pending:
            raise ValueError('feedback does not match outstanding proposal')
        # Object identity alone is insufficient: the returned dataclass contains
        # a mutable dict. Reject tampering before accepting any feedback.
        private = self._proposals[-1]
        if (proposal.sequence != private.sequence or proposal.index != private.index
                or proposal.candidate_id != private.candidate_id
                or proposal.planned_epoch_folds != private.planned_epoch_folds
                or proposal.params != private.params
                or candidate_digest(proposal.params) != private.candidate_id):
            raise ValueError('proposal integrity mismatch')
        if status not in {'evaluated', 'failed'} or type(status) is not str:
            raise ValueError('status must be evaluated or failed')
        if status == 'evaluated':
            if type(score) not in {int, float} or not math.isfinite(score):
                raise ValueError('evaluated feedback requires a finite score')
            score = float(score)
        elif score is not None:
            raise ValueError('failed feedback must not carry a score')
        self._feedback.append(SearchFeedback(proposal.sequence, proposal.candidate_id,
                                             status, score))
        self._pending = None

    def snapshot(self) -> dict[str, Any]:
        from copy import deepcopy
        return deepcopy({
            'version': STRATEGY_VERSION, 'request_digest': self.request.digest,
            'space_digest': self.space.digest, 'strategy': self.request.strategy,
            'seed': self.seed, 'fold_count': self.fold_count,
            'max_trials': self.request.max_trials,
            'max_epoch_folds': self.request.max_epoch_folds,
            'proposed_count': len(self._proposals),
            'completed_count': sum(f.status == 'evaluated' for f in self._feedback),
            'failed_count': sum(f.status == 'failed' for f in self._feedback),
            'reserved_epoch_folds': self._reserved_cost,
            'stop_reason': self._stop_reason,
            'pending_sequence': self._pending.sequence if self._pending else None,
            'proposals': [{'sequence': p.sequence, 'index': p.index,
                           'candidate_id': p.candidate_id, 'params': p.params,
                           'planned_epoch_folds': p.planned_epoch_folds}
                          for p in self._proposals],
            'feedback': [{'sequence': f.sequence, 'candidate_id': f.candidate_id,
                          'status': f.status, 'score': f.score} for f in self._feedback],
        })
