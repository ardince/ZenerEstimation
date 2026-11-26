"""M20-B4: strict, indexed finite parameter spaces (no model execution)."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import prod, isfinite
from typing import Any

from .spaces import (ARIMAParameterSpace, KalmanParameterSpace,
                     LSTMParameterSpace, GRUParameterSpace, HybridNeuralParameterSpace)

NEURAL_FAMILIES = frozenset({'LSTM', 'GRU', 'LinearTrendLSTM', 'KalmanLSTM'})
SPACE_CLASSES = {
    'ARIMA': ARIMAParameterSpace, 'Kalman': KalmanParameterSpace,
    'LSTM': LSTMParameterSpace, 'GRU': GRUParameterSpace,
    'LinearTrendLSTM': HybridNeuralParameterSpace,
    'KalmanLSTM': HybridNeuralParameterSpace,
}
SPACE_VERSION = 'm20_b4_space_v1'


def _canonical(value: Any) -> Any:
    if type(value) is bool:
        return {'type': 'bool', 'value': value}
    if type(value) is int:
        return {'type': 'int', 'value': value}
    if type(value) is float:
        import math
        if not math.isfinite(value):
            raise ValueError('nonfinite parameter')
        return {'type': 'float', 'value': value.hex()}
    if type(value) is str:
        return {'type': 'str', 'value': value}
    if type(value) is tuple:
        return {'type': 'tuple', 'value': [_canonical(v) for v in value]}
    raise TypeError('unsupported parameter type')


def candidate_digest(params: dict[str, Any]) -> str:
    if type(params) is not dict or not params or any(type(k) is not str for k in params):
        raise ValueError('candidate must be a nonempty string-keyed dictionary')
    payload = {k: _canonical(v) for k, v in sorted(params.items())}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return 'sha256:' + sha256(encoded.encode('utf-8')).hexdigest()


@dataclass(frozen=True, slots=True)
class FiniteSearchSpace:
    model_family: str
    names: tuple[str, ...]
    dimensions: tuple[tuple[Any, ...], ...]
    candidate_count: int

    def __post_init__(self) -> None:
        if type(self.model_family) is not str or self.model_family not in SPACE_CLASSES:
            raise ValueError('unsupported model family')
        expected = (('p', 'd', 'q') if self.model_family == 'ARIMA' else
                    ('process_noise', 'drift_noise', 'regime_factor', 'regime_multiplier')
                    if self.model_family == 'Kalman' else None)
        if type(self.names) is not tuple or any(type(n) is not str for n in self.names):
            raise ValueError('invalid dimension names')
        if expected is not None:
            if self.names != expected:
                raise ValueError('invalid family dimension names')
        elif (self.names not in {('window', 'units', 'epochs'),
                                ('window', 'units', 'epochs', 'batch_size')}):
            raise ValueError('invalid neural dimension names')
        if type(self.dimensions) is not tuple or len(self.dimensions) != len(self.names):
            raise ValueError('invalid dimensions')
        for name, values in zip(self.names, self.dimensions):
            if type(values) is not tuple or not values:
                raise ValueError('empty or invalid search dimension')
            seen = set()
            for value in values:
                if self.model_family == 'Kalman':
                    if type(value) is not float or not isfinite(value) or value <= 0:
                        raise ValueError('invalid Kalman parameter')
                elif type(value) is not int or (value < 0 if name in {'p', 'd', 'q'} else value <= 0):
                    raise ValueError('invalid integer parameter')
                identity = json.dumps(_canonical(value), sort_keys=True)
                if identity in seen:
                    raise ValueError('duplicate dimension values')
                seen.add(identity)
        count = prod(len(values) for values in self.dimensions)
        if type(self.candidate_count) is not int or count != self.candidate_count or count > 2**63 - 1:
            raise ValueError('invalid candidate_count')

    @classmethod
    def from_legacy(cls, model_family: str, legacy_space: object) -> 'FiniteSearchSpace':
        expected = SPACE_CLASSES.get(model_family)
        if expected is None or type(legacy_space) is not expected:
            raise TypeError('parameter space does not match model family')
        if model_family == 'ARIMA':
            names = ('p', 'd', 'q')
        elif model_family == 'Kalman':
            names = ('process_noise', 'drift_noise', 'regime_factor', 'regime_multiplier')
        else:
            names = ('window', 'units')
            if legacy_space.epochs is not None:
                names += ('epochs',)
            if legacy_space.batch_size is not None:
                names += ('batch_size',)
            if legacy_space.epochs is None:
                raise ValueError('neural search requires explicit epochs for resource accounting')
        dimensions = tuple(tuple(getattr(legacy_space, name)) for name in names)
        if any(not values for values in dimensions):
            raise ValueError('empty search dimension')
        # Reject duplicate values, including 1 and 1.0 when the legacy space accepts both.
        for values in dimensions:
            identities = [json.dumps(_canonical(v), sort_keys=True) for v in values]
            if len(set(identities)) != len(identities):
                raise ValueError('duplicate dimension values')
        count = prod(len(d) for d in dimensions)
        if count > 2**63 - 1:
            raise ValueError('search space too large for bounded indexing')
        return cls(model_family, names, dimensions, count)

    def candidate_at(self, index: int) -> dict[str, Any]:
        if type(index) is not int or not 0 <= index < self.candidate_count:
            raise IndexError('candidate index outside search space')
        positions = []
        for dimension in reversed(self.dimensions):
            index, remainder = divmod(index, len(dimension))
            positions.append(remainder)
        values = [dimension[pos] for dimension, pos in zip(self.dimensions, reversed(positions))]
        if self.model_family == 'ARIMA':
            return {'order': tuple(values)}
        return dict(zip(self.names, values))

    @property
    def digest(self) -> str:
        payload = {'version': SPACE_VERSION, 'family': self.model_family,
                   'names': self.names,
                   'dimensions': [[_canonical(v) for v in d] for d in self.dimensions]}
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        return 'sha256:' + sha256(encoded.encode()).hexdigest()
