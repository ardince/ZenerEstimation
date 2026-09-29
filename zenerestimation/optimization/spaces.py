"""
Optimization parameter spaces.

Provides deterministic parameter-space generation for
forecasting-model optimization.
"""

from __future__ import annotations

from itertools import product

import math
from typing import Iterable


class ARIMAParameterSpace:
    """
    Generate deterministic ARIMA(p, d, q) candidate configurations.

    Parameters
    ----------
    p:
        Autoregressive orders.

    d:
        Differencing orders.

    q:
        Moving-average orders.
    """

    def __init__(
        self,
        *,
        p,
        d,
        q,
    ) -> None:

        self.p = self._validate_orders(
            p,
            "p",
        )

        self.d = self._validate_orders(
            d,
            "d",
        )

        self.q = self._validate_orders(
            q,
            "q",
        )

    def candidates(
        self,
    ) -> tuple[dict, ...]:
        """
        Return the complete Cartesian product of ARIMA orders.
        """

        return tuple(
            {
                "order": (
                    p_value,
                    d_value,
                    q_value,
                )
            }
            for (
                p_value,
                d_value,
                q_value,
            ) in product(
                self.p,
                self.d,
                self.q,
            )
        )

    @property
    def candidate_count(
        self,
    ) -> int:
        """
        Number of parameter combinations.
        """

        return (
            len(self.p)
            * len(self.d)
            * len(self.q)
        )

    @staticmethod
    def _validate_orders(
        values,
        name: str,
    ) -> tuple[int, ...]:

        if isinstance(
            values,
            (str, bytes),
        ):
            raise TypeError(
                f"{name} must be an iterable of integers"
            )

        try:
            values = tuple(
                values
            )
        except TypeError as exc:
            raise TypeError(
                f"{name} must be an iterable of integers"
            ) from exc

        if not values:
            raise ValueError(
                f"{name} cannot be empty"
            )

        for value in values:

            if (
                not isinstance(value, int)
                or isinstance(value, bool)
            ):
                raise TypeError(
                    f"{name} values must be integers"
                )

            if value < 0:
                raise ValueError(
                    f"{name} values must be "
                    "greater than or equal to zero"
                )

        return values

    def __repr__(
        self,
    ) -> str:

        return (
            "ARIMAParameterSpace("
            f"p={self.p!r}, "
            f"d={self.d!r}, "
            f"q={self.q!r}"
            ")"
        )


class KalmanParameterSpace:
    """Deterministic parameter space for Kalman optimization.

    The parameter space contains only tunable numerical parameters.

    Fixed model configuration such as ``dt`` and ``adaptive`` belongs
    to the model factory rather than the optimization space.
    """

    def __init__(
        self,
        *,
        process_noise: Iterable[float],
        drift_noise: Iterable[float],
        regime_factor: Iterable[float],
        regime_multiplier: Iterable[float],
    ) -> None:
        self.process_noise = self._validate_dimension(
            "process_noise",
            process_noise,
        )
        self.drift_noise = self._validate_dimension(
            "drift_noise",
            drift_noise,
        )
        self.regime_factor = self._validate_dimension(
            "regime_factor",
            regime_factor,
        )
        self.regime_multiplier = self._validate_dimension(
            "regime_multiplier",
            regime_multiplier,
        )

    @staticmethod
    def _validate_dimension(
        name: str,
        values: Iterable[float],
    ) -> tuple[float, ...]:
        """Validate one positive finite numeric search dimension."""

        if isinstance(values, (str, bytes)):
            raise TypeError(
                f"{name} must be a non-empty iterable "
                "of positive finite numbers"
            )

        try:
            items = tuple(values)
        except TypeError as exc:
            raise TypeError(
                f"{name} must be a non-empty iterable "
                "of positive finite numbers"
            ) from exc

        if not items:
            raise ValueError(
                f"{name} must not be empty"
            )

        normalized: list[float] = []

        for value in items:
            if isinstance(value, bool):
                raise TypeError(
                    f"{name} values must be numeric"
                )

            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name} values must be numeric"
                )

            numeric = float(value)

            if not math.isfinite(numeric):
                raise ValueError(
                    f"{name} values must be finite"
                )

            if numeric <= 0.0:
                raise ValueError(
                    f"{name} values must be positive"
                )

            normalized.append(numeric)

        return tuple(normalized)

    @property
    def candidate_count(self) -> int:
        """Return the number of Cartesian-product candidates."""

        return (
            len(self.process_noise)
            * len(self.drift_noise)
            * len(self.regime_factor)
            * len(self.regime_multiplier)
        )

    def candidates(
        self,
    ) -> tuple[dict[str, float], ...]:
        """Return candidates in deterministic Cartesian-product order."""

        return tuple(
            {
                "process_noise": process_noise,
                "drift_noise": drift_noise,
                "regime_factor": regime_factor,
                "regime_multiplier": regime_multiplier,
            }
            for (
                process_noise,
                drift_noise,
                regime_factor,
                regime_multiplier,
            ) in product(
                self.process_noise,
                self.drift_noise,
                self.regime_factor,
                self.regime_multiplier,
            )
        )

    def __repr__(self) -> str:
        return (
            "KalmanParameterSpace("
            f"process_noise={self.process_noise!r}, "
            f"drift_noise={self.drift_noise!r}, "
            f"regime_factor={self.regime_factor!r}, "
            f"regime_multiplier={self.regime_multiplier!r}, "
            f"candidate_count={self.candidate_count}"
            ")"
        )


class LSTMParameterSpace:
    """Explicit Cartesian parameter space for LSTM optimization."""

    def __init__(
        self,
        *,
        window,
        units,
    ):
        self.window = self._validate_dimension(
            "window",
            window,
        )
        self.units = self._validate_dimension(
            "units",
            units,
        )

    @staticmethod
    def _validate_dimension(
        name,
        values,
    ):
        if isinstance(values, (str, bytes)):
            raise TypeError(
                f"{name} must be an iterable of positive integers"
            )

        try:
            values = tuple(values)
        except TypeError as exc:
            raise TypeError(
                f"{name} must be an iterable of positive integers"
            ) from exc

        if not values:
            raise ValueError(
                f"{name} must contain at least one value"
            )

        normalized = []

        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
            ):
                raise TypeError(
                    f"{name} values must be integers"
                )

            if value <= 0:
                raise ValueError(
                    f"{name} values must be positive"
                )

            normalized.append(value)

        return tuple(normalized)

    @property
    def candidate_count(self):
        return (
            len(self.window)
            * len(self.units)
        )

    def candidates(self):
        return [
            {
                "window": window,
                "units": units,
            }
            for window in self.window
            for units in self.units
        ]

    def __repr__(self):
        return (
            "LSTMParameterSpace("
            f"window={self.window!r}, "
            f"units={self.units!r}"
            ")"
        )


class GRUParameterSpace:
    """Explicit Cartesian parameter space for GRU optimization."""

    def __init__(
        self,
        *,
        window,
        units,
    ):
        self.window = self._validate_dimension(
            "window",
            window,
        )
        self.units = self._validate_dimension(
            "units",
            units,
        )

    @staticmethod
    def _validate_dimension(
        name,
        values,
    ):
        if isinstance(values, (str, bytes)):
            raise TypeError(
                f"{name} must be an iterable of positive integers"
            )

        try:
            values = tuple(values)
        except TypeError as exc:
            raise TypeError(
                f"{name} must be an iterable of positive integers"
            ) from exc

        if not values:
            raise ValueError(
                f"{name} must contain at least one value"
            )

        normalized = []

        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
            ):
                raise TypeError(
                    f"{name} values must be integers"
                )

            if value <= 0:
                raise ValueError(
                    f"{name} values must be positive"
                )

            normalized.append(value)

        return tuple(normalized)

    @property
    def candidate_count(self):
        return (
            len(self.window)
            * len(self.units)
        )

    def candidates(self):
        return [
            {
                "window": window,
                "units": units,
            }
            for window in self.window
            for units in self.units
        ]

    def __repr__(self):
        return (
            "GRUParameterSpace("
            f"window={self.window!r}, "
            f"units={self.units!r}"
            ")"
        )


class HybridNeuralParameterSpace:
    """Explicit parameter space for hybrid neural residual models."""

    def __init__(
        self,
        *,
        window,
        units,
    ):
        self.window = self._validate_dimension(
            "window",
            window,
        )
        self.units = self._validate_dimension(
            "units",
            units,
        )

    @staticmethod
    def _validate_dimension(
        name,
        values,
    ):
        if isinstance(
            values,
            (str, bytes),
        ):
            raise TypeError(
                f"{name} must be an iterable of positive integers"
            )

        try:
            values = tuple(values)
        except TypeError as exc:
            raise TypeError(
                f"{name} must be an iterable of positive integers"
            ) from exc

        if not values:
            raise ValueError(
                f"{name} must not be empty"
            )

        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
            ):
                raise TypeError(
                    f"{name} values must be integers"
                )

            if value <= 0:
                raise ValueError(
                    f"{name} values must be positive"
                )

        return values

    @property
    def candidate_count(self):
        return (
            len(self.window)
            * len(self.units)
        )

    def candidates(self):
        return [
            {
                "window": window,
                "units": units,
            }
            for window in self.window
            for units in self.units
        ]

    def __repr__(self):
        return (
            "HybridNeuralParameterSpace("
            f"window={self.window!r}, "
            f"units={self.units!r}"
            ")"
        )