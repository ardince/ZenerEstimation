"""
Forecast comparison result object.
"""

from __future__ import annotations

from copy import deepcopy


class ComparisonResult:
    """
    Immutable container for cross-model comparison results.
    """

    def __init__(
        self,
        battery,
        metrics,
        rankings,
        best_models,
        metadata=None,
    ):

        self.battery = battery

        self._metrics = deepcopy(
            metrics
        )

        self._rankings = deepcopy(
            rankings
        )

        self._best_models = deepcopy(
            best_models
        )

        self._metadata = deepcopy(
            metadata
            if metadata is not None
            else {}
        )

    @property
    def metrics(self):
        return deepcopy(
            self._metrics
        )

    @property
    def rankings(self):
        return deepcopy(
            self._rankings
        )

    @property
    def best_models(self):
        return deepcopy(
            self._best_models
        )

    @property
    def metadata(self):
        return deepcopy(
            self._metadata
        )

    @property
    def models(self):
        return list(
            self._metrics.keys()
        )

    def summary(self):

        return {

            "battery":
                self.battery,

            "models":
                self.models,

            "metrics":
                deepcopy(
                    self._metrics
                ),

            "rankings":
                deepcopy(
                    self._rankings
                ),

            "best_models":
                deepcopy(
                    self._best_models
                ),

            "metadata":
                deepcopy(
                    self._metadata
                ),

        }

    def to_dict(self):
        return self.summary()

    def __repr__(self):

        return (

            "ComparisonResult("
            f"battery='{self.battery}', "
            f"models={len(self.models)})"

        )