"""
Cross-model forecast comparison engine.
"""

from __future__ import annotations

from .result import ComparisonResult


class ForecastComparison:
    """
    Compare stored forecasting experiment results.

    The comparison engine consumes ResultPackage objects.
    It never reruns forecasting models.
    """

    METRICS = (
        "rmse",
        "mae",
        "mape",
    )

    def __init__(
        self,
        runs,
    ):

        self.runs = list(runs)

        if not self.runs:

            raise ValueError(
                "At least one result package is required."
            )

        self._validate_battery()

    def _validate_battery(self):

        batteries = {
            run.battery
            for run in self.runs
        }

        if len(batteries) != 1:

            raise ValueError(
                "All comparison runs must belong "
                "to the same battery."
            )

        self.battery = next(
            iter(batteries)
        )

    def models(self):

        return [
            run.model
            for run in self.runs
        ]

    def _extract_metrics(self):

        metrics = {}

        for run in self.runs:

            if run.evaluation is None:
                continue

            model_metrics = {}

            for metric in self.METRICS:

                value = run.evaluation.get(
                    metric
                )

                if value is not None:

                    model_metrics[metric] = float(
                        value
                    )

            if model_metrics:

                metrics[run.model] = (
                    model_metrics
                )

        return metrics

    def metric_table(self):

        return self._extract_metrics()

    def rank(
        self,
        metric,
    ):

        metric = metric.lower()

        if metric not in self.METRICS:

            raise ValueError(
                f"Unsupported metric: {metric}"
            )

        metrics = self._extract_metrics()

        ranked = []

        for model, values in metrics.items():

            if metric not in values:
                continue

            ranked.append(
                (
                    model,
                    values[metric],
                )
            )

        ranked.sort(
            key=lambda item: item[1]
        )

        return [

            {
                "rank": index,
                "model": model,
                "value": value,
            }

            for index, (
                model,
                value,
            ) in enumerate(
                ranked,
                start=1,
            )

        ]

    def best(
        self,
        metric,
    ):

        ranking = self.rank(
            metric
        )

        if not ranking:
            return None

        return ranking[0]["model"]

    def compare(self):

        metrics = self._extract_metrics()

        rankings = {

            metric:
                self.rank(metric)

            for metric in self.METRICS

        }

        best_models = {

            metric:
                self.best(metric)

            for metric in self.METRICS

        }

        return ComparisonResult(

            battery=self.battery,

            metrics=metrics,

            rankings=rankings,

            best_models=best_models,

            metadata={

                "comparison_metrics":
                    list(self.METRICS),

                "lower_is_better": True,

            },

        )

    def summary(self):

        return self.compare().summary()