"""Persistence for standardized multi-model benchmark evidence."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from zenerestimation.comparison import (
    ComparisonResult,
)


class BenchmarkArtifact:
    """Persist standardized multi-model benchmark evidence.

    BenchmarkArtifact is intentionally a thin serialization and
    persistence layer around ComparisonResult.

    It does not evaluate forecasting models, recompute metrics,
    rerank models, select parameters, or select neural seeds.
    """

    def __init__(
        self,
        result: ComparisonResult,
    ) -> None:

        if not isinstance(
            result,
            ComparisonResult,
        ):
            raise TypeError(
                "result must be a ComparisonResult"
            )

        self._result = result

    @property
    def result(
        self,
    ) -> ComparisonResult:
        """Return the underlying comparison result."""

        return self._result

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Return standardized benchmark evidence."""

        source = self._result.to_dict()

        payload = {
            "status": "benchmarked",
            "battery": source["battery"],
            "models": deepcopy(
                source["models"]
            ),
            "metrics": deepcopy(
                source["metrics"]
            ),
            "rankings": deepcopy(
                source["rankings"]
            ),
            "best_models": deepcopy(
                source["best_models"]
            ),
            "metadata": deepcopy(
                source["metadata"]
            ),
        }

        return deepcopy(
            payload
        )

    def save(
        self,
        directory: str | Path,
    ) -> dict[str, Path]:
        """Persist benchmark evidence as JSON and CSV.

        Parameters
        ----------
        directory:
            Existing or new directory in which benchmark artifacts
            should be stored.

        Returns
        -------
        dict[str, Path]
            Paths of the generated ``benchmark.json`` and
            ``benchmark.csv`` files.
        """

        directory = Path(
            directory
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = (
            directory / "benchmark.json"
        )

        csv_path = (
            directory / "benchmark.csv"
        )

        self._save_json(
            json_path
        )

        self._save_csv(
            csv_path
        )

        return {
            "json": json_path,
            "csv": csv_path,
        }

    def _save_json(
        self,
        filename: Path,
    ) -> None:
        """Write the complete benchmark contract as JSON."""

        with filename.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.to_dict(),
                file,
                indent=2,
            )

    def _save_csv(
        self,
        filename: Path,
    ) -> None:
        """Write the standardized metric table as CSV."""

        metrics = self._result.metrics

        with filename.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "model",
                    "rmse",
                    "mae",
                    "mape",
                ],
            )

            writer.writeheader()

            for model in self._result.models:

                model_metrics = metrics[
                    model
                ]

                writer.writerow(
                    {
                        "model": model,
                        "rmse": model_metrics[
                            "rmse"
                        ],
                        "mae": model_metrics[
                            "mae"
                        ],
                        "mape": model_metrics[
                            "mape"
                        ],
                    }
                )

    def __repr__(
        self,
    ) -> str:

        return (
            "BenchmarkArtifact("
            f"battery={self._result.battery!r}, "
            f"models={len(self._result.models)}"
            ")"
        )