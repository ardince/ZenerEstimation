"""
Experiment metadata container.
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field


@dataclass
class Experiment:
    """
    Stores metadata describing one forecasting experiment.
    """

    battery: str

    model: str

    version: str

    execution_time: float

    horizon: int

    id: int | None = None

    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        """
        Convert experiment into a JSON-serializable dictionary.
        """

        return {
            "id": self.id,
            "battery": self.battery,
            "model": self.model,
            "version": self.version,
            "execution_time": round(
                float(self.execution_time),
                3,
            ),
            "horizon": self.horizon,
            "metadata": self.metadata,
        }

    def save_json(
        self,
        filename,
    ):
        """
        Save experiment as JSON.
        """

        with open(
            filename,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.to_dict(),
                f,
                indent=4,
                ensure_ascii=False,
            )

    def summary(self):
        """
        Human-readable summary.
        """

        summary = {
            "id": self.id,
            "battery": self.battery,
            "model": self.model,
            "version": self.version,
            "execution_time": round(
                float(self.execution_time),
                3,
            ),
            "horizon": self.horizon,
        }

        summary.update(self.metadata)

        return summary

    def __repr__(self):

        return (
            f"Experiment("
            f"id={self.id}, "
            f"battery='{self.battery}', "
            f"model='{self.model}')"
        )