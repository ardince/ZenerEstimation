"""
Experiment model.

Represents a single forecasting experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Experiment:
    """
    A forecasting experiment.
    """

    id: int | None = None

    battery: str = ""

    model: str = ""

    version: str = ""

    created: datetime = field(default_factory=datetime.now)

    execution_time: float = 0.0

    horizon: int = 0

    artifacts: dict = field(default_factory=dict)

    metadata: dict = field(default_factory=dict)

    @property
    def created_iso(self) -> str:
        """
        ISO formatted creation time.
        """
        return self.created.isoformat()

    def to_dict(self) -> dict:
        """
        Convert experiment to a JSON-serializable dictionary.
        """
        return {
            "id": self.id,
            "battery": self.battery,
            "model": self.model,
            "version": self.version,
            "created": self.created_iso,
            "execution_time": self.execution_time,
            "horizon": self.horizon,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }

    def __repr__(self):

        return (
            f"Experiment("
            f"id={self.id}, "
            f"battery='{self.battery}', "
            f"model='{self.model}')"
        )
    
@property
def created_str(self) -> str:
    """
    Human-readable creation timestamp.
    """
    return self.created.strftime("%Y-%m-%d %H:%M:%S")