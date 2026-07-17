"""
Experiment registry.
"""

from __future__ import annotations

import json

from pathlib import Path

from zenerestimation.experiment import Experiment


class ExperimentRegistry:
    """
    Stores experiment history.
    """

    def __init__(

        self,

        filename="results/history.json",

    ):

        self.filename = Path(filename)

        self.filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    
    def _load(self):

        if not self.filename.exists():

            return []

        with open(

            self.filename,

            "r",

            encoding="utf-8",

        ) as f:

            return json.load(f)

    def _save(

        self,

        history,

    ):

        with open(

            self.filename,

            "w",

            encoding="utf-8",

        ) as f:

            json.dump(

                history,

                f,

                indent=4,

            )

    
    def _next_id(self):

        history = self._load()

        if not history:

            return 1

        return history[-1]["id"] + 1

    
    def register(

        self,

        experiment: Experiment,

    ):

        history = self._load()

        experiment.id = self._next_id()

        history.append(experiment.to_dict())

        self._save(history)

        return experiment

    
    
    def count(self):

        return len(

            self._load()

        )

    
    def latest(self):

        history = self._load()

        if not history:

            return None

        return history[-1]