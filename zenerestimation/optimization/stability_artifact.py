"""Persistence support for repeated-seed stability evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .stability import StabilityResult


class StabilityArtifact:
    """Persist repeated-seed stability evidence.

    StabilityResult owns the scientific serialization contract.
    StabilityArtifact is intentionally a thin persistence layer and
    does not recompute, reinterpret, rank, or select seed results.

    Parameters
    ----------
    result:
        Completed repeated-seed stability result.
    """

    FILENAME = "stability.json"

    def __init__(
        self,
        result: StabilityResult,
    ) -> None:
        if not isinstance(
            result,
            StabilityResult,
        ):
            raise TypeError(
                "result must be a StabilityResult"
            )

        self._result = result

    @property
    def result(self) -> StabilityResult:
        """Underlying repeated-seed stability result."""

        return self._result

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical stability payload.

        Serialization is delegated entirely to StabilityResult so that
        the artifact layer cannot diverge from the result contract.
        """

        return self._result.to_dict()

    def save(
        self,
        directory: str | Path,
    ) -> Path:
        """Write stability evidence to ``stability.json``.

        Parameters
        ----------
        directory:
            Existing or new result directory in which the artifact
            should be stored.

        Returns
        -------
        pathlib.Path
            Path to the written stability artifact.
        """

        if not isinstance(
            directory,
            (str, Path),
        ):
            raise TypeError(
                "directory must be a string or pathlib.Path"
            )

        directory = Path(directory)

        if directory.exists() and not directory.is_dir():
            raise ValueError(
                "directory must refer to a directory"
            )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = directory / self.FILENAME

        payload = self.to_dict()

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
            )

            file.write("\n")

        return path