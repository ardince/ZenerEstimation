"""Provenance links for benchmark evidence artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any


class BenchmarkProvenance:
    """Link optimization, stability, and benchmark evidence.

    BenchmarkProvenance records references to scientific evidence
    artifacts. It intentionally does not embed, recompute, reinterpret,
    rank, or select any scientific results.

    Parameters
    ----------
    battery:
        Battery identifier associated with the benchmark.

    optimization:
        Mapping from model name to the corresponding optimization
        artifact path.

    stability:
        Mapping from model name to the corresponding repeated-seed
        stability artifact path. Models without stability analysis do
        not need an entry.

    benchmark:
        Path to the standardized benchmark artifact.

    metadata:
        Optional provenance metadata. This should describe the evidence
        relationship rather than duplicate scientific result payloads.
    """

    FILENAME = "provenance.json"

    def __init__(
        self,
        battery: str,
        optimization: Mapping[
            str,
            str | Path,
        ],
        stability: Mapping[
            str,
            str | Path,
        ],
        benchmark: str | Path,
        metadata: Mapping[
            str,
            Any,
        ]
        | None = None,
    ) -> None:

        if (
            not isinstance(
                battery,
                str,
            )
            or not battery.strip()
        ):
            raise ValueError(
                "battery must be a non-empty string"
            )

        if not isinstance(
            optimization,
            Mapping,
        ):
            raise TypeError(
                "optimization must be a mapping"
            )

        if not isinstance(
            stability,
            Mapping,
        ):
            raise TypeError(
                "stability must be a mapping"
            )

        if not isinstance(
            benchmark,
            (str, Path),
        ):
            raise TypeError(
                "benchmark must be a string or pathlib.Path"
            )

        if (
            metadata is not None
            and not isinstance(
                metadata,
                Mapping,
            )
        ):
            raise TypeError(
                "metadata must be a mapping"
            )

        self._battery = battery

        self._optimization = (
            self._normalize_paths(
                optimization,
                name="optimization",
            )
        )

        self._stability = (
            self._normalize_paths(
                stability,
                name="stability",
            )
        )

        self._benchmark = Path(
            benchmark
        )

        self._metadata = deepcopy(
            dict(metadata)
            if metadata is not None
            else {}
        )

    @staticmethod
    def _normalize_paths(
        values: Mapping[
            str,
            str | Path,
        ],
        *,
        name: str,
    ) -> dict[str, Path]:
        """Validate and normalize a model-to-artifact mapping."""

        normalized: dict[
            str,
            Path,
        ] = {}

        for model, path in (
            values.items()
        ):

            if (
                not isinstance(
                    model,
                    str,
                )
                or not model.strip()
            ):
                raise ValueError(
                    f"{name} model names must be "
                    "non-empty strings"
                )

            if not isinstance(
                path,
                (str, Path),
            ):
                raise TypeError(
                    f"{name} artifact paths must be "
                    "strings or pathlib.Path objects"
                )

            normalized[
                model
            ] = Path(
                path
            )

        return normalized

    @property
    def battery(
        self,
    ) -> str:
        """Battery identifier."""

        return self._battery

    @property
    def optimization(
        self,
    ) -> dict[str, Path]:
        """Optimization artifact references."""

        return deepcopy(
            self._optimization
        )

    @property
    def stability(
        self,
    ) -> dict[str, Path]:
        """Stability artifact references."""

        return deepcopy(
            self._stability
        )

    @property
    def benchmark(
        self,
    ) -> Path:
        """Benchmark artifact reference."""

        return self._benchmark

    @property
    def metadata(
        self,
    ) -> dict[str, Any]:
        """Defensive copy of provenance metadata."""

        return deepcopy(
            self._metadata
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Return the standardized provenance payload."""

        payload = {
            "status": "linked",
            "battery": self._battery,
            "optimization": {
                model: path.as_posix()
                for model, path
                in self._optimization.items()
            },
            "stability": {
                model: path.as_posix()
                for model, path
                in self._stability.items()
            },
            "benchmark": str(
                self._benchmark.as_posix()
            ),
            "metadata": deepcopy(
                self._metadata
            ),
        }

        return deepcopy(
            payload
        )

    def save(
        self,
        directory: str | Path,
    ) -> Path:
        """Persist provenance as ``provenance.json``."""

        if not isinstance(
            directory,
            (str, Path),
        ):
            raise TypeError(
                "directory must be a string or pathlib.Path"
            )

        directory = Path(
            directory
        )

        if (
            directory.exists()
            and not directory.is_dir()
        ):
            raise ValueError(
                "directory must refer to a directory"
            )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = (
            directory
            / self.FILENAME
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.to_dict(),
                file,
                indent=2,
                ensure_ascii=False,
            )

            file.write(
                "\n"
            )

        return path

    def __repr__(
        self,
    ) -> str:

        return (
            "BenchmarkProvenance("
            f"battery={self._battery!r}, "
            f"optimization={len(self._optimization)}, "
            f"stability={len(self._stability)}"
            ")"
        )