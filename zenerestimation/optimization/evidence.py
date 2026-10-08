"""Utilities for assembling benchmark evidence bundles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import shutil

from .provenance import BenchmarkProvenance


def copy_evidence_files(
    sources: Mapping[str, str | Path],
    destinations: Mapping[str, str | Path],
    *,
    root: str | Path,
) -> dict[str, Path]:
    """Copy authoritative evidence files into a benchmark bundle."""

    if set(sources) != set(destinations):
        raise ValueError(
            "sources and destinations must contain "
            "the same evidence keys"
        )

    root = Path(root)

    copied = {}

    for name, source in sources.items():
        source = Path(source)

        if not source.is_file():
            raise FileNotFoundError(
                f"Evidence source does not exist: {source}"
            )

        relative_destination = Path(
            destinations[name]
        )

        if relative_destination.is_absolute():
            raise ValueError(
                "Evidence destinations must be relative paths"
            )

        destination = (
            root / relative_destination
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source,
            destination,
        )

        copied[name] = destination

    return copied


def validate_provenance_references(
    provenance: BenchmarkProvenance,
    *,
    root: str | Path,
) -> None:
    """Verify that all provenance references resolve to files."""

    root = Path(root)

    references = [
        *provenance.optimization.values(),
        *provenance.stability.values(),
        provenance.benchmark,
    ]

    missing = []

    for reference in references:
        reference = Path(reference)

        if reference.is_absolute():
            path = reference
        else:
            path = root / reference

        if not path.is_file():
            missing.append(
                reference
            )

    if missing:
        formatted = ", ".join(
            path.as_posix()
            for path in missing
        )

        raise FileNotFoundError(
            "Missing provenance evidence: "
            f"{formatted}"
        )