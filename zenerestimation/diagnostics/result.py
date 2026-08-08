"""
Hybrid diagnostics result object.

Stores the outcome of a HybridDiagnostics analysis.
"""

from __future__ import annotations

from copy import deepcopy


class HybridDiagnosticsResult:
    """
    Immutable container for hybrid diagnostics.

    Parameters
    ----------
    summary : dict
        Dictionary returned by HybridDiagnostics.summary().

    metadata : dict, optional
        Additional metadata.
    """

    def __init__(
        self,
        summary,
        metadata=None,
    ):

        self._summary = deepcopy(summary)

        self._metadata = deepcopy(metadata) if metadata else {}

    # ---------------------------------------------------------
    # Properties
    # ---------------------------------------------------------

    @property
    def summary_data(self):
        """
        Return the complete diagnostics summary.
        """
        return deepcopy(self._summary)

    @property
    def metadata(self):
        """
        Return metadata.
        """
        return deepcopy(self._metadata)

    @property
    def quality_score(self):
        return self._summary.get("quality_score")

    @property
    def quality_grade(self):
        return self._summary.get("quality_grade")

    @property
    def recommendations(self):
        return list(
            self._summary.get(
                "recommendations",
                [],
            )
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self):
        """
        Return the diagnostics summary.
        """
        return deepcopy(self._summary)

    # ---------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------

    def to_dict(self):
        """
        Convert to a serializable dictionary.
        """

        return {

            "summary": deepcopy(self._summary),

            "metadata": deepcopy(self._metadata),

        }

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self):

        score = self.quality_score

        grade = self.quality_grade

        return (

            f"HybridDiagnosticsResult("
            f"score={score}, "
            f"grade='{grade}')"

        )