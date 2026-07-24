"""
Remaining Useful Life report writer.

Creates a human-readable report for prognostic experiments.
"""

from __future__ import annotations

from pathlib import Path


class RULReportWriter:
    """
    Writes a Remaining Useful Life experiment report.
    """

    @staticmethod
    def save(
        filename,
        dataset,
        result,
        experiment,
    ):

        filename = Path(filename)

        with open(
            filename,
            "w",
            encoding="utf-8",
        ) as f:

            f.write("=" * 60 + "\n")
            f.write("ZenerEstimation\n")
            f.write("Remaining Useful Life Report\n")
            f.write("=" * 60 + "\n\n")

            # --------------------------------------------------
            # Experiment
            # --------------------------------------------------

            f.write("Experiment\n")
            f.write("-" * 60 + "\n")

            f.write(f"ID               : {experiment.id}\n")
            f.write(f"Battery          : {experiment.battery}\n")
            f.write(f"Model            : {experiment.model}\n")
            f.write(f"Framework        : {experiment.version}\n")
            f.write(
                f"Execution Time   : "
                f"{experiment.execution_time:.3f} s\n"
            )

            f.write("\n")

            # --------------------------------------------------
            # Dataset
            # --------------------------------------------------

            f.write("Dataset\n")
            f.write("-" * 60 + "\n")

            summary = dataset.summary()

            f.write(
                f"Rows             : {summary['rows']}\n"
            )

            f.write(
                f"Columns          : {summary['columns']}\n"
            )

            f.write(
                f"Frequency        : {summary['frequency']}\n"
            )

            f.write(
                f"Missing Values   : {summary['missing']}\n"
            )

            f.write(
                f"Missing Periods  : "
                f"{summary['missing_periods']}\n"
            )

            f.write(
                f"Start            : {summary['start']}\n"
            )

            f.write(
                f"End              : {summary['end']}\n"
            )

            f.write("\n")

            # --------------------------------------------------
            # RUL
            # --------------------------------------------------

            f.write("Remaining Useful Life\n")
            f.write("-" * 60 + "\n")

            f.write(
                f"Threshold        : "
                f"{result.threshold:.3f} µV\n"
            )

            f.write(
                f"Mean             : "
                f"{result.mean:.2f} years\n"
            )

            f.write(
                f"Median           : "
                f"{result.median:.2f} years\n"
            )

            f.write(
                f"P10              : "
                f"{result.p10:.2f} years\n"
            )

            f.write(
                f"P90              : "
                f"{result.p90:.2f} years\n"
            )

            f.write(
                f"Simulations      : "
                f"{result.simulations}\n"
            )

            f.write("\n")

            # --------------------------------------------------
            # Metadata
            # --------------------------------------------------

            if result.metadata:

                f.write("Model Metadata\n")
                f.write("-" * 60 + "\n")

                for key, value in result.metadata.items():

                    f.write(
                        f"{key:20} : {value}\n"
                    )

                f.write("\n")

            f.write("=" * 60 + "\n")
            f.write("End of Report\n")
            f.write("=" * 60 + "\n")