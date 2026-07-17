"""
Experiment report writer.

Creates a human-readable report for every experiment.
"""

from __future__ import annotations

from pathlib import Path


class ReportWriter:
    """
    Writes an experiment report.

    The report is intended for humans, while JSON files
    are intended for software.
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
            f.write("Experiment Report\n")
            f.write("=" * 60 + "\n\n")

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

            f.write("Forecast\n")
            f.write("-" * 60 + "\n")

            for date, value in zip(
                result.dates,
                result.forecast,
            ):

                f.write(
                    f"{date:%Y-%m-%d}"
                    f"    {float(value):10.4f}\n"
                )

            f.write("\n")

            if result.metadata:

                f.write("Model Metadata\n")
                f.write("-" * 60 + "\n")

                for key, value in result.metadata.items():

                    f.write(
                        f"{key:15} : {value}\n"
                    )

                f.write("\n")

            f.write("=" * 60 + "\n")
            f.write("End of Report\n")
            f.write("=" * 60 + "\n")