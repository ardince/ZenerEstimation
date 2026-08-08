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
        diagnostics=None,
        experiment=None,
    ):
        """ Save a human-readable experiment report.

        Parameters
        ----------
        filename: Output report filename.
        dataset: BatteryDataset instance.
        result: ForecastResult instance.
        diagnostics: Optional HybridDiagnosticsResult instance.
        experiment: Experiment instance.
        """

        filename = Path(filename)

        with open(
            filename,
            "w",
            encoding="utf-8",
        ) as f:

            # =====================================================
            # Header
            # =====================================================

            f.write("=" * 60 + "\n")
            f.write("ZenerEstimation\n")
            f.write("Experiment Report\n")
            f.write("=" * 60 + "\n\n")

            # =====================================================
            # Experiment
            # =====================================================

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

            # =====================================================
            # Dataset
            # =====================================================

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

            # =====================================================
            # Forecast
            # =====================================================

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

            # =====================================================
            # Hybrid Diagnostics
            # =====================================================

            if diagnostics is not None:

                f.write(
                    "Hybrid Diagnostics\n"
                )

                f.write(
                    "-" * 60 + "\n"
                )

                diagnostic_summary = (
                    diagnostics.summary()
                )

                if diagnostics.quality_score is not None:

                    f.write(
                        f"Quality Score     : "
                        f"{diagnostics.quality_score:.2f}/100\n"
                    )

                if diagnostics.quality_grade is not None:

                    f.write(
                        f"Quality Grade     : "
                        f"{diagnostics.quality_grade}\n"
                    )

                if "variance_explained" in diagnostic_summary:

                    f.write(
                        f"Variance Explained: "
                        f"{100.0 * diagnostic_summary['variance_explained']:.2f}%\n"
                    )

                if "residual_mean" in diagnostic_summary:

                    f.write(
                        f"Residual Mean     : "
                        f"{diagnostic_summary['residual_mean']:.6f}\n"
                    )

                if "residual_std" in diagnostic_summary:

                    f.write(
                        f"Residual Std      : "
                        f"{diagnostic_summary['residual_std']:.6f}\n"
                    )

                if "residual_rmse" in diagnostic_summary:

                    f.write(
                        f"Residual RMSE     : "
                        f"{diagnostic_summary['residual_rmse']:.6f}\n"
                    )

                if "lag1_autocorrelation" in diagnostic_summary:

                    f.write(
                        f"Lag-1 Corr        : "
                        f"{diagnostic_summary['lag1_autocorrelation']:.6f}\n"
                    )

                if "durbin_watson" in diagnostic_summary:

                    f.write(
                        f"Durbin-Watson     : "
                        f"{diagnostic_summary['durbin_watson']:.6f}\n"
                    )

                if "ljung_box_pvalue" in diagnostic_summary:

                    f.write(
                        f"Ljung-Box p-value : "
                        f"{diagnostic_summary['ljung_box_pvalue']:.6f}\n"
                    )

                if "decomposition_ok" in diagnostic_summary:

                    f.write(
                        f"Decomposition     : "
                        f"{diagnostic_summary['decomposition_ok']}\n"
                    )

                f.write("\n")

                recommendations = (
                    diagnostics.recommendations
                )

                if recommendations:

                    f.write(
                        "Recommendations\n"
                    )

                    f.write(
                        "-" * 60 + "\n"
                    )

                    for recommendation in recommendations:

                        f.write(
                            f"- {recommendation}\n"
                        )

                    f.write("\n")

            # =====================================================
            # Model Metadata
            # =====================================================

            if result.metadata:

                f.write("Model Metadata\n")
                f.write("-" * 60 + "\n")

                for key, value in result.metadata.items():

                    f.write(
                        f"{key:20} : {value}\n"
                    )

                f.write("\n")

            # =====================================================
            # Footer
            # =====================================================

            f.write("=" * 60 + "\n")
            f.write("End of Report\n")
            f.write("=" * 60 + "\n")