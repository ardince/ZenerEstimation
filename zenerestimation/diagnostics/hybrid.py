"""
Hybrid model diagnostics.
"""

from __future__ import annotations

import numpy as np

from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

from zenerestimation.diagnostics.result import HybridDiagnosticsResult


class HybridDiagnostics:
    """
    Diagnostics for hybrid forecasting models.

    Performs consistency checks and
    statistical analysis of trend/residual
    decomposition.
    """

    def __init__(self, model):

        self.model = model

        self.original = None
        self.trend = None
        self.residual = None

        self._summary = None


    def run(
        self,
        dataset,
    ):
        """
        Compute diagnostics.
        """

        self.original = dataset.target.to_numpy(dtype=float)

        self.trend = self.model._trend

        self.residual = self.model._residual


        self._summary = None

        self.summary()


        return self


    def result(self):

        return HybridDiagnosticsResult(

            summary=self.summary(),

            metadata={

                "family": "Hybrid",

            },

        )


    def verify_decomposition(
        self,
        atol=1e-8,
    ):
        """
        Verify

        trend + residual = signal
        """

        reconstructed = self.trend + self.residual

        error = reconstructed - self.original

        self.max_error = float(np.max(np.abs(error)))
        self.mean_error = float(np.mean(error))
        self.rmse_error = float(np.sqrt(np.mean(error**2)))

        return self.max_error <= atol


    def trend_variance(self):

        if not hasattr(self, "_trend_variance"):

            self._trend_variance = float(
                np.var(self.trend)
            )

        return self._trend_variance


    def residual_variance(self):

        if not hasattr(self, "_residual_variance"):

            self._residual_variance = float(
                np.var(self.residual)
            )

        return self._residual_variance


    def variance_explained(self):

        total = float(np.var(self.original))

        if total <= 0.0:

            return 1.0

        explained = 1.0 - (
            self.residual_variance() / total
        )

        explained = max(
            0.0,
            min(
                explained,
                1.0,
            ),
        )

        return explained


    def residual_mean(self):

        return float(np.mean(self.residual))


    def residual_std(self):

        return float(np.std(self.residual))


    def residual_rmse(self):

        return float(
            np.sqrt(np.mean(self.residual ** 2))
        )


    def residual_autocorrelation(self, lag=1):
        """
        Compute the autocorrelation of the residuals.
        """

        if lag >= len(self.residual):
            raise ValueError("lag too large")

        return float(

            np.corrcoef(

                self.residual[:-lag],

                self.residual[lag:],

            )[0,1]

        )


    def durbin_watson(self):
        """
        Compute the Durbin-Watson statistic for the residuals.
        """
        return float(durbin_watson(self.residual))


    def ljung_box(self, lags=10):
        """
        Compute the Ljung-Box test for the residuals.
        """
        return acorr_ljungbox(

            self.residual,

            lags=lags,

            return_df=True,

        )


    def summary(self):
        """
        Return diagnostics.
        """

        if self._summary is None:

            ok = self.verify_decomposition()

            lb = self.ljung_box()

            self._summary = {

                "family": "Hybrid",

                "decomposition_ok":
                    ok,

                "trend_variance":
                    self.trend_variance(),

                "residual_variance":
                    self.residual_variance(),

                "variance_explained":
                    self.variance_explained(),

                "max_error":
                    self.max_error,

                "mean_error":
                    self.mean_error,

                "rmse_error":
                    self.rmse_error,

                "residual_mean":
                    self.residual_mean(),

                "residual_std":
                    self.residual_std(),

                "residual_rmse":
                    self.residual_rmse(),

                "lag1_autocorrelation":
                    self.residual_autocorrelation(lag=1),

                "durbin_watson":
                    self.durbin_watson(),

                "ljung_box_statistic":
                    float(lb["lb_stat"].iloc[-1]),

                "ljung_box_pvalue":
                    float(lb["lb_pvalue"].iloc[-1]),

                "quality_score":
                    self.quality_score(),

                "quality_grade":
                    self.quality_grade(),

                "recommendations":
                    self.recommendations(),

            }

        return self._summary


    def quality_score(self):
        """
        Compute an overall hybrid quality score (0–100).
        """

        score = 0.0

        # -------------------------------------------------
        # Variance explained (40)
        # -------------------------------------------------

        score += 40.0 * self.variance_explained()

        # -------------------------------------------------
        # Decomposition accuracy (20)
        # -------------------------------------------------

        if self.verify_decomposition():
            score += 20.0

        # -------------------------------------------------
        # Residual RMSE (15)
        # -------------------------------------------------

        signal_std = np.std(self.original)

        if signal_std > 0:

            ratio = self.residual_rmse() / signal_std

            score += 15.0 * max(0.0, 1.0 - ratio)

        # -------------------------------------------------
        # Autocorrelation (10)
        # -------------------------------------------------

        score += 10.0 * (1.0 - min(abs(
            self.residual_autocorrelation()
        ), 1.0))

        # -------------------------------------------------
        # Durbin–Watson (10)
        # -------------------------------------------------

        dw = self.durbin_watson()

        score += 10.0 * max(
            0.0,
            1.0 - abs(dw - 2.0) / 2.0,
        )

        # -------------------------------------------------
        # Ljung–Box (5)
        # -------------------------------------------------

        lb = self.ljung_box()

        p = float(lb["lb_pvalue"].iloc[-1])

        score += 5.0 * p

        return round(min(score, 100.0), 2)


    def quality_grade(self):
        """
        Convert the quality score into a qualitative grade.
        """

        score = self.quality_score()

        if score >= 95:
            return "Excellent"

        if score >= 85:
            return "Very Good"

        if score >= 70:
            return "Good"

        if score >= 50:
            return "Fair"

        return "Poor"


    def recommendations(self):
        """
        Generate deterministic recommendations.
        """

        rec = []

        if self.variance_explained() >= 0.90:
            rec.append("Trend captures most of the degradation.")
        else:
            rec.append("Trend model could be improved.")

        if abs(self.residual_autocorrelation()) < 0.20:
            rec.append("Residuals are approximately uncorrelated.")
        else:
            rec.append("Residual autocorrelation remains noticeable.")

        if 1.5 <= self.durbin_watson() <= 2.5:
            rec.append("Residual independence is satisfactory.")
        else:
            rec.append("Residual independence should be investigated.")

        lb = self.ljung_box()

        p = float(lb["lb_pvalue"].iloc[-1])

        if p > 0.05:
            rec.append("Residuals resemble white noise.")
        else:
            rec.append("Residuals still contain temporal structure.")

        if self.verify_decomposition():
            rec.append("Hybrid decomposition verified.")

        return rec