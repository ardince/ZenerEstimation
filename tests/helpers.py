"""
Shared helper functions for the test suite.

Provides small synthetic datasets used by the
forecasting models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zenerestimation.data import BatteryDataset


# ---------------------------------------------------------
# Quarterly battery dataset
# ---------------------------------------------------------

def make_dataset(
    periods=24,
):
    """
    Standard quarterly degradation dataset.
    """

    dates = pd.date_range(
        start="2020-01-01",
        periods=periods,
        freq="QS",
    )

    values = np.linspace(
        80.0,
        35.0,
        periods,
    )

    df = pd.DataFrame(
        {
            "ds": dates,
            "microVolt": values,
        }
    )

    return BatteryDataset(df)


# ---------------------------------------------------------
# Monthly dataset
# ---------------------------------------------------------

def make_monthly_dataset(
    periods=36,
):
    """
    Monthly battery dataset.
    """

    dates = pd.date_range(
        start="2020-01-01",
        periods=periods,
        freq="MS",
    )

    values = np.linspace(
        100.0,
        60.0,
        periods,
    )

    df = pd.DataFrame(
        {
            "ds": dates,
            "microVolt": values,
        }
    )

    return BatteryDataset(df)


# ---------------------------------------------------------
# Irregular dataset
# ---------------------------------------------------------

def make_irregular_dataset():
    """
    Small irregularly sampled dataset.
    """

    df = pd.DataFrame(

        {

            "ds": [

                "2020-01-01",

                "2020-04-01",

                "2020-08-15",

                "2021-01-01",

                "2021-07-01",

            ],

            "microVolt": [

                80,

                75,

                70,

                65,

                60,

            ],

        }

    )

    return BatteryDataset(df)