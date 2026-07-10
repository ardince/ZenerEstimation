from __future__ import annotations

from pathlib import Path

import pandas as pd


class SmartDatasetLoader:
    """
    Automatically detects and converts supported
    battery dataset formats.
    """

    def load(self, filename):

        path = Path(filename)

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        return self.normalize(df, path)
    

    def detect_format(self, df):

        cols = set(df.columns)

        if {"Month", "Year", "microVolt"} <= cols:
            return "MONTH_YEAR"

        if {"ds", "Volt", "microVolt"} <= cols:
            return "DS_VOLT"

        if {"ds", "microVolt"} <= cols:
            return "DS"

        if {"Date", "Value"} <= cols:
            return "DATE_VALUE"

        raise ValueError(
            f"Unsupported dataset format: {list(df.columns)}"
        )
    

    def normalize(self, df, path):

        fmt = self.detect_format(df)

        if fmt == "MONTH_YEAR":

            df = df.copy()

            df["ds"] = pd.to_datetime(
                dict(
                year=df["Year"],
                month=df["Month"],
                day=1,
                )
            )

            df = df.drop(
                columns=["Month", "Year"]
            )

        elif fmt == "DS":

            df = df.copy()

            df["ds"] = pd.to_datetime(
                df["ds"],
                dayfirst=True,
            )

        elif fmt == "DS_VOLT":

            df = df.copy()

            df["ds"] = pd.to_datetime(
                df["ds"],
                dayfirst=True,
            )

        elif fmt == "DATE_VALUE":

            df = df.rename(
                columns={
                    "Date": "ds",
                    "Value": "microVolt",
                }
            )

            df["ds"] = pd.to_datetime(df["ds"])

        df = df.sort_values("ds").reset_index(drop=True)

        metadata = {
            "battery_id": path.stem,
            "format": fmt,
            "source_file": path.name,
        }

        return df, metadata