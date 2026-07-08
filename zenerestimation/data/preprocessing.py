"""
Dataset preprocessing utilities.
"""

import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicated timestamps.
    """
    return df.drop_duplicates(subset="ds").reset_index(drop=True)


def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort dataset chronologically.
    """
    return df.sort_values("ds").reset_index(drop=True)