"""
Create a canonical processed battery dataset.

Example
-------
python examples/process_dataset.py --battery 732B-5610410
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from zenerestimation.data.processing import (
    DatasetProcessor,
    ProcessedDatasetWriter,
)

#print("DEBUG: process_dataset.py started")

# ============================================================
# Paths
# ============================================================

RAW_DIRECTORY = Path("datasets/raw")
PROCESSED_DIRECTORY = Path("datasets/processed")


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Create a canonical processed battery dataset."
        )
    )

    parser.add_argument(
        "--battery",
        required=True,
        help=(
            "Battery identifier, for example "
            "732B-5610410"
        ),
    )

    parser.add_argument(
        "--frequency",
        default="QS-MAR",
        help=(
            "Canonical pandas frequency. "
            "Default: QS-MAR"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing processed artifacts."
        ),
    )

    return parser.parse_args()


# ============================================================
# Raw loading
# ============================================================


def load_raw_dataset(
    filename: Path,
) -> pd.DataFrame:
    """
    Load the raw source CSV without modifying it.
    """

    if not filename.exists():

        raise FileNotFoundError(
            "raw dataset not found: "
            f"{filename}"
        )

    data = pd.read_csv(
        filename
    )

    if data.empty:

        raise ValueError(
            "raw dataset is empty"
        )

    return data


def normalize_raw_schema(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert supported raw dataset schemas to the
    canonical input columns expected by DatasetProcessor:

        ds
        microVolt
    """

    frame = data.copy()

    # --------------------------------------------------------
    # Already canonical
    # --------------------------------------------------------

    if {
        "ds",
        "microVolt",
    }.issubset(frame.columns):

        return frame[
            [
                "ds",
                "microVolt",
            ]
        ].copy()

    # --------------------------------------------------------
    # Separate Month / Year columns
    # --------------------------------------------------------

    if {
        "Month",
        "Year",
        "microVolt",
    }.issubset(frame.columns):

        month = pd.to_numeric(
            frame["Month"],
            errors="coerce",
        )

        year = pd.to_numeric(
            frame["Year"],
            errors="coerce",
        )

        if month.isna().any():

            raise ValueError(
                "Month contains invalid or missing values"
            )

        if year.isna().any():

            raise ValueError(
                "Year contains invalid or missing values"
            )

        frame["ds"] = pd.to_datetime(
            {
                "year": year.astype(int),
                "month": month.astype(int),
                "day": 1,
            },
            errors="coerce",
        )

        if frame["ds"].isna().any():

            raise ValueError(
                "Month/Year could not be converted "
                "to valid dates"
            )

        return frame[
            [
                "ds",
                "microVolt",
            ]
        ].copy()

    # --------------------------------------------------------
    # Unsupported schema
    # --------------------------------------------------------

    raise ValueError(
        "unsupported raw dataset schema; "
        "expected either "
        "['ds', 'microVolt'] "
        "or "
        "['Month', 'Year', 'microVolt']. "
        f"Available columns: {list(frame.columns)}"
    )


# ============================================================
# Console output
# ============================================================


def print_processing_summary(
    result,
    paths,
) -> None:
    """
    Print canonical processing results.
    """

    data = result.dataframe()

    missing = data.loc[
        ~data["is_observed"]
    ].copy()

    print()
    print("=" * 70)
    print("ZenerEstimation Dataset Processing")
    print("=" * 70)

    print(
        f"Battery           : "
        f"{result.battery}"
    )

    print(
        f"Frequency         : "
        f"{result.frequency}"
    )

    print(
        f"Source Rows       : "
        f"{result.source_rows}"
    )

    print(
        f"Processed Rows    : "
        f"{result.processed_rows}"
    )

    print(
        f"Observed Rows     : "
        f"{result.observed_rows}"
    )

    print(
        f"Missing Periods   : "
        f"{result.missing_periods}"
    )

    print(
        f"Start Date        : "
        f"{data['ds'].iloc[0].date()}"
    )

    print(
        f"End Date          : "
        f"{data['ds'].iloc[-1].date()}"
    )

    print()

    if missing.empty:

        print(
            "Missing Dates     : None"
        )

    else:

        print(
            "Missing Dates:"
        )

        for value in missing[
            "ds"
        ]:

            print(
                f"  - {value.date()}"
            )

    print()
    print(
        f"Processed CSV     : "
        f"{paths.data}"
    )

    print(
        f"Metadata JSON     : "
        f"{paths.metadata}"
    )

    print("=" * 70)
    print()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    battery = args.battery

    raw_file = (
        RAW_DIRECTORY
        / f"{battery}.csv"
    )

    print()
    print(
        f"Loading raw dataset: "
        f"{raw_file}"
    )

    raw_data = load_raw_dataset(
        raw_file
    )

    raw_data = normalize_raw_schema(
    raw_data
    )

    processor = DatasetProcessor(
        frequency=args.frequency,
        dayfirst=True,
        duplicate_policy="mean",
    )

    try:

        result = processor.process(
            raw_data,
            battery=battery,
            metadata={
                "source": str(
                    raw_file
                ),
            },
        )

    except ValueError as exc:

        print()
        print("=" * 70)
        print("Dataset processing failed")
        print("=" * 70)
        print(
            f"Battery           : {battery}"
        )
        print(
            f"Frequency         : {args.frequency}"
        )
        print(
            f"Reason            : {exc}"
        )
        print()
        print(
            "The raw dataset has been left unchanged."
        )
        print(
            "No processed dataset was written."
        )
        print("=" * 70)
        print()

        raise

    writer = ProcessedDatasetWriter(
        PROCESSED_DIRECTORY
    )

    paths = writer.save(
        result,
        overwrite=args.overwrite,
    )

    print_processing_summary(
        result,
        paths,
    )


if __name__ == "__main__":
    main()