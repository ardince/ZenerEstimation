from pathlib import Path

import pandas as pd

from zenerestimation.data import BatteryDataset


PROCESSED_DIRECTORY = Path("datasets/processed")

BATTERIES = [
    "732B-5610110",
    "732B-5610410",
]


def validate_battery(
    battery: str,
) -> None:

    filename = (
        PROCESSED_DIRECTORY
        / f"{battery}.csv"
    )

    dataset = (
        BatteryDataset
        .from_processed_csv(
            filename
        )
    )

    print()
    print("=" * 70)
    print(
        f"Processed Dataset Loader Validation: "
        f"{battery}"
    )
    print("=" * 70)

    print(
        f"Battery              : "
        f"{dataset.battery}"
    )

    print(
        f"Rows                 : "
        f"{len(dataset.data)}"
    )

    print(
        f"Observed Rows        : "
        f"{dataset.observed_rows}"
    )

    print(
        f"Missing Period Count : "
        f"{dataset.missing_period_count}"
    )

    print(
        f"Source Type          : "
        f"{dataset.source_type}"
    )

    print(
        f"Is Processed         : "
        f"{dataset.is_processed}"
    )

    print(
        f"Source Path          : "
        f"{dataset.source_path}"
    )

    print(
        f"Start Date           : "
        f"{dataset.data['ds'].min().date()}"
    )

    print(
        f"End Date             : "
        f"{dataset.data['ds'].max().date()}"
    )

    print(
        f"NaN microVolt Rows   : "
        f"{dataset.data['microVolt'].isna().sum()}"
    )

    inserted = dataset.data.loc[
        ~dataset.observed_mask,
        [
            "ds",
            "microVolt",
            "is_observed",
        ],
    ]

    print()
    print("Inserted Canonical Periods:")

    if inserted.empty:

        print("  None")

    else:

        for row in inserted.itertuples():

            print(
                "  "
                f"{row.ds.date()}  "
                f"microVolt={row.microVolt}  "
                f"is_observed={row.is_observed}"
            )

    # ----------------------------------------------------
    # Contract checks
    # ----------------------------------------------------

    assert (
        dataset.data[
            "ds"
        ].duplicated().sum()
        == 0
    )

    assert (
        dataset.data[
            "microVolt"
        ].isna().sum()
        == dataset.missing_period_count
    )

    assert (
        (
            ~dataset.observed_mask
        ).sum()
        == dataset.missing_period_count
    )

    assert not (
        dataset.observed_mask
        & dataset.data[
            "microVolt"
        ].isna()
    ).any()

    assert not (
        ~dataset.observed_mask
        & dataset.data[
            "microVolt"
        ].notna()
    ).any()

    print()
    print(
        "Loader validation    : PASS"
    )


def main() -> None:

    for battery in BATTERIES:

        validate_battery(
            battery
        )


if __name__ == "__main__":
    main()