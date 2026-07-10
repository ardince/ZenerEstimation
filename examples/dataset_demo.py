"""
Dataset Demonstration

Sprint Acceptance Test

This example demonstrates every data-processing feature
implemented so far.
"""

from pathlib import Path

from zenerestimation.data import BatteryDataset
from zenerestimation.visualization.plots import (
    DatasetPlotter,
)


def print_header():

    print("=" * 60)
    print("ZENER ESTIMATION")
    print("Sprint Acceptance Test")
    print("=" * 60)


def load_dataset(filename):

    print()

    print("[1] Loading dataset")

    dataset = BatteryDataset.from_csv(filename)

    print("OK")

    return dataset


def validation(dataset):

    print()

    print("[2] Validation")

    dataset.validate()

    print("Passed")


def summary(dataset):

    print()

    print("[3] Dataset Summary")

    s = dataset.summary()

    for k, v in s.items():

        print(f"{k:20} : {v}")


def analysis(dataset):

    print()

    print("[4] Battery Analysis")

    report = dataset.report()

    for k, v in report.items():

        print(f"{k:20} : {v}")


def frequency(dataset):

    print()

    print("[5] Frequency")

    print(dataset.detect_frequency())


def missing(dataset):

    print()

    print("[6] Missing Periods")

    missing = dataset.missing_periods()

    if len(missing) == 0:

        print("None")

    else:

        for m in missing:

            print(m)


def windows(dataset):

    print()

    print("[7] Rolling Windows")

    X, y = dataset.windows(8)

    print("X:", X.shape)

    print("y:", y.shape)


def plots(dataset):

    print()

    print("[8] Visualization")

    plotter = DatasetPlotter(dataset)

    plotter.plot_series()

    plotter.plot_histogram()

    plotter.plot_boxplot()


def diagnostics(dataset):

    print()

    print("[9] Diagnostics")

    report = dataset.report()

    print()

    print("Average drift")

    print(report["estimated_drift"])


def final_report():

    print()

    print("=" * 60)

    print("SPRINT ACCEPTANCE TEST")

    print("STATUS : PASS")

    print("=" * 60)


def main():

    print_header()

    filename = Path(
        "../datasets/battery.csv"
    )

    dataset = load_dataset(filename)

    validation(dataset)

    summary(dataset)

    analysis(dataset)

    frequency(dataset)

    missing(dataset)

    windows(dataset)

    plots(dataset)

    diagnostics(dataset)

    final_report()


if __name__ == "__main__":

    main()


