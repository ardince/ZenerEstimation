import pandas as pd

from zenerestimation.data import BatteryDataset
from zenerestimation.visualization.plots import plot_dataset


def test_plot_dataset():

    df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=3, freq="QS"),
            "microVolt": [180, 179, 178],
        }
    )

    dataset = BatteryDataset(df)

    fig = plot_dataset(dataset)

    assert fig is not None