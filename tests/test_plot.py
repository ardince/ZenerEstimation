import pandas as pd

from zenerestimation.data import BatteryDataset
from zenerestimation.visualization import DatasetPlotter


def test_plot_creation():

    df = pd.DataFrame(
        {
            "ds": [
                "01/01/2024",
                "01/04/2024",
            ],
            "microVolt": [
                180,
                179,
            ],
        }
    )

    dataset = BatteryDataset(df)

    plotter = DatasetPlotter(dataset)

    fig = plotter.plot_series()

    assert fig is not None