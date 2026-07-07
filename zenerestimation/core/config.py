from dataclasses import dataclass

@dataclass
class ForecastConfig:

    random_seed: int = 42

    epochs: int = 200

    batch_size: int = 8

    windows = [6,8,10]

    gru_units = [24,32]

    lstm_units = [24,32]

    forecast_horizon = 5