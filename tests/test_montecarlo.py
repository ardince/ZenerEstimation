import numpy as np

from zenerestimation.prognostics import MonteCarloRUL


def test_run_returns_result():

    engine = MonteCarloRUL(

        simulations=200,

        random_seed=42,

    )

    result = engine.run(

        current_value=160.0,

        current_drift=2.0,

        sigma=0.5,

        threshold=170.0,

        model="Kalman",

    )

    assert result.model == "Kalman"

    assert result.threshold == 170.0

    assert len(result.samples) == 200

    assert result.mean > 0.0

    assert result.p10 <= result.median <= result.p90


def test_reproducible_seed():

    engine1 = MonteCarloRUL(

        simulations=100,

        random_seed=1,

    )

    engine2 = MonteCarloRUL(

        simulations=100,

        random_seed=1,

    )

    result1 = engine1.run(

        current_value=160,

        current_drift=2,

        sigma=0.5,

        threshold=170,

    )

    result2 = engine2.run(

        current_value=160,

        current_drift=2,

        sigma=0.5,

        threshold=170,

    )

    assert np.allclose(

        result1.samples,

        result2.samples,

    )


def test_zero_noise():

    engine = MonteCarloRUL(

        simulations=20,

        random_seed=0,

        fast_probability=0.0,

        fast_noise=0.0,

        slow_noise=0.0,

    )

    result = engine.run(

        current_value=0.0,

        current_drift=1.0,

        sigma=0.0,

        threshold=2.0,

    )

    assert np.all(result.samples == result.samples[0])