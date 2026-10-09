import numpy as np
import pytest
import torch

from hft.ml.models import (
    DailyMLP,
    MinuteCNN,
    MinuteGRU,
    MinuteTransformer,
    fit_torch,
    train_lightgbm,
)

torch.set_num_threads(1)


@pytest.mark.parametrize("model", [MinuteCNN(4, 16), MinuteGRU(4, 16), MinuteTransformer(4, 16)])
def test_minute_models_map_windows_to_one_prediction(model):
    assert model(torch.zeros(5, 60, 4)).shape == (5,)


def _planted(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 60, 4)).astype(np.float32)
    y = (0.5 * x[:, -1, 0] + 0.1 * rng.normal(size=n)).astype(np.float32)  # last return predicts
    return x, y


@pytest.mark.parametrize("cls", [MinuteCNN, MinuteGRU, MinuteTransformer])
def test_minute_models_learn_a_planted_signal(cls):
    x, y = _planted()
    torch.manual_seed(0)
    model = fit_torch(cls(4, 16), x, y, epochs=25, batch=128, lr=3e-3)
    with torch.no_grad():
        mse = float(((model(torch.from_numpy(x)) - torch.from_numpy(y)) ** 2).mean())
    assert mse < 0.5 * float(np.var(y))


def test_daily_models_learn_and_are_deterministic():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3000, 23)).astype(np.float32)
    y = (0.3 * x[:, 5] + 0.05 * rng.normal(size=3000)).astype(np.float32)
    first = train_lightgbm(x[:2400], y[:2400], x[2400:], y[2400:], seed=0)
    second = train_lightgbm(x[:2400], y[:2400], x[2400:], y[2400:], seed=0)
    np.testing.assert_array_equal(first.predict(x[2400:]), second.predict(x[2400:]))
    assert np.corrcoef(first.predict(x[2400:]), y[2400:])[0, 1] > 0.8
    torch.manual_seed(0)
    mlp = fit_torch(DailyMLP(23, 32), x, y, epochs=20, batch=256, lr=3e-3)
    with torch.no_grad():
        prediction = mlp(torch.from_numpy(x)).numpy()
    assert np.corrcoef(prediction, y)[0, 1] > 0.8
