import numpy as np
import pandas as pd
import pytest

from state_space_arb.config import KalmanConfig
from state_space_arb.data import make_synthetic_pair
from state_space_arb.engine import KalmanFilterEngine, KalmanTracker, MatrixSingularityError


def test_converges_to_true_slope_from_a_bad_prior():
    """Deterministic y = 2x + 3. Start from beta=0 with a wide prior: must lock on within 50 steps."""
    rng = np.random.default_rng(0)
    x = 100 + np.cumsum(rng.normal(0, 1, 200))
    y = 2.0 * x + 3.0
    k = KalmanTracker(beta0=0.0, alpha0=0.0, p_beta=10.0, p_alpha=1e4, q=1e-9, r=1e-2)
    for i in range(50):
        res = k.update(y[i], x[i])
    assert abs(res.beta - 2.0) < 0.01


def test_tracks_linear_drift_in_beta():
    """Slope drifts 1.0 -> 1.5 linearly; with a sensible Q the filter stays close at the end."""
    rng = np.random.default_rng(1)
    n = 400
    x = 100 + np.cumsum(rng.normal(0, 1, n))
    beta = np.linspace(1.0, 1.5, n)
    y = beta * x
    k = KalmanTracker.from_ols(y[:50], x[:50], q=1e-6, r=1e-2)
    for i in range(n):
        res = k.update(y[i], x[i])
    assert abs(res.beta - 1.5) < 0.05


@pytest.mark.parametrize("seed", [3, 7, 11, 21, 42])
def test_spread_mean_is_near_zero_and_zscores_are_calibrated(seed):
    d = make_synthetic_pair(n=2000, seed=seed)
    s = KalmanFilterEngine().compute_states(d, "AAA_close", "BBB_close").iloc[100:]
    assert abs(s["spread"].mean()) < 0.1 * s["spread"].std()
    assert 0.7 < s["zscore"].std() < 1.3     # z should be ~unit variance regardless of seed


def test_batch_matches_streaming():
    d = make_synthetic_pair(n=300)
    cfg = KalmanConfig()
    batch = KalmanFilterEngine(cfg).compute_states(d, "AAA_close", "BBB_close")
    y, x = d["AAA_close"].to_numpy(), d["BBB_close"].to_numpy()
    k = KalmanTracker.from_ols(y[:cfg.warmup], x[:cfg.warmup], q=cfg.transition_covariance_q,
                               sigma_halflife=cfg.sigma_halflife)
    streamed = np.array([k.update(a, b).spread for a, b in zip(y, x)])
    np.testing.assert_allclose(batch["spread"].to_numpy(), streamed)


def test_no_lookahead_in_filter():
    """Changing FUTURE prices must not change any past output."""
    d = make_synthetic_pair(n=300)
    full = KalmanFilterEngine().compute_states(d, "AAA_close", "BBB_close")
    d2 = d.copy()
    d2.iloc[200:, d2.columns.get_loc("AAA_close")] *= 1.5
    pert = KalmanFilterEngine().compute_states(d2, "AAA_close", "BBB_close")
    pd.testing.assert_frame_equal(full.iloc[:200], pert.iloc[:200])


def test_constant_reference_raises():
    d = pd.DataFrame({"y": np.linspace(1, 2, 60), "x": np.full(60, 5.0)})
    with pytest.raises(MatrixSingularityError):
        KalmanFilterEngine().compute_states(d, "y", "x")


def test_nan_input_raises():
    k = KalmanTracker(1, 0, 1, 1, 1e-5, 1e-2)
    with pytest.raises(MatrixSingularityError):
        k.update(float("nan"), 1.0)
