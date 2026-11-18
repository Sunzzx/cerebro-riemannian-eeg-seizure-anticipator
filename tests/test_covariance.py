"""Tests for Covariance Estimation, Ledoit-Wolf Shrinkage and Epoching."""

import numpy as np
import pytest

from cerebro_seizure.covariance import (
    sample_covariance,
    ledoit_wolf_shrinkage,
    epoch_signal,
    sliding_window_covariances,
)
from cerebro_seizure.manifold import is_spd


def test_sample_covariance():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(6, 500))
    cov = sample_covariance(X)
    assert cov.shape == (6, 6)
    assert is_spd(cov)
    assert np.allclose(cov, cov.T)


def test_ledoit_wolf_shrinkage():
    rng = np.random.default_rng(42)
    # Ill-conditioned regime: fewer samples than channels (p=10, n=8)
    X_ill = rng.normal(0, 1, size=(10, 8))
    shrunk_cov, rho = ledoit_wolf_shrinkage(X_ill)

    assert shrunk_cov.shape == (10, 10)
    assert is_spd(shrunk_cov)
    assert 0.0 <= rho <= 1.0

    # Well-conditioned regime: many samples (p=4, n=2000)
    X_well = rng.normal(0, 1, size=(4, 2000))
    shrunk_well, rho_well = ledoit_wolf_shrinkage(X_well)
    assert is_spd(shrunk_well)
    # In large sample limit, shrinkage rho should be small
    assert rho_well < 0.2


def test_epoch_signal():
    X = np.zeros((8, 1000))
    # Epoch of 250 samples, hop of 125 samples
    epochs = epoch_signal(X, epoch_samples=250, hop_samples=125)
    # Starts at 0, 125, 250, 375, 500, 625, 750 (total 7 epochs)
    assert epochs.shape == (7, 8, 250)

    # Edge cases
    with pytest.raises(ValueError):
        epoch_signal(X, epoch_samples=2000, hop_samples=100)

    with pytest.raises(ValueError):
        epoch_signal(X, epoch_samples=100, hop_samples=0)


def test_sliding_window_covariances():
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, size=(4, 600))
    covs = sliding_window_covariances(X, window_samples=200, hop_samples=100, estimator="ledoit_wolf")
    assert covs.ndim == 3
    assert covs.shape[1:] == (4, 4)
    for c in covs:
        assert is_spd(c)
