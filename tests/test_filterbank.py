"""Tests for Multi-Band Riemannian Filterbank."""

import numpy as np
import pytest

from cerebro_seizure.filterbank import RiemannianFilterbank, EEG_BANDS
from cerebro_seizure.manifold import is_spd


def test_filterbank_frequency_isolation():
    fs = 250.0
    fb = RiemannianFilterbank(sampling_rate=fs, filter_order=4)

    # Generate a pure 10 Hz alpha wave (4 channels, 2.0 sec)
    t = np.arange(int(2.0 * fs)) / fs
    alpha_wave = np.tile(np.sin(2 * np.pi * 10.0 * t), (4, 1))

    bands_data = fb.filter_all_bands(alpha_wave)
    assert "alpha" in bands_data
    assert "delta" in bands_data

    # Alpha band energy should dominate
    energy_alpha = np.sum(bands_data["alpha"]**2)
    energy_delta = np.sum(bands_data["delta"]**2)
    energy_gamma = np.sum(bands_data["gamma"]**2)

    assert energy_alpha > 5.0 * energy_delta
    assert energy_alpha > 5.0 * energy_gamma


def test_filterbank_covariance_extraction():
    fs = 250.0
    fb = RiemannianFilterbank(sampling_rate=fs)
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(4, 500))

    cov_dict = fb.extract_band_covariances(X)
    assert len(cov_dict) == len(EEG_BANDS)
    for name, c in cov_dict.items():
        assert is_spd(c)
        assert c.shape == (4, 4)


def test_filterbank_composite_tangent_vector():
    fs = 250.0
    fb = RiemannianFilterbank(sampling_rate=fs)
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(4, 500))

    ref_means = fb.extract_band_covariances(X)
    comp_vec = fb.extract_composite_tangent_vector(X, ref_means)

    # For C=4, tangent dim per band = 4*5/2 = 10. For 5 bands = 50
    expected_dim = len(EEG_BANDS) * (4 * 5 // 2)
    assert len(comp_vec) == expected_dim
