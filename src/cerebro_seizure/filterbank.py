"""Filter Bank Decomposition for Riemannian EEG Feature Extraction.

Decomposes multi-channel continuous EEG into physiological frequency bands:
- Delta (0.5 - 4.0 Hz)
- Theta (4.0 - 8.0 Hz)
- Alpha (8.0 - 12.0 Hz)
- Beta (12.0 - 30.0 Hz)
- Gamma (30.0 - 50.0 Hz)

Provides multi-band covariance extraction and concatenated tangent space projections.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, sosfiltfilt
from typing import Sequence

from cerebro_seizure.covariance import ledoit_wolf_shrinkage
from cerebro_seizure.manifold import (
    tangent_space_project,
    tangent_space_vectorize,
    regularize_spd,
)


@dataclass(frozen=True)
class FilterbankBand:
    name: str
    low_freq: float
    high_freq: float


EEG_BANDS: tuple[FilterbankBand, ...] = (
    FilterbankBand("delta", 0.5, 4.0),
    FilterbankBand("theta", 4.0, 8.0),
    FilterbankBand("alpha", 8.0, 12.0),
    FilterbankBand("beta", 12.0, 30.0),
    FilterbankBand("gamma", 30.0, 50.0),
)


class RiemannianFilterbank:
    """Multi-band temporal filterbank with zero-phase filtering and Riemannian covariance extraction."""

    def __init__(
        self,
        sampling_rate: float = 250.0,
        bands: Sequence[FilterbankBand] = EEG_BANDS,
        filter_order: int = 4,
    ) -> None:
        self.sampling_rate = float(sampling_rate)
        self.bands = list(bands)
        self.filter_order = filter_order
        self._sos_filters: dict[str, np.ndarray] = {}
        self._build_filters()

    def _build_filters(self) -> None:
        nyq = 0.5 * self.sampling_rate
        for band in self.bands:
            low = max(0.01, band.low_freq) / nyq
            high = min(nyq - 0.1, band.high_freq) / nyq
            if low >= high or low <= 0 or high >= 1.0:
                # Fallback safeguard
                low = 0.05
                high = 0.9
            sos = butter(self.filter_order, [low, high], btype="bandpass", output="sos")
            self._sos_filters[band.name] = sos

    def filter_band(self, X: np.ndarray, band_name: str) -> np.ndarray:
        """Filter (C, T) EEG array in the specified frequency band."""
        if band_name not in self._sos_filters:
            raise KeyError(f"Unknown band '{band_name}'. Available: {list(self._sos_filters.keys())}")
        sos = self._sos_filters[band_name]
        # Apply zero-phase forward-backward filter along time axis (axis=1)
        # Handle short signals with padlen guard
        t_len = X.shape[1]
        padlen = min(t_len - 1, 3 * (2 * self.filter_order + 1))
        if padlen <= 0:
            return X.copy()
        filtered = sosfiltfilt(sos, X, axis=-1, padlen=padlen)
        return np.ascontiguousarray(filtered, dtype=np.float64)

    def filter_all_bands(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return dictionary of filtered signals for each band: {band_name: (C, T)}."""
        return {b.name: self.filter_band(X, b.name) for b in self.bands}

    def extract_band_covariances(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Extract Ledoit-Wolf regularized covariance for each frequency band."""
        cov_dict = {}
        for band in self.bands:
            filtered = self.filter_band(X, band.name)
            cov, _ = ledoit_wolf_shrinkage(filtered)
            cov_dict[band.name] = regularize_spd(cov)
        return cov_dict

    def extract_composite_tangent_vector(
        self,
        X: np.ndarray,
        reference_means: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Project each band's covariance to its reference tangent space and concatenate."""
        vectors = []
        for band in self.bands:
            filtered = self.filter_band(X, band.name)
            cov, _ = ledoit_wolf_shrinkage(filtered)
            ref = reference_means[band.name]
            tangent_mat = tangent_space_project(cov, ref)
            t_vec = tangent_space_vectorize(tangent_mat)
            vectors.append(t_vec)
        return np.concatenate(vectors, axis=0)
