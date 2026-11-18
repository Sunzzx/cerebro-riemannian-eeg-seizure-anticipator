"""Physiological Multi-Channel EEG Simulator for Epileptic Transition Dynamics.

Simulates realistic multi-lead surface EEG signals across three clinical states:
1. Inter-ictal: Normal resting cortical baseline with low spatial coherence
2. Pre-ictal: Hypersynchronous spatial coupling, spike-wave bursts, and Riemannian manifold drift
3. Ictal: High-amplitude paroxysmal electrographic seizure discharges
"""

from __future__ import annotations

from dataclasses import dataclass
enum_exists = True
from enum import Enum
import numpy as np


class ClinicalState(str, Enum):
    INTER_ICTAL = "inter_ictal"
    PRE_ICTAL = "pre_ictal"
    ICTAL = "ictal"


@dataclass
class SyntheticEEGConfig:
    num_channels: int = 8
    sampling_rate: float = 250.0
    num_sources: int = 12
    epileptogenic_channels: tuple[int, ...] = (2, 3, 4)
    seed: int | None = 42


class SyntheticEEGSimulator:
    """Generates realistic multi-channel continuous EEG with volume conduction and state transitions."""

    def __init__(self, config: SyntheticEEGConfig | None = None) -> None:
        self.config = config or SyntheticEEGConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.C = self.config.num_channels
        self.K = self.config.num_sources
        self.fs = self.config.sampling_rate
        self.epi_chans = self.config.epileptogenic_channels

        # Build random volume conduction lead-field matrix A in R^{C x K}
        raw_leadfield = self.rng.normal(0.0, 1.0, size=(self.C, self.K))
        # Normalize columns (sources)
        self.leadfield = raw_leadfield / np.linalg.norm(raw_leadfield, axis=0, keepdims=True)

    def generate_state_segment(
        self,
        state: ClinicalState,
        duration_sec: float,
        noise_std: float = 0.2,
    ) -> np.ndarray:
        """Generate an EEG segment of shape (C, T) corresponding to the given clinical state."""
        n_samples = int(duration_sec * self.fs)
        t = np.arange(n_samples) / self.fs

        # Source signals (K, n_samples)
        sources = np.zeros((self.K, n_samples), dtype=np.float64)

        # Baseline pink/colored noise generation
        white = self.rng.normal(0.0, 1.0, size=(self.K, n_samples))
        # Integrate slightly for 1/f spectral slope
        for k in range(self.K):
            sources[k] = np.convolve(white[k], np.exp(-np.linspace(0, 3, 15)), mode="same")

        if state == ClinicalState.INTER_ICTAL:
            # Normal resting background: independent alpha (10 Hz) and beta (20 Hz) rhythms
            for k in range(self.K):
                alpha_phase = self.rng.uniform(0, 2 * np.pi)
                beta_phase = self.rng.uniform(0, 2 * np.pi)
                alpha_amp = self.rng.uniform(0.3, 0.8)
                beta_amp = self.rng.uniform(0.1, 0.3)
                sources[k] += alpha_amp * np.sin(2 * np.pi * 10.0 * t + alpha_phase)
                sources[k] += beta_amp * np.sin(2 * np.pi * 20.0 * t + beta_phase)

        elif state == ClinicalState.PRE_ICTAL:
            # Pre-ictal: Hypersynchrony in epileptogenic sources, theta slowing (5-6 Hz), phase locking
            epi_sources = [0, 1, 2]
            shared_phase = self.rng.uniform(0, 2 * np.pi)
            for k in range(self.K):
                if k in epi_sources:
                    # High spatial correlation & 5.5 Hz rhythmic driving
                    sources[k] += 4.0 * np.sin(2 * np.pi * 5.5 * t + shared_phase + 0.1 * k)
                    # High-frequency gamma bursts (40 Hz)
                    sources[k] += 2.0 * np.sin(2 * np.pi * 40.0 * t + shared_phase)
                    # Periodic sharp spike-wave discharges (every ~0.5 sec)
                    spikes = np.exp(-((t % 0.5) - 0.05) ** 2 / 0.001)
                    sources[k] += 5.0 * spikes
                else:
                    # Moderate alpha/theta
                    sources[k] += 0.4 * np.sin(2 * np.pi * 6.0 * t + self.rng.uniform(0, 2 * np.pi))

        elif state == ClinicalState.ICTAL:
            # Ictal: Paroxysmal 3 Hz spike-and-wave seizure discharge across all cortical channels
            seizure_phase = self.rng.uniform(0, 2 * np.pi)
            for k in range(self.K):
                # Massive coherent oscillation (well above inter-ictal RMS)
                wave_component = 12.0 * np.sin(2 * np.pi * 3.0 * t + seizure_phase)
                spike_component = 18.0 * np.exp(-((t % 0.33) - 0.02) ** 2 / 0.0005)
                sources[k] += wave_component + spike_component
                # Suppress baseline colored noise contribution relative to discharge
                sources[k] = sources[k] * 1.0

        # Project to surface electrodes via leadfield: X = A @ S
        eeg = self.leadfield @ sources

        # Add measurement sensor noise
        sensor_noise = self.rng.normal(0.0, noise_std, size=eeg.shape)
        eeg += sensor_noise

        return np.ascontiguousarray(eeg, dtype=np.float64)

    def generate_continuous_transition_recording(
        self,
        inter_ictal_sec: float = 20.0,
        pre_ictal_sec: float = 20.0,
        ictal_sec: float = 10.0,
    ) -> tuple[np.ndarray, list[tuple[float, float, ClinicalState]]]:
        """Generate continuous EEG sequence transitioning through Inter-ictal -> Pre-ictal -> Ictal."""
        seg1 = self.generate_state_segment(ClinicalState.INTER_ICTAL, inter_ictal_sec)
        seg2 = self.generate_state_segment(ClinicalState.PRE_ICTAL, pre_ictal_sec)
        seg3 = self.generate_state_segment(ClinicalState.ICTAL, ictal_sec)

        full_eeg = np.concatenate([seg1, seg2, seg3], axis=1)

        t0 = 0.0
        t1 = inter_ictal_sec
        t2 = t1 + pre_ictal_sec
        t3 = t2 + ictal_sec

        timeline = [
            (t0, t1, ClinicalState.INTER_ICTAL),
            (t1, t2, ClinicalState.PRE_ICTAL),
            (t2, t3, ClinicalState.ICTAL),
        ]
        return full_eeg, timeline
