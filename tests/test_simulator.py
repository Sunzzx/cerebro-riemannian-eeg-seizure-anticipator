"""Tests for Physiological Multi-Channel EEG Simulator."""

import numpy as np
from cerebro_seizure.simulator import (
    ClinicalState,
    SyntheticEEGConfig,
    SyntheticEEGSimulator,
)


def test_simulator_dimensions_and_states():
    cfg = SyntheticEEGConfig(num_channels=6, sampling_rate=200.0, seed=123)
    sim = SyntheticEEGSimulator(cfg)

    dur = 2.0
    n_expected = int(dur * 200.0)

    s_inter = sim.generate_state_segment(ClinicalState.INTER_ICTAL, dur)
    s_pre = sim.generate_state_segment(ClinicalState.PRE_ICTAL, dur)
    s_ictal = sim.generate_state_segment(ClinicalState.ICTAL, dur)

    assert s_inter.shape == (6, n_expected)
    assert s_pre.shape == (6, n_expected)
    assert s_ictal.shape == (6, n_expected)

    # Amplitude of ictal should be substantially larger than inter-ictal
    rms_inter = np.sqrt(np.mean(s_inter**2))
    rms_ictal = np.sqrt(np.mean(s_ictal**2))
    assert rms_ictal > 3.0 * rms_inter


def test_continuous_transition_recording():
    cfg = SyntheticEEGConfig(num_channels=4, sampling_rate=100.0, seed=42)
    sim = SyntheticEEGSimulator(cfg)

    stream, timeline = sim.generate_continuous_transition_recording(5.0, 5.0, 5.0)
    assert stream.shape == (4, 1500)
    assert len(timeline) == 3
    assert timeline[0][2] == ClinicalState.INTER_ICTAL
    assert timeline[1][2] == ClinicalState.PRE_ICTAL
    assert timeline[2][2] == ClinicalState.ICTAL
