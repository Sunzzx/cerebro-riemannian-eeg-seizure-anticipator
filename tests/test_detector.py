"""Tests for Real-Time Online Seizure Monitor & Alarm Engine."""

import numpy as np
import pytest

from cerebro_seizure.detector import OnlineSeizureMonitor, AlarmLevel
from cerebro_seizure.simulator import SyntheticEEGSimulator, SyntheticEEGConfig, ClinicalState


def test_online_monitor_lifecycle():
    cfg = SyntheticEEGConfig(num_channels=4, sampling_rate=200.0, seed=555)
    sim = SyntheticEEGSimulator(cfg)

    monitor = OnlineSeizureMonitor(
        num_channels=4,
        sampling_rate=200.0,
        window_sec=1.0,
        hop_sec=0.2,
        warning_threshold=0.4,
        alert_threshold=0.7,
        consecutive_triggers_required=2,
    )

    # Calibrate with synthetic baseline
    calib_interictal = sim.generate_state_segment(ClinicalState.INTER_ICTAL, 6.0)
    calib_preictal = sim.generate_state_segment(ClinicalState.PRE_ICTAL, 6.0)
    monitor.fit_classifiers(calib_interictal, calib_preictal)

    assert monitor.baseline_mean_ is not None
    assert monitor.tangent_classifier is not None

    # Stream 3.0 seconds of inter-ictal: should remain NORMAL or WARNING, never persistent SEIZURE_ALERT
    test_interictal = sim.generate_state_segment(ClinicalState.INTER_ICTAL, 3.0)
    telemetry_inter = monitor.push_samples(test_interictal)
    assert len(telemetry_inter) > 0
    for telem in telemetry_inter:
        assert telem.alarm_level in (AlarmLevel.NORMAL, AlarmLevel.WARNING)

    # Stream 5.0 seconds of pre-ictal: should trigger SEIZURE_ALERT
    test_preictal = sim.generate_state_segment(ClinicalState.PRE_ICTAL, 5.0)
    telemetry_pre = monitor.push_samples(test_preictal)
    alert_events = [t for t in telemetry_pre if t.alarm_level == AlarmLevel.SEIZURE_ALERT]
    assert len(alert_events) >= 1

    # Stream ictal paroxysm: should transition to ICTAL_ACTIVE
    test_ictal = sim.generate_state_segment(ClinicalState.ICTAL, 2.0)
    telemetry_ictal = monitor.push_samples(test_ictal)
    ictal_events = [t for t in telemetry_ictal if t.alarm_level == AlarmLevel.ICTAL_ACTIVE]
    assert len(ictal_events) >= 1
