"""Tests for Riemannian Classifiers & Anomaly Detection."""

import numpy as np
import pytest

from cerebro_seizure.classifier import (
    RiemannianMDMClassifier,
    TangentSpaceClassifier,
    RiemannianTrajectoryAnomalyDetector,
)
from cerebro_seizure.covariance import sample_covariance
from cerebro_seizure.simulator import SyntheticEEGSimulator, SyntheticEEGConfig, ClinicalState


@pytest.fixture
def synthetic_calibration_data():
    cfg = SyntheticEEGConfig(num_channels=4, sampling_rate=200.0, seed=777)
    sim = SyntheticEEGSimulator(cfg)

    # Generate epochs for class 0 (inter-ictal) and class 1 (pre-ictal)
    covs_0 = [sample_covariance(sim.generate_state_segment(ClinicalState.INTER_ICTAL, 1.0)) for _ in range(8)]
    covs_1 = [sample_covariance(sim.generate_state_segment(ClinicalState.PRE_ICTAL, 1.0)) for _ in range(8)]

    all_covs = covs_0 + covs_1
    labels = [0] * 8 + [1] * 8
    return all_covs, labels, sim


def test_riemannian_mdm_classifier(synthetic_calibration_data):
    covs, labels, sim = synthetic_calibration_data
    clf = RiemannianMDMClassifier(temperature=1.0)
    clf.fit(covs, labels)

    assert len(clf.class_means_) == 2

    # Test predictions
    test_inter = sample_covariance(sim.generate_state_segment(ClinicalState.INTER_ICTAL, 1.0))
    test_pre = sample_covariance(sim.generate_state_segment(ClinicalState.PRE_ICTAL, 1.0))

    pred_0 = clf.predict(test_inter)
    pred_1 = clf.predict(test_pre)

    probs_0 = clf.predict_proba(test_inter)
    assert sum(probs_0.values()) == pytest.approx(1.0, abs=1e-5)
    assert probs_0[0] > probs_0[1]

    probs_1 = clf.predict_proba(test_pre)
    assert probs_1[1] > probs_1[0]


def test_tangent_space_classifier(synthetic_calibration_data):
    covs, labels, sim = synthetic_calibration_data
    clf = TangentSpaceClassifier(l2_reg=1e-2)
    clf.fit(covs, labels)

    assert clf.weights_ is not None
    assert clf.reference_mean_ is not None

    test_inter = sample_covariance(sim.generate_state_segment(ClinicalState.INTER_ICTAL, 1.0))
    test_pre = sample_covariance(sim.generate_state_segment(ClinicalState.PRE_ICTAL, 1.0))

    score_inter = clf.decision_function(test_inter)
    score_pre = clf.decision_function(test_pre)
    assert score_pre > score_inter

    risk_inter = clf.predict_risk_score(test_inter)
    risk_pre = clf.predict_risk_score(test_pre)
    assert 0.0 <= risk_inter <= 1.0
    assert 0.0 <= risk_pre <= 1.0
    assert risk_pre > risk_inter


def test_trajectory_anomaly_detector(synthetic_calibration_data):
    covs, _, sim = synthetic_calibration_data
    interictal_covs = covs[:8]

    detector = RiemannianTrajectoryAnomalyDetector(z_threshold=2.0)
    detector.calibrate_baseline(interictal_covs)

    # Evaluating baseline should produce z near 0
    test_inter = interictal_covs[0]
    z, vel, risk = detector.score_transition(test_inter, dt_sec=1.0)
    assert abs(z) < 2.5

    # Evaluating pre-ictal sample should trigger higher z-score
    test_pre = sample_covariance(sim.generate_state_segment(ClinicalState.PRE_ICTAL, 1.5))
    z_pre, vel_pre, risk_pre = detector.score_transition(test_pre, dt_sec=1.0)
    assert z_pre > z
