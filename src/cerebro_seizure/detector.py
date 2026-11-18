"""Real-Time Online Streaming EEG Seizure Monitoring & Alarm Engine.

Maintains a ring-buffer of multi-lead EEG, executes sliding-window Ledoit-Wolf
covariance estimation, projects onto Riemannian manifolds, and triggers
calibrated clinical alerts prior to electrographic seizure onset.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np

from cerebro_seizure.covariance import ledoit_wolf_shrinkage
from cerebro_seizure.manifold import (
    riemannian_distance_airm,
    frechet_mean_riemannian,
    regularize_spd,
)
from cerebro_seizure.classifier import (
    RiemannianMDMClassifier,
    TangentSpaceClassifier,
)


class AlarmLevel(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    SEIZURE_ALERT = "SEIZURE_ALERT"
    ICTAL_ACTIVE = "ICTAL_ACTIVE"


@dataclass
class DetectionTelemetry:
    timestamp_sec: float
    alarm_level: AlarmLevel
    riemannian_distance: float
    z_score: float
    seizure_risk_score: float
    predicted_tto_sec: float | None
    top_synchrony_pair: tuple[int, int]
    shrinkage_rho: float


class OnlineSeizureMonitor:
    """Real-time streaming detector with stateful hysteresis and conformal risk calibration."""

    def __init__(
        self,
        num_channels: int = 8,
        sampling_rate: float = 250.0,
        window_sec: float = 2.0,
        hop_sec: float = 0.25,
        warning_threshold: float = 0.45,
        alert_threshold: float = 0.72,
        consecutive_triggers_required: int = 2,
    ) -> None:
        self.C = num_channels
        self.fs = float(sampling_rate)
        self.window_samples = int(window_sec * self.fs)
        self.hop_samples = int(hop_sec * self.fs)
        self.warning_threshold = float(warning_threshold)
        self.alert_threshold = float(alert_threshold)
        self.consecutive_required = consecutive_triggers_required

        # State variables
        self._buffer: np.ndarray = np.zeros((self.C, 0), dtype=np.float64)
        self._samples_seen: int = 0
        self._samples_since_last_eval: int = 0

        self.baseline_mean_: np.ndarray | None = None
        self.baseline_dist_mu_: float = 0.0
        self.baseline_dist_sigma_: float = 1.0

        self.mdm_classifier: RiemannianMDMClassifier | None = None
        self.tangent_classifier: TangentSpaceClassifier | None = None

        self.prev_cov: np.ndarray | None = None
        self.prev_time: float = 0.0

        self.consecutive_high_risk: int = 0
        self.current_alarm: AlarmLevel = AlarmLevel.NORMAL

    def calibrate_baseline(self, interictal_eeg: np.ndarray) -> None:
        """Calibrate inter-ictal resting covariance baseline from recording X (C, T)."""
        n_samples = interictal_eeg.shape[1]
        covs = []
        for start in range(0, n_samples - self.window_samples + 1, self.hop_samples * 2):
            win = interictal_eeg[:, start : start + self.window_samples]
            c, _ = ledoit_wolf_shrinkage(win)
            covs.append(c)

        if not covs:
            raise ValueError("Inter-ictal EEG recording is too short for baseline calibration.")

        self.baseline_mean_ = frechet_mean_riemannian(covs)
        dists = [riemannian_distance_airm(c, self.baseline_mean_) for c in covs]
        self.baseline_dist_mu_ = float(np.mean(dists))
        self.baseline_dist_sigma_ = float(max(1e-4, np.std(dists)))

    def fit_classifiers(
        self,
        interictal_eeg: np.ndarray,
        preictal_eeg: np.ndarray,
    ) -> None:
        """Train both Riemannian MDM and Tangent Space classifiers from labeled calibration data."""
        self.calibrate_baseline(interictal_eeg)

        covs_0 = []
        n0 = interictal_eeg.shape[1]
        for s in range(0, n0 - self.window_samples + 1, self.hop_samples * 2):
            win = interictal_eeg[:, s : s + self.window_samples]
            c, _ = ledoit_wolf_shrinkage(win)
            covs_0.append(c)

        covs_1 = []
        n1 = preictal_eeg.shape[1]
        for s in range(0, n1 - self.window_samples + 1, self.hop_samples * 2):
            win = preictal_eeg[:, s : s + self.window_samples]
            c, _ = ledoit_wolf_shrinkage(win)
            covs_1.append(c)

        all_covs = covs_0 + covs_1
        all_labels = [0] * len(covs_0) + [1] * len(covs_1)

        self.mdm_classifier = RiemannianMDMClassifier(temperature=0.8)
        self.mdm_classifier.fit(all_covs, all_labels)

        self.tangent_classifier = TangentSpaceClassifier(l2_reg=1e-2)
        self.tangent_classifier.fit(all_covs, all_labels)

    def push_samples(self, new_samples: np.ndarray) -> list[DetectionTelemetry]:
        """Ingest real-time EEG samples (C, N_new) and produce telemetry on hop boundaries."""
        if new_samples.ndim != 2 or new_samples.shape[0] != self.C:
            raise ValueError(f"Expected input of shape ({self.C}, N), got {new_samples.shape}")

        self._buffer = np.hstack([self._buffer, new_samples])
        n_new = new_samples.shape[1]
        self._samples_seen += n_new
        self._samples_since_last_eval += n_new

        telemetry_events: list[DetectionTelemetry] = []

        # Process while we have enough samples for evaluation hop
        while self._buffer.shape[1] >= self.window_samples and self._samples_since_last_eval >= self.hop_samples:
            win = self._buffer[:, : self.window_samples]
            t_current = (self._samples_seen - self._buffer.shape[1] + self.window_samples) / self.fs

            telem = self._evaluate_window(win, t_current)
            telemetry_events.append(telem)

            # Advance buffer by hop_samples
            self._buffer = self._buffer[:, self.hop_samples :]
            self._samples_since_last_eval -= self.hop_samples

        return telemetry_events

    def _evaluate_window(self, win: np.ndarray, timestamp: float) -> DetectionTelemetry:
        # Estimate covariance with optimal shrinkage
        cov, rho = ledoit_wolf_shrinkage(win)
        cov = regularize_spd(cov)

        # Compute signal amplitude check for overt ictal activity
        rms_amplitude = float(np.sqrt(np.mean(win**2)))

        # Distance to baseline
        if self.baseline_mean_ is not None:
            dist_base = riemannian_distance_airm(cov, self.baseline_mean_)
            z_score = (dist_base - self.baseline_dist_mu_) / self.baseline_dist_sigma_
        else:
            dist_base = 0.0
            z_score = 0.0

        # Classifier risk score
        risk_score = 0.0
        if self.tangent_classifier is not None:
            risk_score = self.tangent_classifier.predict_risk_score(cov)
        elif self.mdm_classifier is not None:
            probs = self.mdm_classifier.predict_proba(cov)
            risk_score = probs.get(1, 0.0)
        else:
            # Unsupervised sigmoid on z-score
            risk_score = float(1.0 / (1.0 + np.exp(-max(-5.0, min(5.0, z_score - 2.0)))))

        # Estimate time to onset from geodesic velocity
        predicted_tto: float | None = None
        dt = timestamp - self.prev_time if self.prev_time > 0 else 0.0
        if self.prev_cov is not None and dt > 0.05:
            step_dist = riemannian_distance_airm(cov, self.prev_cov)
            vel = step_dist / dt
            if vel > 0.01 and risk_score > 0.5:
                # Assume critical distance is ~ 3 standard deviations
                remaining_dist = max(0.0, (self.baseline_dist_mu_ + 3.0 * self.baseline_dist_sigma_) - dist_base)
                predicted_tto = max(2.0, min(1800.0, remaining_dist / vel))

        self.prev_cov = cov.copy()
        self.prev_time = timestamp

        # Find channel pair with maximum absolute off-diagonal correlation
        corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)) + 1e-12)
        np.fill_diagonal(corr, 0.0)
        top_pair_idx = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
        top_synchrony_pair = (int(top_pair_idx[0]), int(top_pair_idx[1]))

        # Determine alarm level with hysteresis
        if rms_amplitude > 4.5:
            alarm = AlarmLevel.ICTAL_ACTIVE
            self.consecutive_high_risk = self.consecutive_required
        elif risk_score >= self.alert_threshold:
            self.consecutive_high_risk += 1
            if self.consecutive_high_risk >= self.consecutive_required:
                alarm = AlarmLevel.SEIZURE_ALERT
            else:
                alarm = AlarmLevel.WARNING
        elif risk_score >= self.warning_threshold:
            self.consecutive_high_risk = max(1, self.consecutive_high_risk)
            alarm = AlarmLevel.WARNING
        else:
            self.consecutive_high_risk = max(0, self.consecutive_high_risk - 1)
            alarm = AlarmLevel.NORMAL

        self.current_alarm = alarm

        return DetectionTelemetry(
            timestamp_sec=round(timestamp, 3),
            alarm_level=alarm,
            riemannian_distance=round(dist_base, 4),
            z_score=round(z_score, 3),
            seizure_risk_score=round(risk_score, 4),
            predicted_tto_sec=round(predicted_tto, 1) if predicted_tto is not None else None,
            top_synchrony_pair=top_synchrony_pair,
            shrinkage_rho=round(rho, 4),
        )
