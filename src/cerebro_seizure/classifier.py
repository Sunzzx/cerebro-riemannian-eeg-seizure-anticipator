"""Riemannian Manifold Classifiers for Seizure Anticipation.

Implements:
- Riemannian Minimum Distance to Mean (MDM) with softmax probability calibration
- Tangent Space Regularized Linear Classifier (Ridge/Logistic)
- Riemannian Geodesic Trajectory Anomaly Detector for pre-ictal warning.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

from cerebro_seizure.manifold import (
    riemannian_distance_airm,
    frechet_mean_riemannian,
    tangent_space_project,
    tangent_space_vectorize,
    regularize_spd,
)


class RiemannianMDMClassifier:
    """Riemannian Minimum Distance to Mean (MDM) classifier on S_{++}^C."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)
        self.classes_: list[int] = []
        self.class_means_: dict[int, np.ndarray] = {}

    def fit(self, covariances: Sequence[np.ndarray], labels: Sequence[int]) -> RiemannianMDMClassifier:
        """Fit MDM by computing the Fréchet mean of covariances for each class."""
        covs = [regularize_spd(c) for c in covariances]
        y = np.array(labels, dtype=int)
        self.classes_ = sorted(list(set(y.tolist())))

        if len(self.classes_) < 2:
            raise ValueError(f"Expected at least 2 distinct classes, got {self.classes_}")

        self.class_means_ = {}
        for cls in self.classes_:
            cls_covs = [covs[i] for i in range(len(covs)) if y[i] == cls]
            if not cls_covs:
                raise ValueError(f"No covariance samples for class {cls}")
            self.class_means_[cls] = frechet_mean_riemannian(cls_covs)

        return self

    def predict_distances(self, cov: np.ndarray) -> dict[int, float]:
        """Compute AIRM distance from input cov to each class Fréchet mean."""
        cov_reg = regularize_spd(cov)
        return {cls: riemannian_distance_airm(cov_reg, self.class_means_[cls]) for cls in self.classes_}

    def predict_proba(self, cov: np.ndarray) -> dict[int, float]:
        """Predict calibrated softmax posterior probabilities using negative squared AIRM distances."""
        dists = self.predict_distances(cov)
        # Softmax over - dist^2 / (2 * temperature^2)
        logits = np.array([- (dists[cls]**2) / (2.0 * max(1e-4, self.temperature**2)) for cls in self.classes_])
        max_l = np.max(logits)
        exp_l = np.exp(logits - max_l)
        probs = exp_l / np.sum(exp_l)
        return {cls: float(probs[i]) for i, cls in enumerate(self.classes_)}

    def predict(self, cov: np.ndarray) -> int:
        """Predict class with minimum Riemannian distance to mean."""
        dists = self.predict_distances(cov)
        return min(dists, key=dists.get)  # type: ignore[arg-type]


class TangentSpaceClassifier:
    """Projects SPD matrices to Euclidean tangent space and fits L2-regularized linear model."""

    def __init__(self, l2_reg: float = 1e-2) -> None:
        self.l2_reg = float(l2_reg)
        self.reference_mean_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0

    def fit(self, covariances: Sequence[np.ndarray], labels: Sequence[int]) -> TangentSpaceClassifier:
        """Compute global Fréchet mean, vectorize to tangent space, and solve regularized least squares."""
        covs = [regularize_spd(c) for c in covariances]
        y = np.array(labels, dtype=np.float64)
        # Map binary labels to {-1, +1}
        y_bin = np.where(y == 0, -1.0, 1.0)

        # Reference mean is the global Fréchet mean
        self.reference_mean_ = frechet_mean_riemannian(covs)

        # Vectorize all matrices in tangent space
        X_tangent = []
        for c in covs:
            S = tangent_space_project(c, self.reference_mean_)
            v = tangent_space_vectorize(S)
            X_tangent.append(v)

        V = np.array(X_tangent, dtype=np.float64)  # (N, D)
        n, d = V.shape

        # Augmented design matrix [V, 1]
        V_aug = np.hstack([V, np.ones((n, 1))])
        # Regularization matrix (do not penalize bias)
        reg_mat = self.l2_reg * np.eye(d + 1)
        reg_mat[-1, -1] = 0.0

        # Normal equations: (V^T V + reg) w = V^T y
        w_aug = np.linalg.solve(V_aug.T @ V_aug + reg_mat, V_aug.T @ y_bin)
        self.weights_ = w_aug[:-1]
        self.bias_ = float(w_aug[-1])
        return self

    def decision_function(self, cov: np.ndarray) -> float:
        """Compute linear decision score w^T v + b in tangent space."""
        if self.reference_mean_ is None or self.weights_ is None:
            raise RuntimeError("Classifier has not been fitted.")
        cov_reg = regularize_spd(cov)
        S = tangent_space_project(cov_reg, self.reference_mean_)
        v = tangent_space_vectorize(S)
        score = float(np.dot(v, self.weights_) + self.bias_)
        return score

    def predict_risk_score(self, cov: np.ndarray) -> float:
        """Predict calibrated sigmoid risk score in [0.0, 1.0]."""
        score = self.decision_function(cov)
        # Sigmoid
        prob = 1.0 / (1.0 + np.exp(-np.clip(score, -20.0, 20.0)))
        return float(prob)

    def predict(self, cov: np.ndarray) -> int:
        """Predict binary class (0 or 1)."""
        return 1 if self.decision_function(cov) >= 0.0 else 0


class RiemannianTrajectoryAnomalyDetector:
    """Tracks continuous Riemannian geodesic migration away from baseline inter-ictal state."""

    def __init__(self, z_threshold: float = 2.5, velocity_weight: float = 0.3) -> None:
        self.z_threshold = float(z_threshold)
        self.velocity_weight = float(velocity_weight)
        self.baseline_mean_: np.ndarray | None = None
        self.baseline_dist_mu_: float = 0.0
        self.baseline_dist_sigma_: float = 1.0
        self.prev_cov_: np.ndarray | None = None

    def calibrate_baseline(self, interictal_covariances: Sequence[np.ndarray]) -> RiemannianTrajectoryAnomalyDetector:
        """Calibrate the inter-ictal baseline distribution (mean, variance of Riemannian distance)."""
        covs = [regularize_spd(c) for c in interictal_covariances]
        self.baseline_mean_ = frechet_mean_riemannian(covs)

        dists = [riemannian_distance_airm(c, self.baseline_mean_) for c in covs]
        self.baseline_dist_mu_ = float(np.mean(dists))
        self.baseline_dist_sigma_ = float(max(1e-4, np.std(dists)))
        return self

    def score_transition(self, current_cov: np.ndarray, dt_sec: float = 1.0) -> tuple[float, float, float]:
        """Evaluate Riemannian trajectory metrics.

        Returns (z_score, geodesic_velocity, combined_risk_index).
        """
        if self.baseline_mean_ is None:
            raise RuntimeError("Baseline not calibrated.")

        cov_reg = regularize_spd(current_cov)
        # Distance to baseline Fréchet center
        d_baseline = riemannian_distance_airm(cov_reg, self.baseline_mean_)
        z_score = (d_baseline - self.baseline_dist_mu_) / self.baseline_dist_sigma_

        # Geodesic velocity (rate of Riemannian drift)
        if self.prev_cov_ is not None and dt_sec > 0:
            step_dist = riemannian_distance_airm(cov_reg, self.prev_cov_)
            velocity = step_dist / dt_sec
        else:
            velocity = 0.0

        self.prev_cov_ = cov_reg.copy()

        # Combined risk index
        risk_index = max(0.0, z_score) + self.velocity_weight * max(0.0, velocity)
        return float(z_score), float(velocity), float(risk_index)
