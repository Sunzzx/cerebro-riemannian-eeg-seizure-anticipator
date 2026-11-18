"""Cerebro: Real-Time Riemannian Manifold EEG Epileptic Seizure Anticipator.

Physiological brain-computer interface and neuromorphic seizure prediction
engine using differential geometry of Symmetric Positive Definite (SPD) manifolds.
"""

__version__ = "0.1.0"
__author__ = "Ayush Kandwal (Sunzzx)"
__license__ = "MIT"

from cerebro_seizure.manifold import (
    matrix_power_sym,
    matrix_log_sym,
    matrix_exp_sym,
    riemannian_distance_airm,
    riemannian_distance_log_euclidean,
    geodesic_interpolate,
    frechet_mean_riemannian,
    tangent_space_project,
    tangent_space_vectorize,
    tangent_space_unvectorize,
    is_spd,
    regularize_spd,
)

from cerebro_seizure.covariance import (
    sample_covariance,
    ledoit_wolf_shrinkage,
    epoch_signal,
    sliding_window_covariances,
)

from cerebro_seizure.filterbank import (
    EEG_BANDS,
    FilterbankBand,
    RiemannianFilterbank,
)

from cerebro_seizure.simulator import (
    ClinicalState,
    SyntheticEEGConfig,
    SyntheticEEGSimulator,
)

from cerebro_seizure.classifier import (
    RiemannianMDMClassifier,
    TangentSpaceClassifier,
    RiemannianTrajectoryAnomalyDetector,
)

from cerebro_seizure.detector import (
    AlarmLevel,
    DetectionTelemetry,
    OnlineSeizureMonitor,
)

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "matrix_power_sym",
    "matrix_log_sym",
    "matrix_exp_sym",
    "riemannian_distance_airm",
    "riemannian_distance_log_euclidean",
    "geodesic_interpolate",
    "frechet_mean_riemannian",
    "tangent_space_project",
    "tangent_space_vectorize",
    "tangent_space_unvectorize",
    "is_spd",
    "regularize_spd",
    "sample_covariance",
    "ledoit_wolf_shrinkage",
    "epoch_signal",
    "sliding_window_covariances",
    "EEG_BANDS",
    "FilterbankBand",
    "RiemannianFilterbank",
    "ClinicalState",
    "SyntheticEEGConfig",
    "SyntheticEEGSimulator",
    "RiemannianMDMClassifier",
    "TangentSpaceClassifier",
    "RiemannianTrajectoryAnomalyDetector",
    "AlarmLevel",
    "DetectionTelemetry",
    "OnlineSeizureMonitor",
]
