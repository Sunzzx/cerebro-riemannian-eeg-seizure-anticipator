"""Covariance Estimation and Epoching for Multi-Channel EEG.

Implements:
- Sample Covariance Matrix (SCM)
- Analytical Ledoit-Wolf Optimal Shrinkage Covariance
- Sliding window epoch extraction and continuous tensor generation.
"""

from __future__ import annotations

import numpy as np
from typing import Literal
from cerebro_seizure.manifold import regularize_spd


def sample_covariance(X: np.ndarray, center: bool = True) -> np.ndarray:
    """Compute Sample Covariance Matrix for channels-by-time array X in R^{C x T}."""
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array (channels x time), got shape {X.shape}")
    c, t = X.shape
    if t < 2:
        raise ValueError(f"Need at least 2 time points, got {t}")

    if center:
        mean_vec = np.mean(X, axis=1, keepdims=True)
        Xc = X - mean_vec
    else:
        Xc = X

    cov = (Xc @ Xc.T) / (t - 1.0)
    return regularize_spd(cov)


def ledoit_wolf_shrinkage(X: np.ndarray, center: bool = True) -> tuple[np.ndarray, float]:
    """Compute Ledoit-Wolf optimal shrinkage covariance matrix (sklearn-compatible).

    Sigma^* = (1 - rho) * S + rho * mu * I
    where mu = trace(S) / p, and rho in [0, 1] is the optimal shrinkage intensity.
    Returns (Sigma^*, rho).
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array (channels x time), got shape {X.shape}")
    p, n = X.shape
    if n < 3:
        scm = sample_covariance(X, center=center)
        return regularize_spd(scm + 1e-4 * np.eye(p)), 1.0

    if center:
        mean_vec = np.mean(X, axis=1, keepdims=True)
        Xc = X - mean_vec
    else:
        Xc = X

    # sklearn layout: observations x features
    Xt = Xc.T  # (n, p)
    emp_cov = (Xt.T @ Xt) / float(n)
    mu = float(np.trace(emp_cov) / p)
    delta_mat = emp_cov.copy()
    delta_mat.flat[:: p + 1] -= mu
    delta = float(np.sum(delta_mat ** 2) / p)
    if delta < 1e-18:
        return regularize_spd(mu * np.eye(p)), 1.0

    X2 = Xt ** 2
    beta_ = float(np.sum(X2.T @ X2) / (n * n))
    beta = (beta_ - float(np.sum(emp_cov ** 2))) / float(n)
    beta = float(min(beta, delta))
    rho = float(np.clip(beta / delta, 0.0, 1.0))

    target = mu * np.eye(p, dtype=np.float64)
    shrunk = (1.0 - rho) * emp_cov + rho * target
    return regularize_spd(shrunk), rho


def epoch_signal(
    X: np.ndarray,
    epoch_samples: int,
    hop_samples: int,
) -> np.ndarray:
    """Slice continuous EEG array X of shape (C, T) into overlapping epochs (N, C, epoch_samples)."""
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array (channels x time), got shape {X.shape}")
    c, t = X.shape
    if epoch_samples > t:
        raise ValueError(f"Epoch length ({epoch_samples}) exceeds total duration ({t}).")
    if hop_samples <= 0:
        raise ValueError(f"Hop samples must be > 0, got {hop_samples}")

    starts = range(0, t - epoch_samples + 1, hop_samples)
    epochs = [X[:, s : s + epoch_samples] for s in starts]
    if not epochs:
        return np.empty((0, c, epoch_samples), dtype=X.dtype)
    return np.stack(epochs, axis=0)


def sliding_window_covariances(
    X: np.ndarray,
    window_samples: int,
    hop_samples: int,
    estimator: Literal["ledoit_wolf", "scm"] = "ledoit_wolf",
) -> np.ndarray:
    """Extract sliding window covariance matrices (N_windows, C, C) from EEG signal."""
    epochs = epoch_signal(X, window_samples, hop_samples)
    n_epochs, c, _ = epochs.shape
    covs = np.zeros((n_epochs, c, c), dtype=np.float64)

    for i in range(n_epochs):
        epoch = epochs[i]
        if estimator == "ledoit_wolf":
            cov, _ = ledoit_wolf_shrinkage(epoch)
        else:
            cov = sample_covariance(epoch)
        covs[i] = cov

    return covs
