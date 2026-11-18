"""Differential Geometry on Symmetric Positive Definite (SPD) Manifolds.

Provides mathematically exact and numerically stable primitives for computing:
- Affine-Invariant Riemannian Metric (AIRM)
- Log-Euclidean Metric (LEM)
- Geodesic paths on the SPD manifold S_{++}^C
- Fréchet / Karcher Geometric Center of Mass
- Tangent space projection and isometric vectorization preserving Frobenius norms.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


def is_spd(A: np.ndarray, tol: float = 1e-10) -> bool:
    """Verify whether matrix A is symmetric positive-definite."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    # Symmetry check
    if not np.allclose(A, A.T, atol=1e-8, rtol=1e-5):
        return False
    eigvals = np.linalg.eigvalsh((A + A.T) * 0.5)
    return bool(np.all(eigvals > tol))


def regularize_spd(A: np.ndarray, min_eig: float = 1e-8, cond_max: float = 1e7) -> np.ndarray:
    """Enforce symmetry and positive-definiteness via eigenvalue thresholding."""
    A_sym = (A + A.T) * 0.5
    w, v = np.linalg.eigh(A_sym)
    max_w = float(np.max(w))
    floor_w = max(min_eig, max_w / cond_max) if max_w > 0 else min_eig
    w_clipped = np.maximum(w, floor_w)
    reconstructed = (v * w_clipped) @ v.T
    return (reconstructed + reconstructed.T) * 0.5


def matrix_power_sym(A: np.ndarray, power: float, min_eig: float = 1e-10) -> np.ndarray:
    """Compute A^power for symmetric matrix A via eigendecomposition."""
    A_sym = (A + A.T) * 0.5
    w, v = np.linalg.eigh(A_sym)
    w_safe = np.maximum(w, min_eig)
    w_powered = w_safe**power
    res = (v * w_powered) @ v.T
    return (res + res.T) * 0.5


def matrix_log_sym(A: np.ndarray, min_eig: float = 1e-12) -> np.ndarray:
    """Matrix logarithm for symmetric positive-definite matrix A."""
    A_sym = (A + A.T) * 0.5
    w, v = np.linalg.eigh(A_sym)
    w_safe = np.maximum(w, min_eig)
    log_w = np.log(w_safe)
    res = (v * log_w) @ v.T
    return (res + res.T) * 0.5


def matrix_exp_sym(A: np.ndarray, clamp: float = 50.0) -> np.ndarray:
    """Matrix exponential for symmetric matrix A."""
    A_sym = (A + A.T) * 0.5
    w, v = np.linalg.eigh(A_sym)
    w_clamped = np.clip(w, -clamp, clamp)
    exp_w = np.exp(w_clamped)
    res = (v * exp_w) @ v.T
    return (res + res.T) * 0.5


def riemannian_distance_airm(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute Affine-Invariant Riemannian Metric (AIRM) distance between P and Q.

    delta_R(P, Q) = || log( P^{-1/2} Q P^{-1/2} ) ||_F
                  = sqrt( sum_k ln^2( lambda_k ) )
    """
    # Symmetrize and regularize inputs
    P_reg = regularize_spd(P)
    Q_reg = regularize_spd(Q)

    P_inv_sqrt = matrix_power_sym(P_reg, -0.5)
    M = P_inv_sqrt @ Q_reg @ P_inv_sqrt
    M_sym = (M + M.T) * 0.5
    eigvals = np.linalg.eigvalsh(M_sym)
    eigvals = np.maximum(eigvals, 1e-12)
    log_eigs = np.log(eigvals)
    dist = float(np.sqrt(np.maximum(0.0, np.sum(log_eigs**2))))
    return dist


def riemannian_distance_log_euclidean(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute Log-Euclidean distance between P and Q.

    delta_{LE}(P, Q) = || log(P) - log(Q) ||_F
    """
    P_reg = regularize_spd(P)
    Q_reg = regularize_spd(Q)
    log_P = matrix_log_sym(P_reg)
    log_Q = matrix_log_sym(Q_reg)
    diff = log_P - log_Q
    return float(np.linalg.norm(diff, ord="fro"))


def geodesic_interpolate(P: np.ndarray, Q: np.ndarray, t: float) -> np.ndarray:
    """Interpolate along the Riemannian geodesic between P and Q at parameter t in [0, 1].

    gamma(t; P, Q) = P^{1/2} ( P^{-1/2} Q P^{-1/2} )^t P^{1/2}
    """
    if t <= 0.0:
        return regularize_spd(P)
    if t >= 1.0:
        return regularize_spd(Q)

    P_reg = regularize_spd(P)
    Q_reg = regularize_spd(Q)
    P_sqrt = matrix_power_sym(P_reg, 0.5)
    P_inv_sqrt = matrix_power_sym(P_reg, -0.5)

    M = P_inv_sqrt @ Q_reg @ P_inv_sqrt
    M_t = matrix_power_sym(M, t)
    interp = P_sqrt @ M_t @ P_sqrt
    return regularize_spd(interp)


def frechet_mean_riemannian(
    covariances: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
    max_iter: int = 60,
    tol: float = 1e-7,
) -> np.ndarray:
    """Compute the Fréchet / Karcher Riemannian Geometric Mean of SPD matrices.

    argmin_{M in S_{++}^C} sum_i w_i * delta_R^2(M, C_i)
    """
    n_covs = len(covariances)
    if n_covs == 0:
        raise ValueError("Cannot compute Fréchet mean of empty sequence.")
    if n_covs == 1:
        return regularize_spd(covariances[0])

    # Normalize weights
    if weights is None:
        w = np.ones(n_covs, dtype=np.float64) / n_covs
    else:
        w = np.array(weights, dtype=np.float64)
        if np.any(w < 0):
            raise ValueError("Weights must be non-negative.")
        total_w = np.sum(w)
        if total_w <= 0:
            raise ValueError("Sum of weights must be positive.")
        w = w / total_w

    # Initialize with Euclidean mean
    M = np.zeros_like(covariances[0], dtype=np.float64)
    for idx, cov in enumerate(covariances):
        M += w[idx] * regularize_spd(cov)
    M = regularize_spd(M)

    # Riemannian gradient descent
    step_size = 1.0
    for iteration in range(max_iter):
        M_sqrt = matrix_power_sym(M, 0.5)
        M_inv_sqrt = matrix_power_sym(M, -0.5)

        tangent_mean = np.zeros_like(M)
        for idx, cov in enumerate(covariances):
            cov_reg = regularize_spd(cov)
            inner = M_inv_sqrt @ cov_reg @ M_inv_sqrt
            tangent_mean += w[idx] * matrix_log_sym(inner)

        grad_norm = float(np.linalg.norm(tangent_mean, ord="fro"))
        if grad_norm < tol:
            break

        # Geodesic update
        update = matrix_exp_sym(step_size * tangent_mean)
        M_next = regularize_spd(M_sqrt @ update @ M_sqrt)

        if not is_spd(M_next):
            step_size *= 0.5
            continue

        M = M_next

    return regularize_spd(M)


def tangent_space_project(C: np.ndarray, C_ref: np.ndarray) -> np.ndarray:
    """Project SPD matrix C onto the tangent space at reference point C_ref.

    S = C_ref^{1/2} * log( C_ref^{-1/2} C C_ref^{-1/2} ) * C_ref^{1/2}
    """
    C_reg = regularize_spd(C)
    ref_reg = regularize_spd(C_ref)
    ref_sqrt = matrix_power_sym(ref_reg, 0.5)
    ref_inv_sqrt = matrix_power_sym(ref_reg, -0.5)

    inner = ref_inv_sqrt @ C_reg @ ref_inv_sqrt
    log_inner = matrix_log_sym(inner)
    tangent_mat = ref_sqrt @ log_inner @ ref_sqrt
    return (tangent_mat + tangent_mat.T) * 0.5


def tangent_space_exp(S: np.ndarray, C_ref: np.ndarray) -> np.ndarray:
    """Map tangent space matrix S back to SPD manifold at reference point C_ref.

    C = C_ref^{1/2} * exp( C_ref^{-1/2} S C_ref^{-1/2} ) * C_ref^{1/2}
    """
    ref_reg = regularize_spd(C_ref)
    ref_sqrt = matrix_power_sym(ref_reg, 0.5)
    ref_inv_sqrt = matrix_power_sym(ref_reg, -0.5)

    inner = ref_inv_sqrt @ S @ ref_inv_sqrt
    exp_inner = matrix_exp_sym(inner)
    C = ref_sqrt @ exp_inner @ ref_sqrt
    return regularize_spd(C)


def tangent_space_vectorize(S: np.ndarray) -> np.ndarray:
    """Isometrically vectorize symmetric tangent matrix S into Euclidean vector.

    Weights off-diagonal elements by sqrt(2) so that ||vec(S)||_2 == ||S||_F.
    Dimension: C * (C + 1) / 2.
    """
    dim = S.shape[0]
    triu_indices = np.triu_indices(dim)
    vec = []
    for r, c in zip(triu_indices[0], triu_indices[1]):
        if r == c:
            vec.append(S[r, c])
        else:
            vec.append(np.sqrt(2.0) * S[r, c])
    return np.array(vec, dtype=np.float64)


def tangent_space_unvectorize(vec: np.ndarray, dim: int) -> np.ndarray:
    """Invert isometric vectorization back to C x C symmetric tangent matrix."""
    expected_len = dim * (dim + 1) // 2
    if len(vec) != expected_len:
        raise ValueError(f"Vector length {len(vec)} does not match expected {expected_len} for dim {dim}.")

    S = np.zeros((dim, dim), dtype=np.float64)
    triu_indices = np.triu_indices(dim)
    for idx, (r, c) in enumerate(zip(triu_indices[0], triu_indices[1])):
        val = vec[idx]
        if r == c:
            S[r, c] = val
        else:
            val_unweighted = val / np.sqrt(2.0)
            S[r, c] = val_unweighted
            S[c, r] = val_unweighted
    return S
