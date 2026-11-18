"""Unit and Invariance Tests for Riemannian Manifold Differential Geometry."""

import numpy as np
import pytest

from cerebro_seizure.manifold import (
    is_spd,
    regularize_spd,
    matrix_power_sym,
    matrix_log_sym,
    matrix_exp_sym,
    riemannian_distance_airm,
    riemannian_distance_log_euclidean,
    geodesic_interpolate,
    frechet_mean_riemannian,
    tangent_space_project,
    tangent_space_exp,
    tangent_space_vectorize,
    tangent_space_unvectorize,
)


@pytest.fixture
def random_spd_matrices():
    rng = np.random.default_rng(1234)
    dim = 4
    mats = []
    for _ in range(5):
        A = rng.normal(0, 1, size=(dim, dim))
        # S = A A^T + eps * I
        S = A @ A.T + 0.2 * np.eye(dim)
        mats.append(S)
    return mats


def test_is_spd_and_regularize(random_spd_matrices):
    for mat in random_spd_matrices:
        assert is_spd(mat)

    # Test non-symmetric matrix regularized to SPD
    non_sym = np.array([[2.0, 1.0], [0.5, 3.0]])
    reg = regularize_spd(non_sym)
    assert is_spd(reg)
    assert np.allclose(reg, reg.T)


def test_matrix_power_inverse(random_spd_matrices):
    P = random_spd_matrices[0]
    P_half = matrix_power_sym(P, 0.5)
    P_recon = P_half @ P_half
    assert np.allclose(P, P_recon, atol=1e-6)

    P_inv = matrix_power_sym(P, -1.0)
    ident = P @ P_inv
    assert np.allclose(ident, np.eye(P.shape[0]), atol=1e-6)


def test_matrix_log_exp_inverse(random_spd_matrices):
    P = random_spd_matrices[1]
    log_P = matrix_log_sym(P)
    exp_log_P = matrix_exp_sym(log_P)
    assert np.allclose(P, exp_log_P, atol=1e-6)


def test_airm_distance_properties(random_spd_matrices):
    P, Q, R = random_spd_matrices[:3]

    # Identity: dist(P, P) == 0
    assert riemannian_distance_airm(P, P) == pytest.approx(0.0, abs=1e-7)

    # Positivity: dist(P, Q) > 0
    assert riemannian_distance_airm(P, Q) > 0.0

    # Symmetry: dist(P, Q) == dist(Q, P)
    d_pq = riemannian_distance_airm(P, Q)
    d_qp = riemannian_distance_airm(Q, P)
    assert d_pq == pytest.approx(d_qp, rel=1e-6)

    # Inversion invariance: dist(P^-1, Q^-1) == dist(P, Q)
    P_inv = matrix_power_sym(P, -1.0)
    Q_inv = matrix_power_sym(Q, -1.0)
    d_inv = riemannian_distance_airm(P_inv, Q_inv)
    assert d_inv == pytest.approx(d_pq, rel=1e-5)

    # Affine Invariance: dist(A P A^T, A Q A^T) == dist(P, Q)
    rng = np.random.default_rng(99)
    A = rng.normal(0, 1, size=P.shape)
    # Ensure invertible A
    while abs(np.linalg.det(A)) < 0.1:
        A = rng.normal(0, 1, size=P.shape)

    APA = A @ P @ A.T
    AQA = A @ Q @ A.T
    d_affine = riemannian_distance_airm(APA, AQA)
    assert d_affine == pytest.approx(d_pq, rel=1e-4)

    # Triangle inequality: dist(P, R) <= dist(P, Q) + dist(Q, R)
    d_pr = riemannian_distance_airm(P, R)
    d_qr = riemannian_distance_airm(Q, R)
    assert d_pr <= (d_pq + d_qr + 1e-6)


def test_geodesic_interpolation(random_spd_matrices):
    P, Q = random_spd_matrices[0], random_spd_matrices[1]
    total_dist = riemannian_distance_airm(P, Q)

    # t = 0 -> P
    g0 = geodesic_interpolate(P, Q, 0.0)
    assert np.allclose(g0, P, atol=1e-6)

    # t = 1 -> Q
    g1 = geodesic_interpolate(P, Q, 1.0)
    assert np.allclose(g1, Q, atol=1e-6)

    # Midpoint t = 0.5
    g_mid = geodesic_interpolate(P, Q, 0.5)
    d_p_mid = riemannian_distance_airm(P, g_mid)
    d_mid_q = riemannian_distance_airm(g_mid, Q)
    assert d_p_mid == pytest.approx(total_dist * 0.5, rel=1e-4)
    assert d_mid_q == pytest.approx(total_dist * 0.5, rel=1e-4)


def test_frechet_mean(random_spd_matrices):
    mean_cov = frechet_mean_riemannian(random_spd_matrices)
    assert is_spd(mean_cov)

    # Single element
    single = frechet_mean_riemannian([random_spd_matrices[0]])
    assert np.allclose(single, random_spd_matrices[0], atol=1e-6)

    # Identical elements
    identical = [random_spd_matrices[0].copy() for _ in range(4)]
    mean_id = frechet_mean_riemannian(identical)
    assert np.allclose(mean_id, random_spd_matrices[0], atol=1e-6)


def test_tangent_space_isometry(random_spd_matrices):
    C_ref = random_spd_matrices[0]
    C_test = random_spd_matrices[1]

    S = tangent_space_project(C_test, C_ref)
    v = tangent_space_vectorize(S)

    # Check Frobenius norm preservation: ||v||_2 == ||S||_F
    norm_v = np.linalg.norm(v)
    norm_S = np.linalg.norm(S, ord="fro")
    assert norm_v == pytest.approx(norm_S, rel=1e-6)

    # Check inverse vectorization
    S_recon = tangent_space_unvectorize(v, dim=C_ref.shape[0])
    assert np.allclose(S, S_recon, atol=1e-6)

    # Check tangent exponential back-projection
    C_recon = tangent_space_exp(S, C_ref)
    assert np.allclose(C_test, C_recon, atol=1e-6)
