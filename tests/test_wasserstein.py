"""Unit tests for the Wasserstein building blocks.

We test three things:

1. Identity of indiscernibles: ``W_2(N, N) == 0``.
2. Symmetry: ``W_2(P, Q) == W_2(Q, P)``.
3. Agreement between the diagonal closed form and the general
   symmetric-PSD form on diagonal covariance matrices.
4. A hand-computed 1D value (the diagonal formula collapses to
   ``sqrt((mu1-mu2)^2 + (sigma1-sigma2)^2)`` so the result for known
   inputs is easy to check).
"""

from __future__ import annotations

import torch

from graphsnd.wasserstein import (
    wasserstein_gaussian,
    wasserstein_gaussian_diag,
)


def test_identity_of_indiscernibles_diag() -> None:
    mu = torch.tensor([[0.5, -1.0, 2.0]])
    sigma = torch.tensor([[0.2, 0.2, 0.2]])
    w = wasserstein_gaussian_diag(mu, sigma, mu, sigma)
    assert torch.allclose(w, torch.zeros_like(w), atol=1e-6)


def test_symmetry_diag() -> None:
    torch.manual_seed(0)
    mu1 = torch.randn(5, 4)
    mu2 = torch.randn(5, 4)
    s1 = torch.rand(5, 4).abs() + 0.05
    s2 = torch.rand(5, 4).abs() + 0.05
    w_ij = wasserstein_gaussian_diag(mu1, s1, mu2, s2)
    w_ji = wasserstein_gaussian_diag(mu2, s2, mu1, s1)
    assert torch.allclose(w_ij, w_ji, atol=1e-6)


def test_hand_computation_1d() -> None:
    mu1 = torch.tensor([0.0])
    mu2 = torch.tensor([3.0])
    s1 = torch.tensor([1.0])
    s2 = torch.tensor([2.0])
    # expected = sqrt((3-0)^2 + (2-1)^2) = sqrt(10)
    w = wasserstein_gaussian_diag(mu1, s1, mu2, s2)
    assert torch.allclose(w, torch.tensor(10.0).sqrt(), atol=1e-5)


def test_hand_computation_3d() -> None:
    mu1 = torch.tensor([[0.0, 0.0, 0.0]])
    mu2 = torch.tensor([[1.0, -2.0, 2.0]])
    s1 = torch.tensor([[1.0, 1.0, 1.0]])
    s2 = torch.tensor([[2.0, 1.0, 3.0]])
    # (1 + 4 + 4) + (1 + 0 + 4) = 9 + 5 = 14
    w = wasserstein_gaussian_diag(mu1, s1, mu2, s2)
    assert torch.allclose(w, torch.tensor(14.0).sqrt(), atol=1e-5)


def test_general_matches_diag_on_diagonal_covariance() -> None:
    torch.manual_seed(42)
    d = 4
    mu1 = torch.randn(d, dtype=torch.float64)
    mu2 = torch.randn(d, dtype=torch.float64)
    sigma1 = torch.rand(d, dtype=torch.float64).abs() + 0.1
    sigma2 = torch.rand(d, dtype=torch.float64).abs() + 0.1
    cov1 = torch.diag(sigma1 ** 2)
    cov2 = torch.diag(sigma2 ** 2)

    w_diag = wasserstein_gaussian_diag(mu1, sigma1, mu2, sigma2)
    w_full = wasserstein_gaussian(mu1, cov1, mu2, cov2)
    assert torch.allclose(w_diag, w_full, atol=1e-6)


def test_general_known_case_equal_isotropic() -> None:
    d = 3
    mu1 = torch.zeros(d, dtype=torch.float64)
    mu2 = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    cov = torch.eye(d, dtype=torch.float64)
    # Same cov: W_2^2 = ||mu1 - mu2||^2 = 3
    w = wasserstein_gaussian(mu1, cov, mu2, cov)
    assert torch.allclose(w, torch.tensor(3.0, dtype=torch.float64).sqrt(), atol=1e-6)


def test_general_commuting_general() -> None:
    torch.manual_seed(1)
    d = 3
    mu1 = torch.randn(d, dtype=torch.float64)
    mu2 = torch.randn(d, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
    lam1 = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    lam2 = torch.tensor([3.0, 0.5, 1.5], dtype=torch.float64)
    c1 = q @ torch.diag(lam1) @ q.T
    c2 = q @ torch.diag(lam2) @ q.T
    # When cov matrices commute in the same eigenbasis, closed form is
    # ||mu1-mu2||^2 + sum_k (sqrt(lam1_k) - sqrt(lam2_k))^2.
    mean_sq = ((mu1 - mu2) ** 2).sum()
    cov_sq = ((lam1.sqrt() - lam2.sqrt()) ** 2).sum()
    expected = torch.sqrt(mean_sq + cov_sq)
    w = wasserstein_gaussian(mu1, c1, mu2, c2)
    assert torch.allclose(w, expected, atol=1e-5)
