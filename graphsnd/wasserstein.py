"""Closed-form 2-Wasserstein distance between multivariate Gaussians.

This module provides the building block of the Graph-SND behavioral
distance. The 2-Wasserstein distance between two multivariate Gaussians
N(mu_1, Sigma_1) and N(mu_2, Sigma_2) has the closed form (Dowson-Landau
1982; Olkin-Pukelsheim 1982):

    W_2^2 = ||mu_1 - mu_2||^2 + tr(Sigma_1 + Sigma_2 - 2 (Sigma_1^{1/2} Sigma_2 Sigma_1^{1/2})^{1/2}).

When both covariances are diagonal (the common case for Gaussian MLP
policies with a learned scalar or per-dim log-sigma), this simplifies to

    W_2^2 = ||mu_1 - mu_2||^2 + ||sigma_1 - sigma_2||^2,

which avoids any matrix square root. We expose both forms.

For commuting covariances (in particular, both diagonal), the Cholesky-
based expression

    W_2^2 = ||mu_1 - mu_2||^2 + ||L_1 - L_2||_F^2

also coincides with the trace formula; this matches the reference
implementation in ``proroklab/HetGPPO/evaluate/distance_metrics.py``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def wasserstein_gaussian_diag(
    mu1: Tensor,
    sigma1: Tensor,
    mu2: Tensor,
    sigma2: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """2-Wasserstein distance between two Gaussians with diagonal covariance.

    Works in arbitrary leading batch dimensions: all inputs must have the
    same shape ``(..., d)`` where ``d`` is the action dimension and
    ``sigma_*`` hold per-dimension standard deviations (not variances).

    W_2 = sqrt( ||mu1 - mu2||^2 + ||sigma1 - sigma2||^2 ).

    Parameters
    ----------
    mu1, mu2: mean vectors with shape ``(..., d)``.
    sigma1, sigma2: per-dim standard deviations, same shape as the means.
        Values must be non-negative. A small ``eps`` is added before the
        final ``sqrt`` to avoid NaN gradients at exactly zero distance.
    eps: numerical floor before the outer ``sqrt``.

    Returns
    -------
    Tensor of shape ``(...)`` (one scalar per batch element).
    """
    if mu1.shape != mu2.shape or mu1.shape != sigma1.shape or mu1.shape != sigma2.shape:
        raise ValueError(
            "shape mismatch: mu1 %s, sigma1 %s, mu2 %s, sigma2 %s"
            % (tuple(mu1.shape), tuple(sigma1.shape), tuple(mu2.shape), tuple(sigma2.shape))
        )
    if (sigma1 < 0).any() or (sigma2 < 0).any():
        raise ValueError("sigma inputs must be non-negative standard deviations")

    mean_sq = ((mu1 - mu2) ** 2).sum(dim=-1)
    cov_sq = ((sigma1 - sigma2) ** 2).sum(dim=-1)
    return torch.sqrt(mean_sq + cov_sq + eps) - torch.sqrt(
        torch.tensor(eps, dtype=mu1.dtype, device=mu1.device)
    )


def wasserstein_gaussian(
    mu1: Tensor,
    cov1: Tensor,
    mu2: Tensor,
    cov2: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """2-Wasserstein distance between multivariate Gaussians, general covariance.

    Uses the Bures metric form with a matrix square root computed via
    symmetric eigendecomposition:

        W_2^2 = ||mu1 - mu2||^2 + tr(S1 + S2 - 2 * (S1^{1/2} S2 S1^{1/2})^{1/2}).

    The closed form is evaluated in float64 for numerical stability, then
    cast back to the input dtype. Both covariances must be symmetric and
    positive semi-definite.

    Parameters
    ----------
    mu1, mu2: mean vectors with shape ``(..., d)``.
    cov1, cov2: covariance matrices with shape ``(..., d, d)``.

    Returns
    -------
    Tensor of shape ``(...)``.
    """
    if cov1.shape[-2:] != cov2.shape[-2:]:
        raise ValueError("covariance shape mismatch")
    if cov1.shape[-1] != mu1.shape[-1]:
        raise ValueError("covariance / mean dim mismatch")

    dtype_in = mu1.dtype
    mu1_d = mu1.to(torch.float64)
    mu2_d = mu2.to(torch.float64)
    c1 = cov1.to(torch.float64)
    c2 = cov2.to(torch.float64)

    mean_sq = ((mu1_d - mu2_d) ** 2).sum(dim=-1)
    c1_sqrt = _psd_sqrtm(c1)
    middle = c1_sqrt @ c2 @ c1_sqrt
    middle_sqrt = _psd_sqrtm(middle)
    trace_term = (
        _trace(c1) + _trace(c2) - 2.0 * _trace(middle_sqrt)
    ).clamp(min=0.0)

    w2_sq = mean_sq + trace_term
    return (torch.sqrt(w2_sq + eps) - torch.sqrt(torch.tensor(eps, dtype=torch.float64))).to(dtype_in)


def _psd_sqrtm(mat: Tensor) -> Tensor:
    """Symmetric-PSD matrix square root via eigendecomposition.

    Eigenvalues below ``0`` (due to numerical error on a PSD matrix) are
    clamped to zero before the square root. Works with arbitrary leading
    batch dimensions.
    """
    sym = 0.5 * (mat + mat.transpose(-1, -2))
    vals, vecs = torch.linalg.eigh(sym)
    vals = vals.clamp(min=0.0)
    sqrt_vals = vals.sqrt()
    return (vecs * sqrt_vals.unsqueeze(-2)) @ vecs.transpose(-1, -2)


def _trace(mat: Tensor) -> Tensor:
    """Trace of a batch of square matrices."""
    return torch.diagonal(mat, dim1=-2, dim2=-1).sum(dim=-1)
