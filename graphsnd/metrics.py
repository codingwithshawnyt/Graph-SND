"""Core metric implementations for Graph-SND.

Functions
---------
- ``pairwise_behavioral_distance``: Monte Carlo estimator of the paper's
  eq. (1), producing the symmetric ``n x n`` matrix ``D`` with
  ``D[i, j] = d(i, j)``.
- ``snd``: full System Neural Diversity, eq. (2).
- ``graph_snd``: weighted Graph-SND over an edge list (Definition 1).
- ``ht_estimator``: Horvitz-Thompson unbiased estimator of ``SND`` over a
  random Bernoulli graph (Proposition 5).
- ``uniform_sample_estimator``: uniform-weight sample mean used in the
  concentration theorem (Remark 2 / Theorem 6).

Shape conventions
-----------------
Let ``n`` be the agent count, ``T`` the number of observation time steps
(after flattening over vectorised envs), and ``d_act`` the action
dimension of a single agent.

- ``means[i]``  -> Tensor[T, d_act] (mean action vector of policy i at
  each observation that appears in the rollout set ``B``).
- ``stds[i]``   -> Tensor[T, d_act] (per-dimension standard deviation).
- ``D``         -> Tensor[n, n], symmetric, zero diagonal.

The paper's eq. (1) is
    d(i, j) = (1 / (|B| * |A|))
               * sum_{o^t in B} sum_{k in A} W_2(pi_i(o^t_k), pi_j(o^t_k)).

Our ``pairwise_behavioral_distance`` interprets the rollout set ``B``
already as the concatenation of every ``o^t_k`` across time and agents.
That is: ``means[i]`` is of shape ``(T_total, d_act)`` where
``T_total = T * n``, and each row corresponds to one evaluation of
policy i on one observation. This removes the inner sum over ``A`` and
is a strictly equivalent formulation once the observations are
flattened appropriately upstream (see ``graphsnd/rollouts.py``).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from graphsnd.graphs import (
    RngLike,
    bernoulli_edges,
    complete_edges,
    uniform_size_edges,
)
from graphsnd.wasserstein import wasserstein_gaussian_diag


def pairwise_distances_on_edges(
    means: Union[Tensor, Sequence[Tensor]],
    stds: Union[Tensor, Sequence[Tensor]],
    edges: Tensor,
) -> Tensor:
    """Compute ``d(i, j)`` for only the pairs in ``edges``.

    This is the function that makes Graph-SND genuinely cheaper than
    full SND: when ``|E| << n*(n-1)/2`` we never materialise the full
    ``D`` matrix, and we never pay for the Wasserstein evaluations on
    pairs outside ``E``.

    Parameters
    ----------
    means, stds: same conventions as :func:`pairwise_behavioral_distance`.
    edges: LongTensor of shape ``(|E|, 2)``.

    Returns
    -------
    Tensor of shape ``(|E|,)`` with ``result[k] = d(edges[k, 0], edges[k, 1])``.
    """
    mu = _stack_if_list(means)
    sigma = _stack_if_list(stds)
    if mu.shape != sigma.shape:
        raise ValueError(
            f"means and stds must share shape; got {tuple(mu.shape)} vs "
            f"{tuple(sigma.shape)}"
        )
    if mu.ndim != 3:
        raise ValueError(
            f"means must have shape (n, T, d_act); got {tuple(mu.shape)}"
        )
    if edges.ndim != 2 or edges.shape[-1] != 2:
        raise ValueError(f"edges must have shape (|E|, 2); got {tuple(edges.shape)}")

    out = torch.zeros(edges.shape[0], dtype=mu.dtype, device=mu.device)
    for k in range(edges.shape[0]):
        i = int(edges[k, 0])
        j = int(edges[k, 1])
        per_obs_w2 = wasserstein_gaussian_diag(
            mu[i], sigma[i], mu[j], sigma[j]
        )
        out[k] = per_obs_w2.mean()
    return out


def graph_snd_from_rollouts(
    means: Union[Tensor, Sequence[Tensor]],
    stds: Union[Tensor, Sequence[Tensor]],
    edges: Tensor,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Graph-SND evaluated directly from per-policy ``(means, stds)``.

    Does not materialise ``D``; calls :func:`pairwise_distances_on_edges`
    on only the edges in ``E`` and then applies the weighted average.
    Returns ``0`` when ``E`` is empty.
    """
    if edges.numel() == 0:
        mu = _stack_if_list(means)
        return torch.zeros((), dtype=mu.dtype, device=mu.device)
    d_vals = pairwise_distances_on_edges(means, stds, edges)
    if weights is None:
        return d_vals.mean()
    if weights.shape != (edges.shape[0],):
        raise ValueError(
            f"weights shape {tuple(weights.shape)} does not match |E|={edges.shape[0]}"
        )
    if (weights < 0).any():
        raise ValueError("weights must be non-negative")
    w = weights.to(d_vals.dtype)
    denom = w.sum()
    if denom.item() == 0.0:
        return torch.zeros_like(d_vals[:1]).squeeze(0)
    return (w * d_vals).sum() / denom


def pairwise_behavioral_distance(
    means: Union[Tensor, Sequence[Tensor]],
    stds: Union[Tensor, Sequence[Tensor]],
) -> Tensor:
    """Compute the ``n x n`` pairwise behavioral-distance matrix.

    Parameters
    ----------
    means, stds: either
        - a Tensor of shape ``(n, T, d_act)``, or
        - a sequence of length ``n`` whose entries each have shape
          ``(T, d_act)``.

        ``means[i]`` holds policy ``i``'s mean action at each of the
        rollout observations. ``stds[i]`` holds the matching per-dim
        standard deviations (not variances). All entries must share
        ``T`` and ``d_act``.

    Returns
    -------
    Tensor of shape ``(n, n)`` with ``D[i, j] = D[j, i]`` and
    ``D[i, i] = 0``. Corresponds to the Monte Carlo estimate of
    ``d(i, j)`` in the paper's eq. (1).
    """
    mu = _stack_if_list(means)
    sigma = _stack_if_list(stds)
    if mu.shape != sigma.shape:
        raise ValueError(
            f"means and stds must share shape; got {tuple(mu.shape)} vs "
            f"{tuple(sigma.shape)}"
        )
    if mu.ndim != 3:
        raise ValueError(
            f"means must have shape (n, T, d_act); got {tuple(mu.shape)}"
        )

    n, t, _ = mu.shape
    if t == 0:
        raise ValueError("rollout set B is empty (T == 0)")

    D = torch.zeros(n, n, dtype=mu.dtype, device=mu.device)
    for i in range(n):
        for j in range(i + 1, n):
            per_obs_w2 = wasserstein_gaussian_diag(
                mu[i], sigma[i], mu[j], sigma[j]
            )
            d_ij = per_obs_w2.mean()
            D[i, j] = d_ij
            D[j, i] = d_ij
    return D


def snd(D: Tensor) -> Tensor:
    """Full System Neural Diversity (eq. 2).

    ``SND(D) = (n choose 2)^{-1} * sum_{i<j} D[i, j]``.
    """
    _validate_distance_matrix(D)
    n = D.shape[0]
    if n < 2:
        return torch.zeros((), dtype=D.dtype, device=D.device)
    i, j = torch.triu_indices(n, n, offset=1, device=D.device)
    return D[i, j].mean()


def graph_snd(
    D: Tensor,
    edges: Tensor,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Graph-SND on an arbitrary weighted graph (Definition 1, eq. 4).

    ``SND_G(D, G) = (1 / W(G)) * sum_{{i,j} in E} w_ij * D[i, j]``.

    Parameters
    ----------
    D: symmetric ``n x n`` distance matrix.
    edges: LongTensor of shape ``(|E|, 2)`` in ``i < j`` order.
    weights: 1D Tensor of shape ``(|E|,)`` with non-negative entries,
        or ``None`` for unit weights.

    Returns
    -------
    0-dim Tensor. Returns ``0`` by convention when ``W(G) == 0``, per
    the paper's degenerate-graph convention.
    """
    _validate_distance_matrix(D)
    if edges.ndim != 2 or edges.shape[-1] != 2:
        raise ValueError(f"edges must have shape (|E|, 2); got {tuple(edges.shape)}")

    n = D.shape[0]
    if edges.numel() == 0:
        return torch.zeros((), dtype=D.dtype, device=D.device)

    if weights is None:
        w = torch.ones(edges.shape[0], dtype=D.dtype, device=D.device)
    else:
        if weights.shape != (edges.shape[0],):
            raise ValueError(
                f"weights shape {tuple(weights.shape)} does not match |E|={edges.shape[0]}"
            )
        if (weights < 0).any():
            raise ValueError("weights must be non-negative")
        w = weights.to(D.dtype)

    i_idx = edges[:, 0].to(torch.long)
    j_idx = edges[:, 1].to(torch.long)
    if (i_idx >= n).any() or (j_idx >= n).any() or (i_idx < 0).any() or (j_idx < 0).any():
        raise ValueError("edge indices are out of range for D")

    numerator = (w * D[i_idx, j_idx]).sum()
    denominator = w.sum()
    if denominator.item() == 0.0:
        return torch.zeros((), dtype=D.dtype, device=D.device)
    return numerator / denominator


def ht_estimator(D: Tensor, p: float, rng: RngLike = None) -> Tensor:
    """Horvitz-Thompson unbiased estimator of ``SND`` (Proposition 5).

    Each pair ``(i, j)`` is included in the random graph ``G_p``
    independently with probability ``p``, with edge weight ``1/p``. The
    estimator

        hatSND(G_p) = |P|^{-1} * sum_{{i,j} in E(G_p)} (1/p) * d(i, j)

    satisfies ``E[hatSND(G_p)] = SND(D)``.

    Returns ``0`` when the sampled edge set is empty (this event is
    unbiased at the population level and does not invalidate
    ``E[hatSND] = SND``).
    """
    _validate_distance_matrix(D)
    if not 0.0 < p <= 1.0:
        raise ValueError("p must lie in (0, 1]")
    n = D.shape[0]
    n_pairs = n * (n - 1) // 2
    if n_pairs == 0:
        return torch.zeros((), dtype=D.dtype, device=D.device)
    edges = bernoulli_edges(n, p, rng)
    if edges.shape[0] == 0:
        return torch.zeros((), dtype=D.dtype, device=D.device)
    i_idx = edges[:, 0].to(torch.long)
    j_idx = edges[:, 1].to(torch.long)
    return D[i_idx, j_idx].sum() / (p * n_pairs)


def uniform_sample_estimator(
    D: Tensor, m: int, rng: RngLike = None
) -> Tensor:
    """Sample-mean estimator used in the concentration theorem.

    Let ``E`` be a uniform sample (without replacement) of ``m`` edges
    from the complete pair set ``P``. Return the sample mean of
    ``D[i, j]`` over those edges. Per Theorem 6, for ``d(i, j) in
    [0, D_max]``,

        P(|estimator - SND| >= t | |E| = m)
            <= 2 * exp(-2 * m * t^2 / D_max^2).
    """
    _validate_distance_matrix(D)
    n = D.shape[0]
    n_pairs = n * (n - 1) // 2
    if m < 1:
        raise ValueError("m must be >= 1")
    if m > n_pairs:
        raise ValueError(f"m={m} exceeds total pairs {n_pairs} for n={n}")
    edges = uniform_size_edges(n, m, rng)
    i_idx = edges[:, 0].to(torch.long)
    j_idx = edges[:, 1].to(torch.long)
    return D[i_idx, j_idx].mean()


def hoeffding_bound(d_max: float, m: int, delta: float) -> float:
    """Hoeffding-type concentration radius at confidence ``1 - delta``.

    Returns ``t`` such that, under the assumptions of Theorem 6,
    ``P(|estimator - SND| > t | |E| = m) <= delta``.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if m < 1:
        raise ValueError("m must be >= 1")
    return d_max * math.sqrt(math.log(2.0 / delta) / (2.0 * m))


def serfling_bound(d_max: float, m: int, n_pairs: int, delta: float) -> float:
    """Serfling-type finite-population concentration radius (Remark 3).

    Adds the sampling-without-replacement correction factor
    ``(1 - (m - 1) / |P|)`` to the Hoeffding exponent.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if m < 1 or m > n_pairs:
        raise ValueError(f"m must lie in [1, n_pairs]={n_pairs}; got m={m}")
    correction = 1.0 - (m - 1) / n_pairs
    correction = max(correction, 0.0)
    return d_max * math.sqrt(correction * math.log(2.0 / delta) / (2.0 * m))


def _stack_if_list(x: Union[Tensor, Sequence[Tensor]]) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return torch.stack(list(x), dim=0)


def _validate_distance_matrix(D: Tensor) -> None:
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be a square matrix; got shape {tuple(D.shape)}")
    if (D < 0).any():
        raise ValueError("D must have non-negative entries")
