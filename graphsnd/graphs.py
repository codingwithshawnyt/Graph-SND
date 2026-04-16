"""Graph construction utilities for Graph-SND.

A graph over ``n`` agents is represented as a LongTensor of edges with
shape ``(|E|, 2)`` in row-wise ``(i, j)`` form with ``i < j``. Weights,
when relevant, live in a parallel tensor of shape ``(|E|,)``.

We never store adjacency matrices: Graph-SND only reads ``D[i, j]`` for
the pairs in the edge list, so the compact edge-list form is enough and
scales linearly in ``|E|`` rather than quadratically in ``n``.

Three graph families cover the paper's two interpretations of
Graph-SND:

- ``complete_edges(n)``: all :math:`\\binom{n}{2}` pairs. Recovers full
  SND on ``K_n`` (Proposition 1).
- ``bernoulli_edges(n, p, rng)``: each pair is included independently
  with probability ``p``. Used with Horvitz-Thompson weights for the
  unbiased estimator (Proposition 5).
- ``uniform_size_edges(n, m, rng)``: exactly ``m`` edges drawn uniformly
  without replacement. The sampling-without-replacement setup for the
  concentration bound (Theorem 6 / Lemma 7).

A ``knn_edges`` placeholder is included so the signature is stable for
the k-NN localized-measure experiments in a later paper iteration.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor


RngLike = Union[np.random.Generator, int, None]


def _as_generator(rng: RngLike) -> np.random.Generator:
    """Coerce an int seed / ``None`` / ``Generator`` to a ``Generator``."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def complete_edges(n: int, device: Optional[torch.device] = None) -> Tensor:
    """Return all :math:`\\binom{n}{2}` unordered pairs ``(i, j)`` with ``i < j``.

    Parameters
    ----------
    n: number of agents.

    Returns
    -------
    LongTensor of shape ``(n*(n-1)/2, 2)``.
    """
    if n < 2:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    i, j = torch.triu_indices(n, n, offset=1, device=device)
    return torch.stack([i, j], dim=-1).to(torch.long)


def bernoulli_edges(n: int, p: float, rng: RngLike = None) -> Tensor:
    """Include each pair ``(i, j)`` independently with probability ``p``.

    Parameters
    ----------
    n: number of agents.
    p: per-pair inclusion probability in ``(0, 1]``.
    rng: ``numpy`` ``Generator``, ``int`` seed, or ``None``.

    Returns
    -------
    LongTensor of shape ``(|E|, 2)`` in ``i < j`` order. May be empty
    when ``p`` is small.
    """
    if not 0.0 < p <= 1.0:
        raise ValueError("p must lie in (0, 1]")
    if n < 2:
        return torch.empty((0, 2), dtype=torch.long)
    gen = _as_generator(rng)
    all_edges = complete_edges(n)
    mask = gen.random(all_edges.shape[0]) < p
    return all_edges[torch.from_numpy(mask)]


def uniform_size_edges(n: int, m: int, rng: RngLike = None) -> Tensor:
    """Sample ``m`` unordered pairs uniformly without replacement.

    This is the finite-population sampling setup of Lemma 7 in the
    paper: the resulting edge set is a uniform random subset of
    :math:`\\binom{\\mathcal{A}}{2}` of size exactly ``m``.

    Parameters
    ----------
    n: number of agents.
    m: number of edges to sample. Must satisfy ``0 <= m <= n*(n-1)/2``.
    rng: ``numpy`` ``Generator``, ``int`` seed, or ``None``.

    Returns
    -------
    LongTensor of shape ``(m, 2)`` with distinct edges in ``i < j`` order.
    """
    num_pairs = n * (n - 1) // 2
    if not 0 <= m <= num_pairs:
        raise ValueError(
            f"m must be in [0, {num_pairs}] for n={n}, got m={m}"
        )
    if m == 0:
        return torch.empty((0, 2), dtype=torch.long)
    gen = _as_generator(rng)
    all_edges = complete_edges(n)
    idx = gen.choice(num_pairs, size=m, replace=False)
    return all_edges[torch.from_numpy(np.asarray(idx))]


def knn_edges(features: Tensor, k: int, symmetric: bool = True) -> Tensor:
    """k-nearest-neighbour graph edges over agent feature vectors.

    For each agent ``i``, find its ``k`` nearest (by Euclidean distance)
    agents ``j != i`` and add the pair. By default the graph is
    undirected, so we return the union over both endpoints.

    Parameters
    ----------
    features: Tensor of shape ``(n, f)`` with one feature vector per
        agent. Typical choices are mean observation embeddings or mean
        policy means.
    k: number of neighbours per agent.
    symmetric: if ``True``, return the undirected union. If ``False``,
        return the directed edge list restricted to ``i < j``.

    Returns
    -------
    LongTensor of shape ``(|E|, 2)``.

    Notes
    -----
    Included for completeness; the experiments in the current paper
    iteration use only ``complete_edges``, ``bernoulli_edges``, and
    ``uniform_size_edges``.
    """
    if features.ndim != 2:
        raise ValueError("features must have shape (n, f)")
    n = features.shape[0]
    if k < 1 or k >= n:
        raise ValueError(f"k must lie in [1, n-1]; got k={k}, n={n}")

    dist = torch.cdist(features, features)
    dist.fill_diagonal_(float("inf"))
    _, nbrs = torch.topk(dist, k=k, largest=False, dim=-1)

    src = torch.arange(n).unsqueeze(-1).expand(-1, k).reshape(-1)
    dst = nbrs.reshape(-1)
    edges = torch.stack([src, dst], dim=-1)

    lo = torch.minimum(edges[:, 0], edges[:, 1])
    hi = torch.maximum(edges[:, 0], edges[:, 1])
    edges = torch.stack([lo, hi], dim=-1)
    edges = torch.unique(edges, dim=0)

    if not symmetric:
        return edges
    return edges
