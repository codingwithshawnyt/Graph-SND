"""Graph construction utilities for Graph-SND.

A graph over ``n`` agents is represented as a LongTensor of edges with
shape ``(|E|, 2)`` in row-wise ``(i, j)`` form with ``i < j``. Weights,
when relevant, live in a parallel tensor of shape ``(|E|,)``.

We never store adjacency matrices: Graph-SND only reads ``D[i, j]`` for
the pairs in the edge list, so the compact edge-list form is enough and
scales linearly in ``|E|`` rather than quadratically in ``n``.

Graph families:

- ``complete_edges(n)``: all :math:`\\binom{n}{2}` pairs. Recovers full
  SND on ``K_n`` (Proposition 1).
- ``bernoulli_edges(n, p, rng)``: each pair is included independently
  with probability ``p``. Used with Horvitz-Thompson weights for the
  unbiased estimator (Proposition 5).
- ``uniform_size_edges(n, m, rng)``: exactly ``m`` edges drawn uniformly
  without replacement. The sampling-without-replacement setup for the
  concentration bound (Theorem 6 / Lemma 7).
- ``random_regular_edges(n, d, rng)``: a random ``d``-regular graph.
  Near-Ramanujan with high probability (Friedman 2003); used in the
  expander sparsification ablation (Corollary 11).
- ``knn_edges(features, k)``: k-nearest-neighbour graph over agent
  feature vectors.

Spectral utilities:

- ``spectral_gap(n, edges)``: compute the second-largest eigenvalue
  and spectral gap of the adjacency matrix induced by an edge list.
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


def random_regular_edges(n: int, d: int, rng: RngLike = None) -> Tensor:
    """Random ``d``-regular graph edges via the configuration model.

    Uses :func:`networkx.random_regular_graph` which samples uniformly
    over the set of all ``d``-regular graphs on ``n`` vertices.
    Friedman's theorem (2003) guarantees that the resulting graph is
    nearly Ramanujan with high probability:
    :math:`\\lambda_2 \\leq 2\\sqrt{d-1} + \\epsilon` for any
    :math:`\\epsilon > 0` as ``n \\to \\infty``.

    Parameters
    ----------
    n: number of agents (vertices).  Must satisfy ``n >= d + 1`` and
        ``n * d`` must be even (necessary condition for a ``d``-regular
        graph to exist).
    d: degree of each vertex.  Must satisfy ``d >= 1``.
    rng: ``numpy`` ``Generator``, ``int`` seed, or ``None``.

    Returns
    -------
    LongTensor of shape ``(n*d/2, 2)`` with ``i < j`` ordering.
    """
    import networkx as nx

    if n < 2:
        raise ValueError(f"n must be >= 2; got n={n}")
    if d < 1:
        raise ValueError(f"d must be >= 1; got d={d}")
    if d >= n:
        raise ValueError(f"d must be < n; got d={d}, n={n}")
    if (n * d) % 2 != 0:
        raise ValueError(
            f"n*d must be even for a d-regular graph to exist; "
            f"got n={n}, d={d}, n*d={n*d}"
        )
    gen = _as_generator(rng)
    seed = int(gen.integers(0, 2**31))
    g = nx.random_regular_graph(d, n, seed=seed)
    edge_list = sorted((min(u, v), max(u, v)) for u, v in g.edges())
    return torch.tensor(edge_list, dtype=torch.long)


def spectral_gap(n: int, edges: Tensor) -> tuple:
    """Compute spectral properties of the adjacency matrix of a graph.

    Builds the ``n x n`` symmetric adjacency matrix from the edge list
    and returns the second-largest eigenvalue and related quantities.

    Parameters
    ----------
    n: number of vertices.
    edges: LongTensor of shape ``(|E|, 2)`` with ``i < j`` ordering.

    Returns
    -------
    tuple of ``(lambda_2, gap, d_max, is_ramanujan)`` where

    - ``lambda_2``: second-largest eigenvalue of the adjacency matrix.
    - ``gap``: spectral gap ``1 - lambda_2 / d_max`` (``1.0`` if
      ``d_max == 0``).
    - ``d_max``: maximum degree of the graph.
    - ``is_ramanujan``: ``True`` if ``lambda_2 <= 2*sqrt(d_max - 1)``
      (the Ramanujan bound for ``d_max``-regular graphs).
    """
    if edges.numel() == 0:
        return (0.0, 1.0, 0, True)

    A = np.zeros((n, n), dtype=np.float64)
    for k in range(edges.shape[0]):
        i = int(edges[k, 0])
        j = int(edges[k, 1])
        A[i, j] = 1.0
        A[j, i] = 1.0
    eigenvalues = np.linalg.eigvalsh(A)
    eigenvalues.sort()
    d_max = int(A.sum(axis=0).max())
    if d_max == 0:
        return (0.0, 1.0, 0, True)
    lambda_2 = float(eigenvalues[-2])
    gap = 1.0 - lambda_2 / d_max
    ramanujan_bound = 2.0 * np.sqrt(max(d_max - 1, 0))
    is_ram = lambda_2 <= ramanujan_bound
    return (lambda_2, gap, d_max, bool(is_ram))
