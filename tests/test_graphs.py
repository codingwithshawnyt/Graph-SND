"""Unit tests for edge-sampling primitives."""

from __future__ import annotations

import numpy as np
import torch

from graphsnd.graphs import (
    bernoulli_edges,
    complete_edges,
    knn_edges,
    uniform_size_edges,
)


def test_complete_edges_count_and_order() -> None:
    n = 6
    edges = complete_edges(n)
    assert edges.shape == (n * (n - 1) // 2, 2)
    assert (edges[:, 0] < edges[:, 1]).all()
    # All entries in range
    assert (edges >= 0).all() and (edges < n).all()
    # Uniqueness
    as_set = {(int(a), int(b)) for a, b in edges.tolist()}
    assert len(as_set) == edges.shape[0]


def test_complete_edges_small_cases() -> None:
    assert complete_edges(0).shape == (0, 2)
    assert complete_edges(1).shape == (0, 2)
    assert complete_edges(2).tolist() == [[0, 1]]


def test_bernoulli_edges_respects_ij_order() -> None:
    rng = np.random.default_rng(123)
    edges = bernoulli_edges(10, 0.5, rng)
    if edges.shape[0] > 0:
        assert (edges[:, 0] < edges[:, 1]).all()


def test_bernoulli_edges_expected_count() -> None:
    rng = np.random.default_rng(0)
    n, p = 30, 0.4
    n_pairs = n * (n - 1) // 2
    counts = []
    for _ in range(200):
        e = bernoulli_edges(n, p, rng)
        counts.append(e.shape[0])
    mean = float(np.mean(counts))
    expected = n_pairs * p
    assert abs(mean - expected) < 3.5 * np.sqrt(expected * (1 - p))


def test_bernoulli_edges_p_one_equals_complete() -> None:
    rng = np.random.default_rng(0)
    for n in (2, 5, 8):
        e = bernoulli_edges(n, 1.0, rng)
        assert e.shape[0] == n * (n - 1) // 2
        assert torch.equal(e, complete_edges(n))


def test_uniform_size_edges_exact_size_and_no_dupes() -> None:
    rng = np.random.default_rng(7)
    n = 12
    n_pairs = n * (n - 1) // 2
    for m in [1, 5, n_pairs // 2, n_pairs]:
        e = uniform_size_edges(n, m, rng)
        assert e.shape == (m, 2)
        assert (e[:, 0] < e[:, 1]).all()
        pairs = {(int(a), int(b)) for a, b in e.tolist()}
        assert len(pairs) == m


def test_uniform_size_edges_uniform_distribution_over_pairs() -> None:
    rng = np.random.default_rng(0)
    n = 6
    n_pairs = n * (n - 1) // 2
    m = 2
    trials = 20000
    counts = {}
    for _ in range(trials):
        e = uniform_size_edges(n, m, rng)
        for a, b in e.tolist():
            counts[(int(a), int(b))] = counts.get((int(a), int(b)), 0) + 1
    # Every pair should appear an expected number of times:
    # trials * m / n_pairs.
    expected = trials * m / n_pairs
    for cnt in counts.values():
        assert abs(cnt - expected) < 5 * np.sqrt(expected)


def test_knn_edges_shape_and_undirected_union() -> None:
    torch.manual_seed(0)
    features = torch.randn(10, 3)
    edges = knn_edges(features, k=2)
    assert edges.ndim == 2 and edges.shape[-1] == 2
    assert (edges[:, 0] < edges[:, 1]).all()
    # No duplicates
    pairs = {(int(a), int(b)) for a, b in edges.tolist()}
    assert len(pairs) == edges.shape[0]
