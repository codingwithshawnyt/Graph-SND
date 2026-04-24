"""Smoke tests for the expander estimator in the DiCo diversity dispatch.

These tests validate the plumbing — not the statistical properties — of the
expander branch added to ``compute_diversity()`` in ``het_control/graph_snd.py``.
Run before committing to a multi-hour training seed.

Must complete in <30 s on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure the fork and repo root are importable.
_FORK_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _FORK_ROOT.parent
for _p in (_FORK_ROOT, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from graphsnd.graphs import random_regular_edges
from het_control.graph_snd import VALID_ESTIMATORS, compute_diversity
from het_control.snd import compute_behavioral_distance


def _synthetic_actions(n: int, action_dim: int = 4, batch: int = 8, seed: int = 0):
    """Return n synthetic Gaussian-mean action tensors for testing."""
    rng = torch.Generator().manual_seed(seed)
    return [torch.randn(batch, action_dim, generator=rng) for _ in range(n)]


# ---------------------------------------------------------------
# Test 1: compute_diversity returns a finite scalar for expander
# ---------------------------------------------------------------
def test_expander_dispatch_returns_finite_scalar():
    n = 10
    actions = _synthetic_actions(n)
    rng = torch.Generator().manual_seed(42)
    result = compute_diversity(
        actions, estimator="expander", expander_d=3, rng=rng, just_mean=True,
    )
    assert result.dim() == 0, f"Expected 0-d scalar, got shape {result.shape}"
    assert torch.isfinite(result), f"Expected finite scalar, got {result.item()}"
    assert result.item() > 0, f"Expected positive scalar, got {result.item()}"


# ---------------------------------------------------------------
# Test 2: random_regular_edges produces a d-regular graph
# ---------------------------------------------------------------
def test_random_regular_edges_d_regularity():
    n, d = 10, 3
    edges = random_regular_edges(n, d, rng=42)
    # Edge count: n*d/2 = 15
    assert edges.shape == (15, 2), f"Expected (15, 2), got {edges.shape}"
    # Check d-regularity: every vertex has degree exactly d
    degree = torch.zeros(n, dtype=torch.long)
    for row in range(edges.shape[0]):
        i, j = int(edges[row, 0]), int(edges[row, 1])
        assert i < j, f"Edge ({i}, {j}) violates i < j ordering"
        degree[i] += 1
        degree[j] += 1
    for v in range(n):
        assert degree[v].item() == d, (
            f"Vertex {v} has degree {degree[v].item()}, expected {d}"
        )


# ---------------------------------------------------------------
# Test 3: expander Graph-SND ratio to full SND is sane
# ---------------------------------------------------------------
def test_expander_ratio_sanity():
    """Smoke-test heuristic: the ratio expander_snd / full_snd should be
    finite, positive, in (0, 2), and at least |E|/|Pairs| - 0.1.

    This is a smoke-test heuristic; the tight theoretical lower bound
    from Theorem 10 involves π(G). The |E|/|Pairs| - 0.1 check catches
    catastrophic bugs (NaN, zero, sign errors) without spuriously failing
    at small n and d.
    """
    n, d = 10, 3
    actions = _synthetic_actions(n, seed=7)
    rng = torch.Generator().manual_seed(99)

    expander_snd = compute_diversity(
        actions, estimator="expander", expander_d=d, rng=rng, just_mean=True,
    )
    full_snd = compute_behavioral_distance(actions, just_mean=True).mean()

    ratio = (expander_snd / full_snd).item()
    n_edges = n * d // 2  # 15
    n_pairs = n * (n - 1) // 2  # 45
    lower = n_edges / n_pairs - 0.1  # ~0.233

    assert torch.isfinite(expander_snd), f"expander_snd is not finite: {expander_snd}"
    assert ratio > 0, f"ratio must be positive, got {ratio}"
    assert ratio < 2, f"ratio must be < 2, got {ratio}"
    assert ratio >= lower, (
        f"ratio {ratio:.4f} < lower bound {lower:.4f} "
        f"(|E|/|Pairs| - 0.1 = {n_edges}/{n_pairs} - 0.1)"
    )


# ---------------------------------------------------------------
# Test 4: "expander" is in VALID_ESTIMATORS
# ---------------------------------------------------------------
def test_expander_in_valid_estimators():
    assert "expander" in VALID_ESTIMATORS, (
        f"'expander' not found in VALID_ESTIMATORS: {VALID_ESTIMATORS}"
    )
