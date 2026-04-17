"""Sanity check: Graph-SND at p=1.0 recovers DiCo's full SND.

This mirrors the recovery property (Proposition 2 in the Graph-SND
paper) but runs on the DiCo codepath, catching any integration bug
that would break equivalence between ``compute_diversity(..., "full")``
and ``compute_diversity(..., "graph_p01", p=1.0)``.

Concentration, unbiasedness and other empirical properties are tested
in the Graph-SND repo itself and are intentionally out of scope here.
"""

from __future__ import annotations

import torch

from het_control.graph_snd import compute_diversity, sample_bernoulli_edges
from het_control.snd import compute_behavioral_distance


def _synthetic_gaussian_policies(
    n_agents: int = 3,
    batch_size: int = 8,
    action_features: int = 4,
    seed: int = 123,
):
    """Construct ``n_agents`` independent Gaussian-mean tensors.

    Uses ``just_mean=True`` semantics (no scale), matching the DiCo
    training path through ``HetControlMlpEmpirical.estimate_snd``.
    """
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn(batch_size, action_features, generator=g)
        for _ in range(n_agents)
    ]


def test_graph_snd_recovers_full_snd_at_p1():
    agent_actions = _synthetic_gaussian_policies()

    full_snd = compute_behavioral_distance(
        agent_actions, just_mean=True
    ).mean()

    rng = torch.Generator().manual_seed(0)
    graph_snd = compute_diversity(
        agent_actions,
        estimator="graph_p01",
        p=1.0,
        rng=rng,
        just_mean=True,
    )

    assert torch.allclose(full_snd, graph_snd, atol=1e-6), (
        "Expected p=1.0 Graph-SND to equal full SND, got "
        f"full={full_snd.item():.8f}, graph={graph_snd.item():.8f}"
    )


def test_compute_diversity_full_equals_full_path():
    """Sanity: dispatch on 'full' returns the exact full-SND scalar."""
    agent_actions = _synthetic_gaussian_policies()

    baseline = compute_behavioral_distance(
        agent_actions, just_mean=True
    ).mean()
    dispatched = compute_diversity(
        agent_actions, estimator="full", p=1.0, just_mean=True
    )

    assert torch.equal(baseline, dispatched)


def test_sample_bernoulli_edges_fallback_when_empty():
    """Empty Bernoulli samples should fall back to the full C(n, 2) set."""
    rng = torch.Generator().manual_seed(0)
    edges = sample_bernoulli_edges(n=4, p=0.0, rng=rng)
    expected_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert edges == expected_pairs
