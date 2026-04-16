"""Unit tests for ``graphsnd.metrics``.

These tests check the mathematical content of the paper's propositions:

- Prop 1 (recovery on K_n): ``graph_snd(D, complete_edges(n)) == snd(D)``
  exactly, to machine precision.
- Prop 2 (non-negativity / identity-of-indiscernibles): ``snd(D) >= 0``,
  and ``snd(D) == 0`` exactly iff every policy is the same (equivalent
  means and stds at every observation).
- Prop 5 (HT unbiasedness): the sample mean of Horvitz-Thompson
  estimates over many random graphs converges to ``snd(D)``.
- Thm 6 (Hoeffding concentration): the empirical tail probability
  ``P(|estimator - SND| > t)`` is below the theoretical Hoeffding
  bound on most trials.
- Matrix conventions: ``D`` is symmetric with zero diagonal, and
  averaging on the full upper triangle reproduces the mean.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from graphsnd.graphs import (
    bernoulli_edges,
    complete_edges,
    uniform_size_edges,
)
from graphsnd.metrics import (
    graph_snd,
    graph_snd_from_rollouts,
    hoeffding_bound,
    ht_estimator,
    pairwise_behavioral_distance,
    pairwise_distances_on_edges,
    serfling_bound,
    snd,
    uniform_sample_estimator,
)


def _synthetic_D(n: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    upper = rng.uniform(0.0, 1.0, size=(n, n))
    D = np.triu(upper, k=1)
    D = D + D.T
    return torch.from_numpy(D).float()


def _synthetic_policies_means_stds(n: int, T: int, d_act: int, seed: int = 0):
    torch.manual_seed(seed)
    means = torch.randn(n, T, d_act)
    stds = 0.1 + torch.rand(n, T, d_act)
    return means, stds


def test_pairwise_distance_symmetry_zero_diag() -> None:
    means, stds = _synthetic_policies_means_stds(n=5, T=20, d_act=3, seed=1)
    D = pairwise_behavioral_distance(means, stds)
    assert torch.allclose(D, D.T, atol=1e-6)
    assert torch.allclose(D.diag(), torch.zeros(D.shape[0]), atol=1e-6)
    assert (D >= 0).all()


def test_prop1_recovery_on_complete_graph() -> None:
    for n in [2, 4, 8, 16]:
        D = _synthetic_D(n, seed=n)
        full = snd(D)
        via_graph = graph_snd(D, complete_edges(n))
        assert torch.allclose(full, via_graph, atol=1e-7), (
            f"Prop 1 broken at n={n}: {full.item()} vs {via_graph.item()}"
        )


def test_prop1_on_real_policy_distances() -> None:
    means, stds = _synthetic_policies_means_stds(n=6, T=30, d_act=2, seed=7)
    D = pairwise_behavioral_distance(means, stds)
    full = snd(D)
    via_graph = graph_snd(D, complete_edges(6))
    assert torch.allclose(full, via_graph, atol=1e-6)


def test_prop2_nonnegativity_and_zero_condition() -> None:
    n, T, d_act = 5, 20, 3
    torch.manual_seed(123)
    mean = torch.randn(T, d_act)
    std = 0.1 + torch.rand(T, d_act)
    means = mean.unsqueeze(0).expand(n, T, d_act).clone()
    stds = std.unsqueeze(0).expand(n, T, d_act).clone()
    D_homog = pairwise_behavioral_distance(means, stds)
    assert torch.allclose(snd(D_homog), torch.zeros(()), atol=1e-6)

    D = _synthetic_D(n, seed=5)
    val = snd(D).item()
    assert val >= 0


def test_graph_snd_empty_graph_returns_zero() -> None:
    D = _synthetic_D(5, seed=0)
    edges = torch.empty((0, 2), dtype=torch.long)
    v = graph_snd(D, edges)
    assert v.item() == 0.0


def test_graph_snd_weighted_consistency_with_unweighted() -> None:
    D = _synthetic_D(6, seed=42)
    edges = complete_edges(6)
    w_uniform = torch.ones(edges.shape[0]) * 3.3
    assert torch.allclose(
        graph_snd(D, edges),
        graph_snd(D, edges, w_uniform),
        atol=1e-7,
    )


def test_prop5_ht_unbiasedness_empirical() -> None:
    """Mean HT estimate over many graphs must converge to snd(D)."""
    n = 10
    D = _synthetic_D(n, seed=17)
    target = snd(D).item()
    p = 0.5
    rng = np.random.default_rng(2024)
    trials = 1200
    samples = np.array(
        [ht_estimator(D, p=p, rng=rng).item() for _ in range(trials)]
    )
    mean = samples.mean()
    # Variance of HT with p=0.5 roughly d_max^2/(p * n_pairs).
    se = samples.std(ddof=1) / math.sqrt(trials)
    assert abs(mean - target) < 4.0 * se, (
        f"HT empirical bias too large: mean={mean}, target={target}, se={se}"
    )


def test_prop5_ht_recovers_snd_at_p_one() -> None:
    n = 7
    D = _synthetic_D(n, seed=3)
    rng = np.random.default_rng(0)
    v = ht_estimator(D, p=1.0, rng=rng)
    assert torch.allclose(v, snd(D), atol=1e-6)


def test_thm6_hoeffding_tail_is_below_bound() -> None:
    """Empirical P(|est - snd| > t) stays under Hoeffding bound."""
    n = 14
    D = _synthetic_D(n, seed=0)
    d_max = float(D.max().item())
    n_pairs = n * (n - 1) // 2
    m = 20
    delta = 0.2
    t = hoeffding_bound(d_max, m, delta)

    rng = np.random.default_rng(11)
    trials = 400
    violations = 0
    target = snd(D).item()
    for _ in range(trials):
        v = uniform_sample_estimator(D, m=m, rng=rng).item()
        if abs(v - target) > t:
            violations += 1
    observed_tail = violations / trials
    # Empirical tail should lie well below delta.
    assert observed_tail <= delta


def test_uniform_sample_estimator_m_equals_n_pairs_matches_snd() -> None:
    n = 8
    D = _synthetic_D(n, seed=1)
    n_pairs = n * (n - 1) // 2
    rng = np.random.default_rng(0)
    v = uniform_sample_estimator(D, m=n_pairs, rng=rng)
    assert torch.allclose(v, snd(D), atol=1e-6)


def test_serfling_tighter_than_hoeffding_for_large_m() -> None:
    """Remark 3: Serfling correction never widens the bound."""
    n_pairs = 120
    d_max = 2.0
    delta = 0.05
    for m in (5, 25, 60, 100, n_pairs):
        t_h = hoeffding_bound(d_max, m, delta)
        t_s = serfling_bound(d_max, m, n_pairs, delta)
        assert t_s <= t_h + 1e-12


def test_pairwise_distances_on_edges_matches_full_matrix() -> None:
    means, stds = _synthetic_policies_means_stds(n=7, T=25, d_act=2, seed=11)
    D = pairwise_behavioral_distance(means, stds)
    edges = complete_edges(7)
    d_edges = pairwise_distances_on_edges(means, stds, edges)
    direct = D[edges[:, 0], edges[:, 1]]
    assert torch.allclose(d_edges, direct, atol=1e-6)


def test_graph_snd_from_rollouts_matches_matrix_path() -> None:
    means, stds = _synthetic_policies_means_stds(n=5, T=30, d_act=2, seed=3)
    D = pairwise_behavioral_distance(means, stds)
    edges = complete_edges(5)
    assert torch.allclose(
        graph_snd_from_rollouts(means, stds, edges),
        graph_snd(D, edges),
        atol=1e-6,
    )

    rng = np.random.default_rng(42)
    edges = bernoulli_edges(5, p=0.6, rng=rng)
    if edges.shape[0] > 0:
        weights = torch.ones(edges.shape[0]) / 0.6
        assert torch.allclose(
            graph_snd_from_rollouts(means, stds, edges, weights),
            graph_snd(D, edges, weights),
            atol=1e-6,
        )


def test_snd_mean_over_upper_triangle_matches_average() -> None:
    n = 6
    D = _synthetic_D(n, seed=9)
    i, j = torch.triu_indices(n, n, offset=1)
    manual = D[i, j].mean()
    assert torch.allclose(snd(D), manual, atol=1e-7)
