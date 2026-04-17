"""Correctness tests for :mod:`graphsnd.batched_policies`.

The whole point of the batched module is to be *numerically equivalent*
to stacking ``n`` independent per-agent ``GaussianMLPPolicy`` instances.
If that equivalence ever breaks, every downstream Graph-SND number
computed from a batched-overnight checkpoint would silently differ from
the per-agent reference. These tests lock the bridge.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch

from graphsnd.batched_policies import (
    BatchedGaussianMLPPolicy,
    BatchedValueMLP,
    load_batched_checkpoint,
    save_batched_checkpoint,
)
from graphsnd.metrics import (
    graph_snd_from_rollouts,
    pairwise_behavioral_distance,
    snd,
)
from graphsnd.graphs import complete_edges
from graphsnd.policies import (
    GaussianMLPPolicy,
    PolicyConfig,
    ValueMLP,
    save_checkpoint,
)


def _make_per_agent_policies(n: int, seed: int = 0):
    cfg = PolicyConfig(obs_dim=8, act_dim=2, hidden_sizes=(16, 16), u_range=1.0)
    policies = []
    values = []
    for i in range(n):
        torch.manual_seed(seed * 1000 + i)
        p = GaussianMLPPolicy(cfg)
        v = ValueMLP(cfg.obs_dim, hidden_sizes=(16, 16))
        policies.append(p)
        values.append(v)
    return policies, values, cfg


def test_batched_forward_equals_stacked_per_agent_forward():
    torch.manual_seed(0)
    n = 5
    B = 7
    policies, _, cfg = _make_per_agent_policies(n, seed=0)
    batched = BatchedGaussianMLPPolicy.from_per_agent_policies(policies)
    obs = torch.randn(n, B, cfg.obs_dim)
    with torch.no_grad():
        mean_b, std_b = batched(obs)
        stacked_mean = torch.stack(
            [policies[i](obs[i])[0] for i in range(n)], dim=0
        )
        stacked_std = torch.stack(
            [policies[i](obs[i])[1] for i in range(n)], dim=0
        )
    assert torch.allclose(mean_b, stacked_mean, atol=1e-6, rtol=1e-6)
    assert torch.allclose(std_b, stacked_std, atol=1e-6, rtol=1e-6)


def test_batched_value_equals_stacked_per_agent_value():
    torch.manual_seed(1)
    n = 4
    B = 11
    _, values, cfg = _make_per_agent_policies(n, seed=1)
    batched_v = BatchedValueMLP.from_per_agent_values(values, obs_dim=cfg.obs_dim)
    obs = torch.randn(n, B, cfg.obs_dim)
    with torch.no_grad():
        v_batched = batched_v(obs)
        v_stacked = torch.stack(
            [values[i](obs[i]) for i in range(n)], dim=0
        )
    assert v_batched.shape == (n, B)
    assert torch.allclose(v_batched, v_stacked, atol=1e-6, rtol=1e-6)


def test_gradient_is_independent_across_agents():
    """Graph-SND's heterogeneity story needs per-agent gradient isolation.

    Concretely: perturbing the loss at only agent ``k`` must not touch
    any other agent's parameter gradients. This is the mathematical
    property Proposition 2 (graph-automorphism invariance) rests on, and
    it is the reason ``bmm`` is correct here where ``DataParallel``
    replication would be wrong.
    """
    torch.manual_seed(2)
    n = 6
    B = 3
    cfg = PolicyConfig(obs_dim=4, act_dim=2, hidden_sizes=(8, 8))
    batched = BatchedGaussianMLPPolicy(n, cfg)
    obs = torch.randn(n, B, cfg.obs_dim)
    mean, _ = batched(obs)
    k = 2
    loss = mean[k].square().sum()
    loss.backward()
    with torch.no_grad():
        for name, param in batched.named_parameters():
            if param.grad is None:
                continue
            per_agent = param.grad
            if per_agent.ndim == 0:
                continue
            for i in range(n):
                if i == k:
                    continue
                assert torch.allclose(
                    per_agent[i], torch.zeros_like(per_agent[i]), atol=0.0
                ), (
                    f"gradient leaked from agent {k} into agent {i} on "
                    f"parameter {name!r}"
                )


def test_round_trip_per_agent_batched_per_agent():
    """Starting from per-agent, going batched, then back should be a no-op.

    Guarantees every overnight-checkpoint can be re-exported as an
    Exp-1-style per-agent file with no drift.
    """
    torch.manual_seed(3)
    n = 3
    B = 4
    policies, _, cfg = _make_per_agent_policies(n, seed=3)
    batched = BatchedGaussianMLPPolicy.from_per_agent_policies(policies)
    restored = batched.to_per_agent_policies()
    obs_single = torch.randn(B, cfg.obs_dim)
    with torch.no_grad():
        for i in range(n):
            m0, s0 = policies[i](obs_single)
            m1, s1 = restored[i](obs_single)
            assert torch.allclose(m0, m1, atol=1e-6, rtol=1e-6)
            assert torch.allclose(s0, s1, atol=1e-6, rtol=1e-6)


def test_save_and_load_batched_checkpoint_preserves_forward(tmp_path: Path):
    """The batched checkpoint file must round-trip bit-wise on forward."""
    torch.manual_seed(4)
    n = 4
    B = 5
    cfg = PolicyConfig(obs_dim=6, act_dim=2, hidden_sizes=(16, 16))
    policy = BatchedGaussianMLPPolicy(n, cfg, seed_base=4242)
    value = BatchedValueMLP(n, obs_dim=cfg.obs_dim, hidden_sizes=(16, 16))
    path = tmp_path / "ckpt.pt"
    save_batched_checkpoint(
        path, policy, value, extra={"iter": 7, "tag": "overnight"}
    )
    loaded_policy, loaded_value, extra = load_batched_checkpoint(path)
    assert extra == {"iter": 7, "tag": "overnight"}
    assert loaded_value is not None
    obs = torch.randn(n, B, cfg.obs_dim)
    with torch.no_grad():
        m0, s0 = policy(obs)
        m1, s1 = loaded_policy(obs)
        v0 = value(obs)
        v1 = loaded_value(obs)
    assert torch.allclose(m0, m1, atol=0.0, rtol=0.0)
    assert torch.allclose(s0, s1, atol=0.0, rtol=0.0)
    assert torch.allclose(v0, v1, atol=0.0, rtol=0.0)


def test_batched_checkpoint_loads_from_existing_per_agent_file(tmp_path: Path):
    """Back-compat: existing Exp-1 ``save_checkpoint`` files must load
    into the batched loader unchanged, so the n=4/8/16 checkpoints the
    repo already ships with immediately gain the fast batched path.
    """
    n = 4
    B = 3
    policies, values, cfg = _make_per_agent_policies(n, seed=5)
    path = tmp_path / "per_agent.pt"
    save_checkpoint(path, policies, values, extra={"mode": "per-agent"})
    batched_policy, batched_value, extra = load_batched_checkpoint(path)
    assert batched_value is not None
    assert extra == {"mode": "per-agent"}
    obs = torch.randn(n, B, cfg.obs_dim)
    with torch.no_grad():
        stacked_mean = torch.stack(
            [policies[i](obs[i])[0] for i in range(n)], dim=0
        )
        m_batched, _ = batched_policy(obs)
    assert torch.allclose(m_batched, stacked_mean, atol=1e-6, rtol=1e-6)


def test_batched_forward_feeds_graph_snd_pipeline():
    """The batched means/stds must be directly consumable by the existing
    Graph-SND metric code (Proposition 2 recovery at n=100 will use
    exactly this shape contract overnight)."""
    torch.manual_seed(6)
    n = 10
    T = 9
    cfg = PolicyConfig(obs_dim=6, act_dim=2, hidden_sizes=(16, 16))
    batched = BatchedGaussianMLPPolicy(n, cfg, seed_base=42)
    flat_obs = torch.randn(T, cfg.obs_dim)
    obs_per_policy = flat_obs.unsqueeze(0).expand(n, T, cfg.obs_dim)
    with torch.no_grad():
        mean, std = batched(obs_per_policy)
    D = pairwise_behavioral_distance(mean, std)
    full = float(snd(D).item())
    direct = float(
        graph_snd_from_rollouts(mean, std, complete_edges(n)).item()
    )
    assert D.shape == (n, n)
    assert (D >= 0).all()
    assert abs(full - direct) < 1e-6, (full, direct)
