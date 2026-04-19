"""Graph-SND dispatch and timing utilities for the DiCo training loop.

This module is a drop-in replacement layer in front of DiCo's full
:func:`compute_behavioral_distance` (over all ``C(n, 2)`` agent pairs).
When ``estimator == "full"`` we delegate bit-identically to the existing
DiCo codepath; when ``estimator in {"graph_p01", "graph_p025"}`` we
measure diversity on a Bernoulli(p) random subgraph with the
uniform-weight Graph-SND estimator (Remark 8 / Theorem 9 in the
Graph-SND paper). When ``estimator == "knn"`` we build a dynamic
k-nearest-neighbour graph from agent spatial coordinates and compute
uniform-weight Graph-SND over that localized subgraph.

All diversity calls are routed through :func:`time_diversity_call`,
which records per-call wall-clock (milliseconds) into a module-level
buffer; :class:`GraphSNDLoggingCallback` drains and averages this
buffer once per PPO iteration.
"""

from __future__ import annotations

import logging
import time
import weakref
from typing import Callable, List, Optional, Tuple

import torch

from het_control.snd import compute_behavioral_distance, compute_statistical_distance

logger = logging.getLogger(__name__)

# Per-policy ``torch.Generator`` for Bernoulli edge draws. These must **not** be
# stored as attributes on ``nn.Module`` because TorchRL's PPO loss deep-copies
# the actor module, and ``torch.Generator`` is not picklable / deep-copyable.
_GRAPH_RNGS: "weakref.WeakKeyDictionary[object, torch.Generator]" = weakref.WeakKeyDictionary()


def get_graph_rng(owner: object) -> torch.Generator:
    """Return a CPU ``torch.Generator`` tied to ``owner`` (typically ``self`` of the policy module).

    The generator is kept in a process-global weak-key registry so it is not
    part of the module's ``__dict__`` and does not break ``deepcopy`` during
    TorchRL loss construction.

    Parameters
    ----------
    owner : object
        The object that owns this RNG stream (use the ``HetControlMlpEmpirical``
        instance). Weak references are used so generators are dropped when the
        module is garbage-collected.

    Returns
    -------
    torch.Generator
        CPU generator; lazily created with ``manual_seed(0)`` on first access.
    """
    g = _GRAPH_RNGS.get(owner)
    if g is None:
        g = torch.Generator(device="cpu")
        g.manual_seed(0)
        _GRAPH_RNGS[owner] = g
    return g


def reseed_graph_rng(owner: object, seed: int) -> None:
    """Reseed the generator associated with ``owner`` (same scheme as the CSV callback).

    Parameters
    ----------
    owner : object
        Policy module instance (``HetControlMlpEmpirical``).
    seed : int
        Integer passed to ``torch.Generator.manual_seed``.
    """
    get_graph_rng(owner).manual_seed(int(seed))


VALID_ESTIMATORS = ("full", "graph_p01", "graph_p025", "knn")


_CURRENT_ITER_TIMES_MS: List[float] = []


def drain_iter_times_ms() -> List[float]:
    """Pop and return all diversity-call timings recorded since the last drain.

    Returns
    -------
    list of float
        Elapsed wall-clock (milliseconds) for each diversity call made
        since the previous :func:`drain_iter_times_ms`. The global
        buffer is cleared in place before returning.
    """
    global _CURRENT_ITER_TIMES_MS
    out = _CURRENT_ITER_TIMES_MS
    _CURRENT_ITER_TIMES_MS = []
    return out


def sample_bernoulli_edges(
    n: int,
    p: float,
    rng: torch.Generator,
) -> List[Tuple[int, int]]:
    """Return a Bernoulli(p) random subgraph over ``n`` agents.

    Every ordered pair ``(i, j)`` with ``i < j`` is included independently
    with probability ``p``. The implementation is vectorised via a
    Boolean mask indexing into the pre-built upper-triangular pair list;
    no Python per-pair loop is used, which keeps this cheap at large
    ``n`` (e.g. ``n = 100``).

    Parameters
    ----------
    n : int
        Number of agents. The full pair set has size ``n * (n - 1) // 2``.
    p : float
        Bernoulli inclusion probability, in ``[0, 1]``.
    rng : torch.Generator
        Torch RNG state used for reproducibility. The caller is expected
        to re-seed this generator once per PPO iteration (the
        ``GraphSNDLoggingCallback`` does this with
        ``seed * 10000 + iter_number``).

    Returns
    -------
    list of tuple of int
        List of ``(i, j)`` agent-index pairs, ``i < j``. If the Bernoulli
        draw produced zero edges (possible at small ``p`` and small
        ``n``), falls back to returning the full ``C(n, 2)`` pair list
        and emits a DEBUG log record; this keeps the downstream
        ``SND_des / SND(t)`` scaling factor well-defined.
    """
    if n < 2:
        return []
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Expected p in [0, 1], got {p!r}")

    idx = torch.triu_indices(n, n, offset=1)  # [2, n_pairs]
    i_idx = idx[0]
    j_idx = idx[1]
    n_pairs = int(i_idx.shape[0])

    probs = torch.full((n_pairs,), float(p))
    mask = torch.bernoulli(probs, generator=rng).bool()

    if not mask.any():
        logger.debug(
            "sample_bernoulli_edges: empty sample at n=%d, p=%.4f; "
            "falling back to full C(n,2) pairs for this iteration.",
            n,
            p,
        )
        return list(zip(i_idx.tolist(), j_idx.tolist()))

    i_sel = i_idx[mask].tolist()
    j_sel = j_idx[mask].tolist()
    return list(zip(i_sel, j_sel))


def compute_graph_snd_uniform(
    agent_actions: List[torch.Tensor],
    edges: List[Tuple[int, int]],
    just_mean: bool = True,
) -> torch.Tensor:
    """Uniform-weight Graph-SND: arithmetic mean of pairwise Wasserstein over ``edges``.

    This is the estimator described in Remark 8 of the Graph-SND paper,
    which Theorem 9's concentration bound applies to. It reuses DiCo's
    own closed-form Gaussian Wasserstein kernel
    (``compute_statistical_distance``) so the numerics are identical to
    the full-SND path on any shared edge.

    Parameters
    ----------
    agent_actions : list of torch.Tensor
        Per-agent action tensors, as produced by DiCo's
        ``HetControlMlpEmpirical.estimate_snd``. Each has shape
        ``[*batch, action_features]`` (or ``[*batch, 2 * action_features]``
        when ``just_mean`` is False).
    edges : list of tuple of int
        The subgraph edge list (typically the output of
        :func:`sample_bernoulli_edges`). Must be non-empty.
    just_mean : bool, optional
        If True, the action tensors are interpreted as Gaussian means
        only (this is what DiCo's training loop uses). If False, each
        tensor is split into ``(loc, scale)`` along the last dimension.

    Returns
    -------
    torch.Tensor
        A 0-d scalar tensor on the same device/dtype as the input
        actions, equal to the arithmetic mean of pairwise Wasserstein
        distances over the sampled edges (averaged across the batch as
        well, matching ``compute_behavioral_distance(...).mean()``).
    """
    if not edges:
        raise ValueError(
            "compute_graph_snd_uniform received empty edges; "
            "callers should fall back via sample_bernoulli_edges."
        )

    pair_results = [
        compute_statistical_distance(
            agent_actions[i], agent_actions[j], just_mean=just_mean
        )
        for (i, j) in edges
    ]
    stacked = torch.stack(pair_results, dim=-1)  # [*batch, n_edges]
    return stacked.mean()


def compute_knn_edges(
    positions: torch.Tensor,
    k: int,
) -> List[Tuple[int, int]]:
    """Build a symmetric k-NN edge list from agent spatial positions.

    Wraps :func:`graphsnd.graphs.knn_edges` and converts the returned
    ``LongTensor(|E|, 2)`` into the ``List[Tuple[int, int]]`` format
    consumed by :func:`compute_graph_snd_uniform`.

    Parameters
    ----------
    positions : torch.Tensor
        Agent positions of shape ``(n_agents, 2)`` (CPU or CUDA).
    k : int
        Number of nearest neighbours per agent. Must satisfy
        ``1 <= k < n_agents``.

    Returns
    -------
    list of tuple of int
        Edge list ``(i, j)`` with ``i < j``. Falls back to the
        complete edge set (all :math:`C(n, 2)` pairs) when
        ``n_agents <= k``, because the k-NN graph is undefined when
        every agent is already a neighbour of every other.
    """
    from graphsnd.graphs import knn_edges

    n = positions.shape[0]
    if n <= k:
        logger.debug(
            "compute_knn_edges: n=%d <= k=%d; falling back to full C(n,2) pairs.",
            n, k,
        )
        idx = torch.triu_indices(n, n, offset=1)
        return list(zip(idx[0].tolist(), idx[1].tolist()))

    positions_cpu = positions.detach().float().cpu()
    edges_tensor = knn_edges(positions_cpu, k=k, symmetric=True)
    return [(int(e[0]), int(e[1])) for e in edges_tensor.tolist()]


def compute_knn_diversity_per_env(
    agent_actions: List[torch.Tensor],
    positions: torch.Tensor,
    k: int,
    just_mean: bool = True,
) -> torch.Tensor:
    """Per-env k-NN Graph-SND: compute a *different* k-NN graph for each
    parallel environment in the batch, then average the per-env diversity
    scalars.

    This is the **scientifically correct** way to handle vectorised RL
    batches.  Agents' spatial layouts differ across parallel environments,
    so a single k-NN graph computed from one environment snapshot would
    impose wrong neighbourhood structure on the rest.

    Parameters
    ----------
    agent_actions : list of torch.Tensor
        Per-agent action tensors, each of shape ``[B, n_agents, action_dim]``
        (the layout produced by :class:`HetControlMlpEmpirical`).
    positions : torch.Tensor
        Agent spatial positions of shape ``[B, n_agents, 2]``.
    k : int
        Number of nearest neighbours per agent.
    just_mean : bool
        Forwarded to :func:`compute_graph_snd_uniform`.

    Returns
    -------
    torch.Tensor
        A 0-d scalar tensor on the same device/dtype as the input actions,
        equal to the arithmetic mean of per-env Graph-SND values.
    """
    B = positions.shape[0]
    n_agents = positions.shape[1]
    snd_vals: List[torch.Tensor] = []

    for b in range(B):
        edges = compute_knn_edges(positions[b], k)
        env_actions = [a[b] for a in agent_actions]
        if not edges:
            snd_vals.append(
                compute_behavioral_distance(env_actions, just_mean=just_mean).mean()
            )
        else:
            snd_vals.append(
                compute_graph_snd_uniform(env_actions, edges, just_mean=just_mean)
            )

    return torch.stack(snd_vals).mean()


def compute_diversity(
    agent_actions: List[torch.Tensor],
    *,
    estimator: str,
    p: Optional[float] = None,
    rng: Optional[torch.Generator] = None,
    just_mean: bool = True,
    knn_k: int = 3,
    knn_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dispatch to the full-SND path or a Graph-SND path based on ``estimator``.

    The full-SND path (``estimator == "full"``) returns
    ``compute_behavioral_distance(agent_actions, just_mean).mean()``,
    bit-identical to the unmodified DiCo computation. At ``p == 1.0``
    the Graph-SND path reduces to the same value (this is the
    recovery property exercised by
    ``tests/test_graph_snd_recovers_full_snd.py``).

    When ``estimator == "knn"``, a **per-env** dynamic k-NN graph is
    built from ``knn_positions`` and diversity is computed as
    uniform-weight Graph-SND over each environment's localised edge set,
    then averaged across environments.  This correctly handles the case
    where different parallel environments have different agent layouts.

    Parameters
    ----------
    agent_actions : list of torch.Tensor
        Per-agent action tensors of shape ``[*batch, action_features]``
        (or ``[*batch, 2 * action_features]`` when ``just_mean`` is False).
    estimator : str
        One of ``"full"``, ``"graph_p01"``, ``"graph_p025"``, ``"knn"``.
        Any other value raises :class:`ValueError`.
    p : float, optional
        Bernoulli inclusion probability used for Graph-SND. If omitted,
        it is parsed from ``estimator`` (``"graph_p01" -> 0.1``,
        ``"graph_p025" -> 0.25``). Ignored for ``estimator in {"full", "knn"}``.
    rng : torch.Generator, optional
        Torch RNG for the Bernoulli draw. If omitted, a freshly
        constructed generator is used (callers in the DiCo loop pass in
        a seeded one for reproducibility). Ignored for
        ``estimator in {"full", "knn"}``.
    just_mean : bool, optional
        Forwarded to the underlying Wasserstein kernel. ``True`` matches
        the training-time DiCo path.
    knn_k : int, optional
        Number of nearest neighbours for the k-NN graph. Only used when
        ``estimator == "knn"``. Defaults to 3.
    knn_positions : torch.Tensor, optional
        Agent spatial positions of shape ``[B, n_agents, 2]`` (one
        layout per parallel environment). Required when
        ``estimator == "knn"``; ignored otherwise.

    Returns
    -------
    torch.Tensor
        A 0-d scalar tensor on the same device/dtype as the input
        actions. The caller is responsible for any shape adjustment
        (e.g. ``.unsqueeze(-1)`` as in ``estimate_snd``).
    """
    if estimator == "full":
        return compute_behavioral_distance(
            agent_actions, just_mean=just_mean
        ).mean()

    if estimator == "knn":
        if knn_positions is None:
            raise ValueError(
                "knn estimator requires knn_positions with shape [B, n_agents, 2]."
            )
        if knn_positions.dim() == 2:
            knn_positions = knn_positions.unsqueeze(0)
        return compute_knn_diversity_per_env(
            agent_actions, knn_positions, knn_k, just_mean=just_mean
        )

    if estimator not in VALID_ESTIMATORS:
        raise ValueError(
            f"Unknown diversity estimator {estimator!r}; "
            f"expected one of {VALID_ESTIMATORS}."
        )

    if p is None:
        p = 0.1 if estimator == "graph_p01" else 0.25

    n = len(agent_actions)
    if rng is None:
        rng = torch.Generator()
    edges = sample_bernoulli_edges(n, float(p), rng)
    return compute_graph_snd_uniform(agent_actions, edges, just_mean=just_mean)


def time_diversity_call(
    fn: Callable[..., torch.Tensor],
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, float]:
    """Invoke ``fn(*args, **kwargs)`` and return its result plus elapsed ms.

    Correct GPU timing requires ``torch.cuda.synchronize()`` on both
    sides of the wall-clock measurement because kernels are launched
    asynchronously; ``time.perf_counter()`` alone would measure launch
    latency rather than execution time. On CPU inputs the synchronize
    calls are skipped (they would be a no-op anyway).

    The elapsed wall-clock (milliseconds) is appended to the module-
    level buffer :data:`_CURRENT_ITER_TIMES_MS`, which is drained by
    the logging callback via :func:`drain_iter_times_ms` at the end of
    every PPO iteration.

    Parameters
    ----------
    fn : callable
        Any callable returning a ``torch.Tensor`` (typically
        :func:`compute_diversity`).
    *args, **kwargs
        Forwarded to ``fn``.

    Returns
    -------
    (torch.Tensor, float)
        Tuple of ``(fn_return_value, elapsed_ms)``.
    """
    device = _infer_device(args, kwargs)
    is_cuda = device is not None and device.type == "cuda"

    if is_cuda:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    if is_cuda:
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    _CURRENT_ITER_TIMES_MS.append(elapsed_ms)
    return result, elapsed_ms


def _infer_device(args: tuple, kwargs: dict) -> Optional[torch.device]:
    """Return the device of the first tensor found in ``args``/``kwargs``."""
    for value in args:
        dev = _tensor_device(value)
        if dev is not None:
            return dev
    for value in kwargs.values():
        dev = _tensor_device(value)
        if dev is not None:
            return dev
    return None


def _tensor_device(value) -> Optional[torch.device]:
    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        if isinstance(first, torch.Tensor):
            return first.device
    return None
