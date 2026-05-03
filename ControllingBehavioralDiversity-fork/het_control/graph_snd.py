"""Graph-SND dispatch and timing utilities for the DiCo training loop.

This module is a drop-in replacement layer in front of DiCo's full
:func:`compute_behavioral_distance` (over all ``C(n, 2)`` agent pairs).
When ``estimator == "full"`` we delegate bit-identically to the existing
DiCo codepath; when ``estimator in {"graph_p01", "graph_p025"}`` we
measure diversity on a Bernoulli(p) random subgraph with the
uniform-weight Graph-SND estimator analyzed in the Graph-SND paper.
When ``estimator == "knn"`` we build a dynamic
k-nearest-neighbour graph from agent spatial coordinates and compute
uniform-weight Graph-SND over that localized subgraph.

All diversity calls are routed through :func:`time_diversity_call`,
which records per-call wall-clock (milliseconds) into a module-level
buffer; :class:`GraphSNDLoggingCallback` drains and averages this
buffer once per PPO iteration.
"""

from __future__ import annotations

import logging
import sys
import time
import weakref
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch

from het_control.snd import compute_behavioral_distance, compute_statistical_distance

logger = logging.getLogger(__name__)

# Lazily resolved ``knn_edges`` from the sibling ``graphsnd`` package (lives at
# the Graph-SND repo root, not inside ``ControllingBehavioralDiversity-fork``).
# Cluster-style setups sometimes forget ``pip install -e ..`` from the repo
# root; we prepend the root to ``sys.path`` once so ``import graphsnd`` works
# anyway.
_knn_edges_impl = None

# Lazily resolved ``random_regular_edges`` from the sibling ``graphsnd`` package.
_random_regular_edges_impl = None


def _import_graphsnd_symbol(symbol_name: str):
    """Import ``graphsnd.graphs.<symbol_name>`` robustly when the fork's own
    working directory shadows the repository-level ``graphsnd`` package.

    On some deployments, the process starts from
    ``ControllingBehavioralDiversity-fork``. Python then resolves
    ``import graphsnd`` against a fork-local package first, which may not
    expose all symbols (e.g., ``random_regular_edges``). In that case we:
      1) prepend the repository root to ``sys.path``;
      2) clear any already-cached ``graphsnd`` modules from ``sys.modules``;
      3) retry the import from the intended package.
    """
    try:
        mod = __import__("graphsnd.graphs", fromlist=[symbol_name])
        return getattr(mod, symbol_name)
    except (ModuleNotFoundError, ImportError, AttributeError):
        repo_root = Path(__file__).resolve().parents[2]
        gs_dir = repo_root / "graphsnd"
        if gs_dir.is_dir() and str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
            logger.info(
                "Prepended %s to sys.path so ``import graphsnd`` resolves "
                "to repository root package.",
                repo_root,
            )
        # If a shadowed graphsnd module was already imported from the fork
        # subtree, drop it so the next import can resolve against repo_root.
        for key in list(sys.modules.keys()):
            if key == "graphsnd" or key.startswith("graphsnd."):
                del sys.modules[key]
        try:
            mod = __import__("graphsnd.graphs", fromlist=[symbol_name])
            return getattr(mod, symbol_name)
        except (ModuleNotFoundError, ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Could not import ``graphsnd.graphs.%s``. Ensure you run from "
                "the Graph-SND repository root (contains both ``graphsnd/`` "
                "and ``ControllingBehavioralDiversity-fork/``) and install "
                "the package with ``pip install -e .``."
                % symbol_name
            ) from exc


def _get_random_regular_edges():
    """Return ``graphsnd.graphs.random_regular_edges``, bootstrapping repo-root ``sys.path`` if needed."""
    global _random_regular_edges_impl
    if _random_regular_edges_impl is not None:
        return _random_regular_edges_impl
    _random_regular_edges_impl = _import_graphsnd_symbol("random_regular_edges")
    return _random_regular_edges_impl


def _get_knn_edges():
    """Return ``graphsnd.graphs.knn_edges``, bootstrapping repo-root ``sys.path`` if needed."""
    global _knn_edges_impl
    if _knn_edges_impl is not None:
        return _knn_edges_impl
    _knn_edges_impl = _import_graphsnd_symbol("knn_edges")
    return _knn_edges_impl


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


VALID_ESTIMATORS = ("full", "graph_p01", "graph_p025", "knn", "expander")


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

    This is the uniform-edge sample mean analyzed in the Graph-SND paper.
    It reuses DiCo's
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
    knn_edges = _get_knn_edges()

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
    subsample_envs: Optional[int] = 128,
    subsample_rng: Optional[torch.Generator] = None,
    use_vectorized: bool = True,
) -> torch.Tensor:
    """Per-env k-NN Graph-SND: compute a *different* k-NN graph for each
    parallel environment in the batch, then average the per-env diversity
    scalars.

    This is the **scientifically correct** way to handle vectorised RL
    batches.  Agents' spatial layouts differ across parallel environments,
    so a single k-NN graph computed from one environment snapshot would
    impose wrong neighbourhood structure on the rest.

    Two implementations share this entry point:

    * ``use_vectorized=True`` (default) runs the entire per-env k-NN +
      pairwise Wasserstein computation as a single fused GPU kernel chain
      (``torch.cdist`` -> ``torch.topk`` -> ``torch.gather`` -> elementwise
      norm + mean). No Python-level per-env loop, so at ``B = 4096`` a
      single call issues O(10) kernel launches instead of the ~120k the
      Python path triggers. This keeps training-time SND estimation fast
      enough that IPPO training isn't dominated by CUDA submission
      overhead. The k-NN graph is built using directed k-NN edges (each
      agent's top-k neighbours); for the uniform-weight mean this gives
      the same population scalar as the symmetric edge list used in the
      Python path, because pairwise Wasserstein is symmetric and mutual
      neighbours only contribute a double count in both the numerator
      and the denominator of the mean.

    * ``use_vectorized=False`` falls back to the Python per-env loop over
      ``compute_knn_edges`` / ``compute_graph_snd_uniform``. Retained
      because ``tests/`` exercises this path against the reference
      ``graphsnd.graphs.knn_edges`` implementation.

    For very large ``B`` the vectorised path still allocates
    ``O(B * n_agents^2)`` intermediate tensors (the pairwise distance
    matrix and the gathered neighbour actions). ``subsample_envs`` caps
    ``B`` by uniformly sampling at most that many envs per call, which
    is an unbiased Monte Carlo estimate of the same population scalar.

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
        Forwarded to :func:`compute_graph_snd_uniform`. ``True`` matches
        the training-time DiCo path (Gaussian mean Wasserstein).
    subsample_envs : int or None, optional
        If set and ``B > subsample_envs``, uniformly sample this many env
        indices without replacement before computation. Defaults to
        ``128``. ``None`` disables subsampling.
    subsample_rng : torch.Generator, optional
        RNG used for the env subsample draw (reproducibility). Defaults
        to the default torch RNG when omitted.
    use_vectorized : bool, optional
        If ``True`` (default), use the fused GPU path. If ``False``, use
        the reference Python per-env loop (for tests / debugging).

    Returns
    -------
    torch.Tensor
        A 0-d scalar tensor on the same device/dtype as the input actions,
        equal to the arithmetic mean of per-env Graph-SND values.
    """
    B = positions.shape[0]

    # Cheap Monte Carlo downsample of envs to keep kernel-launch count
    # bounded (see docstring). At small B (e.g. n_envs in evaluation,
    # tests) the default is effectively a no-op.
    if subsample_envs is not None and B > subsample_envs:
        if subsample_rng is None:
            idx = torch.randperm(B, device=positions.device)[:subsample_envs]
        else:
            idx = torch.randperm(B, generator=subsample_rng)[:subsample_envs]
            idx = idx.to(positions.device)
        positions = positions.index_select(0, idx)
        agent_actions = [a.index_select(0, idx.to(a.device)) for a in agent_actions]
        B = positions.shape[0]

    if use_vectorized:
        return _compute_knn_diversity_per_env_vectorized(
            agent_actions, positions, k, just_mean=just_mean
        )

    # Reference Python path (kept for tests/debugging).
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


def _compute_knn_diversity_per_env_vectorized(
    agent_actions: List[torch.Tensor],
    positions: torch.Tensor,
    k: int,
    just_mean: bool = True,
) -> torch.Tensor:
    """Fully vectorised per-env k-NN Graph-SND (see docstring of
    :func:`compute_knn_diversity_per_env` for the semantic contract).

    Runs every step of the k-NN + Wasserstein computation as a fused
    sequence of bulk tensor ops on the input device, so a single call
    issues O(10) kernel launches regardless of ``B``. This keeps the
    SND estimate from dominating IPPO training time (the Python path
    bottlenecks at ~200 s/iter on Dispersion at ``B = 4096``).
    """
    B = positions.shape[0]
    n_agents_pos = positions.shape[1]
    P = len(agent_actions)
    # agent_actions[i] is shape [B, N, D] where N is the obs-agent axis and
    # D is the per-agent policy output width.
    N = agent_actions[0].shape[-2]
    D = agent_actions[0].shape[-1]

    # k-NN needs at least one valid (non-self) neighbour per agent. With
    # n_agents <= k the per-env k-NN is undefined; fall back to the
    # complete graph (mean over all ordered pairs) on the action tensor.
    # This matches the ``compute_knn_edges`` n <= k fallback.
    k_eff = min(int(k), n_agents_pos - 1)
    if k_eff < 1 or n_agents_pos <= k_eff:
        # Full SND over all agent pairs.
        return compute_behavioral_distance(agent_actions, just_mean=just_mean).mean()

    # Stack agent_actions into [P, B, N, D] -> [B, P, N, D] so gather
    # along the policy-agent axis is contiguous.
    A = torch.stack(agent_actions, dim=0).permute(1, 0, 2, 3).contiguous()
    # A[b, pi, :, :] = policy agent pi's output at env b (over all N obs agents).

    # Pairwise spatial distances per env: [B, n_agents_pos, n_agents_pos].
    # Mask the diagonal so an agent can't be its own neighbour.
    # ``positions`` may live on CPU (the caller does a .cpu() for the old
    # Python path); bring it onto the action device here so cdist + topk
    # stay on GPU.
    positions_dev = positions.to(A.device, non_blocking=True)
    dist = torch.cdist(positions_dev, positions_dev)
    eye = torch.eye(n_agents_pos, device=A.device, dtype=torch.bool)
    dist = dist.masked_fill(eye, float("inf"))

    # Top-k nearest neighbour indices: [B, n_agents_pos, k_eff].
    _, nbr_idx = torch.topk(dist, k=k_eff, dim=-1, largest=False)

    # Gather A_perm along the policy-agent axis using the neighbour
    # indices: tgt[b, pi, kk, :, :] = A[b, nbr_idx[b, pi, kk], :, :].
    # torch.gather requires the index tensor to match the source shape
    # except along the gather dim; expand over (N, D).
    idx_exp = (
        nbr_idx.reshape(B, n_agents_pos * k_eff)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(B, n_agents_pos * k_eff, N, D)
    )
    tgt = A.gather(dim=1, index=idx_exp).reshape(B, n_agents_pos, k_eff, N, D)
    src = A.unsqueeze(2)  # [B, P, 1, N, D], broadcasts against tgt

    if just_mean:
        # Wasserstein (means only) = L2 norm of the mean diff over the
        # output-dim axis; see het_control.snd.wasserstein_distance.
        diff = src - tgt
        w = torch.linalg.norm(diff, ord=2, dim=-1)  # [B, P, k_eff, N]
        return w.mean()

    # Non-just_mean path: interpret the last dim as (loc, scale) and use
    # the Gaussian Wasserstein closed form (diagonal covariance).
    loc_src, scale_src = src.chunk(2, dim=-1)
    loc_tgt, scale_tgt = tgt.chunk(2, dim=-1)
    loc_diff = torch.linalg.norm(loc_src - loc_tgt, ord=2, dim=-1)
    # For a diagonal covariance, the Bures/Fr\xe9chet term reduces to the
    # L2 distance between the scale vectors (elementwise std).
    scale_diff = torch.linalg.norm(scale_src - scale_tgt, ord=2, dim=-1)
    w = torch.sqrt(loc_diff.pow(2) + scale_diff.pow(2))
    return w.mean()


def compute_diversity(
    agent_actions: List[torch.Tensor],
    *,
    estimator: str,
    p: Optional[float] = None,
    rng: Optional[torch.Generator] = None,
    just_mean: bool = True,
    knn_k: int = 3,
    knn_positions: Optional[torch.Tensor] = None,
    knn_subsample_envs: Optional[int] = 128,
    knn_use_vectorized: bool = True,
    expander_d: int = 3,
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
            agent_actions,
            knn_positions,
            knn_k,
            just_mean=just_mean,
            subsample_envs=knn_subsample_envs,
            subsample_rng=rng,
            use_vectorized=knn_use_vectorized,
        )

    if estimator == "expander":
        random_regular_edges = _get_random_regular_edges()
        n = len(agent_actions)
        d = expander_d
        if (n * d) % 2 != 0:
            logger.debug(
                "n*d=%d is odd; using d=%d instead of %d for expander graph.",
                n * d, d - 1, d,
            )
            d = d - 1
        if rng is None:
            rng = torch.Generator()
        rng_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
        edges_tensor = random_regular_edges(n, d, rng_seed)
        edges = [(int(e[0]), int(e[1])) for e in edges_tensor.tolist()]
        return compute_graph_snd_uniform(agent_actions, edges, just_mean=just_mean)

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
