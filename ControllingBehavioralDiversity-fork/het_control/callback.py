#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import csv
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tensordict import TensorDictBase, TensorDict

from benchmarl.experiment.callback import Callback
from het_control.graph_snd import drain_iter_times_ms, reseed_graph_rng
from het_control.models.het_control_mlp_empirical import HetControlMlpEmpirical
from het_control.snd import compute_behavioral_distance
from het_control.utils import overflowing_logits_norm

_callback_logger = logging.getLogger(__name__)


def get_het_model(policy):
    model = policy.module[0]
    while not isinstance(model, HetControlMlpEmpirical):
        model = model[0]
    return model


class SndCallback(Callback):
    """
    Callback used to compute SND during evaluations
    """

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        for group in self.experiment.group_map.keys():
            if not len(self.experiment.group_map[group]) > 1:
                # If agent group has 1 agent
                continue
            policy = self.experiment.group_policies[group]
            # Cat observations over time
            obs = torch.cat(
                [rollout.select((group, "observation")) for rollout in rollouts], dim=0
            )  # tensor of shape [*batch_size, n_agents, n_features]
            model = get_het_model(policy)
            agent_actions = []
            # Compute actions that each agent would take in this obs
            for i in range(model.n_agents):
                agent_actions.append(
                    model._forward(obs, agent_index=i, compute_estimate=False).get(
                        model.out_key
                    )
                )
            # Compute SND
            distance = compute_behavioral_distance(agent_actions, just_mean=True)
            self.experiment.logger.log(
                {f"eval/{group}/snd": distance.mean().item()},
                step=self.experiment.n_iters_performed,
            )


class NormLoggerCallback(Callback):
    """
    Callback to log some training metrics
    """

    def on_batch_collected(self, batch: TensorDictBase):
        for group in self.experiment.group_map.keys():
            keys_to_norm = [
                (group, "f"),
                (group, "g"),
                (group, "fdivg"),
                (group, "logits"),
                (group, "observation"),
                (group, "out_loc_norm"),
                (group, "estimated_snd"),
                (group, "scaling_ratio"),
            ]
            to_log = {}

            for key in keys_to_norm:
                value = batch.get(key, None)
                if value is not None:
                    to_log.update(
                        {"/".join(("collection",) + key): torch.mean(value).item()}
                    )
            self.experiment.logger.log(
                to_log,
                step=self.experiment.n_iters_performed,
            )


class TagCurriculum(Callback):
    """
    Tag curriculum used to freeze the green agents' policies during training
    """

    def __init__(self, simple_tag_freeze_policy_after_frames, simple_tag_freeze_policy):
        super().__init__()
        self.n_frames_train = simple_tag_freeze_policy_after_frames
        self.simple_tag_freeze_policy = simple_tag_freeze_policy
        self.activated = not simple_tag_freeze_policy

    def on_setup(self):
        # Log params
        self.experiment.logger.log_hparams(
            simple_tag_freeze_policy_after_frames=self.n_frames_train,
            simple_tag_freeze_policy=self.simple_tag_freeze_policy,
        )
        # Make agent group homogeneous
        policy = self.experiment.group_policies["agents"]
        model = get_het_model(policy)
        # Set the desired SND of the green agent team to 0
        # This is not important as the green agent team is composed of 1 agent
        model.desired_snd[:] = 0

    def on_batch_collected(self, batch: TensorDictBase):
        if (
            self.experiment.total_frames >= self.n_frames_train
            and not self.activated
            and self.simple_tag_freeze_policy
        ):
            del self.experiment.train_group_map["agents"]
            self.activated = True


class ActionSpaceLoss(Callback):
    """
    Loss to disincentivize actions outside of the space
    """

    def __init__(self, use_action_loss, action_loss_lr):
        super().__init__()
        self.opt_dict = {}
        self.use_action_loss = use_action_loss
        self.action_loss_lr = action_loss_lr

    def on_setup(self):
        # Log params
        self.experiment.logger.log_hparams(
            use_action_loss=self.use_action_loss, action_loss_lr=self.action_loss_lr
        )

    def on_train_step(self, batch: TensorDictBase, group: str) -> TensorDictBase:
        if not self.use_action_loss:
            return
        policy = self.experiment.group_policies[group]
        model = get_het_model(policy)
        if group not in self.opt_dict:
            self.opt_dict[group] = torch.optim.Adam(
                model.parameters(), lr=self.action_loss_lr
            )
        opt = self.opt_dict[group]
        loss = self.action_space_loss(group, model, batch)
        loss_td = TensorDict({"loss_action_space": loss}, [])

        loss.backward()

        grad_norm = self.experiment._grad_clip(opt)
        loss_td.set(
            f"grad_norm_action_space",
            torch.tensor(grad_norm, device=self.experiment.config.train_device),
        )

        opt.step()
        opt.zero_grad()

        return loss_td

    def action_space_loss(self, group, model, batch):
        logits = model._forward(
            batch.select(*model.in_keys), compute_estimate=True, update_estimate=False
        ).get(
            model.out_key
        )  # Compute logits from batch
        if model.probabilistic:
            logits, _ = torch.chunk(logits, 2, dim=-1)
        out_loc_norm = overflowing_logits_norm(
            logits, self.experiment.action_spec[group, "action"]
        )  # Compute how much they overflow outside the action space bounds

        # Penalise the maximum overflow over the agents
        max_overflowing_logits_norm = out_loc_norm.max(dim=-1)[0]

        loss = max_overflowing_logits_norm.pow(2).mean()
        return loss


class GraphSNDLoggingCallback(Callback):
    """Per-iteration CSV logger for Graph-SND DiCo runs.

    Writes one row per PPO iteration with columns
    ``iter, seed, n_agents, estimator, p, snd_t, snd_des, reward_mean,
    metric_time_ms, scaling_ratio_mean, applied_snd, out_loc_norm_mean``
    to a user-supplied CSV path. The row is flushed (with ``os.fsync``) so a
    killed run still has partial data on disk.

    The callback also re-seeds each group's Bernoulli ``torch.Generator`` (via
    :func:`het_control.graph_snd.reseed_graph_rng`) at the start of every iteration
    using ``seed * 10000 + n_iters_performed``,
    which guarantees every ``(seed, iter)`` pair reproduces the same sequence
    of Bernoulli edge samples regardless of how many model forward passes the
    PPO loop performs within an iteration.

    Notes on the three diagnostic columns (added 2026-04 after three
    speculative hyperparameter fixes failed; see DIAGNOSIS.md Postmortem #2):

    - ``scaling_ratio_mean`` is the batch mean of the scaling ratio that
      ``HetControlMlpEmpirical._forward`` writes into the tensordict under
      ``(group, "scaling_ratio")``. With DiCo's invariant
      ``applied_SND = scaling_ratio * raw_SND`` and ``scaling_ratio =
      desired_snd / estimated_snd``, a healthy run has
      ``scaling_ratio_mean`` close to ``snd_des / snd_t`` per row.
    - ``applied_snd`` is ``snd_t * scaling_ratio_mean``, i.e. the
      diversity of the per-agent actions *the environment actually sees*.
      This is the quantity DiCo is trying to keep at ``desired_snd``.
      ``snd_t`` alone can run away (raw per-agent network spread) while
      applied SND stays on target -- only ``applied_snd`` tells us
      whether the control loop itself is working.
    - ``out_loc_norm_mean`` is the batch mean of ``(group, "out_loc_norm")``,
      which ``HetControlMlpEmpirical`` already computes for the action-space
      overflow loss. It surfaces when agent means saturate the action
      bounds (a frequent root cause of reward stagnation).

    All three are best-effort: if BenchMARL's training_td does not carry
    the key for a given algorithm, the column is written as ``nan`` and
    downstream consumers should treat it accordingly.
    """

    CSV_FIELDS: List[str] = [
        "iter",
        "seed",
        "n_agents",
        "estimator",
        "p",
        "snd_t",
        "snd_des",
        "reward_mean",
        "metric_time_ms",
        "scaling_ratio_mean",
        "applied_snd",
        "out_loc_norm_mean",
    ]

    def __init__(
        self,
        log_path: str,
        seed: int,
        n_agents: int,
        estimator: str,
        p: float,
    ) -> None:
        super().__init__()
        self.log_path = Path(log_path)
        self.seed = int(seed)
        self.n_agents = int(n_agents)
        self.estimator = str(estimator)
        self.p = float(p)

        self._csv_fh = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._current_reward_mean: float = float("nan")
        self._current_iter: int = 0
        self._last_written_iter: int = -1

    # ------------------------------------------------------------------
    # CSV lifecycle
    # ------------------------------------------------------------------
    def _ensure_csv_open(self) -> None:
        if self._csv_fh is not None and not self._csv_fh.closed:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (
            not self.log_path.exists() or self.log_path.stat().st_size == 0
        )
        # ``newline=""`` per csv stdlib docs so cross-platform newlines are correct.
        self._csv_fh = self.log_path.open("a", newline="")
        self._csv_writer = csv.DictWriter(self._csv_fh, fieldnames=self.CSV_FIELDS)
        if write_header:
            self._csv_writer.writeheader()
            self._csv_fh.flush()

    def _close_csv(self) -> None:
        if self._csv_fh is not None:
            try:
                self._csv_fh.close()
            except Exception:  # pragma: no cover - defensive
                pass
            self._csv_fh = None
            self._csv_writer = None

    # ------------------------------------------------------------------
    # BenchMARL hooks
    # ------------------------------------------------------------------
    def on_setup(self) -> None:
        self._ensure_csv_open()
        _callback_logger.info(
            "GraphSNDLoggingCallback: logging to %s (estimator=%s, p=%.4f, seed=%d, n_agents=%d)",
            self.log_path,
            self.estimator,
            self.p,
            self.seed,
            self.n_agents,
        )

    def on_batch_collected(self, batch: TensorDictBase) -> None:
        self._current_iter = int(self.experiment.n_iters_performed)

        # Reseed each HetControlMlpEmpirical model's Bernoulli RNG so the
        # (seed, iter) pair reproduces the identical sequence of subgraph
        # samples across the ~45 minibatch PPO forward passes this iter.
        reseed_val = self.seed * 10000 + self._current_iter
        for group in self.experiment.group_map.keys():
            try:
                policy = self.experiment.group_policies[group]
                model = get_het_model(policy)
            except Exception:
                continue
            reseed_graph_rng(model, reseed_val)

        # Stash reward mean for this iter; written at on_train_end time.
        self._current_reward_mean = self._batch_reward_mean(batch)

    def on_train_end(self, training_td: TensorDictBase, group: str) -> None:
        if self._current_iter == self._last_written_iter:
            # Multi-group experiments will fire on_train_end once per group; we
            # only want a single CSV row per iteration.
            return
        self._ensure_csv_open()

        try:
            policy = self.experiment.group_policies[group]
            model = get_het_model(policy)
        except Exception:
            _callback_logger.debug(
                "GraphSNDLoggingCallback: could not resolve HetControlMlpEmpirical "
                "for group %s; skipping row.",
                group,
            )
            return

        snd_t_value = model.estimated_snd.detach()
        snd_t = float(snd_t_value.item()) if snd_t_value.numel() == 1 else float("nan")
        snd_des = float(model.desired_snd.item())

        times = drain_iter_times_ms()
        metric_time_ms = (
            float(sum(times) / len(times)) if times else float("nan")
        )

        scaling_ratio_mean = _batch_tensor_mean(
            training_td, (group, "scaling_ratio")
        )
        out_loc_norm_mean = _batch_tensor_mean(
            training_td, (group, "out_loc_norm")
        )
        if math.isfinite(snd_t) and math.isfinite(scaling_ratio_mean):
            applied_snd = snd_t * scaling_ratio_mean
        else:
            applied_snd = float("nan")

        row: Dict[str, Any] = {
            "iter": self._current_iter,
            "seed": self.seed,
            "n_agents": self.n_agents,
            "estimator": self.estimator,
            "p": self.p,
            "snd_t": snd_t,
            "snd_des": snd_des,
            "reward_mean": self._current_reward_mean,
            "metric_time_ms": metric_time_ms,
            "scaling_ratio_mean": scaling_ratio_mean,
            "applied_snd": applied_snd,
            "out_loc_norm_mean": out_loc_norm_mean,
        }
        assert self._csv_writer is not None
        self._csv_writer.writerow(row)
        self._csv_fh.flush()
        try:
            os.fsync(self._csv_fh.fileno())
        except OSError:
            # fsync can fail on some filesystems (e.g. /tmp on certain CI
            # runners); a crashed process will still have flushed stdio
            # buffers, which is good enough for the spec's "killed run"
            # guarantee on ordinary disks.
            pass

        self._last_written_iter = self._current_iter

    def on_load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Reopen the CSV in append mode so a resumed run keeps appending to
        # the same file without a stale file-handle pointer.
        self._close_csv()
        self._ensure_csv_open()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _batch_reward_mean(self, batch: TensorDictBase) -> float:
        """Best-effort average reward across groups for the current rollout."""
        try:
            group_means: List[float] = []
            for group in self.experiment.group_map.keys():
                reward = _safe_tensordict_get(batch, ("next", group, "reward"))
                if reward is None:
                    reward = _safe_tensordict_get(batch, (group, "reward"))
                if reward is not None:
                    group_means.append(float(reward.float().mean().item()))
            if not group_means:
                return float("nan")
            return float(sum(group_means) / len(group_means))
        except Exception:
            return float("nan")


def _safe_tensordict_get(td: TensorDictBase, key) -> Optional[torch.Tensor]:
    """Return ``td.get(key)`` or None if the key is absent."""
    try:
        return td.get(key)
    except (KeyError, Exception):
        return None


def _batch_tensor_mean(td: TensorDictBase, key) -> float:
    """Return ``td.get(key).float().mean().item()`` if present and finite, else NaN.

    Used by :class:`GraphSNDLoggingCallback` to surface diagnostic quantities
    (``scaling_ratio``, ``out_loc_norm``) that :class:`HetControlMlpEmpirical`
    writes into the per-forward tensordict. Depending on the algorithm and
    BenchMARL version, these keys may or may not survive into the training_td
    seen by ``on_train_end``; we return NaN defensively so the row is still
    written rather than the whole iteration being lost.
    """
    value = _safe_tensordict_get(td, key)
    if value is None:
        return float("nan")
    try:
        mean = float(value.float().mean().item())
    except Exception:
        return float("nan")
    return mean if math.isfinite(mean) else float("nan")
