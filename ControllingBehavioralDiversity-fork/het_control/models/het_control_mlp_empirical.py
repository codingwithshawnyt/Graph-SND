#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

import os
from dataclasses import dataclass, MISSING
from typing import Type, Sequence, Optional, Tuple

import torch
from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor
from torch import nn
from torchrl.modules import MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig
from het_control.graph_snd import compute_diversity, get_graph_rng, time_diversity_call
from het_control.snd import compute_behavioral_distance
from het_control.utils import overflowing_logits_norm
from .utils import squash


class HetControlMlpEmpirical(Model):
    def __init__(
        self,
        activation_class: Type[nn.Module],
        num_cells: Sequence[int],
        desired_snd: float,
        probabilistic: bool,
        scale_mapping: Optional[str],
        tau: float,
        bootstrap_from_desired_snd: bool,
        process_shared: bool,
        diversity_estimator: str = "full",
        diversity_p: float = 1.0,
        diversity_knn_k: int = 3,
        **kwargs,
    ):
        """DiCo policy model

        Args:
            activation_class (Type[nn.Module]): activation class to be used.
            num_cells (int or Sequence[int], optional): number of cells of every layer in between the input and output. If
            an integer is provided, every layer will have the same number of cells. If an iterable is provided,
            the linear layers out_features will match the content of num_cells.
            desired_snd (float): The desired SND diversity
            probabilistic (bool): Whether the model has stochastic actions or not.
            scale_mapping (str, optional): Type of mapping to use to make the std_dev output of the policy positive
            (choices: "softplus", "exp", "relu", "biased_softplus_1")
            tau (float): The soft-update parameter of the estimated diversity. Must be between 0 and 1
            bootstrap_from_desired_snd (bool): Whether on the first iteration the estimated SND should be bootstrapped
            from the desired snd (True) or from the measured SND (False)
            process_shared (bool): Whether to process the homogeneous part of the policy with a tanh squashing operation to the action space domain
            diversity_estimator (str): Which SND estimator drives the DiCo scaling factor. One of ``"full"``
            (the original C(n, 2) mean), ``"graph_p01"`` or ``"graph_p025"`` (uniform-weight Graph-SND
            on a Bernoulli(p) subgraph), or ``"knn"`` (uniform-weight Graph-SND on a dynamic k-NN
            spatial graph). Defaults to ``"full"`` for backward compatibility.
            diversity_p (float): Bernoulli inclusion probability consumed by the Graph-SND path.
            Ignored when ``diversity_estimator in {"full", "knn"}``.
            diversity_knn_k (int): Number of nearest neighbours for the k-NN graph. Only used
            when ``diversity_estimator == "knn"``. Defaults to 3.
        """

        super().__init__(**kwargs)

        self.num_cells = num_cells
        self.activation_class = activation_class
        self.probabilistic = probabilistic
        self.scale_mapping = scale_mapping
        self.tau = tau
        self.bootstrap_from_desired_snd = bootstrap_from_desired_snd
        self.process_shared = process_shared
        self.diversity_estimator = diversity_estimator
        self.diversity_p = float(diversity_p)
        self.diversity_knn_k = int(diversity_knn_k)
        # Bernoulli RNG lives in ``het_control.graph_snd.get_graph_rng(self)`` (weak-key
        # registry), not on this module, so TorchRL can ``deepcopy`` the actor for
        # PPO loss. GraphSNDLoggingCallback reseeds via ``reseed_graph_rng`` each iter.

        self.register_buffer(
            name="desired_snd",
            tensor=torch.tensor([desired_snd], device=self.device, dtype=torch.float),
        )  # Buffer for SND_{des}
        self.register_buffer(
            name="estimated_snd",
            tensor=torch.tensor([float("nan")], device=self.device, dtype=torch.float),
        )  # Buffer for \widehat{SND}

        self.scale_extractor = (
            NormalParamExtractor(scale_mapping=scale_mapping)
            if scale_mapping is not None
            else None
        )  # Components that maps std_dev according to scale_mapping

        self.input_features = self.input_leaf_spec.shape[-1]
        self.output_features = self.output_leaf_spec.shape[-1]

        self.shared_mlp = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=False,
            share_params=True,  # Parameter-shared
            device=self.device,
            activation_class=self.activation_class,
            num_cells=self.num_cells,
        )  # Shared network that outputs mean and std_dev in stochastic policies and just mean in deterministic policies

        agent_outputs = (
            self.output_features // 2 if self.probabilistic else self.output_features
        )
        self.agent_mlps = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=agent_outputs,
            n_agents=self.n_agents,
            centralised=False,
            share_params=False,  # Not parameter-shared
            device=self.device,
            activation_class=self.activation_class,
            num_cells=self.num_cells,
        )  # Per-agent networks that output mean deviations from the shared policy

        # Per-iteration sums for CSV diagnostics. BenchMARL's ``on_train_end``
        # ``training_td`` often does not retain ``(group, "scaling_ratio")`` /
        # ``out_loc_norm`` for IPPO (and similarly for MADDPG), so
        # :class:`GraphSNDLoggingCallback` aggregates these from every
        # ``torch.is_grad_enabled()`` forward in the iteration instead.
        self.reset_csv_metric_accumulators()
        self._dico_debug_prints_done = 0

    def reset_csv_metric_accumulators(self) -> None:
        """Clear sums used by :class:`GraphSNDLoggingCallback` (call once per iter)."""
        self._csv_forward_count = 0
        self._csv_scaling_ratio_sum = 0.0
        self._csv_out_loc_norm_sum = 0.0

    def consume_csv_metric_means(self) -> Tuple[float, float]:
        """Return mean ``scaling_ratio`` and ``out_loc_norm`` for this iter, then reset.

        If no grad-enabled forwards ran this iteration, returns ``(nan, nan)``.
        """
        if getattr(self, "_csv_forward_count", 0) <= 0:
            sr_mean = float("nan")
            ol_mean = float("nan")
        else:
            n = float(self._csv_forward_count)
            sr_mean = self._csv_scaling_ratio_sum / n
            ol_mean = self._csv_out_loc_norm_sum / n
        self.reset_csv_metric_accumulators()
        return sr_mean, ol_mean

    def _perform_checks(self):
        super()._perform_checks()

        if self.centralised or not self.input_has_agent_dim:
            raise ValueError(f"{self.__class__.__name__} can only be used for policies")

        # Run some checks
        if self.input_has_agent_dim and self.input_leaf_spec.shape[-2] != self.n_agents:
            raise ValueError(
                "If the MLP input has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )
        if (
            self.output_has_agent_dim
            and self.output_leaf_spec.shape[-2] != self.n_agents
        ):
            raise ValueError(
                "If the MLP output has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )

    def _forward(
        self,
        tensordict: TensorDictBase,
        agent_index: int = None,
        update_estimate: bool = True,
        compute_estimate: bool = True,
    ) -> TensorDictBase:
        # Gather in_key

        input = tensordict.get(
            self.in_key
        )  # Observation tensor of shape [*batch, n_agents, n_features]
        shared_out = self.shared_mlp.forward(input)
        if agent_index is None:  # Gather outputs for all agents on the obs
            # tensor of shape [*batch, n_agents, n_actions], where the outputs
            # along the n_agent dimension are taken with the different agent networks
            agent_out = self.agent_mlps.forward(input)
        else:  # Gather outputs for one agent on the obs
            # tensor of shape [*batch, n_agents, n_actions], where the outputs
            # along the n_agent dimension are taken with the same (agent_index) agent network
            agent_out = self.agent_mlps.agent_networks[agent_index].forward(input)

        shared_out = self.process_shared_out(shared_out)

        if (
            self.desired_snd > 0
            and torch.is_grad_enabled()  # we are training
            and compute_estimate
            and self.n_agents > 1
        ):
            # Update \widehat{SND}
            distance = self.estimate_snd(input)
            if update_estimate:
                self.estimated_snd[:] = distance.detach()
        else:
            distance = self.estimated_snd
        if self.desired_snd == 0:
            scaling_ratio = 0.0
        elif (
            self.desired_snd == -1  # Unconstrained networks
            or distance.isnan().any()  # It is the first iteration
            or self.n_agents == 1
        ):
            scaling_ratio = 1.0
        else:  # DiCo scaling
            scaling_ratio = torch.where(
                distance != self.desired_snd,
                self.desired_snd / distance,
                1,
            )

        if self.probabilistic:
            shared_loc, shared_scale = shared_out.chunk(2, -1)

            # DiCo scaling
            agent_loc = shared_loc + agent_out * scaling_ratio
            out_loc_norm = overflowing_logits_norm(
                agent_loc, self.action_spec[self.agent_group, "action"]
            )  # For logging
            agent_scale = shared_scale

            out = torch.cat([agent_loc, agent_scale], dim=-1)
        else:
            # DiCo scaling
            out = shared_out + scaling_ratio * agent_out
            out_loc_norm = overflowing_logits_norm(
                out, self.action_spec[self.agent_group, "action"]
            )  # For logging

        # Aggregate control metrics across all grad-mode forwards this iteration
        # (BenchMARL's training_td at on_train_end often omits these keys for IPPO).
        if torch.is_grad_enabled():
            sr_tensor = (
                scaling_ratio
                if isinstance(scaling_ratio, torch.Tensor)
                else torch.tensor(
                    scaling_ratio, device=out.device, dtype=out.dtype
                )
            )
            self._csv_scaling_ratio_sum += float(
                sr_tensor.detach().float().mean().item()
            )
            self._csv_out_loc_norm_sum += float(
                out_loc_norm.detach().float().mean().item()
            )
            self._csv_forward_count += 1

        _dbg_n = int(os.environ.get("HET_CONTROL_DICO_DEBUG_FORWARDS", "0") or "0")
        if (
            _dbg_n > 0
            and torch.is_grad_enabled()
            and compute_estimate
            and self.n_agents > 1
            and self.desired_snd.item() > 0
        ):
            if self._dico_debug_prints_done < _dbg_n:
                with torch.no_grad():
                    if isinstance(distance, torch.Tensor):
                        dist_m = float(distance.detach().float().mean().item())
                    else:
                        dist_m = float(distance)
                    if isinstance(scaling_ratio, torch.Tensor):
                        sr_m = float(scaling_ratio.detach().float().mean().item())
                    else:
                        sr_m = float(scaling_ratio)
                    est_m = float(self.estimated_snd.detach().mean().item())
                    des_m = float(self.desired_snd.detach().mean().item())
                print(
                    "[HetControlMlpEmpirical DiCo debug] "
                    f"dist_mean={dist_m:.6f} est_mean={est_m:.6f} des={des_m:.6f} "
                    f"scale_mean={sr_m:.6f}",
                    flush=True,
                )
                self._dico_debug_prints_done += 1

        tensordict.set(
            (self.agent_group, "estimated_snd"),
            self.estimated_snd.expand(tensordict.get_item_shape(self.agent_group)),
        )
        tensordict.set(
            (self.agent_group, "scaling_ratio"),
            (
                torch.tensor(scaling_ratio, device=self.device).expand_as(out)
                if not isinstance(scaling_ratio, torch.Tensor)
                else scaling_ratio.expand_as(out)
            ),
        )
        tensordict.set((self.agent_group, "logits"), out)
        tensordict.set((self.agent_group, "out_loc_norm"), out_loc_norm)

        tensordict.set(self.out_key, out)

        return tensordict

    def process_shared_out(self, logits: torch.Tensor):
        if not self.probabilistic and self.process_shared:
            return squash(
                logits,
                action_spec=self.action_spec[self.agent_group, "action"],
                clamp=False,
            )
        elif self.probabilistic:
            loc, scale = self.scale_extractor(logits)
            if self.process_shared:
                loc = squash(
                    loc,
                    action_spec=self.action_spec[self.agent_group, "action"],
                    clamp=False,
                )
            return torch.cat([loc, scale], dim=-1)
        else:
            return logits

    # @torch.no_grad()
    def estimate_snd(self, obs: torch.Tensor):
        """
        Update \widehat{SND}
        """
        agent_actions = []
        # Gather what actions each agent would take if given the obs tensor
        for agent_net in self.agent_mlps.agent_networks:
            agent_outputs = agent_net(obs)
            agent_actions.append(agent_outputs)

        # Build per-env k-NN positions when the knn estimator is active.
        # obs has shape [*batch, n_agents, obs_dim]. We flatten all leading
        # batch dims to get [B, n_agents, obs_dim], then slice [:, :, :2]
        # to extract agent (x, y) for every parallel environment.
        # compute_knn_diversity_per_env builds a *different* k-NN graph per
        # environment so that agents only diversify from their true spatial
        # neighbours in each env, not from a single snapshot of env 0.
        knn_positions = None
        if self.diversity_estimator == "knn":
            obs_flat = obs.reshape(-1, self.n_agents, obs.shape[-1])
            knn_positions = obs_flat[:, :, :2].detach().float().cpu()

        # Dispatch to full SND or Graph-SND based on the configured estimator.
        # ``time_diversity_call`` wraps the call in ``torch.cuda.synchronize()``
        # on each side and records the elapsed ms into a module-level buffer,
        # which ``GraphSNDLoggingCallback`` drains once per PPO iteration.
        distance_scalar, _elapsed_ms = time_diversity_call(
            compute_diversity,
            agent_actions,
            estimator=self.diversity_estimator,
            p=self.diversity_p,
            rng=get_graph_rng(self),
            just_mean=True,
            knn_k=self.diversity_knn_k,
            knn_positions=knn_positions,
        ) # Compute the SND of these unscaled policies (scalar tensor)
        distance = distance_scalar.unsqueeze(-1)
        if self.estimated_snd.isnan().any():  # First iteration
            distance = self.desired_snd if self.bootstrap_from_desired_snd else distance
        else:
            # Soft update of \widehat{SND}
            distance = (1 - self.tau) * self.estimated_snd + self.tau * distance

        return distance


@dataclass
class HetControlMlpEmpiricalConfig(ModelConfig):
    activation_class: Type[nn.Module] = MISSING
    num_cells: Sequence[int] = MISSING

    desired_snd: float = MISSING
    tau: float = MISSING
    bootstrap_from_desired_snd: bool = MISSING
    process_shared: bool = MISSING

    probabilistic: Optional[bool] = MISSING
    scale_mapping: Optional[str] = MISSING

    # Graph-SND dispatch knobs. Defaults keep the unmodified DiCo path active.
    diversity_estimator: str = "full"
    diversity_p: float = 1.0
    diversity_knn_k: int = 3

    @staticmethod
    def associated_class():
        return HetControlMlpEmpirical
