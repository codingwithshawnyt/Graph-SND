# Copyright (c) 2024 ProrokLab; Graph-SND fork diagnostic.
"""Full SND at *random policy init* on VMAS Dispersion (no training).

This mirrors the quantity DiCo uses inside ``HetControlMlpEmpirical.estimate_snd``:
for each per-agent private MLP index ``i``, evaluate ``_apply_one_agent_net(i, obs)``
on a batch of observations, then take ``compute_diversity(..., estimator="full",
just_mean=True)`` (mean pairwise L2 / Wasserstein-on-means over agent pairs and
batch indices).  It does **not** use the post-scaling logits that PPO samples from.

Typical use (from ``ControllingBehavioralDiversity-fork/``, same Hydra overrides as
``scripts/launch_dico_n50_feasibility.sh`` for task shape; headless loggers)::

    SND_DIAG_N_RESETS=32 python het_control/run_scripts/measure_random_init_dispersion_full_snd.py \\
        experiment.loggers=[] \\
        graph_snd_log_path=null \\
        experiment.max_n_iters=1 \\
        experiment.evaluation=false \\
        experiment.render=false \\
        experiment.on_policy_collected_frames_per_batch=12000 \\
        experiment.on_policy_n_envs_per_worker=120 \\
        experiment.train_device=cuda:0 \\
        experiment.sampling_device=cuda:0 \\
        experiment.buffer_device=cpu \\
        seed=0 \\
        task.n_agents=50 \\
        task.n_food=50 \\
        task.share_reward=true \\
        model.desired_snd=-1 \\
        model.diversity_estimator=full
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra
import torch
from omegaconf import DictConfig

from het_control.callback import get_het_model
from het_control.graph_snd import compute_diversity
from het_control.run import get_experiment


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="dispersion_ippo_knn_config",
)
def main(cfg: DictConfig) -> None:
    n_resets = int(os.environ.get("SND_DIAG_N_RESETS", "32"))
    if n_resets < 1:
        raise ValueError("SND_DIAG_N_RESETS must be >= 1")

    experiment = get_experiment(cfg=cfg)
    if len(experiment.group_map) != 1:
        raise RuntimeError(
            f"Expected a single agent group for Dispersion, got {experiment.group_map}"
        )
    (group,) = tuple(experiment.group_map.keys())

    policy = experiment.group_policies[group]
    het = get_het_model(policy)
    het.eval()

    device = torch.device(experiment.config.sampling_device)
    env = experiment.env_func().to(device)

    values: list[float] = []
    with torch.no_grad():
        for _ in range(n_resets):
            tensordict = env.reset()
            obs = tensordict.get((group, "observation"))
            if obs.device != het.device:
                obs = obs.to(het.device, non_blocking=True)

            agent_actions = []
            n_nets = 1 if het.agent_mlps.share_params else het.n_agents
            for agent_i in range(n_nets):
                agent_actions.append(het._apply_one_agent_net(agent_i, obs))

            snd = compute_diversity(
                agent_actions,
                estimator="full",
                just_mean=True,
            )
            values.append(float(snd.detach().cpu().item()))

    mean_v = sum(values) / len(values)
    var_v = sum((x - mean_v) ** 2 for x in values) / max(len(values) - 1, 1)
    std_v = var_v**0.5

    n_agents = int(cfg.task.n_agents)
    print(
        "\n=== Random-init full SND (DiCo ``estimate_snd`` tensor definition) ===\n"
        f"  task: vmas/dispersion  n_agents={n_agents}  n_resets={n_resets}  seed={int(cfg.seed)}\n"
        f"  mean full SND: {mean_v:.6f}\n"
        f"  std  (sample): {std_v:.6f}\n"
        f"  min / max:      {min(values):.6f} / {max(values):.6f}\n"
        "\nRebuttal one-liner (adapt wording): at random init, heterogeneous per-agent\n"
        "MLP deviations already yield mean pairwise action-space distance on the order\n"
        f"of ~{mean_v:.2f} under the same measurement as training-time DiCo; targeting\n"
        "SND_des=0.1 therefore requires strong throttling (DiCo set-point / controller),\n"
        "not a limitation of the Graph-SND surrogate itself.\n"
    )


if __name__ == "__main__":
    main()
