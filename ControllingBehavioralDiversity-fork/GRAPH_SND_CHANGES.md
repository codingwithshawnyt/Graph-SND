# Graph-SND changes in this DiCo fork

This fork integrates **uniform-weight Graph-SND** (Remark 8 / Theorem 9 in the
Graph-SND paper) as a drop-in replacement for full SND in DiCo’s diversity
control loop. Training, losses, and rollouts are otherwise unchanged.

## New files

| Path | Purpose |
| :--- | :--- |
| `het_control/graph_snd.py` | Bernoulli edge sampling, `compute_graph_snd_uniform` (reuses DiCo’s `compute_statistical_distance`), `compute_diversity` dispatch, `time_diversity_call` with CUDA synchronization for wall-clock timing. Exposes `get_graph_rng` / `reseed_graph_rng` using a **weak-key registry** so `torch.Generator` is not stored on `nn.Module` (TorchRL deep-copies the actor for PPO loss). |
| `het_control/conf/navigation_ippo_full_config.yaml` | Hydra preset: full SND + CSV log path. |
| `het_control/conf/navigation_ippo_graph_p01_config.yaml` | Hydra preset: Graph-SND at `p = 0.1`. |
| `het_control/conf/navigation_ippo_graph_p025_config.yaml` | Hydra preset: Graph-SND at `p = 0.25`. |
| `tests/test_graph_snd_recovers_full_snd.py` | Asserts `p = 1.0` sampling recovers full SND (sanity check on DiCo’s codepath). |
| `scripts/plot_graph_dico.py` | Two-panel comparison figure from three CSV logs (full vs graph variants). |

## Modified files

| Path | Change |
| :--- | :--- |
| `het_control/models/het_control_mlp_empirical.py` | `estimate_snd` calls `time_diversity_call(compute_diversity, …)` instead of only `compute_behavioral_distance`; adds `diversity_estimator`, `diversity_p`, and passes `get_graph_rng(self)` into `compute_diversity` (generator not stored on the module). |
| `het_control/run.py` | Registers `GraphSNDLoggingCallback` when `graph_snd_log_path` is set. |
| `het_control/callback.py` | Adds `GraphSNDLoggingCallback`: per-iteration CSV (including `scaling_ratio_mean`, `applied_snd`, `out_loc_norm_mean`), `reseed_graph_rng` each iter. Control columns are read from **per-iteration aggregates on** `HetControlMlpEmpirical` (mean over grad-mode forwards), with fallback to `training_td` if present. |
| `het_control/models/het_control_mlp_empirical.py` | Same Graph-SND `estimate_snd` dispatch as above; adds forward-path sums for CSV diagnostics and optional ``HET_CONTROL_DICO_DEBUG_FORWARDS=N`` stderr lines (first ``N`` grad-mode estimate forwards). |
| `het_control/conf/model/hetcontrolmlpempirical.yaml` | Defaults `diversity_estimator: full`, `diversity_p: 1.0`, and `desired_snd: 0.5` so Hydra runs do not leave `desired_snd` as null (which breaks `torch.tensor([desired_snd], …)` in `HetControlMlpEmpirical.__init__`). |
| `het_control/conf/navigation_ippo_config.yaml` | Adds `graph_snd_log_path: null`. |

## Evaluation rendering

The three `navigation_ippo_*_config.yaml` presets set **`experiment.render: false`**
so runs do not invoke DiCo’s navigation `render_callback` (SND heatmap overlay in
`het_control/environments/vmas.py`). That callback can trigger
`assert outputs.shape[0] == pos_shape[0]` inside VMAS for some agent counts during
RGB evaluation; disabling render avoids the failure on headless servers while
leaving training and CSV logging unchanged. Override with `experiment.render=true`
when you need eval videos locally.

## Not modified

- `het_control/snd.py` (Wasserstein / pairwise distance kernel) — imported, not rewritten.
- `SndCallback` evaluation SND path — unchanged (training-time diversity dispatch is in `estimate_snd` only).
- VMAS, TensorDict, TorchRL, BenchMARL — not vendored here; install per upstream DiCo README.
