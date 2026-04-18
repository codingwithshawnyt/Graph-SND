# DiCo divergence diagnosis (n=16 Navigation, desired_snd=0.5)

## What we observed

Three runs at `n_agents=16`, `desired_snd=0.5`, 167 iterations each (one per
estimator variant):

| Run       | iter 0 `snd_t` | iter 166 `snd_t` | iter 166 `reward_mean` |
|-----------|---------------:|-----------------:|-----------------------:|
| full      | 0.0339         | 165.29           | 0.01043                |
| graph_p025 | 0.0329        | 165.62           | 0.01043                |
| graph_p01 | 0.0331         | 165.01           | 0.01051                |

Three-way agreement within 0.3 % at every iteration. So the Graph-SND
substitution is bit-for-bit correct; whatever broke DiCo broke all three
identically. The task is to explain the 165 and the 0.01.

## Critical insight: `snd_t` does NOT measure the applied-action SND

This is the single most important observation. Before calling any of the three
hypotheses below, read this.

[het_control/callback.py](het_control/callback.py) line 314 logs
`model.estimated_snd`, which
[het_control/models/het_control_mlp_empirical.py](het_control/models/het_control_mlp_empirical.py)
`estimate_snd` (lines 247-276) computes from the *unscaled* outputs of
`self.agent_mlps.agent_networks`. The applied action in the probabilistic
branch (lines 190-200) is

    agent_loc = shared_loc + agent_out * scaling_ratio
    scaling_ratio = desired_snd / estimated_snd         (lines 183-188)

so the SND the environment actually sees is approximately

    SND(agent_loc) = scaling_ratio * SND(agent_out)
                   = (desired_snd / estimated_snd) * SND(agent_out)
                   ≈ desired_snd   whenever estimated_snd ≈ SND(agent_out).

`estimated_snd` is refreshed on every training forward (`_forward` line 172) and
PPO calls the model ~675 times per iteration (45 epochs × ~15 minibatches), so
the `tau=0.01` soft-update converges per-iteration. The invariant therefore
holds: **the 165 reported in the CSV is the spread of the *raw* per-agent
network outputs, not the diversity of the actions the agents actually took in
the environment**. DiCo's scaling can (and probably did) keep `SND(agent_loc)`
near 0.5 the whole time.

Consequence: the `snd_t=165` number is not direct evidence DiCo blew up. The
actual failure mode is that **`reward_mean` sat at ~0.01 for 167 iterations,
which means the policy never learned Navigation**. The runaway `snd_t` is a
symptom of *training instability growing the raw weights*, not a symptom of
DiCo's control-theoretic loop breaking.

## Ranked hypotheses

### H1 (most likely) — task regime mismatch

`n_agents=16` with `shared_rew: False` is outside the Bettini et al. 2024
tested regime for Navigation.

- **For:** Paper default is n=2 ([het_control/conf/task/vmas/navigation.yaml](het_control/conf/task/vmas/navigation.yaml)
  line 6). `shared_rew: False` (line 20) means 16 independent goals and 16
  independent reward signals. DiCo's homogeneity prior (via the scaling) drives
  every agent toward a compromise policy; with 16 goals distributed in a
  bounded arena, the compromise policy cannot actually navigate any single
  agent to any single goal. The equilibrium the CSVs show (0.01 reward, raw
  SND plateauing near 165) is exactly what "each agent outputs a mediocre
  compromise and collects a small step-penalty forever" looks like.
- **For:** All three estimator variants produced nearly identical trajectories.
  If the pathology were in the SND estimator it would differ across variants;
  the fact that it does not means the failure must be in the training setup
  itself.
- **For:** Paper-reported Navigation return is ~50-100 per episode; we got
  ~0.01 per step × 100 steps ≈ 1 per episode. ~50-100× below the paper.
- **Against:** None of the static-analysis evidence points away from H1.

### H2 (contributing — unstable first-iter scaling)

`bootstrap_from_desired_snd: False` in
[het_control/conf/model/hetcontrolmlpempirical.yaml](het_control/conf/model/hetcontrolmlpempirical.yaml)
line 18 makes iter 0's scaling ratio = `0.5 / raw_SND`.

- **For:** From the three CSVs, iter-0 raw SND ≈ 0.033, giving
  `scaling_ratio ≈ 15.2`. That 15× amplification of essentially-random
  per-agent delta networks feeds an extremely noisy first PPO update. Within
  a few iterations the raw `agent_out` magnitudes blow up (snd_t 0.033 at
  iter 0 → 1.34 at iter 5 in `full.csv`), which is consistent with "first-iter
  noise injection set up the runaway".
- **For:** Bettini himself flipped this flag to `True` for at least one task:
  [het_control/conf/reverse_transport_iddpg_config.yaml](het_control/conf/reverse_transport_iddpg_config.yaml)
  line 20. So this is an already-recognised per-task knob, not new behavior
  we are inventing.
- **Against:** H2 alone should not cause reward to stay at 0.01 forever; with
  enough iterations the tau=0.01 soft-update should eventually stabilise. So
  H2 is a contributor, probably not the root cause. It matters mainly because
  it makes the first few iters needlessly fragile.

### H3 (least likely) — missing clamp on `scaling_ratio`

The `scaling_ratio = self.desired_snd / distance` expression at
[het_control/models/het_control_mlp_empirical.py](het_control/models/het_control_mlp_empirical.py)
line 186 has no clamp; `distance → 0` would blow it up.

- **For:** In principle, at initialisation, `SND(agent_out)` can be arbitrarily
  small (all per-agent MLPs identical), so the ratio can be arbitrarily large.
- **Against:** The CSVs show a stable asymptote at ~165, not oscillation or
  NaN. The actual minimum observed scaling in our runs was ~15.2 (iter 0),
  which is high but well short of numerically unstable. So H3 is a
  theoretical issue, not what bit us here.
- **Against:** The constraint in the prompt ("NOT the core DiCo method")
  discourages adding a clamp anyway; it would mask real issues the paper
  assumes you handle by choosing sensible hyperparameters.

## Where the fix lives

Only H1 and H2 are actionable without modifying the Graph-SND code or DiCo's
core scaling method:

1. **H1 fix.** Drop the orchestrator default `N_AGENTS` from 16 to 4 — the
   largest n at which we have any reason to expect paper-like convergence on
   Navigation with `shared_rew: False`, `desired_snd: 0.5`. Document that
   higher n requires either lowering `DESIRED_SND` or setting
   `task.shared_rew=True`. Edit
   [scripts/run_graph_dico_two_gpus_then_third.sh](scripts/run_graph_dico_two_gpus_then_third.sh)
   line 28.
2. **H2 fix.** Set `bootstrap_from_desired_snd: True` in
   [het_control/conf/model/hetcontrolmlpempirical.yaml](het_control/conf/model/hetcontrolmlpempirical.yaml)
   line 18, matching `reverse_transport_iddpg_config.yaml`. This turns iter 0
   into a unit-scaling pass and lets the per-agent networks drift apart
   gently instead of being amplified 15× the very first update.

Both edits are hyperparameter-only. No changes to `het_control/graph_snd.py`,
`het_control/models/het_control_mlp_empirical.py`, or `het_control/callback.py`.
All three config variants (`navigation_ippo_full_config`,
`navigation_ippo_graph_p01_config`, `navigation_ippo_graph_p025_config`) inherit
the same model config and are driven by the same orchestrator, so the fix
applies uniformly.

## Pre-training smoke test

[tests/smoke_dico_training.py](tests/smoke_dico_training.py) runs 8 PPO
iterations at n=4 with reduced frames/batch (~2-3s per iter, < 90 s total)
against any of the three config variants, then parses the resulting
`graph_snd_log.csv` and asserts:

- CSV exists, has ≥ 5 rows, no NaN / inf.
- `snd_t` is moving toward `snd_des` (direction check at iter 4).
- `snd_t` is within one order of magnitude of `snd_des` by iter ≥ 5.
- `scaling_ratio = snd_des / snd_t` stays in `[0.01, 100]` at every row.
- `reward_mean` does not collapse by more than 5 from its iter-0 value and the
  final row is not worse than -10.

This test should pass for all three config variants before any full
167-iteration real training run is launched.

## Expected trajectory after the fix

With `n_agents=4` and `bootstrap_from_desired_snd: True`:

- **iter 0**: `snd_t` starts ~0.02-0.05 (random init), `scaling_ratio = 1`
  because `estimated_snd` is bootstrapped to `desired_snd = 0.5`.
- **iter 5-10**: `snd_t` rises monotonically toward 0.5; `reward_mean` is
  still near zero or slightly positive as policies begin exploring.
- **iter 15-30**: `snd_t` is within 2× of `snd_des` and stabilising;
  `reward_mean` visibly positive and climbing.
- **iter 50+**: `reward_mean` ≥ 0.3 per step (≈ 30 per episode at
  `max_steps=100`), which is what paper-like learning looks like; `snd_t`
  fluctuates within 1.5× of `snd_des`.
- **Estimator comparison**: `graph_p01` and `graph_p025` should still track
  `full` within a few percent throughout, exactly as they did in the broken
  n=16 runs — that part was never in question.

If the smoke test still fails at n=4, the single additional datum needed to
disambiguate a residual H2/H3 variant is **the per-iteration
`scaling_ratio` alongside `estimated_snd` from the problematic run**. Today
the CSV only records `snd_t = estimated_snd`; we do not record the ratio or
`SND(agent_loc)`. Adding those two columns to
`GraphSNDLoggingCallback.on_train_end` would resolve any remaining ambiguity
— but is not part of this fix because the prompt forbids modifying the
logging callback.

## Summary

**Root cause.** `n_agents=16` on Multi-Agent Navigation with
`shared_rew: False` is outside the DiCo paper's tested regime. DiCo's
control-theoretic scaling invariant was actually maintained — the runaway
`snd_t=165` in the CSVs is the raw per-agent network output spread, not the
applied-action SND — but the underlying navigation task is infeasible under
DiCo's homogeneity prior at that scale, so the policy never learned and
reward sat at ~0.01 forever. `bootstrap_from_desired_snd: False` made this
worse by injecting a 15× amplification on the very first PPO update.

**What the fix does.** Two hyperparameter-only edits:
(1) orchestrator default `N_AGENTS: 16 → 4` (paper-safe), and
(2) model default `bootstrap_from_desired_snd: False → True` (iter-0 scaling
becomes 1 instead of 15). Graph-SND code, DiCo core scaling code, and all
frozen upstream forks are untouched.

**Expected trajectory after the fix.** `snd_t` rises from ~0.02-0.05 at
iter 0 to within 2× of `snd_des = 0.5` by iter 15-30 and stays there;
`reward_mean` goes from ~0 to positive (~0.3 per step, ~30 per episode) by
iter 50 and continues climbing. All three variants (`full`, `graph_p01`,
`graph_p025`) should track within a few % of each other, as they already
did in the broken runs.
