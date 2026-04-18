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

**Update (instrumented runs, n=2 Navigation).** With per-iteration
`scaling_ratio_mean` and `applied_snd` in `graph_snd_log.csv`, a healthy
paper-style run can show **raw `snd_t` growing large** while
`mean(scaling_ratio)` shrinks so **`applied_snd` stays near `snd_des`**
(roughly `snd_t × scaling_ratio_mean ≈ snd_des`). Do not judge the control
loop from raw `snd_t` alone; use `applied_snd` (and reward) as the primary
outcomes. The pre-training smoke test
([tests/smoke_dico_training.py](tests/smoke_dico_training.py)) checks that
the last-10-iter mean of `applied_snd` is within tolerance of `snd_des`, not
that raw `snd_t` moves toward the target.

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

## Postmortem: the n=4 fix did not work

The original analysis above is preserved verbatim as the record of a
wrong-but-internally-consistent first attempt. After launching the three
n=4 runs with `bootstrap_from_desired_snd: True`, they failed in the exact
same way as the n=16 runs — same plateau, same reward floor. H1 and H2
were both wrong. The real root cause is `desired_snd = 0.5` itself.

### What we re-observed at n=4

Three runs at `n_agents=4`, `desired_snd=0.5`,
`bootstrap_from_desired_snd=True`, 167 iterations each:

| Run        | iter 0 `snd_t` | iter 166 `snd_t` | iter 166 `reward_mean` |
|------------|---------------:|-----------------:|-----------------------:|
| full       | 0.0561         | ~311             | ~0.0116                |
| graph_p025 | 0.056x         | ~311             | ~0.0116                |
| graph_p01  | 0.056x         | ~311             | ~0.0116                |

Same tight three-way agreement, same 300×-of-target `snd_t` plateau, same
stuck-at-zero reward. Reducing `n_agents` from 16 to 4 did not move the
needle: **H1 is falsified**.

### Why H2 was a no-op at real-run granularity

The key datum is `iter 0 snd_t = 0.0561`, not `0.5`. If
`bootstrap_from_desired_snd: True` were actually taking effect at real-run
granularity, the first CSV row should have recorded `snd_t ≈ snd_des = 0.5`,
not the raw init value.

The reason it did not: look at the soft-update in
[het_control/models/het_control_mlp_empirical.py](het_control/models/het_control_mlp_empirical.py)
line 274:

    distance = (1 - self.tau) * self.estimated_snd + self.tau * distance
             = 0.99 * estimated_snd + 0.01 * raw_measurement

On iter 0, `self.estimated_snd` is seeded with `self.desired_snd = 0.5`
(the bootstrap). But `_forward` runs this update on every training forward,
and the PPO inner loop at the real-run config
(`on_policy_n_minibatch_iters=45`, ~15 minibatches per iter) invokes
`_forward` ~675 times *within iter 0*. The residual bootstrap contribution
to `estimated_snd` at the end of iter 0 is therefore

    0.5 * 0.99^675 ≈ 0.5 * 1.1e-3 ≈ 6e-4,

i.e. the bootstrap has fully decayed to the raw measurement before the
first CSV row is written. The log-callback fires `on_train_end`, so the
CSV never sees the bootstrapped value — it sees the raw measurement the
soft-update converged to. **H2 is falsified as a meaningful lever at
this PPO configuration.**

The flag is still correct-by-design for tasks like
`reverse_transport_iddpg`, which has a much smaller number of forwards per
training iteration and so preserves the bootstrap. But for
`navigation_ippo_config` with `n_minibatch_iters=45`, the flag is a no-op
at the iteration granularity the CSV logs, and setting it to `True` has no
observable effect.

### Why the smoke test passed the broken setup

The 12-iter smoke test in the previous plan used
`on_policy_n_minibatch_iters=5` and `minibatch_size=2048` at
`collected_frames_per_batch=6000`, yielding ~15 `_forward` calls per iter.
That leaves

    0.99^15 ≈ 0.86

so the bootstrap retains 86% of its weight across iter 0 of the smoke
test — the smoke test saw `snd_t[0] ≈ 0.44` (bootstrap-dominated) while
real runs saw `snd_t[0] ≈ 0.05` (raw-dominated). Completely different
starting conditions, same code. The smoke test's `stayed_near_target`
fallback at `check 5` then triggered on the (close-to-target) smoke value,
declaring the direction check passed — exactly the combination that lets
a broken setup slip through.

This is the second and third failure of the previous attempt:
- **H2-reasoning failure**: we inferred the wrong dynamics from the
  smoke-test `snd_t[0]` instead of the real-run value.
- **Smoke-test failure**: our direction check had a loophole specifically
  calibrated to the bootstrap-dominated regime the smoke test accidentally
  produces; it would not catch the raw-init regime real runs use.

### The actual root cause: `desired_snd = 0.5` is outside the paper's tested regime

The DiCo README's own Navigation examples use:

| README line | `desired_snd` value | Context                                 |
|-------------|---------------------|-----------------------------------------|
| 49          | `0.1`               | IPPO Navigation reproduction example    |
| 59          | `0.3`               | MADDPG Navigation reproduction example  |
| 64          | `{-1, 0, 0.3}`      | Reported Navigation diversity sweep     |
| 100         | `0.3`               | IPPO Navigation reproduction example    |

0.5 appears nowhere — not in the README, not in any of the upstream
config files. It was invented in this fork when we set a concrete default
for `desired_snd` in `hetcontrolmlpempirical.yaml` (originally `null`, see
`GRAPH_SND_CHANGES.md` line 25). At 0.5 on the n=2-to-n=4 Navigation scale
we are asking for a spread the policy network cannot produce consistently
under PPO's trust-region step sizes. The scaling invariant
`SND(agent_loc) = scaling_ratio × SND(agent_out)` still holds at every
forward, but PPO grows raw `agent_out` unboundedly to try to drag
`SND(agent_loc)` up to `0.5` — `snd_t` climbs 5.3% per iter, reaches ~311
by iter 166, and `reward_mean` is the loser: the raw outputs are now
dominated by magnitude growth in service of the DiCo target, not by
task-relevant spatial navigation.

### Three new edits

1. **Lower `desired_snd` from 0.5 to 0.1** (paper README line 49) in:
   - [het_control/conf/model/hetcontrolmlpempirical.yaml](het_control/conf/model/hetcontrolmlpempirical.yaml)
     line 13
   - [scripts/run_graph_dico_two_gpus_then_third.sh](scripts/run_graph_dico_two_gpus_then_third.sh)
     `DESIRED_SND` default
   - [tests/smoke_dico_training.py](tests/smoke_dico_training.py)
     `SMOKE_OVERRIDES`
2. **Revert `bootstrap_from_desired_snd: True → False`** in
   [het_control/conf/model/hetcontrolmlpempirical.yaml](het_control/conf/model/hetcontrolmlpempirical.yaml).
   It is a no-op at real-run granularity (see above). The explicit `False`
   matches the paper's implicit default for Navigation (upstream
   `navigation_ippo_config.yaml` does not override the flag). Leaving it
   `True` would give a false sense that we had a knob to tune iter-0
   behavior for this task; we do not.
3. **Keep `N_AGENTS=4` in the orchestrator**, but rewrite the justifying
   comment. n=4 is NOT a stability fix — `desired_snd=0.5` at n=4 diverged
   just as hard as at n=16. n=4 is kept purely so the
   between-estimator comparison (`full` vs `graph_p01` vs `graph_p025`)
   has breadth over the n=2 paper baseline.

### Smoke-test rewrite

[tests/smoke_dico_training.py](tests/smoke_dico_training.py) has been
rewritten so the failure mode above cannot slip through again:

- Bumped `max_n_iters` 12 → 20; dropped `on_policy_n_envs_per_worker`
  60 → 30 to hold the budget under 2 minutes.
- Deleted the `stayed_near_target` clause in the direction check.
- Added a **runaway check**: `max(snd_t[-5:]) / mean(snd_t[5:10]) ≤ 1.5`.
  A healthy run plateaus near `snd_des` so the ratio is ~1.0; the broken
  `desired_snd=0.5` dynamics produces a ratio ≈ 1.76 within 20 iters.
- Tightened the final-iter magnitude check from a 10× window to a 3×
  window: `0.3 × snd_des ≤ snd_t[-1] ≤ 3.0 × snd_des`.
- Replaced "no catastrophic collapse" with a **learning-signal** check:
  `mean(reward[-5:]) − mean(reward[:5]) ≥ 0.001`. The old check let any
  run with a tiny positive slope pass; the new one requires the reward
  actually rise during the smoke window.
- Added an informational warning if `snd_t[0] > 2.0 × snd_des`, so users
  notice when the bootstrap regime differs from real runs.

The runaway check is the one that would have caught the broken setup at
20 iters.

### Confidence and escape hatch

The paper README alignment makes lowering `desired_snd` to 0.1 a
high-confidence fix (the upstream authors validated exactly this value on
this exact task). The tightened smoke test is moderate-high confidence; if
`desired_snd=0.1` still produces `snd_t` runaway or zero reward in real
runs, the next diagnostic step is to log the per-iter `scaling_ratio`
alongside `estimated_snd` in
[het_control/callback.py](het_control/callback.py) — a ratio collapsing to
zero would indicate a gradient-flow bug, not a hyperparameter problem.

### Updated expected trajectory (at `desired_snd = 0.1`)

- **iter 0**: raw `snd_t` starts ~0.03-0.06 (random init); `scaling_ratio`
  starts near `0.1 / 0.05 = 2`, small enough to not blow up PPO.
- **iter 5-15**: `snd_t` rises toward 0.1, stabilising in roughly
  [0.05, 0.15]. Smoke-test runaway check must hold here.
- **iter 15-30**: `snd_t` is within ~1.5× of `snd_des = 0.1`; `reward_mean`
  visibly positive and climbing — by iter 30 we expect ≥ ~0.05 per step.
- **iter 50+**: `reward_mean` in the 0.1-0.3 per-step range, consistent
  with paper-like Navigation learning at n=4.
- **iter 167 (full run)**: `reward_mean` plateaus near 0.3-0.5 per step,
  and `snd_t` fluctuates within [0.03, 0.3]. All three estimator variants
  should track within a few percent.

## Postmortem #2: three speculative fixes, no actual fix

The `desired_snd = 0.1` "fix" above is **retracted**. The text from the
first postmortem through `Updated expected trajectory (at desired_snd
= 0.1)` is preserved verbatim as the record of a second
wrong-but-internally-consistent attempt; treat every "expected
trajectory" bullet there as falsified.

### What the 20-iter smoke actually produced at `desired_snd=0.1`

Running the rewritten `tests/smoke_dico_training.py` against
`navigation_ippo_full_config` at `n=4`, `desired_snd=0.1`,
`bootstrap_from_desired_snd=False`:

| iter | `snd_t` | `reward_mean` |
|-----:|--------:|--------------:|
|    0 | 0.2606  | -0.0014       |
|    5 | 0.2820  |  0.0042       |
|   10 | 0.3322  |  0.0048       |
|   15 | 0.4378  |  0.0049       |
|   19 | 0.5427  |  0.0047       |

Direction check failed (`snd_t` moves away from `snd_des` between iter 0
and 5), runaway check failed
(`max(snd_t[15:20]) / mean(snd_t[5:10]) = 0.543 / 0.298 = 1.82 > 1.5`),
and the final-iter magnitude check failed
(`snd_t[-1] = 0.543` vs allowed window `[0.03, 0.30]`). The
reward-learning-signal check passed by 0.003 — barely, and not in a
shape that looks like actual learning. **The shape of the failure is
identical to the `desired_snd=0.5` runs**: monotone ~4%/iter growth of
raw `snd_t`, reward glued near zero. Lowering `desired_snd` from 0.5 to
0.1 only moved the absolute scale; it did not change the qualitative
dynamics.

### Why Postmortem #1's reasoning about iter 0 was still wrong

Postmortem #1 predicted `snd_t[0] ≈ 0.03-0.06` under
`bootstrap_from_desired_snd=False` because the soft-update should have
converged to the raw measurement within one real-run iteration. The
actual smoke value is 0.26 — about 5× what was predicted, and also
inconsistent with "soft-update converged to raw". The math behind the
actual observation:

- Smoke config runs ~15 forwards per iter (`n_minibatch_iters=5`,
  `minibatch_size=2048`, `collected_frames_per_batch=6000`).
- `0.99^15 ≈ 0.86`, so at iter 0's end the soft-update equals
  `0.86 × initial_seed + 0.14 × raw_measurement`.
- `0.86 × initial_seed + 0.14 × raw ≈ 0.26`, which solved for
  `initial_seed ≈ 0.30` if raw is much smaller.

So `estimated_snd` is effectively seeded with ~0.3 at construction time
(not zero, not `desired_snd`), and the smoke regime never equilibrates
away from that seed. The real-run regime (~675 forwards/iter, so
`0.99^675 ≈ 1.1e-3`) does equilibrate — but then we're in a completely
different dynamical system than the smoke is simulating. The smoke test
was therefore diagnosing a different process than the one real runs
execute; passing or failing there carries very little information about
real-run behavior. This is why three successive "postmortem fixes"
based on smoke-test readouts have all been wrong.

### The actual source is that this experiment is research extrapolation

User clarification: the Graph-SND paper does **not** include a
"Section 6.7 Graph-DiCo integration" experiment. Section 6.7 of that
paper is a future-work section that explicitly leaves testing
Graph-SND on DiCo-controlled training to later work.

- The **DiCo** paper validates Navigation at `n_agents=2` only. Every
  DiCo-paper n=4 experiment uses a different task (Dispersion with
  `share_rew=True`, Sampling, Reverse Transport).
  See [het_control/conf/task/vmas/navigation.yaml](het_control/conf/task/vmas/navigation.yaml)
  line 6 and the DiCo README's Navigation examples, which all use n=2.
- The **Graph-SND** paper uses Navigation at `n=4`/`8`/`16` with its
  **own** IPPO trainer and only *measures* SND per-checkpoint; it never
  routes Graph-SND through DiCo's `desired_snd / estimated_snd`
  scaling feedback loop.
- Our `navigation_ippo_*_config.yaml` variants run **DiCo's control
  loop** at n=4 individual-goal Navigation with Graph-SND as the
  diversity signal. That combination — DiCo-control + Graph-SND-signal
  + n=4 + individual goals + Navigation — is validated by neither
  paper.

The fork's `estimator="full"` codepath is bit-identical to upstream
DiCo at `het_control/models/het_control_mlp_empirical.py`. The
commented-out `# @torch.no_grad()` on `estimate_snd` that was suspected
earlier as a fork bug is present identically upstream — red herring.
So "the fork broke DiCo" is not a tenable hypothesis without new
evidence.

### Two forward paths

1. **Task-extrapolated** (most likely, given the DiCo paper's n=2 cap
   on Navigation): the fork is fine; n=4 individual-goal Navigation
   with DiCo control is simply not trainable at these hyperparameters.
   The correct research move is to redesign the Graph-SND-as-control
   experiment around a DiCo-validated task (Dispersion n=4
   `share_rew=True`) or restrict Graph-SND-as-control to n=2
   Navigation.
2. **Fork-broken**: a subtler regression somewhere in the fork's
   dependency stack (BenchMARL / TorchRL / VMAS pinning), config
   defaults, or custom integration code causes divergence on any task
   at n>2. Less likely (the `estimator="full"` code matches upstream
   line-for-line) but must be ruled out.

### What this plan does instead of another speculative fix

No more hyperparameter edits are promised. Three changes give us the
evidence to pick between the two branches above:

1. **CSV instrumentation**
   ([het_control/callback.py](het_control/callback.py))
   adds `scaling_ratio_mean`, `applied_snd`, and `out_loc_norm_mean`
   columns so the CSV shows the quantity DiCo is actually controlling
   (applied SND) rather than only raw per-agent spread (`snd_t`). The
   `snd_t=165` plateau from the first n=16 runs does not by itself
   prove DiCo failed — `applied_snd` does. The first implementation read
   `scaling_ratio` from BenchMARL's `training_td` at `on_train_end`, which
   is empty for IPPO (and often for MADDPG), so those columns were all NaN.
   The fix is to **average `scaling_ratio` and `out_loc_norm` inside
   `HetControlMlpEmpirical._forward` over grad-mode forwards** each
   iteration and have the callback call `consume_csv_metric_means()` when
   writing the row. For ad-hoc prints from the forward path, set
   `HET_CONTROL_DICO_DEBUG_FORWARDS=20` in the environment.
2. **DiCo-validated sanity baseline**: new
   [het_control/conf/dispersion_maddpg_full_config.yaml](het_control/conf/dispersion_maddpg_full_config.yaml),
   [het_control/run_scripts/run_dispersion_maddpg_full.py](het_control/run_scripts/run_dispersion_maddpg_full.py),
   and
   [scripts/run_dispersion_sanity.sh](scripts/run_dispersion_sanity.sh)
   runs a Dispersion n=4 `share_rew=True` MADDPG job (default
   `MAX_ITERS=200`; use `MAX_ITERS=30` only for a quick smoke). MADDPG is
   off-policy: early iters have sparse replay and high exploration
   epsilon, so reward can look flat for dozens of iters unless you
   shorten `experiment.exploration_anneal_frames` or raise
   `experiment.off_policy_init_random_frames` via `HYDRA_EXTRA` (see
   script header). Reward should climb over a long enough horizon if the
   fork is healthy.
3. **Smoke test (revised again)**: an earlier crash-only version avoided
   false conclusions about raw `snd_t`. Once `applied_snd` is logged
   reliably, the smoke test checks **mean(last 10 `applied_snd`) ≈
   `snd_des`** (±0.05) plus mild reward guards — not raw `snd_t` vs
   `snd_des`. See [tests/smoke_dico_training.py](tests/smoke_dico_training.py).

### Decision rule after the sanity baseline

- Dispersion sanity shows reward growth + `applied_snd` tracking
  `desired_snd` ⇒ **task-extrapolated** branch. Next plan:
  redesign the Graph-SND-as-control experiment around
  Dispersion n=4 `share_rew=True`, or scope Navigation runs to
  n=2 with DiCo's paper hyperparameters.
- Dispersion sanity shows reward stuck near zero on Dispersion too
  ⇒ **fork-broken** branch. Next plan: dependency / config / code
  diff against upstream DiCo, verify n=2 Navigation matches paper
  reward, then revisit.

Either branch is much cheaper to execute than another round of
hyperparameter speculation.
