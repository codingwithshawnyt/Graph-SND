"""Pre-training smoke test for DiCo-Navigation with Graph-SND.

Runs 20 PPO iterations at ``n_agents=4`` with reduced frames/batch (~3-5s
per iter, well under 2 minutes total) against one of the three Graph-SND
config variants, then parses ``graph_snd_log.csv`` and asserts that the
DiCo scaling loop is behaving sensibly. The test is intentionally stricter
than "no crash" -- it catches the slow-runaway bug (raw ``snd_t`` growing
~5% per iter at desired_snd=0.5) that an earlier 12-iter version missed
(see DIAGNOSIS.md Postmortem).

Usage (from the DiCo fork root):

    python tests/smoke_dico_training.py --config navigation_ippo_full_config
    python tests/smoke_dico_training.py --config navigation_ippo_graph_p01_config
    python tests/smoke_dico_training.py --config navigation_ippo_graph_p025_config

Exits 0 on pass, non-zero on fail. Any of these three calls passing is
sufficient evidence to launch the matching full 167-iteration real run.

Design notes:
- We do not import DiCo internals; we shell out to the exact same entry
  point real runs use (``het_control/run_scripts/run_navigation_ippo.py``)
  so the smoke test exercises the real codepath.
- CSV parsing uses stdlib ``csv`` only -- no pandas dependency so this is
  runnable on any machine that can run the training itself.
- All subprocess stdio is captured; on failure we print the stderr tail to
  help triage without flooding the terminal on success.
- The smoke config uses a reduced PPO inner loop (``n_minibatch_iters=5``)
  which makes per-iter soft-update dynamics diverge from real runs
  (``n_minibatch_iters=45``). We therefore do NOT rely on the bootstrap
  value matching real-run iter-0 snd_t; checks are written to be robust
  to either regime. See ``check_csv`` for the specific checks.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SUPPORTED_CONFIGS = (
    "navigation_ippo_full_config",
    "navigation_ippo_graph_p01_config",
    "navigation_ippo_graph_p025_config",
)

FORK_ROOT = Path(__file__).resolve().parent.parent
RUNNER_REL = Path("het_control/run_scripts/run_navigation_ippo.py")

# Smoke-test knobs. Tuned so the 20-iter run finishes in < 2 min on a modest
# GPU. Rationale for the specific values:
#  * ``task.n_agents=4``: Navigation with n=4 and our paper-tested
#    desired_snd=0.1 should reach near-target snd_t within ~20 iters.
#  * ``on_policy_collected_frames_per_batch=6000`` with ``max_n_iters=20``:
#    ~120k total frames, ~3-5s/iter on a modern GPU, ~100s total.
#  * ``on_policy_n_envs_per_worker=30``: 200 steps/env/iter (= 2 full
#    Navigation episodes of 100 max_steps), enough to get a stable reward
#    signal early.
#  * ``max_n_iters=20``: long enough to put the runaway check at iters
#    [15, 19] comfortably past the early bootstrap-decay transient.
#  * ``model.desired_snd=0.1``: matches DiCo README line 49, the paper's
#    primary Navigation value. Changing this to 0.5 (the previously
#    broken value) should now cause the smoke test to FAIL via the
#    runaway check.
SMOKE_OVERRIDES: Tuple[str, ...] = (
    "task.n_agents=4",
    "experiment.on_policy_collected_frames_per_batch=6000",
    "experiment.on_policy_n_envs_per_worker=30",
    "experiment.on_policy_n_minibatch_iters=5",
    "experiment.on_policy_minibatch_size=2048",
    "experiment.max_n_iters=20",
    "experiment.max_n_frames=1000000",
    "experiment.evaluation=false",
    "experiment.loggers=[]",
    "experiment.render=false",
    "experiment.create_json=false",
    "experiment.checkpoint_interval=0",
    "model.desired_snd=0.1",
)


# ---------------------------------------------------------------------------
# Running the real trainer
# ---------------------------------------------------------------------------


def run_trainer(config_name: str, workdir: Path, timeout_s: float) -> None:
    """Invoke the DiCo trainer as a subprocess and raise on any failure.

    Fails the smoke test (via ``SystemExit``) rather than returning a
    status code so the caller can assume success on return.
    """
    runner_abs = FORK_ROOT / RUNNER_REL
    if not runner_abs.exists():
        _fail(f"trainer entry point not found at {runner_abs}")

    cmd: List[str] = [
        sys.executable,
        str(runner_abs),
        f"--config-name={config_name}",
        *SMOKE_OVERRIDES,
        f"hydra.run.dir={workdir}",
    ]
    print(f"[smoke] cwd: {FORK_ROOT}")
    print(f"[smoke] cmd: {' '.join(cmd)}")
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(FORK_ROOT),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        _fail(
            f"trainer timed out after {timeout_s:.0f}s; "
            f"last stderr:\n{_tail(exc.stderr or '', 40)}"
        )

    elapsed = time.monotonic() - t0
    print(f"[smoke] trainer exited with code {result.returncode} in {elapsed:.1f}s")
    if result.returncode != 0:
        _fail(
            f"trainer exited non-zero ({result.returncode}); "
            f"last stderr:\n{_tail(result.stderr, 60)}"
        )


# ---------------------------------------------------------------------------
# CSV checks
# ---------------------------------------------------------------------------


def load_csv(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        _fail(
            f"expected CSV at {csv_path} but it does not exist; the "
            f"``graph_snd_log_path`` in the selected config did not get "
            f"redirected into the workdir"
        )
    with csv_path.open("r", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        _fail(f"CSV at {csv_path} has a header but no data rows")
    return rows


def _to_float(value: Optional[str], col: str, iter_idx: int) -> float:
    if value is None or value == "":
        _fail(f"row iter={iter_idx}: missing value for column '{col}'")
    try:
        return float(value)
    except ValueError:
        _fail(f"row iter={iter_idx}: could not parse '{col}'={value!r} as float")
    # unreachable
    raise AssertionError


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def check_csv(rows: List[Dict[str, str]], min_rows: int = 18) -> None:
    """Run all qualitative checks on the training trajectory.

    Each check prints a short one-line status so a passing run emits
    a readable success report; a failing check calls ``_fail`` with
    diagnostic context. The thresholds are calibrated for the default
    ``model.desired_snd=0.1`` paper-good regime; running the smoke test
    against a known-broken setting (e.g. ``desired_snd=0.5`` on
    Navigation) should cause the runaway check (check 6) to fail within
    the 20-iter budget.
    """
    if len(rows) < min_rows:
        _fail(
            f"expected at least {min_rows} CSV rows, got {len(rows)}; "
            f"the trainer did not complete the requested 20 iterations "
            f"(likely a crash mid-run -- inspect stdout.log in the workdir)."
        )
    print(f"[smoke] check 1 OK: CSV has {len(rows)} rows (>= {min_rows})")

    expected_cols = {
        "iter",
        "seed",
        "n_agents",
        "estimator",
        "p",
        "snd_t",
        "snd_des",
        "reward_mean",
        "metric_time_ms",
    }
    missing = expected_cols - set(rows[0].keys())
    if missing:
        _fail(f"CSV is missing columns: {sorted(missing)}")

    # Parse numeric columns once, keyed by iter index.
    snd_t: List[float] = []
    reward: List[float] = []
    metric_ms: List[float] = []
    for i, row in enumerate(rows):
        snd_t.append(_to_float(row["snd_t"], "snd_t", i))
        reward.append(_to_float(row["reward_mean"], "reward_mean", i))
        metric_ms.append(_to_float(row["metric_time_ms"], "metric_time_ms", i))

    snd_des = _to_float(rows[0]["snd_des"], "snd_des", 0)
    for i, row in enumerate(rows[1:], start=1):
        this = _to_float(row["snd_des"], "snd_des", i)
        if this != snd_des:
            _fail(
                f"snd_des changed mid-run: row 0 = {snd_des}, row {i} = {this} "
                f"(DiCo target SND must be a fixed buffer)"
            )

    # Check 2: no NaN / inf anywhere in the signal columns.
    for col, values in (("snd_t", snd_t), ("reward_mean", reward), ("metric_time_ms", metric_ms)):
        for i, v in enumerate(values):
            if math.isnan(v) or math.isinf(v):
                _fail(
                    f"column '{col}' has non-finite value {v!r} at iter={i}; "
                    f"training became numerically unstable"
                )
    print("[smoke] check 2 OK: no NaN / inf in snd_t, reward_mean, metric_time_ms")

    # Check 3: snd_des is a sane positive target.
    if snd_des <= 0 or not math.isfinite(snd_des) or snd_des > 5.0:
        _fail(
            f"snd_des = {snd_des} is outside the expected smoke-test range "
            f"(0, 5]; did the CLI override of model.desired_snd get lost?"
        )
    print(f"[smoke] check 3 OK: snd_des = {snd_des} is sane")

    # Check 4: snd_t stays strictly positive at every iter.
    for i, v in enumerate(snd_t):
        if v <= 0:
            _fail(
                f"snd_t went non-positive at iter={i}: {v}; the scaling ratio "
                f"would blow up on the next forward"
            )
    print("[smoke] check 4 OK: snd_t > 0 at every iter")

    # Check 5: direction check -- pure monotone-toward-target, no "stayed
    # near target" fallback (the previous fallback let the broken n=4
    # desired_snd=0.5 run pass by triggering on the bootstrap-inflated
    # iter-0 value; see DIAGNOSIS.md Postmortem).
    #
    # Rule: if snd_t[0] is off-target, snd_t[5] must be on the correct
    # side of it. If snd_t[0] is already within 10% of snd_des, we only
    # require that snd_t[5] not have overshot snd_des by more than 2x.
    dir_iter = min(5, len(snd_t) - 1)
    start = snd_t[0]
    mid = snd_t[dir_iter]
    start_close = abs(start - snd_des) / snd_des < 0.10
    if start_close:
        if not (mid <= 2.0 * snd_des):
            _fail(
                f"direction check failed: snd_t[0]={start:.4f} was already "
                f"within 10% of snd_des={snd_des:.4f}, but snd_t[{dir_iter}]"
                f"={mid:.4f} is more than 2x past target -- the controller "
                f"is overshooting rather than settling."
            )
    else:
        started_low = start < snd_des
        moved_toward_target = (mid - start) * (snd_des - start) > 0
        if not moved_toward_target:
            direction = "up" if started_low else "down"
            _fail(
                f"direction check failed: snd_t[0]={start:.4f}, "
                f"snd_t[{dir_iter}]={mid:.4f}, snd_des={snd_des:.4f}. "
                f"Expected snd_t to move {direction} toward snd_des by "
                f"iter {dir_iter}; it went the wrong way."
            )
    print(
        f"[smoke] check 5 OK: snd_t[0]={start:.4f} -> snd_t[{dir_iter}]"
        f"={mid:.4f} (snd_des={snd_des:.4f}) moved toward target"
    )

    # Check 6: RUNAWAY CHECK. The previous smoke test (12 iters, 10x
    # magnitude window) missed the broken desired_snd=0.5 run because
    # snd_t grows only ~5% per iter there, staying within 10x of the
    # target for dozens of iters before finally blowing up to 300x. This
    # check compares the late-window peak to the mid-window mean; a
    # healthy run plateaus near snd_des so the ratio should be close to
    # 1.0, while a runaway produces a strictly increasing trajectory
    # with a ratio >> 1.
    #
    # Broken n=4 desired_snd=0.5: snd_t grew 5.3%/iter, so
    # max(snd_t[15:20]) / mean(snd_t[5:10]) ~ 1.76 > 1.5.
    # Healthy desired_snd=0.1: plateaus near target, ratio ~ 1.0-1.3.
    early_window = snd_t[5:10]
    late_window = snd_t[-5:]
    early_mean = _mean(early_window)
    late_peak = max(late_window)
    runaway_ratio = late_peak / early_mean if early_mean > 0 else float("inf")
    runaway_threshold = 1.5
    if runaway_ratio > runaway_threshold:
        _fail(
            f"runaway check failed: max(snd_t[-5:])={late_peak:.4f} vs "
            f"mean(snd_t[5:10])={early_mean:.4f} (ratio {runaway_ratio:.2f} "
            f"> {runaway_threshold}). snd_t is still growing in the late "
            f"window rather than plateauing at snd_des={snd_des:.4f}. "
            f"This is the signature of DiCo being driven outside its "
            f"paper-tested desired_snd range -- PPO is growing raw "
            f"per-agent outputs unboundedly to compensate for a shrinking "
            f"scaling_ratio. See DIAGNOSIS.md Postmortem."
        )
    print(
        f"[smoke] check 6 OK: runaway ratio late/mid = {runaway_ratio:.2f} "
        f"(<= {runaway_threshold})"
    )

    # Check 7: final-iter magnitude. snd_t by the last iter should be
    # within a 3x window of snd_des. At desired_snd=0.1 the working
    # regime has snd_t stabilise in [0.05, 0.2] by iter ~15-20. Allowing
    # a 3x window keeps this robust to stochastic fluctuations without
    # letting the old 10x blow-ups through.
    last_iter = len(snd_t) - 1
    last = snd_t[-1]
    lower = 0.3 * snd_des
    upper = 3.0 * snd_des
    if not (lower <= last <= upper):
        _fail(
            f"magnitude check failed at iter {last_iter}: snd_t={last:.4f} "
            f"is outside [{lower:.4f}, {upper:.4f}] (3x around snd_des="
            f"{snd_des:.4f}). Either the controller has not approached "
            f"the target by the end of the smoke window, or it has "
            f"overshot substantially."
        )
    print(
        f"[smoke] check 7 OK: final snd_t={last:.4f} is within "
        f"[{lower:.4f}, {upper:.4f}]"
    )

    # Check 8: per-iter scaling-ratio sanity. lambda = snd_des / snd_t.
    # Reduced range vs the previous version because at desired_snd=0.1
    # we expect snd_t in [0.03, 0.3] so lambda in [0.33, 3.33]; anything
    # outside [0.1, 10] is strong evidence the scaling loop is broken.
    lam_lower, lam_upper = 0.1, 10.0
    for i, v in enumerate(snd_t):
        lam = snd_des / v
        if not (lam_lower <= lam <= lam_upper):
            _fail(
                f"scaling ratio lambda = snd_des/snd_t = {lam:.4f} at "
                f"iter={i} is outside [{lam_lower}, {lam_upper}]. Extreme "
                f"scaling means DiCo is either degenerate (lambda~0, "
                f"shared policy dominates) or explosive (lambda~inf, "
                f"raw agent_out blows up)."
            )
    print(f"[smoke] check 8 OK: lambda in [{lam_lower}, {lam_upper}] at every iter")

    # Check 9: LEARNING SIGNAL. The previous check only required "no
    # catastrophic collapse" (delta >= -5), which the broken runs
    # trivially passed because reward went from ~-0.001 to ~+0.01 (tiny
    # positive delta but still firmly in the failure regime). We now
    # require a meaningfully rising reward trend over the smoke window.
    # At desired_snd=0.1, 20-iter Navigation typically shows reward[:5]
    # ~ 0.0 and reward[-5:] >= ~0.01-0.05 in the working regime.
    early_reward_mean = _mean(reward[:5])
    late_reward_mean = _mean(reward[-5:])
    reward_delta = late_reward_mean - early_reward_mean
    # Require some meaningful improvement. The exact magnitude of
    # Navigation reward is task-dependent; we use an absolute floor
    # (0.001 per step) rather than a relative one because the broken
    # run HAD +0.01 delta over 167 iters -- we need to flag the fact
    # that 20 iters should produce at least modest learning.
    min_reward_delta = 0.001
    if reward_delta < min_reward_delta:
        _fail(
            f"learning-signal check failed: mean(reward[:5])="
            f"{early_reward_mean:.5f}, mean(reward[-5:])="
            f"{late_reward_mean:.5f}, delta={reward_delta:.5f} "
            f"(< min {min_reward_delta}). The policy is not learning. "
            f"This would have been the first check to fail on the "
            f"broken n=4 desired_snd=0.5 runs, where reward stayed "
            f"~0 for 167 iters. See DIAGNOSIS.md Postmortem."
        )
    if late_reward_mean < -0.5:
        _fail(
            f"learning-signal check failed: mean(reward[-5:])="
            f"{late_reward_mean:.5f} is catastrophically negative. "
            f"PPO may be diverging."
        )
    print(
        f"[smoke] check 9 OK: reward mean[:5]={early_reward_mean:.5f} -> "
        f"mean[-5:]={late_reward_mean:.5f} (delta {reward_delta:+.5f})"
    )

    # Informational: flag suspicious iter-0 snd_t values. The reduced
    # PPO inner loop in the smoke config decays the bootstrap toward
    # raw SND only partially (0.99^15 ~ 0.86 vs real runs' 0.99^675
    # ~ 6e-4), so snd_t[0] can land near snd_des when
    # bootstrap_from_desired_snd=True -- this can mask divergence if
    # check 5's direction logic were relaxed. We no longer have such a
    # loophole, but surface the observation so users are not surprised.
    if start > 2.0 * snd_des:
        print(
            f"[smoke] WARNING: snd_t[0]={start:.4f} is much larger than "
            f"snd_des={snd_des:.4f}; the raw per-agent spread at init is "
            f"higher than the target. Not a failure, but worth noting."
        )
    elif start > 0.9 * snd_des and start < 1.1 * snd_des:
        print(
            f"[smoke] INFO: snd_t[0]={start:.4f} ~ snd_des={snd_des:.4f} "
            f"(likely bootstrap_from_desired_snd=True or the model was "
            f"initialised with well-separated per-agent outputs)."
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _fail(msg: str) -> None:
    """Print a concise failure banner and exit non-zero."""
    print("\n================ SMOKE TEST FAIL ================", file=sys.stderr)
    print(msg, file=sys.stderr)
    print("=================================================", file=sys.stderr)
    sys.exit(1)


def _tail(text: str, n_lines: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n_lines:]) if lines else "(empty)"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="navigation_ippo_full_config",
        choices=SUPPORTED_CONFIGS,
        help="Which DiCo-Navigation config to smoke test (default: full).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help=(
            "Hard wall-clock budget in seconds for the trainer subprocess. "
            "20 smoke iterations typically finish in 100-140s on a modern "
            "GPU (~5-7s/iter including VMAS vectorized env stepping and "
            "the PPO inner loop); default 180s provides ~40s headroom."
        ),
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help=(
            "Hydra output dir for the smoke run. Defaults to a fresh "
            "``tempfile.mkdtemp()``. Kept on disk if the test fails so "
            "the user can inspect ``graph_snd_log.csv`` / stdout.log."
        ),
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Do not delete the workdir on success (useful for debugging).",
    )
    args = parser.parse_args(argv)

    if args.workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="dico_smoke_"))
        created_tempdir = True
    else:
        workdir = Path(args.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        created_tempdir = False

    print(f"[smoke] config  : {args.config}")
    print(f"[smoke] workdir : {workdir}")
    print(f"[smoke] timeout : {args.timeout:.0f}s")
    print(f"[smoke] cwd     : {FORK_ROOT}")
    if not (FORK_ROOT / "het_control").is_dir():
        _fail(
            f"expected fork root at {FORK_ROOT} to contain a 'het_control' "
            f"directory; are you running this from inside the DiCo fork?"
        )

    run_trainer(args.config, workdir, timeout_s=args.timeout)

    csv_path = workdir / "graph_snd_log.csv"
    rows = load_csv(csv_path)
    check_csv(rows)

    print("\n================ SMOKE TEST PASS ================")
    print(f"config  : {args.config}")
    print(f"rows    : {len(rows)}")
    print(f"snd_t   : {rows[0]['snd_t']} -> {rows[-1]['snd_t']}")
    print(f"reward  : {rows[0]['reward_mean']} -> {rows[-1]['reward_mean']}")
    print(f"workdir : {workdir}")
    print("=================================================")

    if created_tempdir and not args.keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
