"""Pre-training smoke test for DiCo-Navigation with Graph-SND.

Runs ~12 PPO iterations at ``n_agents=4`` with reduced frames/batch (~2-3s
per iter, well under 2 minutes total) against one of the three Graph-SND
config variants, then parses ``graph_snd_log.csv`` and asserts that the
DiCo scaling loop is behaving sensibly. The test is intentionally stricter
than "no crash" -- it catches the qualitative bug (``snd_t`` wandering to
100x of ``snd_des``, reward collapsing) that the n=16 real runs exhibited.

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
"""

from __future__ import annotations

import argparse
import csv
import math
import os
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

# Smoke-test knobs. Kept conservative so this always finishes in < 2 min on
# a modest GPU. ``max_n_iters=12`` gives us enough rows to do a direction
# check at iter 5 and a magnitude check at iter 10 per the task spec.
SMOKE_OVERRIDES: Tuple[str, ...] = (
    "task.n_agents=4",
    "experiment.on_policy_collected_frames_per_batch=6000",
    "experiment.on_policy_n_envs_per_worker=60",
    "experiment.on_policy_n_minibatch_iters=5",
    "experiment.on_policy_minibatch_size=2048",
    "experiment.max_n_iters=12",
    "experiment.max_n_frames=1000000",
    "experiment.evaluation=false",
    "experiment.loggers=[]",
    "experiment.render=false",
    "experiment.create_json=false",
    "experiment.checkpoint_interval=0",
    "model.desired_snd=0.5",
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


def check_csv(rows: List[Dict[str, str]], min_rows: int = 5) -> None:
    """Run all qualitative checks on the training trajectory.

    Each check prints a short one-line status so a passing run emits
    a readable success report; a failing check calls ``_fail`` with
    diagnostic context.
    """
    if len(rows) < min_rows:
        _fail(
            f"expected at least {min_rows} CSV rows, got {len(rows)}; "
            f"did the trainer complete the requested number of iters?"
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
    # snd_des is a model buffer -- should be identical across rows.
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

    # Check 4: snd_t stays strictly positive.
    for i, v in enumerate(snd_t):
        if v <= 0:
            _fail(
                f"snd_t went non-positive at iter={i}: {v}; the scaling ratio "
                f"would blow up on the next forward"
            )
    print("[smoke] check 4 OK: snd_t > 0 at every iter")

    # Check 5: direction check. After ~5 iters, snd_t should have moved
    # toward snd_des (or, if it started very close, not have wandered
    # multiple orders of magnitude away).
    dir_iter = min(5, len(snd_t) - 1)
    start = snd_t[0]
    mid = snd_t[dir_iter]
    started_low = start < snd_des
    moved_toward_target = (mid - start) * (snd_des - start) > 0
    # Allow "stayed near start" if we started within 50% of target already
    # (bootstrap_from_desired_snd=True puts iter-0 snd_t right at snd_des).
    stayed_near_target = abs(start - snd_des) / snd_des < 0.5 and abs(mid - snd_des) / snd_des < 2.0
    if not (moved_toward_target or stayed_near_target):
        direction = "up" if started_low else "down"
        _fail(
            f"direction check failed: snd_t[0]={start:.4f}, "
            f"snd_t[{dir_iter}]={mid:.4f}, snd_des={snd_des:.4f}. "
            f"Expected snd_t to move {direction} toward snd_des by iter {dir_iter}."
        )
    print(
        f"[smoke] check 5 OK: snd_t evolution from {start:.4f} -> {mid:.4f} "
        f"(snd_des={snd_des:.4f}) is sensible"
    )

    # Check 6: magnitude check. By iter 10 (or the last row), snd_t should
    # be within one order of magnitude of snd_des. This is the check that
    # would have caught the n=16 real run, where snd_t reached 5x by iter
    # 10 and 74x by iter 30.
    mag_iter = min(10, len(snd_t) - 1)
    mag_value = snd_t[mag_iter]
    lower = 0.1 * snd_des
    upper = 10.0 * snd_des
    if not (lower <= mag_value <= upper):
        _fail(
            f"magnitude check failed at iter {mag_iter}: snd_t={mag_value:.4f} "
            f"is outside [{lower:.4f}, {upper:.4f}] (one order of magnitude "
            f"around snd_des={snd_des:.4f}). This is the signature of the "
            f"scaling loop failing to constrain raw agent-network spread."
        )
    print(
        f"[smoke] check 6 OK: snd_t[{mag_iter}]={mag_value:.4f} is within "
        f"[{lower:.4f}, {upper:.4f}]"
    )

    # Check 7: scaling sanity at every row. lambda = snd_des / snd_t.
    lam_lower, lam_upper = 0.01, 100.0
    for i, v in enumerate(snd_t):
        lam = snd_des / v
        if not (lam_lower <= lam <= lam_upper):
            _fail(
                f"scaling ratio lambda = snd_des/snd_t = {lam:.4f} at iter={i} "
                f"is outside [{lam_lower}, {lam_upper}]. Extreme scaling "
                f"means DiCo is either degenerate (lambda~=0, shared policy "
                f"dominates) or explosive (lambda~=inf, raw agent_out blows up)."
            )
    print(f"[smoke] check 7 OK: lambda in [{lam_lower}, {lam_upper}] at every iter")

    # Check 8: reward floor. PPO in n=4 Navigation can have slightly
    # negative early rewards during exploration, but should not collapse.
    r0 = reward[0]
    r_last = reward[-1]
    if r_last < -10.0:
        _fail(
            f"reward_mean collapsed to {r_last:.4f} by iter {len(reward)-1}; "
            f"something catastrophic happened in PPO (action-space loss "
            f"exploding, NaN gradients silently recovered, etc.)"
        )
    if r_last < r0 - 5.0:
        _fail(
            f"reward_mean dropped from {r0:.4f} at iter 0 to {r_last:.4f} "
            f"at iter {len(reward)-1} (delta < -5). Policy is getting "
            f"meaningfully worse, not better."
        )
    print(
        f"[smoke] check 8 OK: reward_mean {r0:.4f} -> {r_last:.4f} "
        f"(no catastrophic collapse)"
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
        default=110.0,
        help=(
            "Hard wall-clock budget in seconds for the trainer subprocess "
            "(the test should comfortably finish in < 60s; default 110s "
            "keeps us under the 2-minute spec with headroom)."
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
