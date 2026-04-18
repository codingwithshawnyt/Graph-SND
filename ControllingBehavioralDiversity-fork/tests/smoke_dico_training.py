"""Lightweight smoke test for DiCo-Navigation with Graph-SND.

Runs a short PPO job at ``n_agents=4`` against one of the three Graph-SND
config variants, then checks:

1. Trainer exits 0; CSV has enough rows; core numeric columns are finite.
2. **Applied SND** (what the environment sees after DiCo scaling) tracks
   ``snd_des``: the mean of ``applied_snd`` over the **last 10** iterations
   must be within ``APPLIED_SND_TOLERANCE`` of ``snd_des``. Raw ``snd_t``
   (estimated diversity of *unscaled* agent deltas) is **not** asserted to
   move toward ``snd_des`` — DiCo intentionally lets raw spread grow while
   ``scaling_ratio = snd_des / distance`` shrinks so their product stays near
   target (see DIAGNOSIS.md and n=2 paper-style baselines).
3. **Reward** does not catastrophically collapse vs the start of the window.

The smoke PPO inner loop (``n_minibatch_iters=5``) still differs from full
paper runs (45), so tolerances are modest. This test catches broken
instrumentation (NaN ``applied_snd``), missing CSV columns, and gross
control failure.

Usage (from the DiCo fork root):

    python tests/smoke_dico_training.py --config navigation_ippo_full_config
    python tests/smoke_dico_training.py --config navigation_ippo_graph_p01_config
    python tests/smoke_dico_training.py --config navigation_ippo_graph_p025_config

Exits 0 on pass, non-zero on fail.
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

# Enough iterations for a stable last-10 mean of applied_snd (see check_csv).
SMOKE_OVERRIDES: Tuple[str, ...] = (
    "task.n_agents=4",
    "experiment.on_policy_collected_frames_per_batch=6000",
    "experiment.on_policy_n_envs_per_worker=30",
    "experiment.on_policy_n_minibatch_iters=5",
    "experiment.on_policy_minibatch_size=2048",
    "experiment.max_n_iters=15",
    "experiment.max_n_frames=1000000",
    "experiment.evaluation=false",
    "experiment.loggers=[]",
    "experiment.render=false",
    "experiment.create_json=false",
    "experiment.checkpoint_interval=0",
)

MIN_ROWS = 15
LAST_N_APPLIED = 10
APPLIED_SND_TOLERANCE = 0.05

# Columns that must exist and be finite on every row (control + task signals).
REQUIRED_FINITE_COLUMNS: Tuple[str, ...] = (
    "snd_t",
    "snd_des",
    "reward_mean",
    "metric_time_ms",
    "scaling_ratio_mean",
    "applied_snd",
)


# ---------------------------------------------------------------------------
# Running the real trainer
# ---------------------------------------------------------------------------


def run_trainer(config_name: str, workdir: Path, timeout_s: float) -> None:
    """Invoke the DiCo trainer as a subprocess and raise on any failure."""
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


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def check_csv(rows: List[Dict[str, str]]) -> None:
    """Finite columns + applied_snd tracking + mild reward guard."""
    if len(rows) < MIN_ROWS:
        _fail(
            f"expected at least {MIN_ROWS} CSV rows, got {len(rows)}; "
            f"increase experiment.max_n_iters in SMOKE_OVERRIDES or fix crashes."
        )
    print(f"[smoke] check 1 OK: CSV has {len(rows)} rows (>= {MIN_ROWS})")

    for col in REQUIRED_FINITE_COLUMNS:
        if col not in rows[0]:
            _fail(
                f"CSV missing column {col!r}; need graph_snd instrumentation "
                f"(scaling_ratio_mean, applied_snd) on this fork."
            )

    for col in REQUIRED_FINITE_COLUMNS:
        for i, row in enumerate(rows):
            raw = row.get(col, "")
            if raw is None or raw == "":
                _fail(f"row iter={i}: missing value for column '{col}'")
            try:
                value = float(raw)
            except ValueError:
                _fail(
                    f"row iter={i}: could not parse '{col}'={raw!r} as float"
                )
            if math.isnan(value) or math.isinf(value):
                _fail(
                    f"column '{col}' has non-finite value {value!r} at iter={i}. "
                    f"If scaling_ratio_mean/applied_snd are NaN, per-forward "
                    f"metrics are not reaching the CSV (see het_control/callback.py "
                    f"and HetControlMlpEmpirical.consume_csv_metric_means)."
                )
    print(
        f"[smoke] check 2 OK: no NaN / inf in "
        f"{', '.join(REQUIRED_FINITE_COLUMNS)}"
    )

    snd_des_vals = [float(r["snd_des"]) for r in rows]
    snd_des = snd_des_vals[0]
    _eps = max(1e-4, abs(snd_des) * 1e-5)
    if not all(abs(v - snd_des) < _eps for v in snd_des_vals):
        _fail("snd_des changed mid-run (expected fixed Hydra buffer)")
    print(f"[smoke] check 3 OK: snd_des constant = {snd_des:g}")

    # Last-10 mean of applied SND vs target (DiCo invariant on actions).
    tail = rows[-LAST_N_APPLIED:]
    applied_tail = [float(r["applied_snd"]) for r in tail]
    mean_applied = _mean(applied_tail)
    dev = abs(mean_applied - snd_des)
    if dev > APPLIED_SND_TOLERANCE:
        _fail(
            f"applied_snd control check failed: mean(last {LAST_N_APPLIED})="
            f"{mean_applied:.6f}, snd_des={snd_des:.6f}, |delta|={dev:.6f} "
            f"(tolerance {APPLIED_SND_TOLERANCE}). Raw snd_t is not required to "
            f"match snd_des; applied_snd should."
        )
    print(
        f"[smoke] check 4 OK: mean(applied_snd last {LAST_N_APPLIED})="
        f"{mean_applied:.6f} within {APPLIED_SND_TOLERANCE:g} of snd_des="
        f"{snd_des:g}"
    )

    # Reward: no large collapse vs start of run (same window sizes as applied).
    reward_all = [float(r["reward_mean"]) for r in rows]
    head = reward_all[:LAST_N_APPLIED]
    tail_r = reward_all[-LAST_N_APPLIED:]
    mean_head = _mean(head)
    mean_tail_r = _mean(tail_r)
    if mean_tail_r < -0.02:
        _fail(
            f"reward tail mean {mean_tail_r:.6f} is catastrophically negative "
            f"(policy not learning at all in smoke window)"
        )
    if mean_tail_r < mean_head - 0.01:
        _fail(
            f"reward regressed: mean first {LAST_N_APPLIED} iters = {mean_head:.6f}, "
            f"mean last {LAST_N_APPLIED} = {mean_tail_r:.6f} (allowed drop 0.01)"
        )
    print(
        f"[smoke] check 5 OK: reward mean first {LAST_N_APPLIED}={mean_head:.6f}, "
        f"last {LAST_N_APPLIED}={mean_tail_r:.6f}"
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
        default=240.0,
        help=(
            "Hard wall-clock budget in seconds for the trainer subprocess. "
            "15 iterations typically finish in ~90-150 s on a modern GPU."
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
    print(f"applied : {rows[0]['applied_snd']} -> {rows[-1]['applied_snd']}")
    print(f"reward  : {rows[0]['reward_mean']} -> {rows[-1]['reward_mean']}")
    print(f"workdir : {workdir}")
    print("Primary invariant: applied_snd ~ snd_des (not raw snd_t).")
    print("=================================================")

    if created_tempdir and not args.keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
