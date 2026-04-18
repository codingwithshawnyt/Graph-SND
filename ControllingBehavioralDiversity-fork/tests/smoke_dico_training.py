"""Crash-only smoke test for DiCo-Navigation with Graph-SND.

Runs 5 PPO iterations at ``n_agents=4`` against one of the three
Graph-SND config variants and asserts only that

1. the training subprocess exits with code 0,
2. ``graph_snd_log.csv`` exists with at least 5 data rows,
3. none of ``snd_t`` / ``reward_mean`` / ``metric_time_ms`` is NaN / inf.

That is all the smoke test checks. It deliberately does *not* check
whether ``snd_t`` moved toward ``snd_des``, whether reward rose, or
whether the scaling ratio is sane.

Why so weak? Three earlier revisions of this smoke test included
direction / runaway / magnitude / learning-signal checks and each
revision was used to validate a different speculative "fix" that later
failed in the real run. The root cause of the repeated false-positive
/ false-negative behavior, documented in DIAGNOSIS.md Postmortem #2,
is that the smoke PPO regime (``n_minibatch_iters=5``,
``minibatch_size=2048`` -> ~15 ``_forward`` calls per iter) is not the
same dynamical system as the real-run regime
(``n_minibatch_iters=45``, ~675 forwards per iter). The
``tau=0.01`` soft-update inside ``estimate_snd`` produces qualitatively
different ``estimated_snd`` trajectories in the two regimes:
``0.99^15 ~ 0.86`` vs ``0.99^675 ~ 1.1e-3``. Any behavior-based
assertion calibrated on the smoke regime is therefore a coin-flip for
real-run behavior. Reducing the smoke to a crash check honestly
reflects what it can reliably detect without inventing false signal.

Usage (from the DiCo fork root):

    python tests/smoke_dico_training.py --config navigation_ippo_full_config
    python tests/smoke_dico_training.py --config navigation_ippo_graph_p01_config
    python tests/smoke_dico_training.py --config navigation_ippo_graph_p025_config

Exits 0 on pass, non-zero on fail. The orchestrator in
``scripts/run_graph_dico_two_gpus_then_third.sh`` still invokes this
(indirectly via the ``kill -0`` post-launch health check); the check
being crash-only is sufficient there -- it catches missing GPUs,
import-time regressions, and Hydra misconfigurations, which is all a
pre-launch smoke can reliably detect.
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

# Smoke knobs. Budget targets < 60 s on a modern GPU:
#   * 5 iters x ~5 s/iter (VMAS vectorised step + 5 PPO minibatch iters)
#     ~ 25-30 s core training
#   * 10-20 s of Python / TorchRL / VMAS import and setup
# Everything is sized for the *crash check*: we just need the trainer
# to get far enough to write several CSV rows.
SMOKE_OVERRIDES: Tuple[str, ...] = (
    "task.n_agents=4",
    "experiment.on_policy_collected_frames_per_batch=6000",
    "experiment.on_policy_n_envs_per_worker=30",
    "experiment.on_policy_n_minibatch_iters=5",
    "experiment.on_policy_minibatch_size=2048",
    "experiment.max_n_iters=5",
    "experiment.max_n_frames=1000000",
    "experiment.evaluation=false",
    "experiment.loggers=[]",
    "experiment.render=false",
    "experiment.create_json=false",
    "experiment.checkpoint_interval=0",
)

MIN_ROWS = 5

# Columns we require to be present AND finite. Other columns may be
# NaN (the new diagnostic columns -- scaling_ratio_mean, applied_snd,
# out_loc_norm_mean -- are best-effort per callback.py; see
# DIAGNOSIS.md Postmortem #2).
REQUIRED_FINITE_COLUMNS: Tuple[str, ...] = (
    "snd_t",
    "reward_mean",
    "metric_time_ms",
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
# CSV checks (crash-only)
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


def check_csv(rows: List[Dict[str, str]]) -> None:
    """Crash-only CSV assertions.

    We check row count and that the three numeric columns we rely on
    downstream are finite. We deliberately do *not* check direction,
    runaway, magnitude, or learning-signal -- see module docstring and
    DIAGNOSIS.md Postmortem #2 for why.
    """
    if len(rows) < MIN_ROWS:
        _fail(
            f"expected at least {MIN_ROWS} CSV rows, got {len(rows)}; "
            f"the trainer did not complete {MIN_ROWS} iterations "
            f"(likely a crash mid-run -- inspect stdout.log in the workdir)."
        )
    print(f"[smoke] check 1 OK: CSV has {len(rows)} rows (>= {MIN_ROWS})")

    missing: List[str] = []
    for col in REQUIRED_FINITE_COLUMNS:
        if col not in rows[0]:
            missing.append(col)
    if missing:
        _fail(f"CSV is missing required columns: {missing}")

    for col in REQUIRED_FINITE_COLUMNS:
        for i, row in enumerate(rows):
            raw = row.get(col, "")
            if raw is None or raw == "":
                _fail(
                    f"row iter={i}: missing value for required column '{col}'"
                )
            try:
                value = float(raw)
            except ValueError:
                _fail(
                    f"row iter={i}: could not parse '{col}'={raw!r} as float"
                )
            if math.isnan(value) or math.isinf(value):
                _fail(
                    f"column '{col}' has non-finite value {value!r} at "
                    f"iter={i}; training became numerically unstable"
                )
    print(
        f"[smoke] check 2 OK: no NaN / inf in "
        f"{', '.join(REQUIRED_FINITE_COLUMNS)}"
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
        default=120.0,
        help=(
            "Hard wall-clock budget in seconds for the trainer subprocess. "
            "A 5-iteration smoke typically finishes in ~50-70 s on a modern "
            "GPU including imports; default 120 s provides ~50 s headroom."
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
    print("Note: this is a crash-only check. It does NOT verify whether")
    print("the DiCo control loop is converging -- that is only observable")
    print("at real-run PPO granularity. See DIAGNOSIS.md Postmortem #2.")
    print("=================================================")

    if created_tempdir and not args.keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
