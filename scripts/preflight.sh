#!/usr/bin/env bash
# Preflight checks before an overnight run.
#
# Verifies:
#   1. Python environment is healthy (torch, vmas imports, pytest green).
#   2. Every visible CUDA device can allocate a tensor and do a matmul.
#   3. The batched trainer can complete 2 iterations on each visible GPU
#      at the overnight n=100 / n=50 settings, scaled to a tiny rollout
#      so the whole preflight finishes in under two minutes.
#
# If anything fails, we exit non-zero and print what to do. If
# everything passes, you are safe to launch scripts/run_overnight_n100.sh
# and scripts/run_overnight_n50.sh and close the terminal.

set -euo pipefail

cd "$(dirname "$0")/.."

PY_BIN=".venv/bin/python"
if [ ! -x "$PY_BIN" ]; then
    PY_BIN="python"
fi

echo "[preflight] python: $($PY_BIN -V 2>&1)"
echo "[preflight] repo:   $(pwd)"

echo "[preflight] step 1/3: pytest"
$PY_BIN -m pytest tests/ -q

echo "[preflight] step 2/3: CUDA device probe"
$PY_BIN - <<'PYEOF'
import torch
if not torch.cuda.is_available():
    print("  cuda unavailable; running on CPU. Overnight on CPU at n=100 is not recommended.")
    raise SystemExit(0)
n_dev = torch.cuda.device_count()
print(f"  cuda devices: {n_dev}")
for i in range(n_dev):
    props = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    a = torch.randn(2048, 2048, device=f'cuda:{i}')
    b = torch.randn(2048, 2048, device=f'cuda:{i}')
    c = a @ b
    torch.cuda.synchronize(i)
    print(
        f"  cuda:{i}  {props.name}  "
        f"{props.total_memory/1e9:.1f}GB total, "
        f"{free/1e9:.1f}GB free, "
        f"matmul ok (|c|^2={float((c*c).sum().item()):.2e})"
    )
PYEOF

echo "[preflight] step 3/3: 2-iter smoke on each visible CUDA device"
$PY_BIN - <<'PYEOF'
import os, subprocess, sys, torch, shutil
n_dev = torch.cuda.device_count() if torch.cuda.is_available() else 0
if n_dev == 0:
    print("  skipping (no cuda)")
    raise SystemExit(0)
combos = []
if n_dev >= 1:
    combos.append(("cuda:0", 100))
if n_dev >= 2:
    combos.append(("cuda:1", 50))
ok = True
for dev, n in combos:
    work = f"/tmp/graphsnd_preflight_{dev.replace(':','_')}_{n}"
    shutil.rmtree(work, ignore_errors=True)
    cmd = [
        sys.executable, "training/train_navigation_batched.py",
        "--n-agents", str(n), "--iters", "2",
        "--num-envs", "8", "--rollout-steps", "16",
        "--minibatch-size", "64", "--epochs", "1",
        "--device", dev, "--ckpt-every", "2", "--snd-every", "1",
        "--tag", "preflight", "--seed", "0",
        "--checkpoint-dir", work,
    ]
    print(f"  launching {dev}  n={n}")
    try:
        subprocess.run(cmd, check=True, timeout=300)
        print(f"    OK on {dev}")
    except subprocess.CalledProcessError as e:
        print(f"    FAILED on {dev} (exit {e.returncode})")
        ok = False
    except subprocess.TimeoutExpired:
        print(f"    FAILED on {dev} (timeout)")
        ok = False
    finally:
        shutil.rmtree(work, ignore_errors=True)
if not ok:
    raise SystemExit(1)
PYEOF

echo "[preflight] all checks passed. Safe to launch the overnight run."
