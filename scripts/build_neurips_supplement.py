#!/usr/bin/env python3
"""Build the anonymous NeurIPS code supplement.

The supplement is intentionally curated. It contains the code and result
artifacts needed to verify the Graph-SND paper, but excludes local
workspace state, the paper PDF/source, heavyweight checkpoints, raw
Hydra trees, private host scripts, and non-anonymous repository links.

Usage
-----
    python scripts/build_neurips_supplement.py --check

The ``--check`` flag performs the same build plus anonymization and size
checks. It is suitable for the final pre-submission sanity pass.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "dist" / "graph_snd_neurips_supplement.zip"
MAX_BYTES = 100 * 1024 * 1024

TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".cff",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

FORBIDDEN_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in [
        r"\b" + "Sha" + "wn" + r"\b",
        r"\b" + "sha" + "wnr" + r"\b",
        "coding" + "with" + "sh" + "aw" + "nyt",
        r"github\.com/" + "coding" + "with",
        "/" + "Users" + "/",
        "/" + "usr1" + "/",
        "172" + r"\.24\.",
        r"\b" + "rid" + "dle" + r"\b",
        "my_" + "lone" + "star",
        "lone" + "star",
        "@" + "cmu" + r"\.edu",
        "Carnegie" + " Mellon",
    ]
]

TOP_LEVEL_FILES = [
    ".gitignore",
    "LICENSE",
    "README.md",
    "SUPPLEMENT_MANIFEST.md",
    "SUPPLEMENT_README.md",
    "pyproject.toml",
    "requirements.txt",
]

DIR_PATTERNS = [
    "graphsnd/**/*.py",
    "tests/**/*.py",
    "training/**/*.py",
    "experiments/**/*.py",
]

SCRIPT_FILES = [
    "scripts/build_neurips_supplement.py",
    "scripts/launch_mpe_move2.sh",
    "scripts/plot_reward_curves.py",
    "scripts/preflight.sh",
    "scripts/run_n500_timing.sh",
    "scripts/run_overnight_n100.sh",
    "scripts/run_overnight_n50.sh",
]

RESULT_PATTERNS = [
    "checkpoints/*_meta.json",
    "results/dico/*.csv",
    "results/dico/*.pdf",
    "results/dico_expander_move1/*.json",
    "results/dico_n50_bern_vs_full/*.csv",
    "results/dico_n50_bern_vs_full/*.pdf",
    "results/dico_n50_bern_vs_full/*.tex",
    "results/dico_n50_posthoc_full_snd/*.csv",
    "results/dico_n50_posthoc_full_snd/*.pdf",
    "results/dico_n50_sweep/*.csv",
    "results/dico_n50_sweep/*.pdf",
    "results/discrete_tvd_sanity/*.csv",
    "results/exp1/*.csv",
    "results/exp1/*.json",
    "results/exp1/*.pdf",
    "results/exp2/*.csv",
    "results/exp2/*.json",
    "results/exp3/*.csv",
    "results/exp3/*.json",
    "results/exp3/*.pdf",
    "results/scaling/*.csv",
    "results/scaling/*.pdf",
]

DICO_PATTERNS = [
    "ControllingBehavioralDiversity-fork/GRAPH_SND_CHANGES.md",
    "ControllingBehavioralDiversity-fork/CITATION.cff",
    "ControllingBehavioralDiversity-fork/LICENSE",
    "ControllingBehavioralDiversity-fork/het_control/graph_snd.py",
    "ControllingBehavioralDiversity-fork/het_control/callback.py",
    "ControllingBehavioralDiversity-fork/het_control/run.py",
    "ControllingBehavioralDiversity-fork/het_control/models/het_control_mlp_empirical.py",
    "ControllingBehavioralDiversity-fork/het_control/conf/dispersion_ippo_knn_config.yaml",
    "ControllingBehavioralDiversity-fork/het_control/conf/model/hetcontrolmlpempirical.yaml",
    "ControllingBehavioralDiversity-fork/het_control/conf/navigation_ippo_config.yaml",
    "ControllingBehavioralDiversity-fork/het_control/conf/navigation_ippo_full_config.yaml",
    "ControllingBehavioralDiversity-fork/het_control/conf/navigation_ippo_graph_p01_config.yaml",
    "ControllingBehavioralDiversity-fork/het_control/conf/navigation_ippo_graph_p025_config.yaml",
    "ControllingBehavioralDiversity-fork/het_control/conf/navigation_knn_config.yaml",
    "ControllingBehavioralDiversity-fork/het_control/run_scripts/measure_random_init_dispersion_full_snd.py",
    "ControllingBehavioralDiversity-fork/scripts/launch_bern_dico.sh",
    "ControllingBehavioralDiversity-fork/scripts/launch_bern_dico_seeds.sh",
    "ControllingBehavioralDiversity-fork/scripts/launch_expander_dico.sh",
    "ControllingBehavioralDiversity-fork/scripts/launch_expander_dico_seeds.sh",
    "ControllingBehavioralDiversity-fork/scripts/launch_expander_n50.sh",
    "ControllingBehavioralDiversity-fork/scripts/launch_n50_bern_setpoint_sweep.sh",
    "ControllingBehavioralDiversity-fork/scripts/launch_n50_full_setpoint_sweep.sh",
    "ControllingBehavioralDiversity-fork/scripts/launch_n50_posthoc_full_snd_validation.sh",
    "ControllingBehavioralDiversity-fork/scripts/plot_graph_dico.py",
    "ControllingBehavioralDiversity-fork/tests/*.py",
    "ControllingBehavioralDiversity-fork/results/neurips_final_n50/seed0/*/graph_snd_log.csv",
    "ControllingBehavioralDiversity-fork/results/neurips_final_n50_setpoint_sweep/seed*/snd*/*/graph_snd_log.csv",
]

EXCLUDE_PATTERNS = [
    "Paper/*",
    "*.pt",
    "*.pyc",
    "*.log",
    ".git/*",
    ".venv/*",
    ".kiro/*",
    ".pytest_cache/*",
    "graphsnd.egg-info/*",
    "logs/*",
    "dist/*",
    "**/.DS_Store",
    "**/.hydra/*",
    "**/__pycache__/*",
]


@dataclass(frozen=True)
class Entry:
    src: Path
    arcname: Path


def run_git_ls_files() -> set[str]:
    try:
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def iter_matches(patterns: Iterable[str]) -> set[Path]:
    out: set[Path] = set()
    for pattern in patterns:
        matches = ROOT.glob(pattern)
        out.update(path for path in matches if path.is_file())
    return out


def is_excluded(rel_posix: str) -> bool:
    return any(fnmatch.fnmatch(rel_posix, pat) for pat in EXCLUDE_PATTERNS)


def collect_entries() -> list[Entry]:
    tracked = run_git_ls_files()
    allow_untracked = {
        "LICENSE",
        "SUPPLEMENT_MANIFEST.md",
        "SUPPLEMENT_README.md",
        "scripts/build_neurips_supplement.py",
    }
    candidates: set[Path] = set()
    candidates.update(ROOT / path for path in TOP_LEVEL_FILES)
    candidates.update(ROOT / path for path in SCRIPT_FILES)
    candidates.update(iter_matches(DIR_PATTERNS))
    candidates.update(iter_matches(RESULT_PATTERNS))
    candidates.update(iter_matches(DICO_PATTERNS))

    entries: list[Entry] = []
    for src in sorted(candidates):
        if not src.exists() or not src.is_file():
            continue
        rel = src.relative_to(ROOT)
        rel_posix = rel.as_posix()
        if is_excluded(rel_posix):
            continue
        if tracked and rel_posix not in tracked and rel_posix not in allow_untracked:
            continue
        entries.append(Entry(src=src, arcname=rel))
    return entries


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def scan_text(entries: Iterable[Entry]) -> list[str]:
    failures: list[str] = []
    for entry in entries:
        if not is_text_file(entry.src):
            continue
        text = read_text(entry.src)
        for pattern in FORBIDDEN_PATTERNS:
            match = pattern.search(text)
            if match:
                failures.append(f"{entry.arcname}: forbidden match {match.group(0)!r}")
    return failures


def write_zip(entries: Iterable[Entry], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{out_path.stem}.", suffix=".tmp.zip", dir=out_path.parent
    )
    os.close(fd)
    tmp = Path(tmp_name)
    with zipfile.ZipFile(tmp, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for entry in entries:
            zf.write(entry.src, entry.arcname.as_posix())
    tmp.replace(out_path)


def scan_zip(out_path: Path) -> list[str]:
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="graph_snd_supp_") as td:
        extract_root = Path(td)
        with zipfile.ZipFile(out_path, mode="r") as zf:
            zf.extractall(extract_root)
        for path in extract_root.rglob("*"):
            if not path.is_file() or not is_text_file(path):
                continue
            rel = path.relative_to(extract_root)
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in FORBIDDEN_PATTERNS:
                match = pattern.search(text)
                if match:
                    failures.append(f"{rel}: forbidden match {match.group(0)!r}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="run all validation checks")
    parser.add_argument("--list", action="store_true", help="print archive file list")
    args = parser.parse_args()

    entries = collect_entries()
    if not entries:
        print("ERROR: no files selected for supplement", file=sys.stderr)
        return 2

    if args.list:
        for entry in entries:
            print(entry.arcname.as_posix())
        if not args.check:
            return 0

    failures = scan_text(entries)
    if failures:
        print("ERROR: anonymization scan failed before zip creation:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 2

    write_zip(entries, args.out)
    size = args.out.stat().st_size

    zip_failures = scan_zip(args.out)
    if zip_failures:
        print("ERROR: anonymization scan failed after zip extraction:", file=sys.stderr)
        for failure in zip_failures:
            print(f"  - {failure}", file=sys.stderr)
        return 2

    if size > MAX_BYTES:
        print(
            f"ERROR: supplement is {size / 1024 / 1024:.2f} MiB, "
            f"exceeding {MAX_BYTES / 1024 / 1024:.0f} MiB",
            file=sys.stderr,
        )
        return 2

    print(f"Wrote {args.out.relative_to(ROOT)}")
    print(f"Files: {len(entries)}")
    print(f"Size: {size / 1024 / 1024:.2f} MiB")

    if args.check:
        with tempfile.TemporaryDirectory(prefix="graph_snd_supp_check_") as td:
            check_root = Path(td)
            with zipfile.ZipFile(args.out, mode="r") as zf:
                zf.extractall(check_root)
            required = [
                "README.md",
                "SUPPLEMENT_README.md",
                "SUPPLEMENT_MANIFEST.md",
                "pyproject.toml",
                "graphsnd/metrics.py",
                "tests/test_metrics.py",
                "results/exp1/recovery.csv",
                "ControllingBehavioralDiversity-fork/GRAPH_SND_CHANGES.md",
            ]
            missing = [path for path in required if not (check_root / path).exists()]
            if missing:
                print("ERROR: zip is missing required files:", file=sys.stderr)
                for path in missing:
                    print(f"  - {path}", file=sys.stderr)
                return 2
            if shutil.which("python"):
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import sys; sys.path.insert(0, '.'); import graphsnd; "
                        "from graphsnd import snd, graph_snd, complete_edges; "
                        "print(graphsnd.__version__)",
                    ],
                    cwd=check_root,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if proc.returncode != 0:
                    print("ERROR: extracted import smoke test failed:", file=sys.stderr)
                    print(proc.stdout, file=sys.stderr)
                    print(proc.stderr, file=sys.stderr)
                    return proc.returncode
        print("Supplement check passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
