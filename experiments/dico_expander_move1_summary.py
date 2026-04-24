#!/usr/bin/env python3
"""Summarise Move 1 DiCo logs: expander-$G$ at $n=10$ (multi-seed) and optional $n=50$.

Reads ``graph_snd_log.csv`` files from
``ControllingBehavioralDiversity-fork/results/neurips_final/`` and
``.../neurips_final_n50/``, writes
``results/dico_expander_move1/summary.json`` for paper / checklist.

Usage (from repo root)::

    python experiments/dico_expander_move1_summary.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
N10 = _REPO / "ControllingBehavioralDiversity-fork/results/neurips_final"
N50 = _REPO / "ControllingBehavioralDiversity-fork/results/neurips_final_n50"
OUT_DIR = _REPO / "results/dico_expander_move1"


def _late(rows: List[dict], k: int) -> List[dict]:
    return rows[-k:] if len(rows) >= k else rows


def _summarise_n10(late_k: int = 10, des: float = 0.1) -> Dict[str, Any]:
    variants = {
        "ippo": "ippo",
        "full": "full",
        "knn": "knn",
        "bern": "bern",
        "expander": "expander",
    }
    out: Dict[str, Any] = {}
    for key, sub in variants.items():
        per_seed_reward: List[float] = []
        per_seed_applied: List[float] = []
        per_seed_metric_med: List[float] = []
        for seed in (0, 1, 2):
            p = N10 / f"seed{seed}" / sub / "graph_snd_log.csv"
            if not p.is_file():
                raise FileNotFoundError(p)
            rows = list(csv.DictReader(p.open()))
            tail = _late(rows, late_k)
            per_seed_reward.append(float(np.mean([float(r["reward_mean"]) for r in tail])))
            if key != "ippo":
                per_seed_applied.append(
                    float(np.mean([float(r["applied_snd"]) for r in tail]))
                )
                m = [
                    float(r["metric_time_ms"])
                    for r in tail
                    if r.get("metric_time_ms") not in ("", None)
                ]
                per_seed_metric_med.append(float(np.median(m)) if m else float("nan"))
        r_arr = np.array(per_seed_reward)
        block: Dict[str, Any] = {
            "reward_late_mean": float(r_arr.mean()),
            "reward_late_sem": float(r_arr.std(ddof=1) / np.sqrt(len(r_arr))),
        }
        if key != "ippo":
            a_arr = np.array(per_seed_applied)
            block["applied_late_mean"] = float(a_arr.mean())
            block["applied_late_sem"] = float(a_arr.std(ddof=1) / np.sqrt(len(a_arr)))
            block["tracking_rel_err_pct_mean"] = float(
                np.mean(np.abs(a_arr - des) / des * 100.0)
            )
            block["metric_time_ms_median_of_late"] = float(np.nanmean(per_seed_metric_med))
        out[key] = block
    return {"n_agents": 10, "late_iterations": late_k, "snd_des": des, "variants": out}


def _summarise_n50_seed0(late_k: int = 10, des: float = 0.1) -> Dict[str, Any] | None:
    if not (N50 / "seed0").is_dir():
        return None
    block: Dict[str, Any] = {}
    for sub in ("expander", "bern", "ippo"):
        p = N50 / "seed0" / sub / "graph_snd_log.csv"
        if not p.is_file():
            continue
        rows = list(csv.DictReader(p.open()))
        tail = _late(rows, late_k)
        r = float(np.mean([float(x["reward_mean"]) for x in tail]))
        entry: Dict[str, Any] = {"reward_late_mean": r}
        if sub != "ippo":
            a = float(np.mean([float(x["applied_snd"]) for x in tail]))
            m = [float(x["metric_time_ms"]) for x in tail if x.get("metric_time_ms")]
            entry["applied_late_mean"] = a
            entry["tracking_rel_err_pct"] = float(abs(a - des) / des * 100.0)
            entry["metric_time_ms_median_late"] = float(np.median(m)) if m else float("nan")
        block[sub] = entry
    if not block:
        return None
    return {"n_agents": 50, "late_iterations": late_k, "snd_des": des, "seed": 0, "variants": block}


def main() -> None:
    summary: Dict[str, Any] = {"n10": _summarise_n10()}
    s50 = _summarise_n50_seed0()
    if s50 is not None:
        summary["n50_seed0"] = s50
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
