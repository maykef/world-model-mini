#!/usr/bin/env python3
"""
analysis.py — turn benchmark results into tidy CSV + tables with proper intervals.

Consumes results JSONLs (episode metrics, written by mind_driver benchmarks) and
their trace files (for latency; guard-checked). Emits:
  - raw per-agent-episode rows as CSV (re-analysable without re-running)
  - per-condition summary: mean, std, bootstrap 95% CI for continuous metrics;
    Wilson 95% CI for survival proportion
  - paired analysis (for --paired runs): within-episode planner-minus-off
    differences with bootstrap CI over episodes

Used by ablation.py and grid.py; runnable standalone on any results set.
"""
import csv
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wmcommon import load_trace_records  # noqa: E402

METRICS = ["survived_s", "eaten", "dist_m", "final_energy_kJ"]
BOOT = 10000
SEED = 0


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values, stat=np.mean, n_boot=BOOT, seed=SEED):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    stats = stat(v[idx], axis=1)
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def rows_from_results(results_file, trace_file=None, condition=None):
    """One row per agent-episode. Latency and planner role come from the trace."""
    lat = {}    # (episode, agent) -> [latencies]
    role = {}   # (episode, agent) -> planner mode
    if trace_file and os.path.exists(trace_file):
        _, records = load_trace_records(trace_file, require_real=True)
        for r in records:
            if r.get("kind") != "decision":
                continue
            k = (r.get("episode"), r.get("agent"))
            lat.setdefault(k, []).append(r.get("latency_s", 0))
            role[k] = r.get("planner", "unknown")
    rows = []
    with open(results_file) as f:
        for line in f:
            rec = json.loads(line)
            for m in rec["metrics"]:
                k = (rec["episode"], m["name"])
                rows.append({
                    "condition": condition or rec.get("planner", "?"),
                    "model": rec.get("model", "?"),
                    "planner": (rec.get("roles") or {}).get(m["name"], rec.get("planner", "off")),
                    "paired": bool(rec.get("paired")),
                    "episode": rec["episode"],
                    "terrain_seed": rec["cfg"].get("seed"),
                    "food_seed": rec["cfg"].get("food_seed"),
                    "agent": m["name"],
                    "alive": int(bool(m["alive"])),
                    "survived_s": m["survived_s"],
                    "eaten": m["eaten"],
                    "dist_m": m["dist_m"],
                    "final_energy_kJ": m["final_energy_kJ"],
                    "mean_latency_s": round(float(np.mean(lat[k])), 2) if k in lat else None,
                    "trace_role": role.get(k),
                })
    return rows


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(rows):
    """Per (condition) summary with intervals."""
    out = {}
    by = {}
    for r in rows:
        by.setdefault(r["condition"], []).append(r)
    for cond, rs in sorted(by.items()):
        n = len(rs)
        alive = sum(r["alive"] for r in rs)
        s = {"n_agent_episodes": n,
             "survival": alive / n if n else 0,
             "survival_ci95": wilson_ci(alive, n)}
        for m in METRICS:
            v = [r[m] for r in rs]
            s[m] = {"mean": float(np.mean(v)), "std": float(np.std(v)),
                    "ci95": bootstrap_ci(v)}
        lats = [r["mean_latency_s"] for r in rs if r["mean_latency_s"] is not None]
        s["mean_latency_s"] = float(np.mean(lats)) if lats else None
        out[cond] = s
    return out


def paired_analysis(rows):
    """Within-episode planner-minus-off differences (rows from a --paired run)."""
    by_ep = {}
    for r in rows:
        if not r["paired"]:
            continue
        by_ep.setdefault(r["episode"], {})[r["planner"]] = r
    diffs = {m: [] for m in METRICS + ["alive"]}
    for ep, pair in sorted(by_ep.items()):
        on = next((v for k, v in pair.items() if k != "off"), None)
        off = pair.get("off")
        if not on or not off:
            continue
        for m in METRICS + ["alive"]:
            diffs[m].append(on[m] - off[m])
    result = {"n_pairs": len(diffs["survived_s"])}
    for m, v in diffs.items():
        if v:
            result[m] = {"mean_paired_diff": float(np.mean(v)),
                         "ci95": bootstrap_ci(v)}
    return result


def fmt_table(summary, title):
    lines = [f"\n# {title}", "",
             "| condition | n | survival [95% CI] | survived_s | final_kJ | eaten | dist_m | latency_s |",
             "|---|---|---|---|---|---|---|---|"]
    for cond, s in summary.items():
        ci = s["survival_ci95"]
        def m(k):
            d = s[k]
            return f"{d['mean']:.1f} [{d['ci95'][0]:.1f},{d['ci95'][1]:.1f}]"
        lines.append(f"| {cond} | {s['n_agent_episodes']} | "
                     f"{s['survival']:.2f} [{ci[0]:.2f},{ci[1]:.2f}] | "
                     f"{m('survived_s')} | {m('final_energy_kJ')} | {m('eaten')} | {m('dist_m')} | "
                     f"{s['mean_latency_s'] if s['mean_latency_s'] is not None else '—'} |")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="results jsonl files (traces auto-derived)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    rows = []
    for rf in args.results:
        rows += rows_from_results(rf, rf.replace(".jsonl", "_trace.jsonl"))
    if args.csv:
        write_csv(rows, args.csv)
    print(fmt_table(summarize(rows), "summary"))
    pa = paired_analysis(rows)
    if pa.get("n_pairs"):
        print("\npaired:", json.dumps(pa, indent=1))
