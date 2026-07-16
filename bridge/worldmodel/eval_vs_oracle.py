#!/usr/bin/env python3
"""
eval_vs_oracle.py — how lossy is the learned model against the analytic oracle?

On held-out TEST-split observations, query both planners over the 8 compass
headings and compare 1-step and 3-step predictions. Reports per-heading MAE
(energy, displacement) and — the decision-relevant number — how often both
planners agree on the cheapest heading (top-1 agreement).

Writes results into metrics.json under "learned_vs_analytic".
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import RESULTS_DIR, split_of  # noqa: E402
from planner import AnalyticPlanner, LearnedPlanner  # noqa: E402
from wmcommon import load_trace_records  # noqa: E402

N_OBS = 250
SEED = 0


def test_observations():
    """Observations from test-split episodes only (episode-level split integrity)."""
    obs_list = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_trace.jsonl"))):
        _, records = load_trace_records(path, require_real=True)
        for rec in records:
            if rec.get("kind") != "decision":
                continue
            ep_key = f"{os.path.basename(path)}|{rec.get('episode', -1)}"
            if split_of(ep_key) != "test":
                continue
            obs = rec.get("obs") or {}
            if "slope_pct" in obs and "pos_m" in obs and "terrain" in obs:
                obs_list.append(obs)
    rng = np.random.default_rng(SEED)
    if len(obs_list) > N_OBS:
        obs_list = [obs_list[i] for i in rng.choice(len(obs_list), N_OBS, replace=False)]
    return obs_list


def compare(horizon=1):
    """Horizon-1 comparison is the clean one: both planners predict the same
    single step (learned from observed slopes, oracle from true terrain).
    Beyond one step the learned planner's input depends on the agent's explored
    map, which is an agent property, not a model property."""
    learned = LearnedPlanner()
    oracle = AnalyticPlanner()
    de_err, disp_err, top1, regret = [], [], [], []
    try:
        for obs in test_observations():
            obs = dict(obs, food=[])            # compass-only candidates
            or_ = {r["heading_deg"]: r for r in oracle.plan(obs, horizon=horizon)}
            le = {r["heading_deg"]: r for r in learned.plan(obs, {}, horizon=horizon)}
            common = sorted(set(or_) & set(le))
            if not common:
                continue
            for h in common:
                de_err.append(abs(le[h]["net_dE_kJ"] - or_[h]["net_dE_kJ"]))
                disp_err.append(abs(le[h]["disp_m"] - or_[h]["disp_m"]))
            le_pick = max(common, key=lambda h: le[h]["net_dE_kJ"])
            or_pick = max(common, key=lambda h: or_[h]["net_dE_kJ"])
            top1.append(le_pick == or_pick)
            # regret: extra TRUE cost of following the learned model's choice —
            # near-tie disagreements on flat ground cost ~nothing, so this is the
            # decision-relevant number, not top-1 agreement
            regret.append(or_[or_pick]["net_dE_kJ"] - or_[le_pick]["net_dE_kJ"])
    finally:
        oracle.close()
    return {
        "horizon_steps": horizon,
        "n_obs": len(top1),
        "per_heading_dE_MAE_kJ": float(np.mean(de_err)),
        "per_heading_disp_MAE_m": float(np.mean(disp_err)),
        "cheapest_heading_top1_agreement": float(np.mean(top1)),
        "mean_choice_regret_kJ": float(np.mean(regret)),
        "p90_choice_regret_kJ": float(np.percentile(regret, 90)),
    }


def main():
    out = {"h1": compare(1)}
    path = os.path.join(HERE, "metrics.json")
    metrics = json.load(open(path)) if os.path.exists(path) else {}
    metrics["learned_vs_analytic"] = out
    with open(path, "w") as f:
        json.dump(metrics, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
