#!/usr/bin/env python3
"""
dataset.py — parse decision traces (bridge/results/*_trace.jsonl) into
single-step transitions for the learned dynamics model.

A transition is two consecutive decisions of the SAME agent in the SAME
episode ~2 s apart:
    INPUT  x (13): [energy_kJ, slope_toward_heading_pct,
                    slope_pct N NE E SE S SW W NW, action_speed,
                    wading_flag, metabolic_scale]
    TARGET y (2):  [delta_energy_kJ, displacement_m]

The model learns the COST FUNCTION (terrain+action -> energy/movement), not
the map: position is reconstructed downstream from heading + displacement.

Excluded transitions (documented, deliberate):
  - eating events (obs2.eaten > obs1.eaten): the energy jump is a discrete
    game event the planner accounts for separately, not locomotion dynamics
  - gaps (dt outside [1.5, 2.5] s): a decision was skipped/laggy
Splits are BY EPISODE (stable hash), never by row.
"""
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wmcommon import COMPASS, contiguous, load_trace_records, slope_toward  # noqa: E402

RESULTS_DIR = os.path.join(HERE, "..", "results")
FEATURE_NAMES = (["energy_kJ", "slope_heading_pct"]
                 + [f"slope_{d}_pct" for d in COMPASS]
                 + ["speed_ms", "wading", "metab"])
TARGET_NAMES = ["delta_energy_kJ", "displacement_m"]


def obs_features(obs, action):
    wading = 1.0 if (obs.get("water_level_m", 0) > 0
                     and obs["pos_m"]["alt"] < obs["water_level_m"]) else 0.0
    return [
        float(obs["energy_kJ"]),
        slope_toward(obs["slope_pct"], action["heading_deg"]),
        *[float(obs["slope_pct"][d]) for d in COMPASS],
        float(action["speed"]),
        wading,
        float(obs.get("metabolic_scale", 10)),
    ]


def transitions_from_file(path):
    """Yield (episode_key, x, y) transitions from one trace file.
    Enforces the synthetic-data guard: raises SyntheticDataError on untagged
    or implausible (frozen-position) data."""
    _, records = load_trace_records(path, require_real=True)
    by_agent = {}
    for rec in records:
        if rec.get("kind") != "decision":
            continue
        obs = rec.get("obs") or {}
        if "slope_pct" not in obs or "pos_m" not in obs:
            continue
        key = (rec.get("episode", -1), rec.get("agent", "?"))
        by_agent.setdefault(key, []).append(rec)

    for (ep, agent), recs in by_agent.items():
        recs.sort(key=lambda r: r["t_s"])
        ep_key = f"{os.path.basename(path)}|{ep}"
        seq_key = f"{ep_key}|{agent}"
        for r1, r2 in zip(recs, recs[1:]):
            o1, o2 = r1["obs"], r2["obs"]
            if not contiguous(o1["t_s"], o2["t_s"]):
                continue
            if o2.get("eaten", 0) > o1.get("eaten", 0):
                continue                                    # discrete eat event
            x = obs_features(o1, r1["action"])
            dx = o2["pos_m"]["x"] - o1["pos_m"]["x"]
            dz = o2["pos_m"]["z"] - o1["pos_m"]["z"]
            y = [float(o2["energy_kJ"] - o1["energy_kJ"]), math.hypot(dx, dz)]
            yield ep_key, seq_key, float(o1["t_s"]), x, y


def split_of(ep_key):
    h = int(hashlib.md5(ep_key.encode()).hexdigest(), 16) % 10
    return "train" if h < 8 else ("val" if h == 8 else "test")


def build(results_dir=RESULTS_DIR, out_dir=HERE):
    data = {s: {"X": [], "Y": [], "ep": [], "seq": [], "t": []} for s in ("train", "val", "test")}
    files = sorted(glob.glob(os.path.join(results_dir, "*_trace.jsonl")))
    for path in files:
        for ep_key, seq_key, t, x, y in transitions_from_file(path):
            s = split_of(ep_key)                # split by EPISODE: both agents same side
            data[s]["X"].append(x)
            data[s]["Y"].append(y)
            data[s]["ep"].append(ep_key)
            data[s]["seq"].append(seq_key)      # rollout chains stay per-agent
            data[s]["t"].append(t)

    arrays, counts = {}, {}
    for s, d in data.items():
        arrays[f"X_{s}"] = np.asarray(d["X"], dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
        arrays[f"Y_{s}"] = np.asarray(d["Y"], dtype=np.float32).reshape(-1, len(TARGET_NAMES))
        arrays[f"ep_{s}"] = np.asarray(d["ep"])
        arrays[f"seq_{s}"] = np.asarray(d["seq"])
        arrays[f"t_{s}"] = np.asarray(d["t"], dtype=np.float32)
        counts[s] = len(d["X"])

    xm = arrays["X_train"].mean(0)
    xs = arrays["X_train"].std(0) + 1e-6
    ym = arrays["Y_train"].mean(0)
    ys = arrays["Y_train"].std(0) + 1e-6
    norm = {
        "x_mean": xm.tolist(), "x_std": xs.tolist(),
        "y_mean": ym.tolist(), "y_std": ys.tolist(),
        "feature_names": FEATURE_NAMES, "target_names": TARGET_NAMES,
        "counts": counts, "files": [os.path.basename(f) for f in files],
    }
    np.savez_compressed(os.path.join(out_dir, "data.npz"), **arrays)
    with open(os.path.join(out_dir, "norm.json"), "w") as f:
        json.dump(norm, f, indent=1)
    return counts


if __name__ == "__main__":
    counts = build()
    print("transitions:", counts)
