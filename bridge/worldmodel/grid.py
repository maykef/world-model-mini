#!/usr/bin/env python3
"""
grid.py — the full minds × planner-modes grid on one shared seed set.

Grid: {heuristic} ∪ {served models} × {off, learned, analytic}. The heuristic
ignores planner mode and runs once as the fixed baseline row.

Delphi serves ONE model at a time. Two ways to cover multiple models:
  - default: run the grid for whatever model is currently served; re-run
    grid.py after switching models in Delphi — results accumulate and the
    combined table is rebuilt from everything on disk each time.
  - --auto-switch: let grid.py switch models itself via the Delphi deploy
    script (RESTARTS the live vLLM service — only use when nobody needs it).

One command for the full grid (with consent to switch):
  venv/bin/python bridge/worldmodel/grid.py --episodes 15 \
      --models qwen3.5-122b,gemma-4-31b-heretic,xortron-123b --auto-switch
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis  # noqa: E402
from ablation import model_tag, run_condition, wipe_memory, CONDITION_FLAGS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
SWITCH_SCRIPT = "/mnt/nvme8tb/qwen3.5-cpa/deploy_qwen35_vllm.sh"
GRID_DIR = os.path.join(HERE, "grid")
PLANNER_MODES = ["off", "learned", "analytic"]


def switch_model(key, timeout_s=900):
    print(f"switching Delphi to {key} (this restarts the live vLLM service)…", flush=True)
    subprocess.run(["bash", SWITCH_SCRIPT, "switch", key], check=True)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        tag = model_tag()
        if tag == key:
            print(f"Delphi now serving {key}", flush=True)
            return True
        time.sleep(10)
    raise TimeoutError(f"Delphi did not come up with {key} within {timeout_s}s")


def load_all_rows():
    rows = []
    for f in sorted(glob.glob(os.path.join(GRID_DIR, "rows_*.json"))):
        rows += json.load(open(f))
    return rows


def combined_table(rows):
    """Model × planner-mode table with CIs."""
    keyed = {}
    for r in rows:
        key = f"{r['model']} · {r['condition']}"
        keyed.setdefault(key, []).append(dict(r, condition=key))
    flat = [r for rs in keyed.values() for r in rs]
    return analysis.fmt_table(analysis.summarize(flat), "Grid — minds × planner modes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--duration", type=int, default=240)
    ap.add_argument("--terrain", default="perlin")
    ap.add_argument("--seeds", default="1337,42,2718,9001,31415")
    ap.add_argument("--food-seed", type=int, default=777)
    ap.add_argument("--metab", type=int, default=15)
    ap.add_argument("--size", type=int, default=50)
    ap.add_argument("--models", default=None,
                    help="comma list of Delphi model keys; default: whatever is served now")
    ap.add_argument("--auto-switch", action="store_true",
                    help="switch Delphi between models automatically (restarts the service)")
    ap.add_argument("--skip-heuristic", action="store_true")
    args = ap.parse_args()

    os.makedirs(GRID_DIR, exist_ok=True)
    served = model_tag()
    targets = [m.strip() for m in (args.models or served or "").split(",") if m.strip()]
    if not targets:
        sys.exit("no model served and none requested")

    done_keys = {(r["model"], r["condition"]) for r in load_all_rows()}

    def run_and_store(condition, flags, model_name):
        if (model_name, condition) in done_keys:
            print(f"skip {model_name} · {condition} (already on disk)", flush=True)
            return
        wipe_memory(model_name)
        rf, tf = run_condition(f"{condition} ({model_name})", flags, args)
        if not rf:
            print(f"WARNING: no results for {model_name} · {condition}", flush=True)
            return
        rows = analysis.rows_from_results(rf, tf, condition=condition)
        for r in rows:
            r["model"] = model_name
        with open(os.path.join(GRID_DIR, f"rows_{model_name}_{condition}_{int(time.time())}.json".replace("/", "_")), "w") as f:
            json.dump(rows, f)

    if not args.skip_heuristic:
        run_and_store("heuristic", CONDITION_FLAGS["mock"], "heuristic")

    for model in targets:
        if model_tag() != model:
            if args.auto_switch:
                switch_model(model)
            else:
                print(f"\n{model} is not being served — switch Delphi to it and re-run "
                      "grid.py (results so far are kept), or pass --auto-switch.", flush=True)
                continue
        for mode in PLANNER_MODES:
            run_and_store(mode, CONDITION_FLAGS[mode], model)

    rows = load_all_rows()
    table = combined_table(rows)
    print(table)
    path = os.path.join(GRID_DIR, "grid_table.md")
    with open(path, "w") as f:
        f.write(table + "\n")
    analysis.write_csv(rows, os.path.join(GRID_DIR, "grid_rows.csv"))
    print(f"\nsaved -> {path} (+ grid_rows.csv)")


if __name__ == "__main__":
    main()
