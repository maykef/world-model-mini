#!/usr/bin/env python3
"""
ablation.py — run the SAME deterministic seed set under multiple conditions and
compare outcomes. Conditions: heuristic baseline (mock), LLM without the world
model, LLM with --worldmodel. Memory files for the condition's tag are wiped
before each run so every condition starts equally ignorant.

Orchestrates: mind_driver subprocess (exits after its benchmark) + a headless
Chromium running the real sim connected over ws://. Requires serve.py on :8388
and, for LLM conditions, Delphi serving a model on :8000.

Usage: venv/bin/python bridge/worldmodel/ablation.py --episodes 3 --duration 240
"""
import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BRIDGE = os.path.dirname(HERE)
ROOT = os.path.dirname(BRIDGE)
VENV_PY = os.path.join(ROOT, "venv", "bin", "python")
RESULTS = os.path.join(BRIDGE, "results")
MEMORY = os.path.join(BRIDGE, "memory")
CHROMIUM = shutil.which("chromium") or "/snap/bin/chromium"
SIM_URL = "http://127.0.0.1:8388/?control=llm&ws=ws://127.0.0.1:8390"


def model_tag():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as r:
            return json.load(r)["data"][0]["id"]
    except Exception:
        return None


def wipe_memory(tag):
    import re
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(tag))
    for f in glob.glob(os.path.join(MEMORY, safe + "*")):
        os.remove(f)


def run_condition(name, extra_flags, args):
    print(f"\n=== condition: {name} ===", flush=True)
    before = set(glob.glob(os.path.join(RESULTS, "*.jsonl")))
    cmd = [VENV_PY, os.path.join(BRIDGE, "mind_driver.py"),
           "--episodes", str(args.episodes), "--duration", str(args.duration),
           "--terrain", args.terrain, "--seed", str(args.seed),
           "--food-seed", str(args.food_seed), "--metab", str(args.metab),
           "--size", str(args.size)] + extra_flags
    driver = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(2)
    browser = subprocess.Popen(
        [CHROMIUM, "--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
         "--remote-debugging-port=9223", "--window-size=1200,800", SIM_URL],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + args.episodes * (args.duration + 60) + 300
    try:
        while driver.poll() is None and time.time() < deadline:
            line = driver.stdout.readline()
            if line:
                print("  " + line.rstrip(), flush=True)
        if driver.poll() is None:
            print("  TIMEOUT — killing driver", flush=True)
            driver.send_signal(signal.SIGTERM)
    finally:
        browser.terminate()
        try:
            driver.wait(timeout=10)
        except subprocess.TimeoutExpired:
            driver.kill()
    new = sorted(set(glob.glob(os.path.join(RESULTS, "*.jsonl"))) - before, key=os.path.getmtime)
    res = [f for f in new if not f.endswith("_trace.jsonl")]
    trace = [f for f in new if f.endswith("_trace.jsonl")]
    return (res[-1] if res else None), (trace[-1] if trace else None)


def summarize(results_file, trace_file):
    if not results_file:
        return None
    eps = [json.loads(l) for l in open(results_file)]
    per_agent = {}
    for e in eps:
        for m in e["metrics"]:
            per_agent.setdefault(m["name"], []).append(m)
    lat = []
    if trace_file:
        for line in open(trace_file):
            r = json.loads(line)
            if r.get("kind") == "decision":
                lat.append(r.get("latency_s", 0))
    def avg(v):
        return sum(v) / len(v) if v else 0
    rows = {}
    for name, ms in per_agent.items():
        rows[name] = {
            "episodes": len(ms),
            "survival_rate": avg([1 if m["alive"] else 0 for m in ms]),
            "mean_survived_s": avg([m["survived_s"] for m in ms]),
            "mean_eaten": avg([m["eaten"] for m in ms]),
            "mean_dist_m": avg([m["dist_m"] for m in ms]),
            "mean_final_kJ": avg([m["final_energy_kJ"] for m in ms]),
        }
    return {"agents": rows, "mean_decision_latency_s": round(avg(lat), 2), "decisions": len(lat)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--duration", type=int, default=240)
    ap.add_argument("--terrain", default="perlin")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--food-seed", type=int, default=777)
    ap.add_argument("--metab", type=int, default=15)
    ap.add_argument("--size", type=int, default=50)
    ap.add_argument("--conditions", default="mock,off,on",
                    help="comma list from: mock, off, on")
    args = ap.parse_args()

    tag = model_tag()
    conditions = args.conditions.split(",")
    if ("off" in conditions or "on" in conditions) and not tag:
        sys.exit("LLM conditions requested but Delphi (:8000) is not reachable")

    out = {}
    for cond in conditions:
        if cond == "mock":
            wipe_memory("mock")
            files = run_condition("heuristic baseline (mock)", ["--mock"], args)
        elif cond == "off":
            wipe_memory(tag)
            files = run_condition(f"{tag} without world model", [], args)
        elif cond == "on":
            wipe_memory(tag)
            files = run_condition(f"{tag} WITH world model", ["--worldmodel"], args)
        else:
            continue
        out[cond] = summarize(*files)

    # table
    cols = ["survival_rate", "mean_survived_s", "mean_eaten", "mean_dist_m", "mean_final_kJ"]
    lines = [f"\n# Ablation — {tag or 'no-LLM'} · {args.episodes} episodes × {args.duration}s · "
             f"terrain {args.terrain}:{args.seed} · metab {args.metab}x · food_seed {args.food_seed}",
             "", "| condition | agent | " + " | ".join(cols) + " | mean_latency_s |",
             "|" + "---|" * (len(cols) + 3)]
    for cond, s in out.items():
        if not s:
            lines.append(f"| {cond} | — | (no results) |")
            continue
        for agent, r in s["agents"].items():
            lines.append(f"| {cond} | {agent} | " +
                         " | ".join(f"{r[c]:.2f}" for c in cols) +
                         f" | {s['mean_decision_latency_s']:.2f} |")
    table = "\n".join(lines)
    print(table)
    path = os.path.join(HERE, f"ablation_{int(time.time())}.md")
    with open(path, "w") as f:
        f.write(table + "\n")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
