#!/usr/bin/env python3
"""
ablation.py — controlled comparison of planner conditions on byte-identical
seed sets. Conditions:
  mock      heuristic baseline (real physics, scripted policy, no LLM)
  off       LLM, no planner
  learned   LLM + learned dynamics model  (--planner learned)
  analytic  LLM + ground-truth oracle     (--planner analytic)
  paired    LLM, within-episode control: one agent plans (learned), the other
            doesn't; roles swap each episode (counterbalanced)

Memory files for the condition's tag are wiped before each run so every
condition starts equally ignorant. Seeds cycle through --seeds so the episode
set spans multiple terrains; the same seed list is passed to every condition.

Outputs: per-condition results/trace files, one raw CSV of agent-episode rows,
and a summary table with bootstrap/Wilson 95% CIs (analysis.py).

Usage:
  venv/bin/python bridge/worldmodel/ablation.py --episodes 15 \
      --conditions mock,off,learned,analytic,paired
"""
import argparse
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis  # noqa: E402

BRIDGE = os.path.dirname(HERE)
ROOT = os.path.dirname(BRIDGE)
VENV_PY = os.path.join(ROOT, "venv", "bin", "python")
RESULTS = os.path.join(BRIDGE, "results")
MEMORY = os.path.join(BRIDGE, "memory")
CHROMIUM = shutil.which("chromium") or "/snap/bin/chromium"


def sim_url(port):
    return f"http://127.0.0.1:8388/?control=llm&ws=ws://127.0.0.1:{port}"

CONDITION_FLAGS = {
    "mock": ["--mock"],
    "off": [],
    "learned": ["--planner", "learned"],
    "analytic": ["--planner", "analytic"],
    "paired": ["--planner", "learned", "--paired"],
}


def model_tag():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as r:
            return json.load(r)["data"][0]["id"]
    except Exception:
        return None


def wipe_memory(tag):
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(tag))
    for f in glob.glob(os.path.join(MEMORY, safe + "*")):
        os.remove(f)


def run_condition(name, extra_flags, args):
    print(f"\n=== condition: {name} ===", flush=True)
    before = set(glob.glob(os.path.join(RESULTS, "*.jsonl")))
    port = getattr(args, "port", 8390)
    cmd = [VENV_PY, os.path.join(BRIDGE, "mind_driver.py"),
           "--episodes", str(args.episodes), "--duration", str(args.duration),
           "--terrain", args.terrain, "--seeds", args.seeds,
           "--food-seed", str(args.food_seed), "--metab", str(args.metab),
           "--size", str(args.size), "--port", str(port)] + extra_flags
    driver = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(2)
    if driver.poll() is not None:               # instant death (port conflict, crash): say why
        print(f"  DRIVER DIED at startup (exit {driver.returncode}):", flush=True)
        for line in driver.stdout.read().splitlines()[-8:]:
            print("  ! " + line, flush=True)
        return None, None
    # unique debug port + profile dir so parallel ablation instances coexist
    browser = subprocess.Popen(
        [CHROMIUM, "--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
         f"--remote-debugging-port={9223 + port - 8390}",
         f"--user-data-dir=/tmp/hillsim-chrome-{port}",
         "--window-size=1200,800", sim_url(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + args.episodes * (args.duration + 90) + 300
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
        if driver.poll() is None:               # never leave a driver squatting on the port
            driver.terminate()
        try:
            driver.wait(timeout=10)
        except subprocess.TimeoutExpired:
            driver.kill()
    new = sorted(set(glob.glob(os.path.join(RESULTS, "*.jsonl"))) - before, key=os.path.getmtime)
    res = [f for f in new if not f.endswith("_trace.jsonl")]
    trace = [f for f in new if f.endswith("_trace.jsonl")]
    return (res[-1] if res else None), (trace[-1] if trace else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--duration", type=int, default=240)
    ap.add_argument("--terrain", default="perlin")
    ap.add_argument("--seeds", default="1337,42,2718,9001,31415",
                    help="terrain seeds cycled across episodes — the documented spread")
    ap.add_argument("--food-seed", type=int, default=777)
    ap.add_argument("--metab", type=int, default=15)
    ap.add_argument("--size", type=int, default=50)
    ap.add_argument("--conditions", default="mock,off,learned,analytic")
    ap.add_argument("--port", type=int, default=8390,
                    help="driver WS port — use distinct ports to run instances in parallel "
                         "(NOTE: only the CPU-only mock condition parallelises fairly; "
                         "LLM conditions share Delphi's single sequence slot)")
    args = ap.parse_args()

    tag = model_tag()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in CONDITION_FLAGS]
    if unknown:
        sys.exit(f"unknown conditions: {unknown}")
    if any(c != "mock" for c in conditions) and not tag:
        sys.exit("LLM conditions requested but Delphi (:8000) is not reachable")

    stamp = int(time.time())
    all_rows = []
    for cond in conditions:
        wipe_memory("mock" if cond == "mock" else tag)
        rf, tf = run_condition(f"{cond} ({'mock' if cond == 'mock' else tag})",
                               CONDITION_FLAGS[cond], args)
        if rf:
            all_rows += analysis.rows_from_results(rf, tf, condition=cond)
        else:
            print(f"  WARNING: no results for condition {cond}", flush=True)

    csv_path = os.path.join(HERE, f"ablation_{stamp}.csv")
    analysis.write_csv(all_rows, csv_path)
    title = (f"Ablation — {tag or 'no-LLM'} · {args.episodes} eps × {args.duration}s · "
             f"{args.terrain} seeds [{args.seeds}] · metab {args.metab}x")
    table = analysis.fmt_table(analysis.summarize(all_rows), title)
    paired = analysis.paired_analysis(all_rows)
    out = table
    if paired.get("n_pairs"):
        out += "\n\n## Paired within-episode differences (planner − off)\n"
        out += json.dumps(paired, indent=1)
    print(out)
    md_path = os.path.join(HERE, f"ablation_{stamp}.md")
    with open(md_path, "w") as f:
        f.write(out + "\n")
    print(f"\nsaved -> {md_path}\nraw rows -> {csv_path}")


if __name__ == "__main__":
    main()
