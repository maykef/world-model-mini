#!/usr/bin/env python3
"""
mind_driver.py — bridges the Hill Physics Sandbox agents to an LLM served by
Delphi (vLLM, OpenAI-compatible, port 8000).

The sim (index.html) connects here over WebSocket when "LLM bridge" mode is on.
Protocol (JSON messages):
  sim -> driver:  {type:"hello", agents:[names]}
                  {type:"obs", t_s, episode_active, agents:[{id, alive, obs}]}
                  {type:"reset_done", cfg}
                  {type:"episode_end", t_s, metrics:[...]}
  driver -> sim:  {type:"act", id, heading_deg, speed, reason}
                  {type:"reset", cfg:{terrain,seed,size,water,metab,
                                      food_seed,food_interval_s,first_food_s,episode_s}}

Modes:
  --mock            heuristic policy instead of the LLM (protocol testing, no GPU)
  --probe           one LLM call against a canned observation, print it, exit
  --episodes N      benchmark: N reset->run->episode_end cycles, JSONL results
  (no flags)        free-play: just answer observations with LLM decisions

Delphi serves ONE model at a time (MAX_NUM_SEQS=1) — requests are serialized
via a lock. Results are tagged with the model reported by /v1/models, so run
the same benchmark, switch models in Delphi, and run it again to compare.
"""
import argparse, asyncio, json, logging, math, os, re, sys, time, urllib.request
logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets openai (use the project venv)")

API_BASE = os.environ.get("DELPHI_API", "http://127.0.0.1:8000/v1")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory")


def terrain_key(terrain, seed, size):
    """Journal key: lessons only transfer to the same landscape."""
    return f"{terrain}:{seed if terrain == 'perlin' else 0}:{size}"


def obs_terrain_key(obs):
    return terrain_key(obs.get("terrain", "?"), obs.get("terrain_seed", 0), obs.get("world_size_m", 0))

SYSTEM_PROMPT = """You are {name}, a creature living on rugged 3D terrain. Survive by eating food (balls that drop from the sky) while managing a limited energy budget.

Physics you must respect:
- Moving costs energy per metre; the STEEPER the uphill slope, the more it costs (a +40% slope costs ~4x more than flat). Downhill is cheap. Standing still still burns basal energy.
- Water (if present) slows you and quadruples movement cost.
- If energy reaches 0 you die.
- EATING: one ball is a FULL recharge, but you can only eat when your energy is below the eat_below_pct threshold (check can_eat in the observation). Food you don't eat STAYS where it fell.
- You only SEE food within your vision range. Food you walked away from is invisible until you return — MEMORIZE locations (positions are in metres, x/z) so you can go back when your energy runs low, or take the risk of exploring for new drops.
- Another creature competes for the same food; whoever arrives first (and is hungry) eats it.

You receive observations as JSON: your energy, can_eat flag, position, terrain slope % in 8 compass directions (positive = uphill), and currently-visible food with compass bearing / distance / elevation change.

Reply with ONLY a JSON object, no other text:
{{"heading_deg": <0-360, compass: 0=N 90=E 180=S 270=W>, "speed": <0.0-2.0 m/s>, "reason": "<max 10 words>", "memory": "<notes to your future self; max 80 words>"}}

Your "memory" is echoed back to you on the next decision — it is your ONLY memory within this life. Use it for BOTH:
- food locations you leave behind (x/z coordinates), and
- terrain knowledge: which areas are steep, where ridges give cheap walking, which valleys to avoid (e.g. "NE corner very steep; ridge along x≈20 is a cheap north-south route").

Strategy: when sated, scout and map; when energy drops, weigh returning to a remembered ball against exploring; walk ridges rather than climbing steep valley walls."""

PROBE_OBS = {
    "name": "Amber", "t_s": 12.0, "energy_kJ": 310, "energy_pct": 39,
    "can_eat": True, "eat_below_pct": 55,
    "pos_m": {"x": 18.2, "z": 30.5, "alt": 4.1}, "heading_deg": 90, "speed_ms": 1.1,
    "slope_pct": {"N": 22, "NE": 31, "E": 38, "SE": 18, "S": 2, "SW": -6, "W": -14, "NW": 3},
    "food": [
        {"bearing_deg": 75, "dist_m": 26.0, "elev_m": 7.5, "in_water": False},
        {"bearing_deg": 190, "dist_m": 34.0, "elev_m": -1.2, "in_water": False},
    ],
    "other_agent": {"bearing_deg": 300, "dist_m": 22, "alive": True},
    "water_level_m": 0, "world_size_m": 50, "metabolic_scale": 10,
}


class WorldMemory:
    """Accurate, experience-grounded memory per (agent, terrain): only what the agent
    itself observed, recorded losslessly by the driver. The LLM reasons over it —
    this is 'training' as accumulated experience, not weight updates."""

    CELL = 5  # metres

    def __init__(self, path):
        self.path = path
        self.maps = {}       # "agent|terrain_key" -> {"cells":{}, "food":[], "rivals":{}, "iters_seen":[]}
        self._eaten_seen = {}
        self._dirty = 0
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.maps = json.load(f)
            except Exception:
                pass

    def _m(self, agent, key):
        return self.maps.setdefault(f"{agent}|{key}", {"cells": {}, "food": [], "rivals": {}, "iters_seen": []})

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.maps, f)
        self._dirty = 0

    def update(self, obs, iteration):
        name = obs.get("name", "?")
        key = obs_terrain_key(obs)
        m = self._m(name, key)
        if iteration not in m["iters_seen"]:
            m["iters_seen"] = (m["iters_seen"] + [iteration])[-50:]
        x, z = obs["pos_m"]["x"], obs["pos_m"]["z"]

        # terrain: visited cell -> altitude + worst slope seen there
        cell = f"{int(x // self.CELL) * self.CELL},{int(z // self.CELL) * self.CELL}"
        steep = max(abs(v) for v in obs["slope_pct"].values()) if obs.get("slope_pct") else 0
        c = m["cells"].get(cell, {"alt": 0, "steep": 0, "visits": 0})
        c["alt"] = obs["pos_m"]["alt"]
        c["steep"] = max(c["steep"], steep)
        c["visits"] += 1
        m["cells"][cell] = c

        # food: bearing/distance -> exact coordinates; dedupe within 2.5 m
        vision = 45 * obs.get("world_size_m", 50) / 50
        seen_now = []
        for f in obs.get("food", []):
            b = math.radians(f["bearing_deg"])
            fx = round(x + f["dist_m"] * math.sin(b), 1)
            fz = round(z - f["dist_m"] * math.cos(b), 1)
            seen_now.append((fx, fz))
            for known in m["food"]:
                if abs(known["x"] - fx) <= 2.5 and abs(known["z"] - fz) <= 2.5:
                    known.update(x=fx, z=fz, status="available", last_seen_t=obs["t_s"])
                    break
            else:
                m["food"].append({"x": fx, "z": fz, "status": "available",
                                  "first_iter": iteration, "last_seen_t": obs["t_s"]})
        # remembered food well inside current vision but NOT seen now -> someone took it
        for known in m["food"]:
            if known["status"] == "available" and math.hypot(known["x"] - x, known["z"] - z) < vision * 0.55:
                if not any(abs(known["x"] - sx) <= 2.5 and abs(known["z"] - sz) <= 2.5 for sx, sz in seen_now):
                    known["status"] = f"gone (taken, noticed iter {iteration})"

        # my own meals: eaten counter increased -> nearest remembered food was mine
        ek = f"{name}|{key}"
        prev = self._eaten_seen.get(ek, 0)
        if obs.get("eaten", 0) > prev and m["food"]:
            near = min(m["food"], key=lambda kf: math.hypot(kf["x"] - x, kf["z"] - z))
            if math.hypot(near["x"] - x, near["z"] - z) < 4:
                near["status"] = f"eaten by me (iter {iteration})"
        self._eaten_seen[ek] = obs.get("eaten", 0)

        # rivals
        oa = obs.get("other_agent")
        if oa:
            m["rivals"]["rival"] = {"alive": oa.get("alive"), "last_bearing": oa.get("bearing_deg"),
                                    "last_dist": oa.get("dist_m"), "t": obs["t_s"]}
        self._dirty += 1
        if self._dirty >= 25:
            self.save()

    def new_life(self, iteration, same_layout=True):
        """New episode. Groundhog (same_layout): food respawns identically -> reset availability.
        Different food seed: remembered positions would be false memories -> forget food, keep terrain."""
        self._eaten_seen = {}
        for m in self.maps.values():
            if same_layout:
                for f in m["food"]:
                    if str(f.get("status", "")).startswith(("eaten", "gone")):
                        f["status"] = "available"
            else:
                m["food"] = []

    def render(self, agent, key, world_size):
        mkey = f"{agent}|{key}"
        m = self.maps.get(mkey)
        if not m or not m["cells"]:
            return ""
        cells = m["cells"]
        total_cells = max(1, (int(world_size) // self.CELL) ** 2)
        explored = round(100 * len(cells) / total_cells)
        steep = sorted(((c, v) for c, v in cells.items() if v["steep"] >= 25),
                       key=lambda cv: -cv[1]["steep"])[:8]
        cheap = [c for c, v in cells.items() if v["steep"] <= 8][:8]
        lines = [f"\n\nYOUR WORLD MAP for this terrain (accurate — from your own eyes, {len(m['iters_seen'])} lives here, {explored}% explored):"]
        if m["food"]:
            lines.append("Food locations known (x, z):")
            for f in m["food"][:16]:
                lines.append(f"  - ({f['x']}, {f['z']}) {f['status']}")
        if steep:
            lines.append("Steep zones (5 m cells, worst slope %): " +
                         "; ".join(f"({c}) {v['steep']}%" for c, v in steep))
        if cheap:
            lines.append("Easy flat cells: " + "; ".join(f"({c})" for c in cheap))
        if m["rivals"].get("rival"):
            r = m["rivals"]["rival"]
            lines.append(f"Rivals: 1 other creature (last seen bearing {r['last_bearing']}deg, "
                         f"{r['last_dist']} m away, t={r['t']}s, alive={r['alive']}).")
        lines.append("Unvisited areas are NOT in this map — explore to complete it.")
        return "\n".join(lines)


def get_model_name():
    try:
        with urllib.request.urlopen(API_BASE + "/models", timeout=5) as r:
            return json.load(r)["data"][0]["id"]
    except Exception as e:
        return f"unknown({e.__class__.__name__})"


def parse_action(text):
    """Extract the first JSON object with heading_deg from model output."""
    for m in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "heading_deg" in o:
            h = float(o["heading_deg"]) % 360
            s = max(0.0, min(2.0, float(o.get("speed", 1.2))))
            return {"heading_deg": h, "speed": s, "reason": str(o.get("reason", ""))[:60],
                    "memory": str(o.get("memory", ""))[:400]}
    return None


def mock_policy(obs):
    """Greedy heuristic: nearest food when hungry, else wander. Used with --mock."""
    if obs.get("food") and obs.get("can_eat", True):
        f = min(obs["food"], key=lambda x: x["dist_m"])
        return {"heading_deg": f["bearing_deg"], "speed": 1.6, "reason": "mock: nearest food", "memory": ""}
    return {"heading_deg": (obs.get("heading_deg", 0) + 30) % 360, "speed": 1.0, "reason": "mock: wander", "memory": ""}


class Driver:
    def __init__(self, args):
        self.args = args
        self.model = None if args.mock else get_model_name()
        self.llm_lock = asyncio.Lock()          # MAX_NUM_SEQS=1 on Delphi
        self.latest_obs = {}                    # id -> obs (only newest matters)
        self.obs_event = asyncio.Event()
        self.episode_end = None                 # asyncio.Future during a benchmark
        self.history = {}                       # name -> [(obs_json, action_json), ...]
        self.memory = {}                        # name -> model-managed scratchpad string
        self.trace_path = None                  # per-run decision trace (JSONL)
        self.ep_idx = -1
        self.tag = "mock" if args.mock else self.model
        self.journal_path = os.path.join(
            MEMORY_DIR, re.sub(r"[^A-Za-z0-9._-]", "_", str(self.tag)) + ".json")
        self.journal = {}                       # terrain_key -> [lesson entries]
        if os.path.exists(self.journal_path):
            try:
                with open(self.journal_path) as f:
                    self.journal = json.load(f)
            except Exception:
                pass
        self.worldmem = WorldMemory(os.path.join(
            MEMORY_DIR, re.sub(r"[^A-Za-z0-9._-]", "_", str(self.tag)) + "__worldmap.json"))
        self.iteration = 0
        if not args.mock:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(base_url=API_BASE, api_key="none")

    def save_journal(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        for k in self.journal:
            self.journal[k] = self.journal[k][-20:]
        with open(self.journal_path, "w") as f:
            json.dump(self.journal, f, indent=1)

    def journal_text(self, key):
        entries = self.journal.get(key, [])
        if not entries:
            return ""
        return ("\n\nLESSONS FROM YOUR PAST LIVES on this exact terrain — trust and use them:\n"
                + "\n".join(f"- {e['lesson']}" for e in entries[-4:]))

    def trace(self, record):
        if self.trace_path:
            with open(self.trace_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    async def llm_decide(self, obs):
        name = obs.get("name", "Agent")
        hist = self.history.get(name, [])
        key = obs_terrain_key(obs)
        system = (SYSTEM_PROMPT.format(name=name)
                  + self.worldmem.render(name, key, obs.get("world_size_m", 50))
                  + self.journal_text(key))
        msgs = [{"role": "system", "content": system}]
        for h_obs, h_act in hist[-3:]:          # short window: last 3 decisions
            msgs.append({"role": "user", "content": h_obs})
            msgs.append({"role": "assistant", "content": h_act})
        user = json.dumps(obs)
        mem = self.memory.get(name)
        if mem:
            user += f'\nYour saved notes: "{mem}"'
        msgs.append({"role": "user", "content": user})
        async with self.llm_lock:
            t0 = time.time()
            resp = await self.client.chat.completions.create(
                model=self.model, messages=msgs,
                temperature=0.0, max_tokens=self.args.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": self.args.think}},
            )
            dt = time.time() - t0
        text = resp.choices[0].message.content or ""
        act = parse_action(text)
        if act:
            if act.get("memory"):
                self.memory[name] = act["memory"]
            self.history.setdefault(name, []).append((json.dumps(obs), json.dumps(act)))
            self.history[name] = self.history[name][-6:]
        return act, dt, text

    async def agent_worker(self, ws, agent_id):
        """Consume the newest observation for this agent, decide, send action."""
        last_t = -1.0
        while True:
            await self.obs_event.wait()
            obs = self.latest_obs.get(agent_id)
            if obs is None or obs["t_s"] == last_t:
                await asyncio.sleep(0.05)
                continue
            last_t = obs["t_s"]
            try:
                if self.args.mock:
                    act, dt = mock_policy(obs), 0.0
                else:
                    act, dt, _ = await self.llm_decide(obs)
            except Exception as e:
                print(f"  [{agent_id}] LLM error: {e}", flush=True)
                await asyncio.sleep(1)
                continue
            if act:
                await ws.send(json.dumps({"type": "act", "id": agent_id,
                                          "heading_deg": act["heading_deg"],
                                          "speed": act["speed"], "reason": act["reason"]}))
                self.trace({"kind": "decision", "episode": self.ep_idx, "agent": obs["name"],
                            "t_s": obs["t_s"], "obs": obs, "action": act,
                            "latency_s": round(dt, 2)})
                print(f"  [{obs['name']}] t={obs['t_s']:>6}s E={obs['energy_kJ']}kJ "
                      f"-> {act['heading_deg']:.0f}deg @{act['speed']:.1f}m/s "
                      f"({act['reason']}) [{dt:.1f}s]", flush=True)

    def episode_cfg(self, i):
        a = self.args
        return {"terrain": a.terrain, "seed": a.seed, "size": a.size, "water": a.water,
                "metab": a.metab, "food_seed": a.food_seed + i,
                "food_interval_s": a.food_interval, "first_food_s": 4, "episode_s": a.duration}

    async def reflect_all(self, end_msg):
        for metric in end_msg.get("metrics", []):
            try:
                await self.reflect(metric["name"], metric, end_msg["cfg"])
            except Exception as e:
                print(f"  reflection failed for {metric.get('name')}: {e}", flush=True)

    async def reflect(self, name, metric, cfg):
        """Post-episode reflection: distil durable terrain lessons into the journal."""
        key = terrain_key(cfg["terrain"], cfg["seed"], cfg["size"])
        outcome = (f"{'survived' if metric['alive'] else 'DIED at ' + str(metric['survived_s']) + 's'}, "
                   f"ate {metric['eaten']}, walked {metric['dist_m']} m, "
                   f"final energy {metric['final_energy_kJ']} kJ")
        if self.args.mock:
            lesson = f"(mock ep{self.ep_idx}) {outcome}; drops cover all quadrants."
        else:
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT.format(name=name)},
                {"role": "user", "content":
                    f"Your life just ended. Outcome: {outcome}. Your final notes were: "
                    f"\"{self.memory.get(name, '')}\".\n"
                    "In <=60 words, write the durable lessons about THIS terrain and strategy for "
                    "your next life here: steep areas (by coordinates/compass), cheap routes/ridges, "
                    "water hazards, what you would do differently. Plain text only."},
            ]
            async with self.llm_lock:
                resp = await self.client.chat.completions.create(
                    model=self.model, messages=msgs, temperature=0.0, max_tokens=1024,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            lesson = (resp.choices[0].message.content or "").strip()[:500]
        if lesson:
            self.journal.setdefault(key, []).append(
                {"ts": time.time(), "episode": self.ep_idx, "agent": name,
                 "outcome": outcome, "lesson": lesson})
            self.save_journal()
            print(f"  [{name}] lesson saved: {lesson[:100]}", flush=True)

    async def handle(self, ws):
        print(f"sim connected ({'MOCK' if self.args.mock else self.model})", flush=True)
        if not self.args.episodes and not self.trace_path:   # free-play sessions record too
            os.makedirs(RESULTS_DIR, exist_ok=True)
            self.trace_path = os.path.join(
                RESULTS_DIR,
                f"session_{re.sub(r'[^A-Za-z0-9._-]', '_', str(self.tag))}_{int(time.time())}_trace.jsonl")
        workers = []
        try:
            async def guarded():
                try:
                    await self.run_benchmark(ws)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    import traceback; traceback.print_exc()
            benchmark = asyncio.create_task(guarded()) if self.args.episodes else None
            async for raw in ws:
                m = json.loads(raw)
                if m["type"] == "hello":
                    print(f"hello: agents={m['agents']}", flush=True)
                    for w in workers: w.cancel()
                    workers = [asyncio.create_task(self.agent_worker(ws, i))
                               for i in range(len(m["agents"]))]
                elif m["type"] == "obs":
                    self.iteration = m.get("iteration", self.iteration)
                    if m.get("episode_active") or not self.args.episodes:
                        for a in m["agents"]:
                            if a["alive"] and a["obs"]:
                                self.worldmem.update(a["obs"], self.iteration)
                                self.latest_obs[a["id"]] = a["obs"]
                        self.obs_event.set()
                elif m["type"] == "reset_done":
                    self.trace({"kind": "reset_done", "episode": self.ep_idx,
                                "iteration": m.get("iteration"), "cfg": m.get("cfg")})
                    if not self.args.episodes:          # new life: scratchpad dies; map & journal survive
                        self.latest_obs.clear(); self.history.clear(); self.memory.clear()
                        same = (m.get("cfg") or {}).get("food_mode") == "preset"
                        self.worldmem.new_life(m.get("iteration", 0), same_layout=same)
                        self.worldmem.save()
                elif m["type"] == "episode_end":
                    self.trace({"kind": "episode_end", "episode": self.ep_idx,
                                "iteration": m.get("iteration"), "t_s": m.get("t_s"),
                                "food_left": m.get("food_left"), "metrics": m.get("metrics")})
                    if self.episode_end and not self.episode_end.done():
                        self.episode_end.set_result(m)
                    elif not self.args.episodes and m.get("cfg") and m.get("metrics"):
                        # sim-driven episode (Groundhog loop): learn between iterations
                        asyncio.create_task(self.reflect_all(m))
        finally:
            for w in workers: w.cancel()
            if benchmark: benchmark.cancel()
            print("sim disconnected", flush=True)

    async def run_benchmark(self, ws):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        tag = "mock" if self.args.mock else self.model
        stem = f"{re.sub(r'[^A-Za-z0-9._-]', '_', str(tag))}_{int(time.time())}"
        out = os.path.join(RESULTS_DIR, stem + ".jsonl")
        self.trace_path = os.path.join(RESULTS_DIR, stem + "_trace.jsonl")
        await asyncio.sleep(1.0)
        for i in range(self.args.episodes):
            cfg = self.episode_cfg(i)
            self.ep_idx = i
            self.latest_obs.clear(); self.history.clear(); self.memory.clear()
            self.worldmem.new_life(i, same_layout=False); self.worldmem.save()  # food_seed varies per episode
            self.episode_end = asyncio.get_event_loop().create_future()
            await ws.send(json.dumps({"type": "reset", "cfg": cfg}))
            print(f"episode {i+1}/{self.args.episodes} started: {cfg}", flush=True)
            end = await self.episode_end
            rec = {"model": tag, "episode": i, "cfg": cfg, "t_end_s": end["t_s"],
                   "metrics": end["metrics"], "ts": time.time()}
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            for metric in end["metrics"]:               # cross-episode learning
                try:
                    await self.reflect(metric["name"], metric, cfg)
                except Exception as e:
                    print(f"  reflection failed for {metric['name']}: {e}", flush=True)
            print(f"episode {i+1} done: " + " | ".join(
                f"{x['name']}: {'alive' if x['alive'] else 'DIED@'+str(x['survived_s'])+'s'}"
                f" ate={x['eaten']} dist={x['dist_m']}m E={x['final_energy_kJ']}kJ"
                for x in end["metrics"]), flush=True)
        print(f"benchmark complete -> {out}", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8390)
    ap.add_argument("--mock", action="store_true", help="heuristic policy, no LLM")
    ap.add_argument("--probe", action="store_true", help="single LLM call test, then exit")
    ap.add_argument("--episodes", type=int, default=0, help="run N benchmark episodes")
    ap.add_argument("--duration", type=int, default=300, help="episode length, sim-seconds")
    ap.add_argument("--terrain", default="perlin")
    ap.add_argument("--seed", type=int, default=1337, help="terrain seed")
    ap.add_argument("--food-seed", type=int, default=777)
    ap.add_argument("--food-interval", type=int, default=12)
    ap.add_argument("--size", type=int, default=50)
    ap.add_argument("--water", type=float, default=0)
    ap.add_argument("--metab", type=int, default=10)
    ap.add_argument("--think", action="store_true", help="enable model reasoning/thinking")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    if args.probe:
        d = Driver(args)
        print(f"probing model: {d.model}")
        act, dt, text = await d.llm_decide(PROBE_OBS)
        print(f"raw ({dt:.1f}s): {text[:500]}")
        print(f"parsed action: {act}")
        return

    d = Driver(args)
    print(f"mind_driver on ws://0.0.0.0:{args.port}  "
          f"({'MOCK policy' if args.mock else 'model: ' + str(d.model)})", flush=True)
    async with websockets.serve(d.handle, "0.0.0.0", args.port, max_size=2**22):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
