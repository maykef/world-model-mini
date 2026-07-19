#!/usr/bin/env python3
"""
mind_driver.py — bridges the Hill Physics Sandbox agents to an LLM served by
Delphi (vLLM, OpenAI-compatible, port 8000).

The sim (index.html) connects here over WebSocket when "LLM bridge" mode is on.
Protocol (JSON messages):
  sim -> driver:  {type:"hello", agents:[names]}
                  {type:"obs", t_s, episode_active, agents:[{id, alive, obs,
                               frame?:{jpg_b64, ts_sim, seq, keyframe}}]}
                  {type:"reset_done", cfg}
                  {type:"episode_end", t_s, metrics:[...]}
  driver -> sim:  {type:"act", id, heading_deg, speed, reason}
                  {type:"reset", cfg:{terrain,seed,size,water,metab,
                                      food_seed,food_interval_s,first_food_s,episode_s,
                                      vision?,regime?}}

Observation regimes (--obs): what the MODEL is shown (the sim always sends full oracle
fields for ground-truth logging; the driver decides what to reveal):
  oracle          8-dir slopes + food bearings/distances + rival, as JSON (default; the ceiling)
  pixels          egocentric camera frame + minimal interoception JSON (energy/heading/speed/...)
  pixels+proprio  frame + full interoception (adds position, grade underfoot, wading)
The survival gap between regimes on byte-identical seeds is the "perception tax".

Modes:
  --mock            heuristic policy instead of the LLM (protocol testing, no GPU)
  --probe           one decision against a canned obs (+ synthetic/bundled frame in pixel
                    regimes), print the assembled request and parsed response, exit
  --episodes N      benchmark: N reset->run->episode_end cycles, JSONL results
  (no flags)        free-play: just answer observations with LLM decisions

Delphi serves ONE model at a time (MAX_NUM_SEQS=1) — requests are serialized
via a lock. Results are tagged with the model reported by /v1/models, so run
the same benchmark, switch models in Delphi, and run it again to compare.
"""
import argparse, asyncio, base64, json, logging, math, os, re, sys, time, urllib.request
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

# Pixel regimes: exteroceptive omniscience (slopes, food bearings, rival position) is REMOVED —
# the model must read the world from a first-person frame. Only body-sense stays in JSON.
SYSTEM_PROMPT_PIXEL = """You are {name}, a creature living on rugged 3D terrain. Survive by eating food while managing a limited energy budget.

You do NOT get a map or food coordinates. You PERCEIVE the world through your own eyes: a first-person camera image looking in your direction of travel (~100 deg field of view, eye height ~1.7 m). Food is a ball resting on the ground. Read the IMAGE to find food, judge which way is uphill/downhill, and spot the rival creature.

Physics you must respect:
- Moving costs energy per metre; the STEEPER the uphill, the more it costs (a +40% slope costs ~4x flat). Downhill is cheap. Standing still still burns basal energy.
- Water (if present) slows you and quadruples movement cost.
- If energy reaches 0 you die.
- EATING: one ball is a FULL recharge, but only when your energy is below eat_below_pct (see can_eat). Food you don't eat STAYS where it fell — remember where you saw it.
- A rival competes for the same food; whoever arrives first (and is hungry) eats it.

Alongside the image you get an INTEROCEPTION JSON (body sense only): energy, can_eat, compass heading, speed, sim time, terrain name/seed{proprio}. Compass: 0=N, 90=E, 180=S, 270=W. In the image, straight ahead is your current heading; turn by choosing a new heading_deg.

Reply with ONLY a JSON object, no other text:
{{"heading_deg": <0-360>, "speed": <0.0-2.0 m/s>, "reason": "<max 10 words>", "memory": "<notes to your future self; max 80 words>", "sightings": [{{"type": "food"|"rival"|"landmark", "est_x": <metres>, "est_z": <metres>, "confidence": <0-1>}}]}}

"sightings" is OPTIONAL: list what you SEE in the image and your best guess of its ground position (x, z in metres, same axes as any position you are given). Omit it if unsure — a wrong guess is worse than none. "memory" is echoed back next turn — it is your ONLY memory within this life; record food locations and terrain you learn.

Strategy: when sated, scout and memorize; when energy drops, return to remembered food or explore for new drops. Prefer ridges/contours over climbing steep slopes head-on."""

# Tiny 16x16 gradient JPEG (base64) — lets --mock/--probe exercise the multimodal path with no GPU/sim.
SYNTH_JPEG_B64 = ("/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA0JCgsKCA0LCgsODg0PEyAVExISEyccHhcgLikxMC4pLSwzOko+MzZGNywtQFdBRkxOUlNSMj5a"
                  "YVpQYEpRUk//2wBDAQ4ODhMREyYVFSZPNS01T09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0//wA"
                  "ARCAAQABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQR"
                  "BRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3"
                  "R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb"
                  "3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMo"
                  "EIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaH"
                  "iImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxE"
                  "APwDk7TTOny1uWemdPlrWs9M6fLW5aaZ0+WiEwyzM9tT/2Q==")

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


def _iter_json_objects(text):
    """Yield balanced-brace {...} substrings (string-aware) so nested objects/arrays survive."""
    depth = 0; start = -1; instr = False; esc = False
    for i, ch in enumerate(text):
        if instr:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': instr = False
            continue
        if ch == '"': instr = True
        elif ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start:i + 1]


def parse_action(text):
    """Extract the first JSON object with heading_deg. Tolerates nested `sightings` (est_x/est_z,
    est_y accepted as a z alias); missing sightings is fine."""
    for chunk in _iter_json_objects(text):
        try:
            o = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if not isinstance(o, dict) or "heading_deg" not in o:
            continue
        act = {"heading_deg": float(o["heading_deg"]) % 360,
               "speed": max(0.0, min(2.0, float(o.get("speed", 1.2)))),
               "reason": str(o.get("reason", ""))[:60],
               "memory": str(o.get("memory", ""))[:400]}
        sights = []
        for s in (o.get("sightings") or []):
            if not isinstance(s, dict):
                continue
            try:
                x = float(s["est_x"]); z = float(s.get("est_z", s.get("est_y")))
            except (KeyError, TypeError, ValueError):
                continue
            sights.append({"type": str(s.get("type", "?"))[:12], "est_x": x, "est_z": z,
                           "confidence": max(0.0, min(1.0, float(s.get("confidence", 0.5) or 0.5)))})
        if sights:
            act["sightings"] = sights
        return act
    return None


def mock_policy(obs):
    """Greedy heuristic: nearest food when hungry, else wander. Used with --mock."""
    if obs.get("food") and obs.get("can_eat", True):
        f = min(obs["food"], key=lambda x: x["dist_m"])
        return {"heading_deg": f["bearing_deg"], "speed": 1.6, "reason": "mock: nearest food", "memory": ""}
    return {"heading_deg": (obs.get("heading_deg", 0) + 30) % 360, "speed": 1.0, "reason": "mock: wander", "memory": ""}


# ── Slope-probe: how well can the VLM read terrain grade from a frame, per render config? ─────────
# A perception micro-benchmark with NO episode logic: load captured frames + ground-truth grades
# (from make_slope_probe.py), ask the VLM to estimate grade ahead/left/right at 5 m & 15 m, and
# score MAE + Spearman rank-correlation per render config.
SLOPE_SLOTS = ["ahead_5m", "ahead_15m", "left_5m", "left_15m", "right_5m", "right_15m"]

SLOPE_PROMPT = (
    "This is a first-person view of terrain (about 100 deg field of view, eye height ~1.7 m, with a "
    "GRAVITY-LEVELLED horizon: when the ground is flat the horizon sits across the image centre). "
    "Estimate the ground SLOPE as GRADE PERCENT (rise/run x100; POSITIVE = uphill, NEGATIVE = downhill) "
    "in each direction, at two look-distances:\n"
    "- ahead (straight forward) at 5 m and 15 m\n"
    "- to your left at 5 m and 15 m\n"
    "- to your right at 5 m and 15 m\n"
    'Reply with ONLY this JSON (integers, roughly -60..60): '
    '{"ahead_5m":N,"ahead_15m":N,"left_5m":N,"left_15m":N,"right_5m":N,"right_15m":N}')


def parse_slopes(text):
    """First JSON object carrying any of the six grade slots -> {slot: float}."""
    for chunk in _iter_json_objects(text):
        try:
            o = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if not isinstance(o, dict):
            continue
        got = {}
        for slot in SLOPE_SLOTS:
            if slot in o:
                try:
                    got[slot] = float(o[slot])
                except (TypeError, ValueError):
                    pass
        if got:
            return got
    return {}


def mock_slopes(b64):
    """Deterministic image-hash pseudo-estimates — a BLIND baseline (uncorrelated with truth),
    so --mock exercises the full pipeline + metrics without a GPU."""
    import hashlib
    h = int(hashlib.md5(b64.encode()).hexdigest(), 16)
    out = {}
    for slot in SLOPE_SLOTS:
        h = (h * 1103515245 + 12345) & 0x7fffffff
        out[slot] = float(h % 121 - 60)
    return out


def _rankdata(a):
    """Ranks with average-tie handling (pure Python; no numpy dependency)."""
    order = sorted(range(len(a)), key=lambda i: a[i])
    ranks = [0.0] * len(a)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x, y):
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = sum((a - mx) ** 2 for a in x)
    sy = sum((b - my) ** 2 for b in y)
    if sx == 0 or sy == 0:
        return None
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    return sxy / (sx * sy) ** 0.5


def _spearman(x, y):
    return _pearson(_rankdata(x), _rankdata(y)) if len(x) >= 2 else None


async def slope_query(driver, b64):
    detail = getattr(driver.args, "image_detail", "auto")
    msgs = [{"role": "system", "content": "You are a careful observer estimating terrain slope from a photo. Reply only with the requested JSON."},
            {"role": "user", "content": [
                {"type": "text", "text": SLOPE_PROMPT},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64, "detail": detail}}]}]
    async with driver.llm_lock:
        resp = await driver.client.chat.completions.create(
            model=driver.model, messages=msgs, temperature=0.0,
            max_tokens=driver.args.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": driver.args.think}})
    return parse_slopes(resp.choices[0].message.content or "")


async def run_slope_probe(args):
    root = args.slope_probe
    manifest = os.path.join(root, "manifest.jsonl")
    if not os.path.exists(manifest):
        sys.exit(f"no manifest at {manifest} — generate one with make_slope_probe.py")
    entries = [json.loads(l) for l in open(manifest) if l.strip()]
    driver = Driver(args)
    tag = "mock" if args.mock else driver.model
    by_cfg = {}
    for e in entries:
        by_cfg.setdefault(e.get("render_config", "?"), []).append(e)
    print(f"slope-probe: {len(entries)} frames, {len(by_cfg)} configs, model={tag}", flush=True)

    rows = []
    for cfg, items in sorted(by_cfg.items()):
        gts, ests = [], []
        for e in items:
            with open(os.path.join(root, e["image"]), "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            est = mock_slopes(b64) if args.mock else await slope_query(driver, b64)
            for slot in SLOPE_SLOTS:
                if slot in e.get("grade_pct", {}) and est.get(slot) is not None:
                    gts.append(float(e["grade_pct"][slot]))
                    ests.append(est[slot])
            print(f"  [{cfg}] {e['image']}: {len(est)}/6 slots", flush=True)
        mae = (sum(abs(a - b) for a, b in zip(ests, gts)) / len(gts)) if gts else None
        rows.append({"render_config": cfg, "frames": len(items), "pairs": len(gts),
                     "mae_pct": round(mae, 2) if mae is not None else None,
                     "spearman": (round(_spearman(gts, ests), 3) if _spearman(gts, ests) is not None else None)})

    print(f"\n{'render_config':<16}{'frames':>7}{'pairs':>7}{'MAE_%':>9}{'Spearman':>10}")
    for r in rows:
        mae = f"{r['mae_pct']:.2f}" if r["mae_pct"] is not None else "-"
        rho = f"{r['spearman']:.3f}" if r["spearman"] is not None else "-"
        print(f"{r['render_config']:<16}{r['frames']:>7}{r['pairs']:>7}{mae:>9}{rho:>10}")
    out = os.path.join(root, "slope_probe_results.jsonl")
    with open(out, "w") as f:
        f.write(json.dumps({"model": tag, "ts": time.time(), "rows": rows}) + "\n")
    print(f"\nwrote {out}", flush=True)


class Driver:
    def __init__(self, args):
        self.args = args
        self.model = None if args.mock else get_model_name()
        self.llm_lock = asyncio.Lock()          # MAX_NUM_SEQS=1 on Delphi
        self.latest_obs = {}                    # id -> obs (only newest matters)
        self.obs_regime = getattr(args, "obs", "oracle")   # what the model is shown
        self.latest_frame = {}                  # id -> {jpg_b64, ts_sim, seq, keyframe} (newest frame)
        self._last_img_seq = {}                 # name -> seq of last image attached (keyframe gating)
        self._last_seen_frame = {}              # name -> frame currently informing the model's belief
        self._reset_vis_stats()
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
        self.done = None                        # set in main() for benchmark runs
        self.planner = None
        self.paired = getattr(args, "paired", False)
        mode = getattr(args, "planner", "off")
        if getattr(args, "worldmodel", False) and mode == "off":
            mode = "learned"                    # legacy alias
        self.planner_mode = mode
        if mode != "off":                       # lazy: torch/node only load when needed
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "worldmodel"))
            from planner import make_planner
            self.planner = make_planner(mode)
            print(f"planner loaded: {mode}", flush=True)
        if not args.mock:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(base_url=API_BASE, api_key="none")

    def planner_for(self, name):
        """Planner for this agent. In --paired mode one agent plans and the other
        doesn't, with roles swapping every episode (counterbalances spawn asymmetry)."""
        if not self.planner:
            return None
        if not self.paired:
            return self.planner
        agent_idx = 0 if name == "Amber" else 1
        return self.planner if (self.ep_idx + agent_idx) % 2 == 0 else None

    def roles(self):
        if not self.paired or not self.planner:
            return {n: self.planner_mode for n in ("Amber", "Cyan")} if self.planner else None
        return {n: (self.planner_mode if self.planner_for(n) else "off")
                for n in ("Amber", "Cyan")}

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
            if not os.path.exists(self.trace_path):   # first write: provenance meta record
                with open(self.trace_path, "w") as f:
                    f.write(json.dumps({"kind": "meta", "real": True, "source": "sim-ws",
                                        "policy": str(self.tag)}) + "\n")
            with open(self.trace_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def _reset_vis_stats(self):
        self.vis_stats = {"frames_seen": 0, "staleness": []}   # per-episode vision accounting

    @staticmethod
    def _grade_ahead(obs):
        """The slope % you feel underfoot in your heading — proprioceptive, kept in pixel regimes."""
        slopes = obs.get("slope_pct") or {}
        if not slopes:
            return None
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        h = obs.get("heading_deg", 0) or 0
        return slopes.get(dirs[int((h + 22.5) // 45) % 8])

    def interoception(self, obs):
        """Body-sense JSON for pixel regimes — no exteroceptive omniscience (slopes/food/rival)."""
        io = {k: obs.get(k) for k in ("name", "t_s", "energy_pct", "can_eat", "eat_below_pct",
                                      "heading_deg", "speed_ms", "terrain", "terrain_seed",
                                      "world_size_m", "metabolic_scale")}
        if self.obs_regime == "pixels+proprio":       # full interoception adds path-integration + gait
            io["pos_m"] = obs.get("pos_m")
            io["energy_kJ"] = obs.get("energy_kJ")
            io["grade_ahead_pct"] = self._grade_ahead(obs)
            wl = obs.get("water_level_m") or 0
            alt = (obs.get("pos_m") or {}).get("alt")
            io["wading"] = bool(wl and alt is not None and alt < wl)
        return io

    def _frame_policy(self, name, obs, frame):
        """Attach an image on a NEW keyframe (default) or every decision (--every-frame). Track the
        frame currently informing the model's belief so vision staleness is measured per decision."""
        attach = False
        if frame:
            new_kf = bool(frame.get("keyframe")) and frame.get("seq") != self._last_img_seq.get(name)
            attach = bool(getattr(self.args, "every_frame", False) or new_kf)
            if attach:
                self._last_img_seq[name] = frame.get("seq")
                self._last_seen_frame[name] = frame
        seen = self._last_seen_frame.get(name)
        stale = None
        if seen and obs.get("t_s") is not None and seen.get("ts_sim") is not None:
            stale = round(obs["t_s"] - seen["ts_sim"], 2)
        return attach, seen, stale

    def build_messages(self, obs, frame=None):
        """Assemble the chat request for one decision. Returns (msgs, meta, hist_user).
        Oracle regime is byte-for-byte the old behaviour; pixel regimes swap in a first-person
        image + interoception and drop the exteroceptive oracle fields."""
        name = obs.get("name", "Agent")
        hist = self.history.get(name, [])
        key = obs_terrain_key(obs)
        planner = self.planner_for(name)
        pixel = self.obs_regime != "oracle"
        meta = {"regime": self.obs_regime}

        if pixel:
            proprio = (", your position (x,z,alt), the grade underfoot, and wading state"
                       if self.obs_regime == "pixels+proprio" else "")
            system = SYSTEM_PROMPT_PIXEL.format(name=name, proprio=proprio)
        else:
            system = SYSTEM_PROMPT.format(name=name)
            system += self.worldmem.render(name, key, obs.get("world_size_m", 50))
        # NOTE (Phase 3): a *believed* map (from the model's own sightings) will be injected here
        # for pixel regimes — deliberately absent now so we never leak oracle food coordinates.
        system += self.journal_text(key)
        if planner and not pixel:
            system += ("\n\nA PLANNER block may follow each observation: a world model's "
                       "prediction of energy cost and movement per candidate heading. It is usually "
                       "accurate on explored terrain and unreliable where marked UNKNOWN TERRAIN.")
        msgs = [{"role": "system", "content": system}]
        for h_user, h_act in hist[-3:]:            # short window: last 3 decisions (text only)
            msgs.append({"role": "user", "content": h_user})
            msgs.append({"role": "assistant", "content": h_act})

        mem = self.memory.get(name)
        if pixel:
            hist_user = "INTEROCEPTION (body sense):\n" + json.dumps(self.interoception(obs))
            if mem:
                hist_user += f'\nYour saved notes: "{mem}"'
            attach, seen, stale = self._frame_policy(name, obs, frame)
            meta.update(image_attached=attach, frame_seq=(seen or {}).get("seq"),
                        frame_ts_sim=(seen or {}).get("ts_sim"), vision_staleness_s=stale)
            if attach and seen:
                content = [{"type": "text", "text": hist_user},
                           {"type": "image_url", "image_url": {
                               "url": "data:image/jpeg;base64," + seen["jpg_b64"],
                               "detail": getattr(self.args, "image_detail", "auto")}}]
            else:
                content = hist_user                # text-only turn (no fresh keyframe)
            msgs.append({"role": "user", "content": content})
        else:
            user = json.dumps(obs)
            if planner:
                cells = self.worldmem.maps.get(f"{name}|{key}", {}).get("cells", {})
                try:
                    user += planner.render(planner.plan(obs, cells), obs)
                except Exception as e:
                    print(f"  planner failed: {e}", flush=True)
            if mem:
                user += f'\nYour saved notes: "{mem}"'
            msgs.append({"role": "user", "content": user})
            hist_user = user
        return msgs, meta, hist_user

    async def llm_decide(self, obs, frame=None):
        t0 = time.time()                        # total decision latency incl. planning
        name = obs.get("name", "Agent")
        msgs, meta, hist_user = self.build_messages(obs, frame)
        if meta.get("image_attached"):
            self.vis_stats["frames_seen"] += 1
        if meta.get("vision_staleness_s") is not None:
            self.vis_stats["staleness"].append(meta["vision_staleness_s"])
        async with self.llm_lock:
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
            self.history.setdefault(name, []).append((hist_user, json.dumps(act)))
            self.history[name] = self.history[name][-6:]
        return act, dt, text, meta

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
            frame = self.latest_frame.get(agent_id)
            meta = {}
            try:
                if self.args.mock:
                    act, dt = mock_policy(obs), 0.0
                    if self.obs_regime != "oracle":     # still exercise frame flow/staleness in mock
                        attach, seen, stale = self._frame_policy(obs["name"], obs, frame)
                        if attach:
                            self.vis_stats["frames_seen"] += 1
                        if stale is not None:
                            self.vis_stats["staleness"].append(stale)
                        meta = {"regime": self.obs_regime, "image_attached": attach,
                                "frame_seq": (seen or {}).get("seq"), "vision_staleness_s": stale}
                else:
                    act, dt, _, meta = await self.llm_decide(obs, frame)
            except Exception as e:
                print(f"  [{agent_id}] LLM error: {e}", flush=True)
                await asyncio.sleep(1)
                continue
            if act:
                await ws.send(json.dumps({"type": "act", "id": agent_id,
                                          "heading_deg": act["heading_deg"],
                                          "speed": act["speed"], "reason": act["reason"]}))
                rec = {"kind": "decision", "episode": self.ep_idx, "agent": obs["name"],
                       "t_s": obs["t_s"], "obs": obs, "action": act,
                       "planner": self.planner_mode if self.planner_for(obs["name"]) else "off",
                       "latency_s": round(dt, 2)}
                if self.obs_regime != "oracle":
                    rec["vision"] = meta
                self.trace(rec)
                vtag = ""
                if self.obs_regime != "oracle":
                    vtag = f" img={'Y' if meta.get('image_attached') else '·'} stale={meta.get('vision_staleness_s')}s"
                print(f"  [{obs['name']}] t={obs['t_s']:>6}s E={obs['energy_kJ']}kJ "
                      f"-> {act['heading_deg']:.0f}deg @{act['speed']:.1f}m/s "
                      f"({act['reason']}){vtag} [{dt:.1f}s]", flush=True)

    def episode_cfg(self, i):
        a = self.args
        seeds = [int(s) for s in str(a.seeds).split(",")] if getattr(a, "seeds", None) else [a.seed]
        if getattr(a, "groundhog", False):
            # Groundhog benchmark: IDENTICAL world + food layout every episode —
            # memory (world map, journal) carries across lives; the metric is the
            # learning curve, not just the mean.
            cfg = {"terrain": a.terrain, "seed": seeds[0], "size": a.size,
                   "water": a.water, "metab": a.metab, "food_seed": a.food_seed,
                   "food_mode": "preset", "food_count": 12, "episode_s": a.duration}
        else:
            cfg = {"terrain": a.terrain, "seed": seeds[i % len(seeds)], "size": a.size,
                   "water": a.water, "metab": a.metab, "food_seed": a.food_seed + i,
                   "food_interval_s": a.food_interval, "first_food_s": 4, "episode_s": a.duration}
        if getattr(self, "obs_regime", "oracle") != "oracle":  # ask the sim to stream egocentric frames
            cfg["vision"] = True
            cfg["regime"] = self.obs_regime
        return cfg

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
                                self.worldmem.update(a["obs"], self.iteration)  # ground-truth recorder
                                self.latest_obs[a["id"]] = a["obs"]
                            if a.get("frame"):          # newest egocentric frame per agent (pixel regimes)
                                self.latest_frame[a["id"]] = a["frame"]
                        self.obs_event.set()
                elif m["type"] == "reset_done":
                    self.trace({"kind": "reset_done", "episode": self.ep_idx,
                                "iteration": m.get("iteration"), "cfg": m.get("cfg")})
                    if not self.args.episodes:          # new life: scratchpad dies; map & journal survive
                        self.latest_obs.clear(); self.history.clear(); self.memory.clear()
                        self.latest_frame.clear(); self._last_img_seq.clear()
                        self._last_seen_frame.clear(); self._reset_vis_stats()
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
            self.latest_frame.clear(); self._last_img_seq.clear()
            self._last_seen_frame.clear(); self._reset_vis_stats()
            # groundhog: identical layout -> remembered food respawns; else forget food
            self.worldmem.new_life(i, same_layout=getattr(self.args, "groundhog", False))
            self.worldmem.save()
            self.episode_end = asyncio.get_event_loop().create_future()
            await ws.send(json.dumps({"type": "reset", "cfg": cfg}))
            print(f"episode {i+1}/{self.args.episodes} started: {cfg}", flush=True)
            end = await self.episode_end
            stale = self.vis_stats["staleness"]
            rec = {"model": tag, "episode": i, "cfg": cfg, "t_end_s": end["t_s"],
                   "planner": self.planner_mode, "paired": self.paired,
                   "roles": self.roles(), "obs_regime": self.obs_regime,
                   "frames_seen": self.vis_stats["frames_seen"],
                   "mean_vision_staleness_s": round(sum(stale) / len(stale), 2) if stale else None,
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
        if self.done and not self.done.done():
            self.done.set_result(True)          # benchmark runs exit when finished


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
    ap.add_argument("--worldmodel", action="store_true",
                    help="deprecated alias for --planner learned")
    ap.add_argument("--planner", choices=["off", "learned", "analytic"], default="off",
                    help="lookahead source injected into the prompt")
    ap.add_argument("--paired", action="store_true",
                    help="within-episode control: one agent plans, the other doesn't; roles swap per episode")
    ap.add_argument("--groundhog", action="store_true",
                    help="benchmark on the IDENTICAL world+food every episode; memory persists across lives")
    ap.add_argument("--seeds", default=None,
                    help="comma list of terrain seeds cycled across episodes (overrides --seed)")
    ap.add_argument("--obs", choices=["oracle", "pixels", "pixels+proprio"], default="oracle",
                    help="observation regime the MODEL sees (default oracle; pixel regimes need vision frames)")
    ap.add_argument("--every-frame", action="store_true",
                    help="attach the image on every decision (default: only on a new keyframe)")
    ap.add_argument("--image-detail", choices=["low", "high", "auto"], default="auto",
                    help="OpenAI image_url detail passthrough")
    ap.add_argument("--probe-image", default=None,
                    help="path to a JPEG to send with --probe in a pixel regime (else a bundled synthetic frame)")
    ap.add_argument("--slope-probe", default=None, metavar="DIR",
                    help="perception benchmark: score VLM grade estimates vs ground truth per render config (no episodes)")
    args = ap.parse_args()

    if args.slope_probe:
        await run_slope_probe(args)
        return

    if args.probe:
        d = Driver(args)
        obs = dict(PROBE_OBS)
        frame = None
        if d.obs_regime != "oracle":
            b64 = SYNTH_JPEG_B64
            if args.probe_image:
                with open(args.probe_image, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode()
            frame = {"jpg_b64": b64, "ts_sim": obs["t_s"] - 0.5, "seq": 1, "keyframe": True}
        msgs, meta, _ = d.build_messages(obs, frame)
        last = msgs[-1]["content"]
        parts = "+".join(p.get("type") for p in last) if isinstance(last, list) else "text"
        print(f"regime: {d.obs_regime}  image_attached: {meta.get('image_attached')}  "
              f"user-content: [{parts}]  messages: {len(msgs)}")
        if args.mock:
            print(f"parsed action (mock): {mock_policy(obs)}")
            return
        print(f"probing model: {d.model}")
        act, dt, text, _ = await d.llm_decide(obs, frame)
        print(f"raw ({dt:.1f}s): {text[:500]}")
        print(f"parsed action: {act}")
        return

    d = Driver(args)
    d.done = asyncio.get_running_loop().create_future() if args.episodes else None
    print(f"mind_driver on ws://0.0.0.0:{args.port}  "
          f"({'MOCK policy' if args.mock else 'model: ' + str(d.model)})"
          f"{'  [worldmodel ON]' if d.planner else ''}", flush=True)
    async with websockets.serve(d.handle, "0.0.0.0", args.port, max_size=2**22):
        await (d.done if d.done else asyncio.Future())
    print("benchmark finished — exiting", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
