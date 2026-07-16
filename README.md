# world-model-mini

**A testbed for AI models living under real terrain conditions.**

A self-contained 3D physics world where embodied agents must survive by foraging for food while paying realistic energy costs — the steeper the terrain, the more energy every metre costs. Local LLMs act as the agents' minds through a WebSocket bridge, accumulate *experiential* memory across lives, and can be benchmarked against each other on identical, fully deterministic episodes.

> "Training" here means **accumulated experience, not weight updates**: agents learn the terrain, remember where food fell, and optimise energy-efficient routes — Generative-Agents-style memory architecture, with the model weights untouched.

![](https://img.shields.io/badge/deps-zero%20(vendored)-brightgreen) ![](https://img.shields.io/badge/physics-real%20units-blue)

---

## The world (`index.html`)

One self-contained HTML file — no build step, no CDN, no network. Open it in any browser.

**Physics** (hand-written rigid-body engine, real units — metres/kg/seconds, fixed 1/240 s substep):
- Gravity presets (Earth / Moon / Mars / Jupiter) or free slider
- Coulomb friction with proper rolling↔sliding transition (impulse-based, solid-sphere inertia ⅖ m r²), rolling resistance, restitution with bounce threshold
- Quadratic air drag; **buoyancy** on the submerged spherical cap (wood floats at the theoretical depth, steel rolls along the lakebed)
- Ball–ball collisions; quaternion orientation (visible spin)

**Terrain**: full free-standing landforms (gentle/steep mounds, twin peaks, bowl, volcano crater, bumpy field) plus **procedural Perlin terrain** — seeded fBm noise (deterministic per seed) with feature-scale / height / octave controls and water level for lakes. World size 50–300 m acts as a pure landscape scale factor while balls and agents stay real-sized.

**Rendering**: Three.js (r147, vendored inline), sun + soft shadows, slope-aware terrain colouring, orbit/pan/zoom camera with view presets, ball-follow chase cam, auto-orbit, ⌖ Centre button, live GPU/CPU/RAM gauges.

## The agents

Two embodied creatures (Amber & Cyan), 70 kg walkers with **real energy physiology**:
- Metabolic cost of gradient walking from *Minetti et al. 2002* (J/kg/m — a +40 % slope costs ~4× flat), 2 W/kg basal burn, ×4 cost wading, slower gait uphill
- Death at 0 energy. **Satiation**: one food ball is a full recharge, but eating is only possible below 55 % energy — so agents must memorise uneaten food and decide later: go back, or keep exploring?
- Built-in heuristic policy (energy-aware steering by progress-per-joule → emergent contouring around steep slopes) as the baseline to beat

## The minds (`bridge/mind_driver.py`)

The sim streams JSON observations (energy, slope % in 8 compass directions, visible food bearings/distances, rival position) over WebSocket; the driver queries a local LLM (vLLM, OpenAI-compatible API) and returns `{heading_deg, speed, reason}` — asynchronously, so physics never blocks on inference; slow models act on stale information, a real cost the benchmark captures.

**Three memory layers** (the "learning"):
1. **Scratchpad** — ≤80-word notes the model writes each decision, echoed back; dies with the agent
2. **World map** — *accurate experiential memory*, recorded losslessly by the driver from the agent's own observations: visited cells (altitude, worst slope → steep zones / cheap corridors), food sightings converted to exact coordinates with status (`available` / `eaten by me` / `gone — rival took it`), rival contacts, % explored. Per agent, per terrain, persistent across lives
3. **Journal** — post-episode reflection ("durable lessons about THIS terrain") injected into future lives on the same terrain only

**Benchmarking**: episodes are fully deterministic — terrain seed + quadrant-stratified food schedule seed — so different models face byte-identical worlds. Results (`bridge/results/*.jsonl`) log survival, food eaten, distance, final energy; decision traces record every obs/action/latency.

**🔁 Groundhog mode**: repeat an identical episode ad infinitum (same terrain, same 12-ball food layout, same spawns). Episode ends when both agents die; each death triggers reflection; each new life starts with a better map. Watch exploration turn into efficient farming routes.

## Quick start

```bash
# 1. serve the sim (or just open index.html in a browser)
python3 serve.py &            # static files + /stats gauges on :8388

# 2. start the mind driver (needs a vLLM/OpenAI-compatible server; see DELPHI_API env)
python3 -m venv venv && venv/bin/pip install websockets openai
venv/bin/python bridge/mind_driver.py                 # free play
venv/bin/python bridge/mind_driver.py --episodes 5 --duration 300   # benchmark
venv/bin/python bridge/mind_driver.py --mock          # no-GPU protocol test
venv/bin/python bridge/mind_driver.py --probe         # single live LLM call test

# 3. open the sim, pick a terrain/seed, click "🤖 LLM bridge"
#    (optionally tick "🔁 Loop identical episode" for Groundhog mode)
```

Model comparison on one GPU: run the benchmark, switch the served model, run it again with the same seeds — per-model results and memory files are tagged automatically.

## Development notes

- **Physics is testable headless**: everything above the `// ===== rendering` marker in the app script is DOM- and THREE-free. Extract it, append assertions, run with `node`. All mechanics in this repo were verified that way (rolling speed vs. theory, energy monotonicity, buoyancy equilibrium vs. Archimedes, Minetti cost ratios, food-layout determinism).
- **Rendering is testable headless** too: `chromium --headless=new --use-angle=swiftshader --enable-unsafe-swiftshader --screenshot=…`.
- `hill2d.html` is the original 2D prototype, kept for quick physics experiments.
- See `STATUS.md` for current state and decisions, `CLAUDE.md` for conventions and standing instructions.

## Learned world model (`bridge/worldmodel/`)

Memory answers *"what did I find at cell X"*; the world model answers *"what would
happen if I did Y"* — including in places the agent has never stood. A small MLP
(13 features → 2 targets, ~35k params, CPU-only by design so it never touches the
LLM's VRAM) learns the **dynamics/cost function**: given local slopes, commanded
heading+speed, wading flag and metabolic scale, predict `[Δenergy, displacement]`
for one 2-second step. It deliberately does **not** predict terrain at unseen
coordinates — the planner flags those paths as out-of-distribution instead.

- **Data**: `collect_data.js` extracts the sim's physics verbatim (file untouched)
  and generates seeded episodes with held 2 s actions — 6.2k real transitions over
  13 world configs. `dataset.py` builds episode-split (never row-split) train/val/test.
- **Accuracy** (held-out test): Δenergy MAE **2.9 kJ** (target std 16.8), displacement
  MAE **0.26 m** (std 1.14). Autoregressive rollout error *saturates* instead of
  compounding: 2.9 → 6.0 → 7.7 → 8.1 kJ at K = 1/3/5/10 steps (`plots/`).
- **Planner** (`planner.py`): for each candidate heading (8 compass + food bearings)
  roll the model ~3 steps; step 1 uses observed slopes, later steps use the agent's
  own world-map cells; leaving explored territory raises an OOD flag. ~1 ms per plan.
- **Integration**: `mind_driver.py --worldmodel` injects a compact per-heading
  prediction block into the prompt. Flag-gated, additive; protocol and physics untouched.

**Scaled ablation** (n = 30 agent-episodes per condition; 15 episodes × 2 agents,
byte-identical seed sets across conditions, terrain seeds 1337/42/2718/9001/31415,
240 s, metab 15×, memory wiped per condition; bootstrap/Wilson 95 % CIs):

| condition | n | survival [95% CI] | survived s | final kJ | eaten | latency s |
|---|---|---|---|---|---|---|
| heuristic (mock)     | 30 | 0.57 [0.39, 0.73] | 153 [117, 187] | 331 | 3.0 | 0.0 |
| qwen3.5, planner off | 30 | 0.37 [0.22, 0.54] | 114 [80, 149]  | 207 | 1.3 | 3.5 |
| qwen3.5, learned WM  | 30 | 0.40 [0.25, 0.58] | 125 [92, 161]  | 226 | 1.0 | 3.9 |
| qwen3.5, analytic WM | 30 | **0.60 [0.42, 0.75]** | **159 [123, 193]** | **358** | 2.0 | 4.1 |

**What held and what didn't.** The earlier n=6 pilot's headline (learned WM 67 % vs
50 %) did **not** survive scaling: learned (0.40) vs off (0.37) is a null result, and
the paired within-episode design (one agent with the learned planner, one without,
roles counterbalanced, 15 pairs) confirms it cleanly — paired survival-time
difference −12 s [−94, +69], alive-rate difference −0.07 [−0.47, +0.33]. The
**analytic oracle**, injecting *ground-truth* per-heading costs in the identical
prompt format, DID work: +23 points survival over planner-off, +45 s lifespan,
+73 % final energy. So it is not "any foresight helps" — it is **accurate foresight
helps, and the learned model isn't accurate enough where it matters**: its held-out
prediction error looks small on average (per-heading ΔE MAE 3.5 kJ vs oracle; mean
choice regret 1.8 kJ, p90 4.2 kJ), but the pred-vs-actual scatter shows it
*underpredicts exactly the extreme slope costs that kill agents*, and with a fresh
map each episode its rollout truncates to 1 step (OOD) while the oracle sees three.
Also humbling: the energy-greedy heuristic (0.57) statistically matches the
oracle-equipped LLM (0.60) at zero latency — in this world, deliberation adds
nothing beyond accurate greedy foresight. All raw rows: `scaled_ablation_combined.csv`.

## Roadmap

- Scale the ablation (more seeds/episodes) and add the other two Delphi models
- More agents + a communication channel (cooperation, gossip, emergent language) in a richer "real-like" world
