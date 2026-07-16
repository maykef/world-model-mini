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

## Roadmap

- Learned world model trained on the recorded decision traces (predict next state from state+action), then offered to the LLM as a planning tool
- More agents + a communication channel (cooperation, gossip, emergent language) in a richer "real-like" world
