# STATUS — world-model-mini

_Last updated: 2026-07-16_

## Purpose (revealed 2026-07-16)

Not a ball-physics toy: this is a **testbed for AI models in real terrain conditions**. Embodied agents live on the terrain, forage for food (falling balls, placeholder), and pay realistic energy costs (steeper = more expensive). The three LLMs served by Delphi (qwen3.5-122b / gemma-4-31b-heretic / xortron-123b — one at a time, single GPU) are benchmarked as the agents' "minds" on identical seeded episodes.

## Agent layer

- Two embodied agents (Amber, Cyan), 70 kg walkers: capsule bodies, heading nose, billboarded energy bars, dead pose.
- **Energy physiology in real units:** Minetti et al. 2002 metabolic cost of gradient walking (J/kg/m, ~4× flat cost at +40% slope), 2 W/kg basal rate, wading ×4, slower walking uphill. `Metab ×` slider (default 10×) compresses survival timescales. Death at 0 J.
- **Satiation & memory pressure:** one ball = one FULL recharge, but eating is only possible below 55% energy (`can_eat`/`eat_below_pct` in obs). Uneaten food persists but is only observable within vision — so minds must memorize food locations and weigh returning vs exploring. The LLM gets a self-managed `memory` field (echoed back each decision, its only persistent memory); agents spawn at 50% (hungry).
- Built-in policy: energy-aware — when hungry picks cheapest-to-reach food and steers by progress-per-joule (emergent contouring); when sated, explores.
- Observations (`buildObs`): energy, position, slope % in 8 compass directions, up to 6 visible food items (bearing/dist/elevation), other agent, water — JSON, designed to be LLM-readable.

## LLM bridge (bridge/mind_driver.py)

- Sim ("LLM bridge" mode, or `?control=llm&ws=...`) streams observations every 2 sim-s over WebSocket; driver queries Delphi (OpenAI API, port 8000, requests serialized — MAX_NUM_SEQS=1) and returns `{heading_deg, speed, reason}` actions asynchronously (physics never blocks on the LLM).
- **Episodes:** driver commands `reset` with terrain seed + food seed → sim rebuilds world, spawns agents, drops food on a deterministic schedule → `episode_end` returns metrics (survival, eaten, distance, final energy) → JSONL in `bridge/results/`, tagged with the served model.
- Benchmark: `venv/bin/python bridge/mind_driver.py --episodes 5 --duration 300` then open the sim in LLM mode; switch Delphi models and repeat with the same seeds.
- `--mock` (heuristic, no GPU) for protocol tests; `--probe` for a single live LLM call. Probe on qwen3.5-122b: clean JSON action in 0.7 s (thinking disabled; `--think` enables it).
- **Full recording:** decision traces (obs/action/memory/latency), `reset_done` and `episode_end` events — in benchmark AND free-play sessions (`bridge/results/*_trace.jsonl`).
- **Learning across lives — three memory layers** ("training" here = accumulated experience, never weight updates):
  1. *Scratchpad* (`memory` field, ≤80 words): the model's working notes within a life; dies on reset.
  2. *World map* (`bridge/memory/<model>__worldmap.json`, `WorldMemory` class): **accurate experiential memory**, recorded losslessly by the driver from what each agent actually observed — per (agent, terrain-key): visited 5 m cells (altitude, worst slope → "steep zones"/"easy cells"), food sightings converted from bearing+distance to exact (x,z) with status tracking (available / eaten by me / gone-taken), rival contact log, % explored. Injected into every decision prompt. Groundhog resets restore food to `available` (identical layout); benchmark episodes (different food seed) wipe food memory to prevent false memories while keeping terrain knowledge.
  3. *Journal* (`bridge/memory/<model>.json`): post-episode reflections — distilled strategy lessons per terrain key, last 4 injected.
- Long-term vision (user): scale to many agents that communicate, collaborate, gossip — Generative-Agents-style; current 2-agent survival world is the deliberately basic starting point.
- **🔁 Groundhog mode** (checkbox in Agents card): repeats an identical episode ad infinitum — same terrain, same 12 preset food balls (quadrant-stratified, byte-identical layout each iteration), same agent spawns. No food drops mid-episode; the episode ends when both agents are dead (`food_left` in metrics shows whether they cleared the map), then auto-restarts 2.5 s later with iteration counter in the HUD. With the LLM bridge on, each death triggers reflection → the journal grows → later iterations start with accumulated terrain knowledge.
- WS bridge on the tailnet: `wss://microscopy-rig-system.tail53cc58.ts.net:8444` (tailscale serve → 8390); the page auto-picks wss when loaded over https.
- Verified end-to-end: headless Chromium + mock driver ran a full episode (reset → foraging → death → metrics logged).

## Current state

v2.1 is a **3D** physics sandbox: `index.html` renders a 50 × 50 m heightfield world in WebGL (Three.js r147, vendored inline — still one self-contained file) with sun + soft shadows and slope-aware terrain coloring. Terrains are full free-standing hills (gentle/steep mound, twin peaks, bowl, volcano crater, bumpy field) — balls roll down every side — plus a **Procedural (Perlin)** mode: seeded classic Perlin noise with fBm octaves (deterministic per seed; controls for seed/🎲, feature scale, height, detail octaves), smoothstep edge falloff to plains, and automatic peak detection for "Drop at peak" / ball rain. Noise lives in the DOM-free physics section (mulberry32 PRNG → permutation table), so terrain is Node-testable; @6 octaves contact queries for 150 balls cost ~25 ms per simulated second.

**World size** is adjustable 50–300 m (slider) and acts as a pure scale factor for the *landscape only*: presets and Perlin terrain (feature size, height, edge falloff, water level) are authored at a 50 m reference and magnified with the world — same seed → same landscape, bigger — while balls stay real-world sized (0.2–3 m). So a 300 m world means ~50 m mountains with tiny balls rolling 200 m down them. Geometry, fog, sun/shadow frustum, camera views and orbit limits all rescale on slider release. A **Follow selected ball** chase-camera checkbox makes the big worlds navigable (min zoom 2 m).

**Water bodies** (procedural terrain): a water-level slider (0–10 m) floods the valleys. Physics is real buoyancy — Archimedes on the submerged spherical-cap volume (ρ_water = 1000), quadratic drag in the effective fluid, viscous + rotational damping. Wood (700 kg/m³) floats at the theoretical 0.65 depth/diameter, ice floats high, rubber (1100) barely sinks, steel sinks and rolls along the lakebed. Ball states now include floating/submerged. Camera: left-drag/one-finger orbit with damping, wheel/pinch zoom, right-drag pan, N/S/E/W/Top/Iso view buttons, auto-orbit toggle; interaction has Camera mode (click = drop ball) and Launch mode (drag = aim a shot). Physics is a hand-written full-3D rigid-body engine in real units: gravity presets (Earth/Moon/Mars/Jupiter), quadratic air drag, restitution with bounce threshold, Coulomb friction with rolling↔sliding transition (impulse-based, 3-D slip vector, solid-sphere inertia 2/5·m·r²), rolling resistance, quaternion orientation (visible spin via two-tone texture), ball–ball collisions. 5 terrain presets, 4 materials (rubber/steel/wood/ice), left-click/drag to drop/launch balls, "rain 20 balls", trails, per-ball energy readouts.

Verified: Node smoke tests on the physics core (rolls downhill and settles exactly on the surface, energy decays monotonically over 60 s in the bowl, ice slides where rubber rolls, ball–ball collisions separate, quaternion stays unit) + headless-Chromium screenshot check of the actual WebGL rendering.

The earlier 2D side-view version is kept as `hill2d.html`.

**Serving:** the project directory is broadcast over Tailscale at **https://microscopy-rig-system.tail53cc58.ts.net:8443/** (tailnet-only; Delphi keeps the root domain on 443). Chain: systemd user service `hillsim.service` (`python3 serve.py` — static files on 127.0.0.1:8388 **plus a `/stats` endpoint**: GPU util/VRAM/temp/power via nvidia-smi, CPU %, RAM) ← `tailscale serve --https=8443`. The GUI polls `/stats` every 2.5 s into a System gauges card (hidden on `file://`). WS bridge: `tailscale serve --https=8444` → 8390. Manage with `systemctl --user status hillsim`, `tailscale serve status`.

**GUI layout:** map is sticky (always visible while the control sidebar scrolls independently, capped to viewport); `⌖ Centre` button recenters target + iso view and cancels ball-follow. Driver also writes per-decision traces (`bridge/results/*_trace.jsonl`: obs, action, memory, latency) — the future world-model training data.

## Decisions made

- 2026-07-16 — Hand-written physics + browser rendering, single self-contained HTML file per app. Three.js is vendored *inline* (UMD r147) so `file://` still works with zero network.
- 2026-07-16 — Went 3D (user requirement: true 3D rendering; machine has an RTX PRO 6000 Blackwell, 96 GB — display runs on it via Xorg).
- Physics substep fixed at 1/240 s; world is 50 × 50 m.

## Repo

- GitHub: `maykef/world-model-mini` (repo created by the user 2026-07-16; local git initialised the same day). Data dirs (`bridge/results/`, `bridge/memory/`, `venv/`) are gitignored — experiment data stays local.

## Learned world model (2026-07-16, bridge/worldmodel/)

Dynamics model (state, action) → [Δenergy, displacement] per 2 s step; MLP 13→128→128→2, CPU-only (never competes with Delphi's VRAM). Trained on 6.2k real transitions from `collect_data.js` (physics extracted verbatim from index.html — sim untouched), episode-split. **Test MAE 2.9 kJ / 0.26 m; rollout error saturates at ~8 kJ by K=10.** `planner.py` rolls candidates 3 steps (observed slopes → world-map cells → OOD flag on unexplored terrain, ~1 ms). `mind_driver.py --worldmodel` (default OFF) injects predictions into the prompt. Existing pre-worldmodel traces were synthetic protocol tests → quarantined in `results/protocol-tests/`.

**Ablation** (3×240 s, perlin:1337, metab 15, identical seeds, fresh memory per condition): heuristic 50 % survival / 144 s mean; qwen3.5 without WM 50 % / 137 s; qwen3.5 **with WM 67 % / 188 s** and +34 % final energy, costing +0.5 s per decision (4.1→4.6 s). Small n (6 agent-episodes/condition) — suggestive, not significant. Notably bare qwen did not beat the heuristic; the planner is what moved the needle. Table: `bridge/worldmodel/ablation_1784212367.md`.

## Open decisions
- [ ] Scale the world-model ablation (more seeds/episodes; gemma + xortron conditions).
- [ ] Multi-agent communication channel (talk/collaborate/gossip) — the next step toward the "real-like" world.

## Done

- 2026-07-16 — Project directory created; CLAUDE.md, STATUS.md, and project memory set up.
- 2026-07-16 — v1 2D hill physics sandbox built and smoke-tested (now `hill2d.html`).
- 2026-07-16 — v2 3D WebGL version (`index.html`): full-3D physics engine + Three.js rendering, verified by Node physics tests and headless-browser screenshots.

## Next steps

1. Get user feedback on the 3D sandbox (feel, missing physics, more object types?).
2. `git init` + first commit.
3. Possible extensions: draw-your-own terrain / heightmap import, wind, boxes & cylinders, trajectory export (CSV/JSON) for later world-model training, first-person "follow ball" camera.
