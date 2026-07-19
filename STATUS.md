# STATUS — world-model-mini

_Last updated: 2026-07-18_

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

**Scaled ablation (2026-07-16, n=30/condition, 5 terrain seeds, CIs):** the n=6 pilot's learned-WM advantage **did not replicate** — learned 0.40 [0.25,0.58] vs off 0.37 [0.22,0.54]; the paired within-episode design (15 counterbalanced pairs) confirms the null (Δsurvival-time −12 s [−94,+69]). The **analytic oracle** (ground-truth costs via `oracle.js`, identical prompt shape) genuinely helps: 0.60 [0.42,0.75], +45 s lifespan, +73 % final energy vs off. Learned-vs-oracle prediction error is small on average (ΔE MAE 3.5 kJ/heading, mean choice regret 1.8 kJ) but the model underpredicts extreme slope costs — the lethal tail — and empty-map episodes truncate its rollout to 1 step. Heuristic baseline (0.57) ≈ oracle-LLM (0.60) at zero latency. Conclusion: accurate foresight helps; lossy foresight doesn't; deliberation adds nothing beyond accurate greedy in this world. Data: `bridge/worldmodel/scaled_ablation_combined.{md,csv}`; guard: all metrics from provenance-tagged real-physics traces only. Grid runner (`grid.py`) ready for gemma/xortron (needs Delphi model switch — user consent).

**Groundhog learning experiment (2026-07-16, n=20/condition, 10 identical lives, memory persists):** two humbling findings. (1) **No learning curve**: late-vs-early-lives differences are null or negative in every condition (e.g. learned Δsurvived −2 s [−105,+66]; off −40 s [−132,+60]) — the memory layers preserve facts faithfully, but qwen does not convert them into measurably improving performance over 10 lives. Heuristic control: perfectly flat 240 s × 10, as designed. (2) **The planner ranking flipped in the easy preset-food regime**: learned 0.85 [0.64,0.95] > off 0.55 [0.34,0.74] > analytic 0.40 [0.22,0.61] — near-inverse of the scarce-drops world. Data: `groundhog_learning.md`, `groundhog_combined.csv`. The energy-greedy heuristic saturates this world (1.00 survival, 5 meals/life).

**Root cause found & fixed (2026-07-17, from decision traces):** analytic's collapse was a **planner-harness bug, not regime-dependence**. Rollouts ran the full 6 s horizon and could not eat (the oracle's ghost world has no balls), so a food heading was billed for walking *past* the ball up whatever terrain lay beyond — at Amber's spawn the block quoted −124 kJ for food 3.2 m away (true cost-to-ball ≈ −50 kJ) and ranked the best move in the world dead last of 10. Trace evidence against the earlier "biased away from food" story: hungry agents chose food headings at the same ~80 % rate in *all* conditions; instead the inflated costs made qwen panic-sprint (84 % of analytic decisions at 2.0 m/s vs 77 % at 1.5 m/s for learned), and with the 4–6 s action lag it overshot balls, zigzagged, and starved in ~22–45 s with 0 eaten (7/10 Amber + 4/10 Cyan lives). Verified NOT broken: oracle numbers match real transitions (terrain identity mean err 0.018 m), cheapest-first wording is sign-correct, and the block rendered in 811/811 decisions. **Fix (planner.py / oracle.js):** food-heading rollouts now stop AT the food (`stop_at_m` in the oracle; fractional final step in the learned planner), the block tags them "REACHES food#N — dE is the full cost to get there; eating refills you", and the header states the 1.4 m/s speed assumption. Regression test `test_food_rollout_stops_at_food` added; suite 8/8. Groundhog numbers above predate the fix.

## Vision mode — egocentric pixel perception (2026-07-18)

Replacing oracle JSON perception with **egocentric pixel perception** as a switchable observation regime, to measure the "perception tax" — the survival/efficiency gap between regimes on byte-identical seeds. Three-regime ablation ladder planned: `oracle` (ceiling/control) · `pixels` (frames + minimal interoception) · `pixels+proprio` (frames + full interoception). See the vision-mode brief for the full plan (Phases 1–4 + benchmark).

**Phase 1 — egocentric cameras + frame streaming (sim side, `index.html`) — DONE.**
- Per-agent head camera (1.7 m above contact, FOV 100°, oriented to locomotion heading) whose pose derives entirely from agent state each render tick (no independent state → cannot drift). Rendered to an offscreen 448×448 `WebGLRenderTarget`, encoded to JPEG base64 (q0.7). Constants at top of the vision module (`FRAME_SIZE`, `FRAME_FOV`, `HEAD_HEIGHT`, …).
- **Keyframe gating**: 32×32 luma thumbnail of the last *sent* frame per agent; a new frame is emitted when mean-abs luma diff > `LUMA_THRESH`, OR `MAX_FRAME_AGE_S` (2.0 sim-s) elapsed, OR a discrete event fires (episode start, ate-in-view, a ball spawned/consumed nearby). `keyframe=true` for change/event; age-only heartbeats are `keyframe=false`.
- **Protocol** (backward compatible): each per-agent obs entry gains an optional `frame:{jpg_b64, ts_sim, seq, keyframe}` — present only in vision mode, so with vision off the WS bytes are identical to before. Obs also gains `regime` + `oracle_hidden:true` in pixel regimes; the sim still computes the full oracle fields (for ground-truth logging) and the driver decides what the model sees. `ts_sim` on every frame extends staleness accounting to vision.
- **UI**: 👁 Vision toggle in the Agents card; two PiP viewports (top-right of the map) show each agent's egocentric view with a `#seq ◆KEY / ·hb` indicator; HUD shows per-agent frames-sent counters. `?vision=1` enables at load; reset `cfg.vision`/`cfg.regime` lets the driver request it per-episode.
- **Verified headless** (both fixed-step, wall-clock-independent — see README): `?selftest=vision` → 448×448 JPEG, 6.6 KB, keyframe (perlin terrain + preset food in view); `?selftest=traj&vision={0,1}` → **bit-identical** agent trajectories + energy over 30 sim-s (determinism is sacred; frame capture is render-only and never touches sim state or the clock). Live-page screenshot confirms PiP + toggle + counters.
- **Decisions/deviations**: (a) frame capture runs on its own `VISION_CAPTURE_S`=1.0 sim-s cadence inside the render half, decoupled from the 2 sim-s obs cadence — the newest gated frame is attached at obs time and its `ts_sim` logged, so vision staleness falls out naturally. (b) "death" as a keyframe trigger is a no-op in practice (dead agents send `obs:null`, so no frame attaches) — dropped for now. (c) The sim's `regime` label is informational; the **driver** (Phase 2) is the authority on what the model actually sees.

**Phase 2 — multimodal mind driver (`bridge/mind_driver.py`) — DONE.**
- New `--obs {oracle,pixels,pixels+proprio}` (default `oracle`, fully backward compatible). The **driver** is the authority on what the model sees; the sim always sends full oracle fields (marked `oracle_hidden`) for ground-truth logging.
- Message assembly refactored into `build_messages(obs, frame)` (shared by `llm_decide` and `--probe`). Pixel regimes: system prompt swapped for `SYSTEM_PROMPT_PIXEL` (perceive via first-person image; no map/coords given), user turn = interoception JSON + the frame as an OpenAI `image_url` data-URI. Exteroception (8-dir `slope_pct`, `food` bearings/dists, `other_agent`) is dropped; **verified** stripped by test. Interoception: `pixels` = energy/can_eat/heading/speed/time/terrain; `pixels+proprio` adds `pos_m` (path-integration), `energy_kJ`, `grade_ahead_pct` (slope underfoot in heading), `wading`.
- History replayed text-only per regime (no oracle leak, no image re-send). Response schema unchanged + optional `sightings:[{type,est_x,est_z,confidence}]`; **`parse_action` rewritten** with a string-aware balanced-brace scanner so nested sightings survive (the old flat-regex would have broken on them). `est_y` accepted as a `z` alias; missing/malformed sightings dropped leniently.
- Frame handling: newest frame per agent used at decision time; **default keyframe-only prompting** (attach the image only on a new keyframe) vs `--every-frame`; `--image-detail {low,high,auto}` passthrough. Vision staleness (`obs t_s − frame ts_sim` of the frame informing the model) logged per decision; `frames_seen` + `mean_vision_staleness_s` + `obs_regime` written to results JSONL; per-decision `vision` block added to traces.
- `--mock` fabricates a synthetic 16×16 JPEG (`SYNTH_JPEG_B64`) so the full multimodal path runs with no GPU; `--probe` builds+prints the assembled request (and, in pixel regimes, sends a synthetic or `--probe-image` frame) then the parsed response. `bridge/test_frame.jpg` is a real 448×448 ego frame bundled for real-VLM probes.
- The driver requests frames from the sim by adding `vision:true`/`regime` to the reset `cfg` in pixel regimes; sim streams accordingly.
- **Verified**: 12/12 headless tests pass (4 new: sightings parsing, oracle-stripping+image-attach, oracle-unchanged, keyframe-vs-every-frame). Full **end-to-end round-trip** with `--mock --obs pixels+proprio` + headless sim (live RAF): frames streamed, images attached, `frames_seen=18`, `mean_vision_staleness_s≈0.81`, JSONL/traces carry the new fields.
- **Real-VLM probe CONFIRMED (2026-07-18)**: Delphi's default **`qwen3.5-122b` is a vision model** (per `qwen3.5-cpa/CLAUDE.md`; verified — it ingested a real 448×448 ego frame via `image_url` with no error and returned valid JSON incl. `sightings` in ~1.3 s). Start Delphi with `cd /mnt/nvme8tb/qwen3.5-cpa && ./deploy_qwen35_vllm.sh serve` (loads ~79 GB, 2–5 min; NEVER wrap in `timeout` — SIGTERM to the group kills vLLM mid-load). Real-frame probe: `venv/bin/python bridge/mind_driver.py --probe --obs pixels+proprio --probe-image bridge/test_frame.jpg`.

**Perception-realism pass (2026-07-19) — camera + terrain shader + slope-probe.**
- **Gravity-stabilized head camera**: `aimAgentCamera` now uses world-up + a horizontal optical axis (look target at eye height) → the horizon stays level and centred, never pitching/rolling with the terrain the agent stands on. Removed the old downward-bias factor. Pose still derives purely from agent state (no drift).
- **Terrain shader** (`terrainMat.onBeforeCompile`, shader-only — physics untouched), driven by a `renderConfig` flag with three levels:
  - `naturalistic` — the original continuous elevation+steepness tint.
  - `bands` — discrete cost bands at Minetti grade thresholds **10/25/40 %** → green/olive/brown/grey (linear-space uniforms), from the world-space normal in the fragment shader.
  - `bands+contours` — bands plus faint **2 m isolines** (screen-consistent width via `fwidth`).
  All three add a **seeded, deterministic ~1 m world-space ground texture** (2-octave value noise keyed to the terrain seed) so the ground foreshortens honestly (world-space UVs, not stretched). UI selector in the World card; `?render=` URL param (encode `+` as `%2B`); `cfg.render` over the reset protocol. Switching config is a live uniform update (no recompile). Frames: `docs/vision/shading-*.png`.
- **Headless frame test rewritten** (`?selftest=vision`): decodes the JPEG (async `Image`), asserts dims are exactly 448×448 and the frame is **non-degenerate** (≥64 unique colours on a 64² downsample OR per-channel variance > 4 — a uniform sky/black frame fails both, a valid low-entropy terrain frame passes). Byte size logged as telemetry only, no longer gated. Verified: naturalistic uniq≈197/var≈6817, bands uniq≈300, bands+contours uniq≈311 — OK.
- **Determinism re-checked** (required after render changes): `?selftest=traj` vision on vs off → **bit-identical** physics over 30 sim-s. Camera/shader are render-only.
- **Slope-probe perception benchmark** (NO episode logic): `?selftest=slopeset&render=<cfg>&samples=N` renders deterministic (seed-4242) egocentric viewpoints on perlin — **identical viewpoints across configs, only the shader differs** — with ground-truth grade % ahead/left/right at 5 m & 15 m. `bridge/make_slope_probe.py` drives headless Chromium per config → `data/slopeset/{manifest.jsonl, <config>/frame_NNN.jpg}`. `mind_driver.py --slope-probe DIR` asks the VLM to estimate the six grades per frame and reports **MAE + Spearman rank-correlation per render config** (pure-Python metrics, no numpy dep). `--mock` = blind image-hash baseline; verified end-to-end against **qwen3.5-122b** (all 6 slots/frame). Real conclusions need `--samples 30+` (the 3-frame runs are plumbing only). Tests: 13/13 (added slope parsing + Spearman/rank/tie).

**Phase 3–4 + main benchmark**: not started. Phase 3 note: `build_messages` has the injection point marked for the *believed* map (from the model's own `sightings`), kept deliberately empty now so no oracle food coords leak into pixel regimes.

## Open decisions
- [ ] Re-run the groundhog ablation with the fixed planner block (expect analytic ≥ off; also re-check the scaled-ablation conclusions, which used the buggy block too).
- [ ] Longer Groundhog horizons (25–50 lives) and/or scarcer preset food to remove the ceiling and give learning room to show.
- [ ] gemma + xortron grid segments (one command each; needs Delphi model switch — user consent).
- [ ] Multi-agent communication channel (talk/collaborate/gossip) — the next step toward the "real-like" world.

## Done

- 2026-07-16 — Project directory created; CLAUDE.md, STATUS.md, and project memory set up.
- 2026-07-16 — v1 2D hill physics sandbox built and smoke-tested (now `hill2d.html`).
- 2026-07-16 — v2 3D WebGL version (`index.html`): full-3D physics engine + Three.js rendering, verified by Node physics tests and headless-browser screenshots.
- 2026-07-17 — Diagnosed groundhog analytic collapse (planner billed food headings for overshooting the ball) and fixed rollout truncation in both planners; regression test added.

## Next steps

1. Get user feedback on the 3D sandbox (feel, missing physics, more object types?).
2. `git init` + first commit.
3. Possible extensions: draw-your-own terrain / heightmap import, wind, boxes & cylinders, trajectory export (CSV/JSON) for later world-model training, first-person "follow ball" camera.
