# world-model-mini

A testbed for **AI models living in real terrain conditions**: a 3D physics world (WebGL, real units) where embodied agents forage for food under realistic energy costs (the steeper the terrain, the more energy spent), with LLMs served by Delphi acting as the agents' minds via a WebSocket bridge. GitHub: `maykef/world-model-mini`.

## Owner's instructions (standing)

- **"Training" means experiential memory, NOT weight updates.** The agents "learn" by accumulating an accurate memory of the environment from their own observations (terrain map, food locations, rivals, cheap routes) — Generative-Agents-style. Never propose fine-tuning as the default meaning of "train".
- **Balls-as-food are a placeholder** — they work for now, but the food abstraction may change.
- **Fairness matters:** identical seeded conditions across model comparisons (terrain seed, food layout, spawns); food must cover the whole map, not cluster.
- **One GPU:** Delphi serves ONE model at a time (qwen3.5-122b / gemma-4-31b-heretic / xortron-123b) — benchmark models sequentially against the same seeds; never try to run them simultaneously.
- **Long-term vision:** scale to many agents that communicate, collaborate, gossip, even invent language in a richer "real-like" world. Keep the current setup deliberately basic and build toward that.
- Never clobber Delphi's Tailscale mapping (root domain, port 443 → 33213).

## Project state

- Started 2026-07-16. See `STATUS.md` for current progress, decisions, and next steps.
- `index.html` — the world: hand-written 3D physics engine + agents + Three.js (r147, vendored inline) in one self-contained file, served at https://microscopy-rig-system.tail53cc58.ts.net:8443/
- `bridge/mind_driver.py` — the minds: WebSocket bridge sim↔Delphi (OpenAI API :8000), episodes, benchmark logging, three-layer memory (scratchpad / accurate world map / reflection journal).
- `serve.py` — static server + `/stats` (GPU/CPU/RAM gauges), run by the `hillsim` systemd user service.
- `hill2d.html` — the earlier 2D side-view version, kept for quick physics experiments.

## Working conventions

- **Keep it little.** Prefer the simplest thing that works; no premature abstraction, no config frameworks, no multi-file sprawl before there's something to sprawl.
- **STATUS.md is the source of truth for progress.** Update it whenever a milestone lands or the plan changes.
- Single self-contained HTML file per app — no CDNs, no npm, no build step, so it opens from `file://` anywhere. Third-party libs (Three.js) are allowed but must be **vendored inline** into the HTML. Any future ML/training code defaults to Python.
- **Verify physics headlessly.** The physics section of the app script is DOM-free and THREE-free by design (everything above the `// ===== rendering` marker in the last `<script>` block). To test: extract that section, append assertions, run with `node`. Keep physics and rendering separated so this stays possible.
- **Verify rendering headlessly too:** `/snap/bin/chromium --headless=new --use-angle=swiftshader --enable-unsafe-swiftshader --screenshot=... file://...` (paths must be under `$HOME` — snap confinement), then look at the PNG. Headless Firefox hangs; don't use it. Note: `--virtual-time-budget` freezes `requestAnimationFrame` timestamps, so the sim clock reading 0 in screenshots is an artifact.
- Real units everywhere: metres, kilograms, seconds. Physics substep is fixed (1/240 s), decoupled from the render loop.
- This machine drives its display (GNOME/Xorg) on an RTX PRO 6000 Blackwell (96 GB, usually ~90 % VRAM-occupied by a vLLM server — graphics fine, but check `nvidia-smi` before assuming free VRAM for training).
- Large artifacts (recorded trajectories, checkpoints, videos) go under `data/` and `checkpoints/` — keep them out of git.
- This machine has a large NVMe scratch volume (`/mnt/nvme8tb`), so disk space is not a constraint; GPU availability should be checked before assuming training capacity.

## Layout

```
world-model-mini/
├── CLAUDE.md            # this file — conventions & owner's instructions
├── STATUS.md            # current state, decisions, next steps
├── index.html           # the world: physics + agents + Three.js (inlined) + UI, one file
├── hill2d.html          # earlier 2D side-view version (canvas), kept for quick tests
├── serve.py             # static server + /stats gauges (hillsim systemd user service)
├── bridge/
│   ├── mind_driver.py   # LLM bridge: WS server, episodes, benchmark, memory layers
│   ├── results/         # episode metrics + decision traces, JSONL (gitignored)
│   └── memory/          # per-model journal + world maps, JSON (gitignored)
└── venv/                # python deps: websockets, openai (gitignored)
```
