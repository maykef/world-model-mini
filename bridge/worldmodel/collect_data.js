#!/usr/bin/env node
'use strict';
// collect_data.js — generate REAL physics transitions for world-model training.
//
// Extracts the DOM-free physics section from index.html verbatim (the sim file
// is never modified — same pattern as the node smoke tests, see CLAUDE.md) and
// runs scripted episodes: agents take held 2-second external actions exactly
// like the LLM loop, and every (obs, action) decision is logged in the same
// JSONL schema mind_driver.py writes. Fully seeded -> reproducible.
//
// Usage: node collect_data.js [outfile]

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const scripts = html.split('<script>');
const app = scripts[scripts.length - 1];
const physics = app.slice(0, app.indexOf('// ===== rendering'));
if (!physics.includes('function stepWorld')) throw new Error('physics extraction failed');

const OUT = process.argv[2] ||
  path.join(ROOT, 'bridge', 'results', `collect_${Math.floor(Date.now() / 1000)}_trace.jsonl`);
const out = fs.createWriteStream(OUT);
const emit = o => out.write(JSON.stringify(o) + '\n');

// episode matrix: terrain variety + water + metabolic scales, all seeded
const CONFIGS = [];
for (const terrain of ['gentle', 'steep', 'twin', 'bowl', 'volcano', 'bumpy']) {
  CONFIGS.push({ terrain, seed: 1337, size: 50, water: 0, metab: 10 });
}
for (const seed of [1337, 42, 2718, 9001]) {
  CONFIGS.push({ terrain: 'perlin', seed, size: 50, water: 0, metab: 10 });
}
CONFIGS.push({ terrain: 'perlin', seed: 1337, size: 50, water: 3, metab: 10 });
CONFIGS.push({ terrain: 'perlin', seed: 42, size: 50, water: 2.5, metab: 25 });
CONFIGS.push({ terrain: 'steep', seed: 1337, size: 50, water: 0, metab: 25 });

const EPISODES_PER_CFG = 3;
const DURATION_S = 240;
const DECIDE_S = 2.0;

const body = `
(function main() {
  const OBS_STEPS = Math.round(${DECIDE_S} / DT);
  let epIdx = 0;
  for (const cfg of CONFIGS) {
    for (let ep = 0; ep < ${EPISODES_PER_CFG}; ep++) {
      // --- world setup (mirrors applyReset, minus rendering) ---
      P.terrain = cfg.terrain; P.pSeed = cfg.seed >>> 0;
      noise2 = buildNoise(P.pSeed);
      setWorldSize(cfg.size);
      P.water = cfg.water; P.metab = cfg.metab;
      P.agentsOn = true; P.control = 'llm';
      if (P.terrain === 'perlin') findPerlinPeak();
      balls = []; selected = null; simTime = 0; colorIdx = 0;
      spawnAgents();
      for (const a of agents) a.external = true;
      const foodSeed = 777 + epIdx * 13;
      beginEpisode({ food_seed: foodSeed, food_interval_s: 12, first_food_s: 4,
                     episode_s: ${DURATION_S} });
      const fullCfg = Object.assign({ food_seed: foodSeed, food_interval_s: 12,
                                      first_food_s: 4, episode_s: ${DURATION_S} }, cfg);
      emit({ kind: 'reset_done', episode: epIdx, iteration: 0, cfg: fullCfg });

      // per-agent seeded policy rngs
      const rngs = agents.map((a, i) => mulberry32(1e6 + epIdx * 977 + i * 131));

      while (episode.active) {
        // decide: held external action per agent, exactly like the LLM loop
        for (let i = 0; i < agents.length; i++) {
          const a = agents[i];
          if (!a.alive) continue;
          const obs = buildObs(a);
          const r = rngs[i];
          const hungry = a.energy < 0.55 * a.maxEnergy;
          let heading_deg;
          if (obs.food.length && hungry && r() < 0.7) {
            heading_deg = (obs.food[0].bearing_deg + (r() - 0.5) * 30 + 360) % 360;
          } else {
            heading_deg = r() * 360;
          }
          const speed = [0.4, 0.8, 1.2, 1.6, 2.0][Math.floor(r() * 5)];
          const d = dirFromCompass(heading_deg);
          a.extAction = { heading: Math.atan2(d.z, d.x), speed };
          emit({ kind: 'decision', episode: epIdx, agent: a.name, t_s: obs.t_s,
                 obs, action: { heading_deg: Math.round(heading_deg * 10) / 10, speed,
                                reason: 'collect', memory: '' },
                 latency_s: 0 });
        }
        // advance one decision interval
        for (let s = 0; s < OBS_STEPS && episode.active; s++) {
          stepWorld();
          if (simTime >= episode.endAt || agents.every(a => !a.alive)) episode.active = false;
        }
      }
      emit({ kind: 'episode_end', episode: epIdx, iteration: 0,
             t_s: Math.round(simTime), food_left: balls.length,
             cfg: fullCfg, metrics: episodeMetrics() });
      epIdx++;
    }
  }
  console.log('episodes:', epIdx);
})();
`;

eval(physics + '\n' + body);
out.end(() => {
  const lines = fs.readFileSync(OUT, 'utf8').trim().split('\n');
  const dec = lines.filter(l => l.includes('"kind": "decision"')).length;
  console.log(`wrote ${OUT}`);
  console.log(`records: ${lines.length}  decisions: ${dec}`);
});
