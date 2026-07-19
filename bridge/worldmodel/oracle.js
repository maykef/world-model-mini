#!/usr/bin/env node
'use strict';
// oracle.js — analytic ground-truth planner service.
//
// Answers per-heading [delta_energy, displacement] queries by running the SIM'S
// OWN physics (extracted verbatim from index.html — Minetti cost, basal burn,
// wading, walls, terrain) for a ghost agent taking a held action. No learning,
// no re-derivation: this is the same stepAgent the engine executes.
//
// Protocol: line-delimited JSON on stdin -> one JSON line on stdout per request.
//   request:  {cfg:{terrain,seed,size,water,metab}, energy_kJ, x, z,
//              candidates:[{heading_deg, stop_at_m?, ...}], horizon_steps, step_s, speed}
//   response: {results:[{heading_deg, net_dE_kJ, disp_m, steps, ood:false}]}
// stop_at_m: end the rollout once the ghost has moved that far (food candidates —
// the ghost world has no balls, so without this the quoted cost includes walking
// past the food it would actually have eaten).

const fs = require('fs');
const path = require('path');
const readline = require('readline');

const ROOT = path.resolve(__dirname, '..', '..');
const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const scripts = html.split('<script>');
const app = scripts[scripts.length - 1];
const physics = app.slice(0, app.indexOf('// ===== rendering'));
if (!physics.includes('function stepAgent')) throw new Error('physics extraction failed');

const service = `
(function () {
  let lastCfg = '';
  function configure(cfg) {
    const key = JSON.stringify(cfg);
    if (key === lastCfg) return;
    lastCfg = key;
    P.terrain = cfg.terrain;
    P.pSeed = (cfg.seed >>> 0) || 0;
    noise2 = buildNoise(P.pSeed);
    setWorldSize(cfg.size || 50);
    P.water = cfg.water || 0;
    P.metab = cfg.metab || 10;
    balls = [];               // ghost world: no food, no rival
    agents = [];
    episode = null;
  }

  global.handleRequest = function (req) {
    configure(req.cfg);
    const H = req.horizon_steps || 3;
    const stepS = req.step_s || 2.0;
    const speed = req.speed == null ? 1.4 : req.speed;
    const subSteps = Math.round(stepS / DT);
    const results = [];
    for (const cand of req.candidates) {
      const a = makeAgent(0, req.x, req.z);
      agents = [a];                       // stepAgent's eat-scan sees empty balls
      a.energy = req.energy_kJ * 1000;
      a.maxEnergy = Math.max(a.maxEnergy, a.energy * 2);  // never satiation-gated
      a.external = true;
      const d = dirFromCompass(cand.heading_deg || 0);
      a.extAction = { heading: Math.atan2(d.z, d.x), speed };
      a.decideAt = Infinity;
      const x0 = a.p.x, z0 = a.p.z, e0 = a.energy;
      const stopAt = cand.stop_at_m == null ? Infinity : cand.stop_at_m;
      for (let s = 0; s < H * subSteps && a.alive; s++) {
        stepAgent(a);
        if (Math.hypot(a.p.x - x0, a.p.z - z0) >= stopAt) break;
      }
      results.push({
        heading_deg: Math.round((cand.heading_deg || 0)),
        net_dE_kJ: Math.round((a.energy - e0) / 100) / 10,
        disp_m: Math.round(Math.hypot(a.p.x - x0, a.p.z - z0) * 10) / 10,
        steps: H, ood: false,
        food_idx: cand.food_idx == null ? null : cand.food_idx,
      });
    }
    agents = [];
    return { results };
  };
})();
`;

eval(physics + '\n' + service);

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', line => {
  if (!line.trim()) return;
  try {
    const req = JSON.parse(line);
    process.stdout.write(JSON.stringify(global.handleRequest(req)) + '\n');
  } catch (e) {
    process.stdout.write(JSON.stringify({ error: String(e && e.message || e) }) + '\n');
  }
});
process.stdout.write(JSON.stringify({ ready: true }) + '\n');
