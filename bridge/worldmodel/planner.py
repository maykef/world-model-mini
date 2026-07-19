#!/usr/bin/env python3
"""
planner.py — lookahead over candidate headings, two interchangeable sources:

  LearnedPlanner   the trained dynamics MLP (model.pt). Step 1 uses observed
                   slopes; later steps use the agent's world-map cells; leaving
                   explored territory raises an OOD flag (the model predicts
                   consequences of actions, not unseen terrain).
  AnalyticPlanner  ground-truth oracle: a persistent node subprocess running the
                   sim's own physics (oracle.js) — exact Minetti costs on the
                   true terrain. No OOD flags: it knows the world.

Both return the same result shape and render the same prompt block, so agent
comparisons isolate exactly one variable: the prediction source. Both are
deterministic. A full plan is ~1 ms (learned) / ~10 ms (analytic).
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train import DynamicsMLP  # noqa: E402
from wmcommon import COMPASS, candidates_for, slope_toward  # noqa: E402

CELL = 5  # must match WorldMemory.CELL
DEFAULT_SPEED = 1.4
HORIZON = 3
STEP_S = 2.0


def _attach_food(r, cand, obs):
    if cand["food_idx"] is not None:
        f = obs["food"][cand["food_idx"]]
        r["food_dist_m"] = f["dist_m"]
        r["reaches_food"] = r["disp_m"] + 0.1 >= f["dist_m"]
    return r


def render(results, obs):
    """Compact prompt block — identical for both planner sources by design.

    Food-heading rollouts STOP at the food, so their dE is the cost to reach it
    — not the cost of overshooting it for the full horizon, which made adjacent
    food look like the worst option (the groundhog analytic failure)."""
    lines = ["\nPLANNER (world model) — predicted cost per heading, walking at "
             f"{DEFAULT_SPEED} m/s for up to ~{HORIZON * STEP_S:.0f} s "
             "(negative dE = energy spent; faster walking spends proportionally more per second):"]
    for r in sorted(results, key=lambda r: r["net_dE_kJ"], reverse=True):
        tag = ""
        if r.get("food_idx") is not None:
            f = obs["food"][r["food_idx"]]
            if r.get("reaches_food"):
                tag = (f" → REACHES food#{r['food_idx']} ({f['dist_m']} m away): "
                       "dE is the full cost to get there; eating refills you")
            else:
                tag = f" → toward food#{r['food_idx']} ({f['dist_m']} m away, beyond this prediction)"
        unc = "  [UNKNOWN TERRAIN beyond {:.0f} m — unexplored]".format(r["disp_m"]) if r.get("ood") else ""
        lines.append(f"  heading {r['heading_deg']:>3}°: dE {r['net_dE_kJ']:+.1f} kJ, "
                     f"moves {r['disp_m']:.1f} m{tag}{unc}")
    lines.append("Prefer cheap headings that make progress; treat UNKNOWN TERRAIN predictions as unreliable.")
    return "\n".join(lines)


class LearnedPlanner:
    mode = "learned"

    def __init__(self, model_path=os.path.join(HERE, "model.pt")):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        torch.manual_seed(cfg.get("seed", 0))
        self.model = DynamicsMLP(cfg["n_in"], cfg["n_out"], cfg["hidden"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.norm = ckpt["norm"]
        self.xm = np.asarray(self.norm["x_mean"], dtype=np.float32)
        self.xs = np.asarray(self.norm["x_std"], dtype=np.float32)
        self.ym = np.asarray(self.norm["y_mean"], dtype=np.float32)
        self.ys = np.asarray(self.norm["y_std"], dtype=np.float32)

    def _predict(self, X):
        Xn = torch.tensor((np.asarray(X, dtype=np.float32) - self.xm) / self.xs)
        with torch.no_grad():
            Yn = self.model(Xn).numpy()
        return Yn * self.ys + self.ym            # [:, (delta_energy_kJ, displacement_m)]

    @staticmethod
    def _cell_alt(cells, x, z):
        c = cells.get(f"{int(x // CELL) * CELL},{int(z // CELL) * CELL}")
        return c["alt"] if c else None

    def plan(self, obs, cells=None, horizon=HORIZON, step_s=STEP_S, speed=DEFAULT_SPEED):
        cells = cells or {}
        wading = 1.0 if (obs.get("water_level_m", 0) > 0
                         and obs["pos_m"]["alt"] < obs["water_level_m"]) else 0.0
        metab = float(obs.get("metabolic_scale", 10))
        results = []
        for cand in candidates_for(obs):
            h = cand["heading_deg"]
            hr = math.radians(h)
            ux, uz = math.sin(hr), -math.cos(hr)
            x, z = obs["pos_m"]["x"], obs["pos_m"]["z"]
            alt = obs["pos_m"]["alt"]
            energy = float(obs["energy_kJ"])
            food_stop = (obs["food"][cand["food_idx"]]["dist_m"]
                         if cand["food_idx"] is not None else None)
            net_e, net_d, steps, ood = 0.0, 0.0, 0, False
            for k in range(horizon):
                if k == 0:
                    slope = slope_toward(obs["slope_pct"], h)
                else:
                    step_len = speed * step_s
                    a2 = self._cell_alt(cells, x + ux * step_len, z + uz * step_len)
                    a1 = self._cell_alt(cells, x, z) or alt
                    if a2 is None:
                        ood = True
                        break
                    slope = 100 * (a2 - a1) / max(1e-6, step_len)
                feats = [energy + net_e, slope,
                         *[obs["slope_pct"][d] for d in COMPASS],
                         speed, wading, metab]
                de, disp = self._predict([feats])[0]
                de, disp = float(de), float(disp)
                if food_stop is not None and disp > 0 and net_d + disp >= food_stop:
                    # rollout stops AT the food: charge only the fraction of this
                    # step needed to reach it, never the terrain beyond
                    frac = max(0.0, (food_stop - net_d) / disp)
                    net_e += de * frac
                    net_d += disp * frac
                    steps += 1
                    break
                net_e += de
                net_d += disp
                x += ux * disp
                z += uz * disp
                steps += 1
            r = {"heading_deg": round(h), "net_dE_kJ": round(net_e, 1),
                 "disp_m": round(net_d, 1), "steps": steps, "ood": ood,
                 "food_idx": cand["food_idx"]}
            results.append(_attach_food(r, cand, obs))
        return results

    def render(self, results, obs):
        return render(results, obs)


class AnalyticPlanner:
    """Ground-truth oracle via the engine's own physics (oracle.js subprocess)."""
    mode = "analytic"

    def __init__(self, oracle_path=os.path.join(HERE, "oracle.js")):
        self.proc = subprocess.Popen(
            ["node", oracle_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)
        ready = json.loads(self.proc.stdout.readline())
        if not ready.get("ready"):
            raise RuntimeError(f"oracle failed to start: {ready}")

    def _ask(self, req):
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        resp = json.loads(self.proc.stdout.readline())
        if "error" in resp:
            raise RuntimeError(f"oracle error: {resp['error']}")
        return resp["results"]

    def plan(self, obs, cells=None, horizon=HORIZON, step_s=STEP_S, speed=DEFAULT_SPEED):
        size = obs.get("world_size_m", 50)
        cfg = {"terrain": obs.get("terrain", "gentle"),
               "seed": obs.get("terrain_seed", 0),
               "size": size,
               "water": (obs.get("water_level_m", 0) / (size / 50)) if obs.get("water_level_m", 0) else 0,
               "metab": obs.get("metabolic_scale", 10)}
        cands = candidates_for(obs)
        for c in cands:                           # ghost can't eat: stop the rollout at the food
            if c["food_idx"] is not None:
                c["stop_at_m"] = obs["food"][c["food_idx"]]["dist_m"]
        results = self._ask({"cfg": cfg, "energy_kJ": obs["energy_kJ"],
                             "x": obs["pos_m"]["x"], "z": obs["pos_m"]["z"],
                             "candidates": cands, "horizon_steps": horizon,
                             "step_s": step_s, "speed": speed})
        return [_attach_food(r, c, obs) for r, c in zip(results, cands)]

    def render(self, results, obs):
        return render(results, obs)

    def close(self):
        try:
            self.proc.terminate()
        except Exception:
            pass


def make_planner(mode):
    if mode == "learned":
        return LearnedPlanner()
    if mode == "analytic":
        return AnalyticPlanner()
    return None


# backwards-compat alias (tests, earlier integrations)
Planner = LearnedPlanner
