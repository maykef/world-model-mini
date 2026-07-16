#!/usr/bin/env python3
"""
planner.py — lookahead over candidate headings using the learned dynamics model.

For each candidate heading (8 compass directions + bearings to visible food),
roll the model forward a short horizon and report predicted net energy change,
displacement, and an OOD flag. Step 1 uses the actually-observed local slopes;
later steps estimate slope from the agent's world-map cells (altitude of
visited 5 m cells). When the path leaves explored territory the rollout stops
and the candidate is flagged uncertain — the model predicts consequences of
actions, not terrain it has never seen (that extrapolation is out of scope).

CPU-only, batched, deterministic; a full plan is ~1 ms and never blocks physics.
"""
import json
import math
import os

import numpy as np
import torch

from train import DynamicsMLP  # noqa: E402  (same dir; driver adds it to sys.path)

HERE = os.path.dirname(os.path.abspath(__file__))
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
CELL = 5  # must match WorldMemory.CELL


def slope_toward(slope_pct, heading_deg):
    h = heading_deg % 360
    i = int(h // 45)
    frac = (h - i * 45) / 45
    return slope_pct[COMPASS[i]] * (1 - frac) + slope_pct[COMPASS[(i + 1) % 8]] * frac


class Planner:
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

    def candidates_for(self, obs):
        cands = [{"heading_deg": d * 45, "food_idx": None} for d in range(8)]
        for i, f in enumerate(obs.get("food", [])):
            b = f["bearing_deg"] % 360
            near = min(cands, key=lambda c: min(abs(c["heading_deg"] - b), 360 - abs(c["heading_deg"] - b)))
            gap = min(abs(near["heading_deg"] - b), 360 - abs(near["heading_deg"] - b))
            if gap <= 10:
                near["food_idx"] = i if near["food_idx"] is None else near["food_idx"]
            else:
                cands.append({"heading_deg": b, "food_idx": i})
        return cands

    def plan(self, obs, cells=None, horizon=3, step_s=2.0, speed=1.4):
        """Returns per candidate: predicted net dE (kJ), displacement (m),
        steps rolled, ood flag, food_idx (if the heading aims at visible food)."""
        cells = cells or {}
        wading = 1.0 if (obs.get("water_level_m", 0) > 0
                         and obs["pos_m"]["alt"] < obs["water_level_m"]) else 0.0
        metab = float(obs.get("metabolic_scale", 10))
        results = []
        for cand in self.candidates_for(obs):
            h = cand["heading_deg"]
            hr = math.radians(h)
            ux, uz = math.sin(hr), -math.cos(hr)
            x, z = obs["pos_m"]["x"], obs["pos_m"]["z"]
            alt = obs["pos_m"]["alt"]
            energy = float(obs["energy_kJ"])
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
                         *[obs["slope_pct"][d] for d in COMPASS],  # local context (step-1 frame)
                         speed, wading, metab]
                de, disp = self._predict([feats])[0]
                net_e += float(de)
                net_d += float(disp)
                x += ux * float(disp)
                z += uz * float(disp)
                steps += 1
            r = {"heading_deg": round(h), "net_dE_kJ": round(net_e, 1),
                 "disp_m": round(net_d, 1), "steps": steps, "ood": ood,
                 "food_idx": cand["food_idx"]}
            if cand["food_idx"] is not None:
                f = obs["food"][cand["food_idx"]]
                r["food_dist_m"] = f["dist_m"]
                r["reaches_food"] = net_d >= f["dist_m"]
            results.append(r)
        return results

    def render(self, results, obs):
        """Compact prompt block for the LLM."""
        lines = ["\nPLANNER (learned world model) — predicted consequences over the "
                 "next ~6 s per heading (negative dE = energy spent):"]
        for r in sorted(results, key=lambda r: r["net_dE_kJ"], reverse=True):
            tag = ""
            if r["food_idx"] is not None:
                tag = f" → food#{r['food_idx']} ({obs['food'][r['food_idx']]['dist_m']} m away)"
            unc = "  [UNKNOWN TERRAIN beyond {:.0f} m — unexplored]".format(r["disp_m"]) if r["ood"] else ""
            lines.append(f"  heading {r['heading_deg']:>3}°: dE {r['net_dE_kJ']:+.1f} kJ, "
                         f"moves {r['disp_m']:.1f} m{tag}{unc}")
        lines.append("Prefer cheap headings that make progress; treat UNKNOWN TERRAIN predictions as unreliable.")
        return "\n".join(lines)
