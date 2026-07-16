"""Headless tests: dataset shapes, model forward pass, rollout determinism,
planner behaviour incl. the OOD flag. Run: venv/bin/python -m pytest bridge/worldmodel/"""
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dataset  # noqa: E402
from train import DynamicsMLP  # noqa: E402


def _obs(x, z, energy, t, eaten=0, slopes=None):
    return {
        "name": "Amber", "t_s": t, "eaten": eaten, "energy_kJ": energy, "energy_pct": 50,
        "can_eat": True, "eat_below_pct": 55,
        "pos_m": {"x": x, "z": z, "alt": 2.0},
        "heading_deg": 0, "speed_ms": 1.2,
        "slope_pct": slopes or {"N": 10, "NE": 5, "E": 0, "SE": -5, "S": -10, "SW": -5, "W": 0, "NW": 5},
        "food": [], "other_agent": None, "water_level_m": 0,
        "world_size_m": 50, "metabolic_scale": 10,
        "terrain": "gentle", "terrain_seed": 0,
    }


def _write_trace(path, n=8):
    with open(path, "w") as f:
        for i in range(n):
            rec = {"kind": "decision", "episode": 0, "agent": "Amber", "t_s": 2.0 * (i + 1),
                   "obs": _obs(10 + i, 10, 400 - 5 * i, 2.0 * (i + 1),
                               eaten=1 if i >= 5 else 0),
                   "action": {"heading_deg": 90.0, "speed": 1.2, "reason": "t", "memory": ""},
                   "latency_s": 0}
            f.write(json.dumps(rec) + "\n")


def test_dataset_shapes_and_eat_exclusion(tmp_path):
    _write_trace(tmp_path / "t_trace.jsonl", n=8)
    counts = dataset.build(results_dir=str(tmp_path), out_dir=str(tmp_path))
    d = np.load(tmp_path / "data.npz", allow_pickle=True)
    total = sum(counts.values())
    # 8 records -> 7 consecutive pairs, minus 1 eat transition (eaten 0 -> 1)
    assert total == 6
    for s in ("train", "val", "test"):
        X, Y = d[f"X_{s}"], d[f"Y_{s}"]
        assert X.shape[1] == len(dataset.FEATURE_NAMES) == 13
        assert Y.shape[1] == len(dataset.TARGET_NAMES) == 2
    with open(tmp_path / "norm.json") as f:
        norm = json.load(f)
    assert len(norm["x_mean"]) == 13 and len(norm["y_std"]) == 2


def test_slope_interpolation():
    slopes = {"N": 0, "NE": 40, "E": 0, "SE": 0, "S": 0, "SW": 0, "W": 0, "NW": 0}
    assert dataset.slope_toward(slopes, 45) == 40
    assert abs(dataset.slope_toward(slopes, 22.5) - 20) < 1e-9
    assert dataset.slope_toward(slopes, 180) == 0


def test_forward_pass_shape():
    torch.manual_seed(0)
    m = DynamicsMLP(13, 2)
    y = m(torch.zeros(7, 13))
    assert y.shape == (7, 2)


def test_trained_model_and_rollout_determinism():
    path = os.path.join(HERE, "model.pt")
    if not os.path.exists(path):
        import pytest
        pytest.skip("model.pt not trained yet")
    from planner import Planner
    p1, p2 = Planner(path), Planner(path)
    obs = _obs(25, 25, 300, 10.0)
    obs["food"] = [{"bearing_deg": 90, "dist_m": 8.0, "elev_m": 0, "in_water": False}]
    cells = {f"{cx},{cz}": {"alt": 2.0, "steep": 5, "visits": 1}
             for cx in range(0, 50, 5) for cz in range(0, 50, 5)}
    r1 = p1.plan(obs, cells)
    r2 = p2.plan(obs, cells)
    assert r1 == r2                                   # deterministic across instances
    assert any(r["food_idx"] is not None for r in r1)  # food candidate present
    assert all(not r["ood"] for r in r1)               # fully explored map -> no OOD
    # empty map -> everything beyond step 1 is unknown
    r3 = p1.plan(obs, {})
    assert all(r["ood"] for r in r3)
    assert all(r["steps"] == 1 for r in r3)
    text = p1.render(r1, obs)
    assert "PLANNER" in text and "heading" in text
