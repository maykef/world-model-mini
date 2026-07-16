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


def _write_trace(path, n=8, meta=True, frozen=False):
    with open(path, "w") as f:
        if meta:
            f.write(json.dumps({"kind": "meta", "real": True, "source": "physics-collector"}) + "\n")
        for i in range(n):
            rec = {"kind": "decision", "episode": 0, "agent": "Amber", "t_s": 2.0 * (i + 1),
                   "obs": _obs(10 if frozen else 10 + i, 10, 400 - 5 * i, 2.0 * (i + 1),
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


def test_synthetic_guard_fires():
    import pytest
    from wmcommon import SyntheticDataError, load_trace_records
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # untagged file (pre-guard / fabricated) -> refused
        p1 = os.path.join(td, "untagged_trace.jsonl")
        _write_trace(p1, meta=False)
        with pytest.raises(SyntheticDataError, match="no real-provenance"):
            load_trace_records(p1)
        # tagged but frozen positions (fabrication signature) -> refused
        p2 = os.path.join(td, "frozen_trace.jsonl")
        _write_trace(p2, meta=True, frozen=True)
        with pytest.raises(SyntheticDataError, match="frozen positions"):
            load_trace_records(p2)
        # tagged, plausible -> accepted
        p3 = os.path.join(td, "good_trace.jsonl")
        _write_trace(p3)
        meta, records = load_trace_records(p3)
        assert meta["real"] is True and len(records) == 8


def test_analytic_planner_matches_engine_cost():
    """Oracle must reproduce Minetti + basal cost, independently integrated over
    the known analytic 'gentle' terrain (h = 1 + 13·exp(−r²/200))."""
    from planner import AnalyticPlanner

    def terrain_h(x, z):
        return 1 + 13 * math.exp(-((x - 25) ** 2 + (z - 25) ** 2) / 200)

    def cot(g):
        g = max(-0.45, min(0.45, g))
        return max(1.5, 280.5 * g**5 - 58.7 * g**4 - 76.8 * g**3 + 51.9 * g * g + 19.6 * g + 2.5)

    mass, metab, speed, dt = 70.0, 10, 1.4, 1 / 240
    x, z, expected = 5.0, 45.0, 0.0
    for _ in range(3 * 480):                     # 3 steps × 2 s at 240 Hz, heading east
        nx = x + speed * dt
        dh = terrain_h(nx, z) - terrain_h(x, z)
        horiz = speed * dt
        expected += mass * 2.0 * dt              # basal, 2 W/kg
        expected += mass * cot(dh / horiz) * math.hypot(horiz, dh)
        x = nx
    expected_kJ = expected * metab / 1000

    oracle = AnalyticPlanner()
    try:
        obs = _obs(5, 45, 400, 10.0)
        obs.update(terrain="gentle", terrain_seed=0, world_size_m=50, metabolic_scale=10)
        r = {c["heading_deg"]: c for c in oracle.plan(obs, horizon=3, speed=1.4)}[90]
        assert abs(-r["net_dE_kJ"] - expected_kJ) / expected_kJ < 0.02, (r, expected_kJ)
        assert abs(r["disp_m"] - 8.4) < 0.5, r
        assert r["ood"] is False
        r2 = {c["heading_deg"]: c for c in oracle.plan(obs, horizon=3, speed=1.4)}[90]
        assert r == r2                           # deterministic
    finally:
        oracle.close()


def test_paired_roles_counterbalance():
    """Both agents share one sim episode by construction; the runner must swap
    planner roles across episodes so spawn asymmetry cancels."""
    import argparse
    sys.path.insert(0, os.path.join(HERE, ".."))
    import mind_driver
    args = argparse.Namespace(mock=True, planner="learned", paired=True, worldmodel=False,
                              episodes=2, seeds=None, seed=1, food_seed=1, terrain="perlin",
                              size=50, water=0, metab=10, duration=60, food_interval=12,
                              max_tokens=10, think=False)
    d = mind_driver.Driver.__new__(mind_driver.Driver)   # no __init__: pure role logic
    d.args = args
    d.paired = True
    d.planner = object()
    d.planner_mode = "learned"
    d.ep_idx = 0
    assert d.planner_for("Amber") and not d.planner_for("Cyan")
    assert d.roles() == {"Amber": "learned", "Cyan": "off"}
    d.ep_idx = 1
    assert not d.planner_for("Amber") and d.planner_for("Cyan")
    assert d.roles() == {"Amber": "off", "Cyan": "learned"}
    # identical world state: paired mode changes NO cfg fields
    d2 = mind_driver.Driver.__new__(mind_driver.Driver)
    d2.args = args
    unpaired_args = argparse.Namespace(**{**vars(args), "paired": False})
    d3 = mind_driver.Driver.__new__(mind_driver.Driver)
    d3.args = unpaired_args
    for i in range(3):
        d2.ep_idx = d3.ep_idx = i
        assert mind_driver.Driver.episode_cfg(d2, i) == mind_driver.Driver.episode_cfg(d3, i)


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
