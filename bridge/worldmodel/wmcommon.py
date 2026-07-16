#!/usr/bin/env python3
"""
wmcommon.py — shared utilities for the world-model pipeline.

Includes the SYNTHETIC-DATA GUARD (see CLAUDE.md history): every trace consumed
for training or evaluation must carry a provenance meta record written at
generation time ({"kind":"meta","real":true,...}) AND pass a plausibility check
(agent positions must actually vary — fabricated protocol-test traces had frozen
positions). Files failing either check raise SyntheticDataError; they are never
silently skipped into a metric.
"""
import json
import math
import os

COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


class SyntheticDataError(RuntimeError):
    pass


def slope_toward(slope_pct, heading_deg):
    """Interpolate the 8 compass slope readings toward an arbitrary heading."""
    h = heading_deg % 360
    i = int(h // 45)
    frac = (h - i * 45) / 45
    return slope_pct[COMPASS[i]] * (1 - frac) + slope_pct[COMPASS[(i + 1) % 8]] * frac


def candidates_for(obs, merge_deg=10):
    """Candidate headings: 8 compass directions + bearings to visible food
    (merged into a compass direction when within merge_deg)."""
    cands = [{"heading_deg": d * 45, "food_idx": None} for d in range(8)]
    for i, f in enumerate(obs.get("food", [])):
        b = f["bearing_deg"] % 360
        near = min(cands, key=lambda c: min(abs(c["heading_deg"] - b), 360 - abs(c["heading_deg"] - b)))
        gap = min(abs(near["heading_deg"] - b), 360 - abs(near["heading_deg"] - b))
        if gap <= merge_deg:
            near["food_idx"] = i if near["food_idx"] is None else near["food_idx"]
        else:
            cands.append({"heading_deg": b, "food_idx": i})
    return cands


def contiguous(t1, t2, lo=1.5, hi=2.5):
    """One decision interval apart — eating exclusions/gaps break contiguity."""
    return lo <= t2 - t1 <= hi


def load_trace_records(path, require_real=True):
    """Load a trace file, enforcing the synthetic-data guard.

    Returns (meta, records). Raises SyntheticDataError when require_real and
    the file has no real-provenance meta record, or when decision positions are
    implausibly frozen (the signature of fabricated protocol-test data)."""
    records, meta = [], None
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "meta" and meta is None:
                meta = rec
            else:
                records.append(rec)

    if require_real:
        if not meta or meta.get("real") is not True:
            raise SyntheticDataError(
                f"{os.path.basename(path)}: no real-provenance meta record — refusing to use "
                "(pre-guard or synthetic/protocol-test data; see bridge/results/protocol-tests/)")
        # plausibility: within each (episode, agent), positions must move
        span = {}
        for r in records:
            if r.get("kind") != "decision":
                continue
            p = (r.get("obs") or {}).get("pos_m")
            if not p:
                continue
            k = (r.get("episode"), r.get("agent"))
            s = span.setdefault(k, [p["x"], p["x"], p["z"], p["z"], 0])
            s[0] = min(s[0], p["x"]); s[1] = max(s[1], p["x"])
            s[2] = min(s[2], p["z"]); s[3] = max(s[3], p["z"])
            s[4] += 1
        for k, s in span.items():
            if s[4] >= 5 and math.hypot(s[1] - s[0], s[3] - s[2]) < 0.5:
                raise SyntheticDataError(
                    f"{os.path.basename(path)}: episode/agent {k} has frozen positions over "
                    f"{s[4]} decisions — synthetic data signature, refusing to use")
    return meta, records


def meta_record(source, **extra):
    """Provenance record written as the FIRST line of every generated trace."""
    return {"kind": "meta", "real": True, "source": source, **extra}
