#!/usr/bin/env python3
"""Validate that the solved PiP crop actually contains the opponent.

Re-derives the "legitimate opponent" candidate set with the same band +
static/tall blacklist logic as 2b_solve_pip.py, then for every sample that
has at least one legitimate candidate, checks whether one of them lands
inside the solved crop rect at that timestamp.  Skill target: >= 99%.
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--detections", required=True)
ap.add_argument("--cmds", required=True)
ap.add_argument("--x-min", type=float, default=None)
ap.add_argument("--x-max", type=float, default=None)
ap.add_argument("--y2-min", type=float, default=None)
ap.add_argument("--y2-max", type=float, default=None)
ap.add_argument("--max-h-ratio", type=float, default=1.8)
ap.add_argument("--static-min-lifetime-s", type=float, default=1.5)
ap.add_argument("--static-max-x-range", type=float, default=40)
ap.add_argument("--min-tid-count", type=int, default=5)
ap.add_argument("--show", type=int, default=15, help="how many misses to print")
a = ap.parse_args()

data = json.loads(Path(a.detections).read_text())
fps, W, H, samples = data["fps"], data["W"], data["H"], data["samples"]
meta = json.loads(Path(a.cmds + ".meta.json").read_text())
pip_w, pip_h = meta["pip_w"], meta["pip_h"]

x_min = a.x_min if a.x_min is not None else 0.33 * W
x_max = a.x_max if a.x_max is not None else 0.67 * W
y2_min = a.y2_min if a.y2_min is not None else 0.42 * H
y2_max = a.y2_max if a.y2_max is not None else 0.72 * H
in_band = lambda p: x_min <= p["x"] <= x_max and y2_min <= p["y2"] <= y2_max

tid_xs, tid_ts, tid_hs, all_h = defaultdict(list), defaultdict(list), defaultdict(list), []
for s in samples:
    for p in s["people"]:
        if in_band(p):
            tid_xs[p["tid"]].append(p["x"]); tid_ts[p["tid"]].append(s["t"])
            tid_hs[p["tid"]].append(p["h"]); all_h.append(p["h"])
h_cap = a.max_h_ratio * float(np.median(all_h))
static, tall = set(), set()
for tid in tid_xs:
    if (max(tid_ts[tid]) - min(tid_ts[tid]) >= a.static_min_lifetime_s
            and max(tid_xs[tid]) - min(tid_xs[tid]) < a.static_max_x_range):
        static.add(tid)
    if np.median(tid_hs[tid]) > h_cap:
        tall.add(tid)
band_count = {t: len(v) for t, v in tid_xs.items() if t not in static and t not in tall}

# ---- parse solved crop@pip x/y timeline ----
tx, xs, ys = [], [], []
cur_x = cur_y = None
for line in Path(a.cmds).read_text().splitlines():
    line = line.strip().rstrip(";")
    if not line:
        continue
    t_str, rest = line.split(None, 1)
    for cmd in rest.split(","):
        parts = cmd.split()
        if len(parts) >= 3 and parts[0] == "crop@pip":
            if parts[1] == "x": cur_x = float(parts[2])
            elif parts[1] == "y": cur_y = float(parts[2])
    if cur_x is not None and cur_y is not None:
        tx.append(float(t_str)); xs.append(cur_x); ys.append(cur_y)
tx, xs, ys = np.array(tx), np.array(xs), np.array(ys)
print(f"parsed {len(tx)} crop@pip keyframes; pip crop {pip_w}x{pip_h}")

covered = missed = skipped = 0
miss_rows = []
for s in samples:
    cands = [p for p in s["people"] if in_band(p) and p["tid"] not in static
             and p["tid"] not in tall and band_count.get(p["tid"], 0) >= a.min_tid_count
             and p["h"] <= 1.15 * h_cap]
    if not cands:
        skipped += 1
        continue
    k = int(np.clip(np.searchsorted(tx, s["t"]) - 1, 0, len(tx) - 1))
    cx0, cy0 = xs[k], ys[k]
    ok = any(cx0 <= p["x"] <= cx0 + pip_w and cy0 <= p["y"] <= cy0 + pip_h
             for p in cands)
    if ok:
        covered += 1
    else:
        missed += 1
        b = min(cands, key=lambda p: abs(p["x"] - (cx0 + pip_w / 2)))
        miss_rows.append((s["t"], b["x"], b["y"], b["tid"], cx0, cy0))

tot = covered + missed
print(f"samples with a legit opponent candidate: {tot}  (no-candidate: {skipped})")
print(f"COVERAGE: {covered}/{tot} = {100.0*covered/tot:.2f}%")
if miss_rows:
    print(f"\nfirst {min(a.show,len(miss_rows))} misses (t, opp x/y, tid, crop x/y):")
    for r in miss_rows[:a.show]:
        print(f"  t={r[0]:7.2f}  opp=({r[1]:6.0f},{r[2]:6.0f}) tid={r[3]:<5} "
              f"crop=({r[4]:6.0f},{r[5]:6.0f})  rect x[{r[4]:.0f},{r[4]+pip_w:.0f}] y[{r[5]:.0f},{r[5]+pip_h:.0f}]")
