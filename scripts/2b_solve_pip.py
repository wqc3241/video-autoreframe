#!/usr/bin/env python3
"""Solve a smooth picture-in-picture (PiP) crop path that tracks the OPPONENT
(the far-court player), for overlay by 3_encode.py --pip-cmds.

Reads the same detections JSON as 2_solve_path.py, but the detection pass
must have been run with --min-h ~35 and --imgsz 1280: a far-court opponent
is only ~50-60px tall in 1080p and is dropped entirely by the default
--min-h 80 (this is why PiP needs its own detection run).

Subject selection differs from the main picker:
  - Candidates are restricted to a far-court band (y2 in [y2-min, y2-max])
    and an x window covering the player's own court across the net.
  - A HEIGHT CAP (1.8 x the median in-band bbox height) rejects the near
    player when they walk deep into the court to collect balls: at far-court
    depth the opponent is ~50px tall while any near person entering the band
    is 120px+. Applied both per-tid (median h) and per-sample.
  - Long-lived static tids (lifetime >= 6s with x-range < 40px) are
    blacklisted as bystanders/fixtures. Do NOT require motion of every tid:
    ByteTrack fragments a 50px opponent into many short tids, and each
    fragment individually may not move (standing between rallies).
  - Cold start picks the candidate whose tid has the most in-band samples
    across the whole video (the opponent dominates presence in-band).

Emits sendcmd lines targeting the named filter instance `crop@pip` for BOTH
x and y, plus a .meta.json with the PiP source-crop dimensions.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
import osqp


def parse_aspect(s):
    w, h = s.split(":")
    return int(w), int(h)


def solve_axis(N, fit_target, fit_mask, lb, ub, w_vel, w_acc, w_fit):
    rows = np.arange(N - 1)
    D1 = sparse.csr_matrix(
        (np.r_[-np.ones(N - 1), np.ones(N - 1)],
         (np.r_[rows, rows], np.r_[rows, rows + 1])),
        shape=(N - 1, N))
    rows2 = np.arange(N - 2)
    D2 = sparse.csr_matrix(
        (np.r_[np.ones(N - 2), -2 * np.ones(N - 2), np.ones(N - 2)],
         (np.r_[rows2, rows2, rows2],
          np.r_[rows2, rows2 + 1, rows2 + 2])),
        shape=(N - 2, N))
    F = sparse.diags(fit_mask, format="csr")
    P = 2 * (w_vel * (D1.T @ D1) + w_acc * (D2.T @ D2) + w_fit * F)
    q = -2 * w_fit * (fit_mask * fit_target)
    A = sparse.eye(N, format="csr")
    solver = osqp.OSQP()
    solver.setup(P=P.tocsc(), q=q.astype(np.float64),
                 A=A.tocsc(),
                 l=lb.astype(np.float64), u=ub.astype(np.float64),
                 verbose=False, eps_abs=1e-4, eps_rel=1e-4,
                 max_iter=20000, polish=True, polish_refine_iter=5)
    res = solver.solve()
    if res.info.status_val not in (1, 2):
        raise SystemExit(f"PiP QP failed: {res.info.status}")
    return res.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True)
    ap.add_argument("--out-cmds", required=True)
    ap.add_argument("--x-min", type=float, default=None,
                    help="Opponent search window left edge (default 0.33*W)")
    ap.add_argument("--x-max", type=float, default=None,
                    help="Opponent search window right edge (default 0.67*W)")
    ap.add_argument("--y2-min", type=float, default=None,
                    help="Opponent feet-y2 lower bound (default 0.42*H)")
    ap.add_argument("--y2-max", type=float, default=None,
                    help="Opponent feet-y2 upper bound (default 0.72*H)")
    ap.add_argument("--max-h-ratio", type=float, default=1.8,
                    help="Reject candidates taller than this multiple of the "
                    "median in-band bbox height (filters the near player "
                    "walking into the far-court band to collect balls)")
    ap.add_argument("--max-h", type=float, default=None,
                    help="Absolute height cap in px; overrides --max-h-ratio")
    ap.add_argument("--static-min-lifetime-s", type=float, default=1.5,
                    help="Blacklist a tid as a static bystander when it "
                    "lives at least this long AND moves less than "
                    "--static-max-x-range. Keep this LOW (~1.5s): a static "
                    "phantom (seated spectator/furniture) that survives the "
                    "blacklist gets ADOPTed during a brief opponent "
                    "detection gap and then poisons the position anchor for "
                    "a whole tid-memory window. The real opponent's own "
                    "still fragments are also blacklisted by this, which is "
                    "fine — the hold-last-position behavior keeps them in "
                    "frame precisely because they are standing still.")
    ap.add_argument("--static-max-x-range", type=float, default=40)
    ap.add_argument("--min-tid-count", type=int, default=5,
                    help="ADOPT/COLD only consider tids with at least this "
                    "many in-band samples over the whole video. A one-off "
                    "flicker detection (e.g. a 2-sample phantom at the band "
                    "edge) must not be adoptable during a brief opponent "
                    "dropout; real opponent fragments persist >=0.5s.")
    ap.add_argument("--max-jump-px", type=float, default=150)
    ap.add_argument("--tid-memory-s", type=float, default=4.0,
                    help="After losing the lock, adopt a NEW tid appearing "
                    "within this window if it is near the last position; "
                    "past the window, cold-start re-picks by presence")
    ap.add_argument("--adopt-radius-px", type=float, default=250,
                    help="Max x-distance from last position for adopting a "
                    "new tid during the memory window. Must cover the "
                    "position jump across a rally-cut boundary (SwingVision "
                    "sources are auto-edited rally clips: the opponent can "
                    "teleport ~200px across a cut)")
    ap.add_argument("--pip-src-h", type=int, default=None,
                    help="PiP source-crop height in px. Default: "
                    "2.5 * median opponent bbox height, clamped [110, H/3]. "
                    "Keep this TIGHT: the near player's head reaches into "
                    "the far-court y-band whenever the two players x-align, "
                    "and a loose crop fills with the near player's back.")
    ap.add_argument("--pip-up-bias", type=float, default=0.05,
                    help="Shift the PiP center up by this fraction of the "
                    "crop height (crops less below the opponent's feet, "
                    "where the near player's head intrudes)")
    ap.add_argument("--pip-aspect", default="16:9")
    ap.add_argument("--w-vel", type=float, default=5.0)
    ap.add_argument("--w-acc", type=float, default=1500.0)
    ap.add_argument("--w-fit", type=float, default=1.0)
    ap.add_argument("--invisible-fit-weight", type=float, default=0.15)
    ap.add_argument("--safety-margin-frac", type=float, default=0.18,
                    help="Keep the opponent center this fraction of the crop "
                    "size away from the PiP crop edges")
    args = ap.parse_args()

    data = json.loads(Path(args.detections).read_text())
    fps = data["fps"]
    W_SRC = data["W"]
    H_SRC = data["H"]
    samples = data["samples"]

    x_min = args.x_min if args.x_min is not None else 0.33 * W_SRC
    x_max = args.x_max if args.x_max is not None else 0.67 * W_SRC
    y2_min = args.y2_min if args.y2_min is not None else 0.42 * H_SRC
    y2_max = args.y2_max if args.y2_max is not None else 0.72 * H_SRC
    print(f"Opponent band: x [{x_min:.0f}, {x_max:.0f}]  "
          f"y2 [{y2_min:.0f}, {y2_max:.0f}]")

    def in_band(p):
        return (x_min <= p["x"] <= x_max) and (y2_min <= p["y2"] <= y2_max)

    # ---- tid-level stats over the whole video (offline, retroactive) ----
    tid_xs = defaultdict(list)
    tid_ts = defaultdict(list)
    tid_hs = defaultdict(list)
    all_h = []
    for s in samples:
        for p in s["people"]:
            if in_band(p):
                tid_xs[p["tid"]].append(p["x"])
                tid_ts[p["tid"]].append(s["t"])
                tid_hs[p["tid"]].append(p["h"])
                all_h.append(p["h"])
    if not all_h:
        raise SystemExit(
            "No detections in the far-court band. Check that detection ran "
            "with --min-h ~35 --imgsz 1280, or widen the band.")

    med_band_h = float(np.median(all_h))
    h_cap = args.max_h if args.max_h is not None \
        else args.max_h_ratio * med_band_h
    print(f"In-band median h: {med_band_h:.0f}px -> height cap "
          f"{h_cap:.0f}px")

    static_tids = set()
    tall_tids = set()
    for tid in tid_xs:
        lifetime = max(tid_ts[tid]) - min(tid_ts[tid])
        x_range = max(tid_xs[tid]) - min(tid_xs[tid])
        if (lifetime >= args.static_min_lifetime_s
                and x_range < args.static_max_x_range):
            static_tids.add(tid)
        if np.median(tid_hs[tid]) > h_cap:
            tall_tids.add(tid)
    tid_band_count = Counter(
        {tid: len(xs) for tid, xs in tid_xs.items()
         if tid not in static_tids and tid not in tall_tids})
    print(f"In-band tids: {len(tid_xs)}, blacklisted static: "
          f"{sorted(static_tids)}, too tall: {sorted(tall_tids)}")
    if not tid_band_count:
        raise SystemExit("All in-band tids were filtered; widen the band or "
                         "raise --max-h-ratio.")

    # ---- Per-sample opponent pick ----
    n_s = len(samples)
    sample_cx = np.full(n_s, np.nan)
    sample_cy = np.full(n_s, np.nan)
    sample_h = np.full(n_s, np.nan)
    prev_tid = None
    prev_x = None
    prev_t = None
    reasons = Counter()

    per_sample_h_cap = 1.15 * h_cap
    for i, s in enumerate(samples):
        cands = [p for p in s["people"]
                 if in_band(p)
                 and p["tid"] not in static_tids
                 and p["tid"] not in tall_tids
                 and p["h"] <= per_sample_h_cap
                 and (tid_band_count[p["tid"]] >= args.min_tid_count
                      or p["tid"] == prev_tid)]
        if not cands:
            reasons["no_cands"] += 1
            continue
        picked = None
        dt = (s["t"] - prev_t) if prev_t is not None else 1e9

        # 1) LOCKED: same tid, plausible jump.
        if prev_tid is not None:
            same = [p for p in cands if p["tid"] == prev_tid]
            if same:
                cand = same[0]
                jump = abs(cand["x"] - prev_x) if prev_x is not None else 0
                if jump <= args.max_jump_px + 1500 * dt:
                    picked = cand
                    reasons["locked"] += 1

        # 2) ADOPT: lock lost recently; a new tid appears near the last
        #    position (ByteTrack id switch on the same person).
        if (picked is None and prev_x is not None
                and dt <= args.tid_memory_s):
            near = [p for p in cands
                    if abs(p["x"] - prev_x) <= args.adopt_radius_px]
            if near:
                picked = min(near, key=lambda p: abs(p["x"] - prev_x))
                reasons["adopted"] += 1

        # 3) COLD START: pick the tid with the most in-band presence.
        if picked is None and (prev_tid is None or dt > args.tid_memory_s):
            picked = max(cands, key=lambda p: tid_band_count[p["tid"]])
            reasons["cold_start"] += 1

        if picked is None:
            continue
        sample_cx[i] = picked["x"]
        sample_cy[i] = picked["y"]
        sample_h[i] = picked["h"]
        prev_tid = picked["tid"]
        prev_x = picked["x"]
        prev_t = s["t"]

    valid = ~np.isnan(sample_cx)
    print(f"Opponent picks: {valid.mean()*100:.1f}%  reasons: {dict(reasons)}")
    if valid.sum() < 10:
        raise SystemExit("Too few opponent picks; aborting PiP solve.")

    # ---- PiP crop size ----
    med_h = float(np.nanmedian(sample_h))
    pa_w, pa_h = parse_aspect(args.pip_aspect)
    if args.pip_src_h is not None:
        pip_h = args.pip_src_h
    else:
        pip_h = int(round(np.clip(2.5 * med_h, 110, H_SRC / 3)))
    if pip_h % 2:
        pip_h += 1
    pip_w = int(round(pip_h * pa_w / pa_h))
    if pip_w % 2:
        pip_w += 1
    print(f"Opponent median h: {med_h:.0f}px -> PiP source crop "
          f"{pip_w}x{pip_h}")

    # ---- Frame grid + forward-filled targets ----
    sample_t = np.array([s["t"] for s in samples])
    N = int(round(sample_t[-1] * fps)) + 1
    frame_t = np.arange(N) / fps

    def ffill(vals):
        v = vals.copy()
        last = None
        for i in range(len(v)):
            if np.isnan(v[i]):
                v[i] = last if last is not None else np.nan
            else:
                last = v[i]
        first = next((i for i, ok in enumerate(~np.isnan(v)) if ok), None)
        if first is not None:
            v[:first] = v[first]
        return v

    scx = ffill(sample_cx)
    scy = ffill(sample_cy) - args.pip_up_bias * pip_h
    subj_x = np.interp(frame_t, sample_t, scx)
    subj_y = np.interp(frame_t, sample_t, scy)

    det_times = sample_t[valid]
    visible = np.zeros(N, dtype=bool)
    for i, ft in enumerate(frame_t):
        k = np.searchsorted(det_times, ft)
        nearest = float("inf")
        if k < len(det_times):
            nearest = min(nearest, det_times[k] - ft)
        if k > 0:
            nearest = min(nearest, ft - det_times[k - 1])
        if nearest <= 0.15:
            visible[i] = True
    print(f"Opponent visible frames: {visible.mean()*100:.1f}%")

    # ---- Visibility-ramped box constraints, then QP per axis ----
    RAMP = int(round(0.4 * fps))
    dist = np.full(N, RAMP, dtype=int)
    last_inv = -10**9
    for i in range(N):
        if not visible[i]:
            last_inv = i
        else:
            dist[i] = min(dist[i], i - last_inv)
    last_inv = 10**9
    for i in range(N - 1, -1, -1):
        if not visible[i]:
            last_inv = i
        else:
            dist[i] = min(dist[i], last_inv - i)

    fit_mask = np.where(visible, 1.0, float(args.invisible_fit_weight))

    def bounds_for(subj, crop_sz, src_sz, margin):
        max_pos = src_sz - crop_sz
        lb = np.zeros(N)
        ub = np.full(N, float(max_pos))
        for i in range(N):
            if visible[i]:
                frac = min(1.0, dist[i] / RAMP) if RAMP > 0 else 1.0
                m = margin * frac
                lo = max(0.0, subj[i] - (crop_sz - m))
                hi = min(float(max_pos), subj[i] - m)
                if lo > hi:
                    center = np.clip(subj[i] - crop_sz / 2, 0, max_pos)
                    lb[i] = max(0.0, center - 4)
                    ub[i] = min(float(max_pos), center + 4)
                else:
                    lb[i] = lo
                    ub[i] = hi
        return lb, ub

    mx = args.safety_margin_frac * pip_w
    my = args.safety_margin_frac * pip_h
    lbx, ubx = bounds_for(subj_x, pip_w, W_SRC, mx)
    lby, uby = bounds_for(subj_y, pip_h, H_SRC, my)
    fit_x = np.clip(subj_x - pip_w / 2, 0, W_SRC - pip_w)
    fit_y = np.clip(subj_y - pip_h / 2, 0, H_SRC - pip_h)

    crop_x = np.clip(
        solve_axis(N, fit_x, fit_mask, lbx, ubx,
                   args.w_vel, args.w_acc, args.w_fit), 0, W_SRC - pip_w)
    crop_y = np.clip(
        solve_axis(N, fit_y, fit_mask, lby, uby,
                   args.w_vel, args.w_acc, args.w_fit), 0, H_SRC - pip_h)

    # ---- Emit sendcmd targeting crop@pip (same half-frame-early timing
    #      trick as the main path; see SKILL.md sendcmd gotcha) ----
    half_dt = 0.5 / fps
    by_t = {}
    prev_xv = prev_yv = None
    for i in range(N):
        xv = int(round(crop_x[i]))
        yv = int(round(crop_y[i]))
        t_cmd = max(0.0, frame_t[i] - half_dt)
        cmds = []
        if xv != prev_xv:
            cmds.append(f"crop@pip x {xv}")
            prev_xv = xv
        if yv != prev_yv:
            cmds.append(f"crop@pip y {yv}")
            prev_yv = yv
        if cmds:
            by_t[t_cmd] = cmds
    lines = [f"{t:.7f} " + ", ".join(cmds) + ";"
             for t, cmds in sorted(by_t.items())]
    Path(args.out_cmds).write_text("\n".join(lines) + "\n")

    meta = {
        "fps": fps,
        "src_w": W_SRC, "src_h": H_SRC,
        "pip_w": pip_w, "pip_h": pip_h,
        "pip_aspect": args.pip_aspect,
    }
    Path(args.out_cmds + ".meta.json").write_text(json.dumps(meta))

    velx = np.abs(np.diff(crop_x))
    print(f"Wrote {len(lines)} command lines to {args.out_cmds}")
    print(f"pip crop_x range {crop_x.min():.0f}-{crop_x.max():.0f}  "
          f"crop_y range {crop_y.min():.0f}-{crop_y.max():.0f}")
    print(f"pip x px/frame: mean={velx.mean():.2f} max={velx.max():.2f}")


if __name__ == "__main__":
    main()
