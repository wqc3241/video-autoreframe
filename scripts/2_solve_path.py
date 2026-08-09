#!/usr/bin/env python3
"""Solve smooth camera path via QP and emit an ffmpeg sendcmd file.

Reads detections JSON from 1_detect.py, selects the target subject per frame
(temporal-coherent + static-fixture filter), and solves a constrained QP to
find a smooth camera-x trajectory. Emits a sendcmd file usable by 3_encode.py.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
import osqp


def parse_aspect(s):
    w, h = s.split(":")
    return int(w), int(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True)
    ap.add_argument("--out-cmds", required=True)
    ap.add_argument("--target-aspect", default="9:16",
                    help='Output aspect ratio, e.g. "9:16" or "4:5"')
    ap.add_argument("--subject-position",
                    choices=["near", "far", "any"], default="near",
                    help="Which person to track: near (closest to camera), "
                    "far (farthest), or any (largest bbox)")
    ap.add_argument("--y2-min", type=float, default=600,
                    help="Min bottom-of-bbox y2 for 'near' subject")
    ap.add_argument("--safety-margin", type=int, default=90,
                    help="Subject must be >= this many px from crop edges")
    ap.add_argument("--w-vel", type=float, default=5.0)
    ap.add_argument("--w-acc", type=float, default=1500.0)
    ap.add_argument("--w-fit", type=float, default=1.0)
    ap.add_argument("--invisible-fit-weight", type=float, default=0.15,
                    help="Fit-mask value applied to invisible frames "
                    "(forward-filled to last-known subject_x). Prevents the "
                    "camera from drifting toward the next-visible position "
                    "during long gaps with no subject. 0 = full freedom "
                    "(old behavior), 1 = same pull as visible.")
    ap.add_argument("--reacquire-window-s", type=float, default=0.8,
                    help="If locked tid disappears, allow re-acquire by "
                    "size similarity for this many seconds before cold-start")
    ap.add_argument("--tid-memory-s", type=float, default=5.0,
                    help="How long to remember prev_tid for the locked-path "
                    "after losing detections. If the same tid reappears "
                    "within this window (even at low h, e.g. at the far "
                    "baseline), re-engage immediately. Should be >> the "
                    "reacquire-window-s.")
    ap.add_argument("--reacquire-min-h-ratio", type=float, default=0.6,
                    help="Re-acquire candidate must have h >= this * prev_h")
    ap.add_argument("--reacquire-min-area", type=float, default=3500,
                    help="Re-acquire candidate must also have bbox area "
                    "(h*w) >= this. Filters narrow distant bystanders that "
                    "pass the h-ratio check when the locked subject was at "
                    "the far baseline (where prev_h is also small).")
    ap.add_argument("--cold-start-min-h", type=float, default=170,
                    help="Cold-start (no recent lock) requires a candidate "
                    "with at least this bbox height. Filters distant walkers.")
    args = ap.parse_args()

    data = json.loads(Path(args.detections).read_text())
    fps = data["fps"]
    W_SRC = data["W"]
    H_SRC = data["H"]
    samples = data["samples"]

    # Compute target crop dimensions from source + target aspect.
    ta_w, ta_h = parse_aspect(args.target_aspect)
    # Keep source height; width = height * (target_w / target_h).
    crop_h = H_SRC
    crop_w = int(round(H_SRC * ta_w / ta_h))
    if crop_w % 2:
        crop_w += 1
    max_x = W_SRC - crop_w
    print(f"Source: {W_SRC}x{H_SRC}  crop: {crop_w}x{crop_h}  max_x: {max_x}")

    sample_t = np.array([s["t"] for s in samples])
    total_duration = sample_t[-1]
    N = int(round(total_duration * fps)) + 1
    frame_t = np.arange(N) / fps

    # ---- Static-fixture filter (reject motionless bystanders) ----
    BIN = 30
    FIXTURE_MIN = 50      # min occurrences across video to count as fixture
    FAR_Y2_UPPER = 720    # above this -> always subject, skip filter
    bins = Counter()
    for s in samples:
        for p in s["people"]:
            if p["y2"] < args.y2_min:
                continue
            xb = int(p["x"] // BIN) * BIN
            yb = int(p["y2"] // BIN) * BIN
            bins[(xb, yb)] += 1
    fixtures = {k for k, v in bins.items() if v >= FIXTURE_MIN}
    fixture_cells = set()
    for xb, yb in fixtures:
        if yb < FAR_Y2_UPPER:
            for dx in (-BIN, 0, BIN):
                for dy in (-BIN, 0, BIN):
                    fixture_cells.add((xb + dx, yb + dy))

    def is_fixture(p):
        if p["y2"] >= FAR_Y2_UPPER:
            return False
        xb = int(p["x"] // BIN) * BIN
        yb = int(p["y2"] // BIN) * BIN
        return (xb, yb) in fixture_cells

    # ---- Tid-locked subject picker ----
    # State machine:
    #   LOCKED (have prev_tid): keep picking that tid as it appears
    #   BRIEF LOSS (tid gone, <reacquire-window-s since seen): allow switch
    #     to a candidate of similar size (filters out small walkers)
    #   COLD START (no recent lock): require h >= cold-start-min-h to pick.
    #     This prevents latching onto distant walking bystanders during
    #     between-rally moments when the main subject is briefly off-screen.
    sample_x = np.full(len(samples), np.nan)
    prev_tid = None
    prev_x = None
    prev_y2 = None
    prev_h = None
    prev_t = None
    MAX_JUMP_PX = 150
    reasons = Counter()  # diagnostics

    if args.subject_position == "near":
        depth_key = lambda p: p["y2"]
    elif args.subject_position == "far":
        depth_key = lambda p: -p["y2"]
    else:
        depth_key = lambda p: p["h"] * p["w"]

    for i, s in enumerate(samples):
        cands = [p for p in s["people"]
                 if p["y2"] >= args.y2_min and not is_fixture(p)]
        if not cands:
            reasons["no_cands"] += 1
            continue

        picked = None
        dt = (s["t"] - prev_t) if prev_t is not None else 1e9

        # 1) LOCKED: try to pick the same tid as last frame.
        if prev_tid is not None:
            same = [p for p in cands if p["tid"] == prev_tid]
            if same:
                cand = same[0]
                jump = abs(cand["x"] - prev_x) if prev_x is not None else 0
                if jump <= MAX_JUMP_PX + 1500 * dt:
                    picked = cand
                    reasons["locked"] += 1

        # 2) BRIEF LOSS: allow re-acquire to a similar-sized candidate
        #    only if we lost the lock recently. Filters out walkers whose
        #    bbox is dramatically smaller than the real subject.
        #    The area floor catches narrow far-edge detections where the
        #    h-ratio alone is too lenient (e.g. when prev_h is also small
        #    because the subject was at the far baseline).
        if (picked is None and prev_tid is not None and prev_h is not None
                and dt <= args.reacquire_window_s):
            min_h = args.reacquire_min_h_ratio * prev_h
            sim = [p for p in cands
                   if p["h"] >= min_h
                   and p["h"] * p["w"] >= args.reacquire_min_area]
            if sim:
                # Among similar-sized candidates, prefer closest in x to
                # the last position, with a mild bonus for higher y2.
                max_y2 = max(depth_key(p) for p in sim)
                picked = max(sim,
                             key=lambda p: -abs(p["x"] - prev_x)
                                           - 0.5 * (max_y2 - depth_key(p)))
                jump = abs(picked["x"] - prev_x)
                if jump > MAX_JUMP_PX + 1500 * dt:
                    picked = None
                    reasons["reacquire_jumpy"] += 1
                else:
                    reasons["reacquired"] += 1
            else:
                reasons["reacquire_no_similar"] += 1

        # 3) COLD START: no lock or long-since-seen. Require minimum size
        #    so we don't latch onto a distant walking bystander.
        if picked is None and (prev_tid is None
                               or dt > args.reacquire_window_s):
            big = [p for p in cands if p["h"] >= args.cold_start_min_h]
            if big:
                # Pick the closest-to-camera candidate (largest depth_key).
                # Among ties prefer largest bbox area.
                picked = max(big, key=lambda p: (depth_key(p),
                                                 p["h"] * p["w"]))
                reasons["cold_start"] += 1
            else:
                reasons["cold_no_big"] += 1

        if picked is None:
            # Drop the lock only after a long absence. We keep prev_tid for
            # tid-memory-s seconds so that if the same tid reappears (even
            # at low h, e.g. main player reaching the far baseline), the
            # locked path re-engages instead of forcing cold-start.
            if prev_t is not None and dt > args.tid_memory_s:
                prev_tid = None
            continue

        sample_x[i] = picked["x"]
        prev_tid = picked.get("tid", prev_tid)
        prev_x = picked["x"]
        prev_y2 = picked["y2"]
        prev_h = picked["h"]
        prev_t = s["t"]

    valid = ~np.isnan(sample_x)
    print(f"Valid subject picks: {valid.mean()*100:.1f}%")
    print(f"Pick reasons: {dict(reasons)}")

    # ---- Per-frame subject estimate (for fit term + box bounds) ----
    sx = sample_x.copy()
    last = None
    for i in range(len(sx)):
        if np.isnan(sx[i]):
            sx[i] = last if last is not None else W_SRC / 2
        else:
            last = sx[i]
    first_valid = next((i for i, v in enumerate(valid) if v), None)
    if first_valid is not None:
        sx[:first_valid] = sample_x[first_valid]
    subject_x = np.interp(frame_t, sample_t, sx)

    # Per-frame visibility mask (real detection within 0.15s).
    det_times = sample_t[valid]
    visible = np.zeros(N, dtype=bool)
    if len(det_times):
        for i, ft in enumerate(frame_t):
            k = np.searchsorted(det_times, ft)
            nearest = float("inf")
            if k < len(det_times):
                nearest = min(nearest, det_times[k] - ft)
            if k > 0:
                nearest = min(nearest, ft - det_times[k - 1])
            if nearest <= 0.15:
                visible[i] = True
    print(f"Visible frames: {visible.sum()} ({visible.mean()*100:.1f}%)")

    # ---- Box constraints with ramp-in at visibility edges ----
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

    lb = np.zeros(N)
    ub = np.full(N, max_x, dtype=float)
    for i in range(N):
        if visible[i]:
            frac = min(1.0, dist[i] / RAMP) if RAMP > 0 else 1.0
            m = args.safety_margin * frac
            s = subject_x[i]
            lo = max(0.0, s - (crop_w - m))
            hi = min(float(max_x), s - m)
            if lo > hi:
                center = np.clip(s - crop_w / 2, 0, max_x)
                lb[i] = max(0.0, center - 4)
                ub[i] = min(float(max_x), center + 4)
            else:
                lb[i] = lo
                ub[i] = hi

    fit_target = np.clip(subject_x - crop_w / 2, 0, max_x)
    # Strong fit pull during visible frames; weak pull during invisible
    # toward the forward-filled last-known subject_x. Without the weak pull,
    # long invisible gaps let the camera drift in a U-shape between the
    # surrounding visible boundaries (minimizing acceleration produces a
    # detour that may center an irrelevant bystander mid-gap).
    fit_mask = np.where(visible, 1.0, float(args.invisible_fit_weight))

    # ---- Build QP matrices ----
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

    P = 2 * (args.w_vel * (D1.T @ D1)
             + args.w_acc * (D2.T @ D2)
             + args.w_fit * F)
    q = -2 * args.w_fit * (fit_mask * fit_target)
    A = sparse.eye(N, format="csr")

    print(f"QP: {N} vars. Solving...")
    solver = osqp.OSQP()
    solver.setup(P=P.tocsc(), q=q.astype(np.float64),
                 A=A.tocsc(),
                 l=lb.astype(np.float64), u=ub.astype(np.float64),
                 verbose=False, eps_abs=1e-4, eps_rel=1e-4,
                 max_iter=20000, polish=True, polish_refine_iter=5)
    res = solver.solve()
    print(f"Status: {res.info.status}  iters: {res.info.iter}")
    if res.info.status_val not in (1, 2):
        raise SystemExit("QP failed")

    crop_x = np.clip(res.x, 0, max_x)

    # ---- Emit sendcmd with PRECISE timing ----
    # CRITICAL: cmd time must be STRICTLY LESS than frame PTS to avoid the
    # 60fps float-precision bug that causes every 3rd frame to stick.
    # Place each cmd halfway between its frame and the previous frame.
    lines = []
    prev = -1
    half_dt = 0.5 / fps
    for i in range(N):
        xv = int(round(crop_x[i]))
        if xv == prev:
            continue
        t_cmd = max(0.0, frame_t[i] - half_dt)
        lines.append(f"{t_cmd:.7f} crop x {xv};")
        prev = xv
    Path(args.out_cmds).write_text("\n".join(lines) + "\n")

    # Also write a tiny metadata sidecar for the encoder.
    meta = {
        "fps": fps,
        "src_w": W_SRC, "src_h": H_SRC,
        "crop_w": crop_w, "crop_h": crop_h,
        "target_aspect": args.target_aspect,
    }
    Path(args.out_cmds + ".meta.json").write_text(json.dumps(meta))

    vel = np.abs(np.diff(crop_x))
    print(f"Wrote {len(lines)} commands to {args.out_cmds}")
    print(f"crop_x range {crop_x.min():.0f}-{crop_x.max():.0f}")
    print(f"px/frame velocity: mean={vel.mean():.2f}  "
          f"p95={np.percentile(vel, 95):.2f}  max={vel.max():.2f}")


if __name__ == "__main__":
    main()
