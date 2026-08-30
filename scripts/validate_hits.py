#!/usr/bin/env python3
"""Audio-visual cross-validation of detected hits (neighbour-court filter).

Adjacent courts produce hit sounds too. For every audio event, run
yolov8m-pose in a +/-0.45s window and check whether OUR near player or
the far opponent shows a stroke (normalized wrist-speed peak). A sound
with BOTH players assessable and BOTH idle is foreign (or a bounce) and
can be rejected; anything less than that is KEPT — on off-center wide
cameras the near player provably returns balls from outside the FOV, so
lack of visual proof must never delete evidence.

Two pose passes per sampled frame:
  near: full frame @ imgsz 960; slot = box with y2 >= --near-y2 and
        area >= --near-area (the near player dwarfs everyone else)
  far:  --far-crop x0 y0 x1 y1 upscaled --far-up x @ conf 0.1. A far
        opponent against a dark windscreen is ~50-80 px and needs the
        2.5-3x upscale to detect at all. Boxes touching the crop bottom
        are the NEAR player's clipped upper body — skip them; also
        x-gate out bench spectators (--far-x-min, original px).

Wrist speed = wrist displacement / dt / person height [heights/s] at
~15 Hz. Consecutive observations whose box centers jump > 0.8 heights
are identity switches, not motion (they fake 90-120 h/s peaks) — skip;
speeds > 15 are residual artifacts — discard.

Pose observations are cached to <outdir>/pose_obs.json so thresholds can
be re-scored without re-inference (delete the cache to re-infer).

Usage:
  validate_hits.py --src match.mov --hits work/hits.json --outdir work \
      --far-crop 430 500 1180 700 --far-x-min 500
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

WIN = 0.45
STRIDE = 4
NEAR_KP_CONF = 0.30
FAR_KP_CONF = 0.20


def snap_fps(fps):
    for std in (23.976023976, 24.0, 25.0, 29.97002997, 30.0, 50.0,
                59.94005994, 60.0, 119.88011988, 120.0):
        if abs(fps - std) / std < 0.002:
            return std
    return fps


def wrist_speed(track, kp_conf_min, fps):
    best = 0.0
    for a, b in zip(track, track[1:]):
        dt = b[0] - a[0]
        if dt <= 0 or dt > STRIDE / fps * 2.5:
            continue
        h = (a[1] + b[1]) / 2
        if abs(b[4] - a[4]) > 0.8 * h:
            continue
        for w in range(2):
            if a[3][w] < kp_conf_min or b[3][w] < kp_conf_min:
                continue
            d = np.hypot(b[2][w][0] - a[2][w][0], b[2][w][1] - a[2][w][1])
            v = d / dt / h
            if v < 15.0:
                best = max(best, v)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--hits", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--near-y2", type=float, default=750)
    ap.add_argument("--near-area", type=float, default=25000)
    ap.add_argument("--far-crop", nargs=4, type=int, required=True,
                    help="x0 y0 x1 y1 of the far-player band, original px")
    ap.add_argument("--far-up", type=float, default=2.5)
    ap.add_argument("--far-x-min", type=float, default=0,
                    help="reject far boxes left of this (bench spectators)")
    ap.add_argument("--far-y2-max", type=float, default=1e9,
                    help="reject far boxes whose feet are below this")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.hits) as f:
        hits = json.load(f)

    cap = cv2.VideoCapture(args.src)
    fps = snap_fps(cap.get(cv2.CAP_PROP_FPS))

    need = set()
    for h in hits:
        f0, f1 = int((h["t"] - WIN) * fps), int((h["t"] + WIN) * fps)
        need.update(fi for fi in range(f0, f1 + 1) if fi % STRIDE == 0)

    obs_path = os.path.join(args.outdir, "pose_obs.json")
    if os.path.exists(obs_path):
        with open(obs_path) as f:
            c = json.load(f)
        near_obs = {int(k): v for k, v in c["near"].items()}
        far_obs = {int(k): v for k, v in c["far"].items()}
        print("loaded cached pose obs — delete pose_obs.json to re-infer")
    else:
        model = YOLO("yolov8m-pose.pt")
        fx0, fy0, fx1, fy1 = args.far_crop
        fw, fh = int((fx1 - fx0) * args.far_up), int((fy1 - fy0) * args.far_up)
        near_obs, far_obs = {}, {}
        idx = done = 0
        t0 = time.time()
        while True:
            if not cap.grab():
                break
            if idx in need:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                t = idx / fps
                r = model.predict(frame, imgsz=960, conf=0.25, verbose=False,
                                  device=args.device)[0]
                best = None
                for b, kp in zip(r.boxes, r.keypoints):
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    area = (x2 - x1) * (y2 - y1)
                    if y2 >= args.near_y2 and area >= args.near_area:
                        if best is None or area > best[0]:
                            wxy = kp.xy[0][[9, 10]].tolist()
                            wc = (kp.conf[0][[9, 10]].tolist()
                                  if kp.conf is not None else [0, 0])
                            best = (area, [t, y2 - y1, wxy, wc, (x1 + x2) / 2])
                if best:
                    near_obs[idx] = best[1]

                up = cv2.resize(frame[fy0:fy1, fx0:fx1], (fw, fh),
                                interpolation=cv2.INTER_LANCZOS4)
                r = model.predict(up, imgsz=fw, conf=0.10, verbose=False,
                                  device=args.device)[0]
                best = None
                for b, kp in zip(r.boxes, r.keypoints):
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    if y2 >= fh - 4:  # near player's clipped upper body
                        continue
                    if fy0 + y2 / args.far_up > args.far_y2_max:
                        continue
                    if fx0 + x2 / args.far_up < args.far_x_min:
                        continue
                    conf = float(b.conf[0])
                    if best is None or conf > best[0]:
                        wxy = kp.xy[0][[9, 10]].tolist()
                        wc = (kp.conf[0][[9, 10]].tolist()
                              if kp.conf is not None else [0, 0])
                        best = (conf, [t, y2 - y1, wxy, wc, (x1 + x2) / 2])
                if best:
                    far_obs[idx] = best[1]
                done += 1
                if done % 300 == 0:
                    print(f"  {done}/{len(need)}  "
                          f"{done/(time.time()-t0):.1f} inf-fps", flush=True)
            idx += 1
        with open(obs_path, "w") as f:
            json.dump({"near": near_obs, "far": far_obs}, f)
    cap.release()

    results = []
    for h in hits:
        f0, f1 = int((h["t"] - WIN) * fps), int((h["t"] + WIN) * fps)
        nt = [near_obs[i] for i in range(f0, f1 + 1) if i in near_obs]
        ft = [far_obs[i] for i in range(f0, f1 + 1) if i in far_obs]
        results.append({
            "t": h["t"], "snr": h["snr"],
            "near_speed": round(wrist_speed(nt, NEAR_KP_CONF, fps), 2),
            "far_speed": round(wrist_speed(ft, FAR_KP_CONF, fps), 2),
            "near_assessable": len(nt) >= 5, "far_assessable": len(ft) >= 5,
        })
    out = os.path.join(args.outdir, "hits_validated.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"{'t':>8} {'snr':>5} {'nearV':>6} {'farV':>6}")
    for r in results:
        print(f"{r['t']:8.2f} {r['snr']:5.1f} {r['near_speed']:6.2f} "
              f"{r['far_speed']:6.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
