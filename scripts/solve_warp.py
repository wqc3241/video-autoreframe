#!/usr/bin/env python3
"""Solve a straightening warp for fixed-camera wide-angle footage.

Model: pure camera re-rotation H = K R(yaw, roll) K^-1 with the principal
point at frame center and an assumed focal length. Roll levels the world;
a partial yaw removes the transverse-line slope disparity caused by a
camera standing off the court centerline while aiming back at it. Because
the model is a rotation (not an arbitrary homography), out-of-plane
content (players, buildings) stays physically consistent — no fake shear.

Input: a JSON file of reference lines measured from gridded zooms
(scripts/zoom_grid.py). Every line listed should be horizontal in the
real world; weight = how visible it is in the final crop.

  {"width": 1920, "height": 1080, "focal_px": 720,
   "lines": [
     {"name": "baseline",  "a": [258, 878], "b": [1408, 815], "weight": 3},
     {"name": "net_band",  "a": [545, 638], "b": [1010, 632], "weight": 2},
     {"name": "horizon",   "a": [1300, 613], "b": [1920, 603], "weight": 2},
     {"name": "service_r", "a": [846, 731], "b": [1148, 727], "weight": 1}
   ]}

focal_px ~720 fits iPhone ultrawide (0.5x) video at 1920 wide (~106 deg
hfov); small errors barely matter at small angles.

Outputs <out>.json with solved angles, residuals, the combined transform
(safe-crop folded in), and a ready ffmpeg `perspective` filter string.
Extra image arguments are warped into *_warp.jpg previews — always eyeball
one before batch-encoding.

Usage:
  solve_warp.py --lines lines.json --out warp.json [frame1.jpg ...]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares


def rot(yaw_deg, roll_deg):
    y, r = np.radians(yaw_deg), np.radians(roll_deg)
    Ry = np.array(
        [[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]
    )
    Rz = np.array(
        [[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]]
    )
    return Ry @ Rz


def apply_h(Hm, pts):
    pts = np.asarray(pts, float).reshape(-1, 2)
    q = np.hstack([pts, np.ones((len(pts), 1))]) @ Hm.T
    return q[:, :2] / q[:, 2:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("previews", nargs="*")
    args = ap.parse_args()

    with open(args.lines) as f:
        cfg = json.load(f)
    W, H_IMG = cfg["width"], cfg["height"]
    F = float(cfg.get("focal_px", 720))
    K = np.array([[F, 0, W / 2], [0, F, H_IMG / 2], [0, 0, 1]])
    K_INV = np.linalg.inv(K)
    LINES = [(l["name"], l["a"], l["b"], float(l.get("weight", 1)))
             for l in cfg["lines"]]

    def homog(yaw, roll):
        return K @ rot(yaw, roll) @ K_INV

    def line_angles(Hm):
        out = {}
        for name, pa, pb, w in LINES:
            qa, qb = apply_h(Hm, [pa, pb])
            out[name] = (np.degrees(np.arctan2(qb[1] - qa[1], qb[0] - qa[0])), w)
        return out

    def residuals(p):
        return [np.sqrt(w) * a for a, w in line_angles(homog(*p)).values()]

    fit = least_squares(residuals, x0=[0.0, 0.0], method="lm")
    yaw, roll = fit.x
    Hm = homog(yaw, roll)
    print(f"solved: yaw={yaw:+.3f} deg  roll={roll:+.3f} deg")
    before, after = line_angles(np.eye(3)), line_angles(Hm)
    for name in before:
        print(f"  {name:12s} {before[name][0]:+6.2f} -> {after[name][0]:+6.2f}")

    # largest centered same-aspect rect inside the warped frame quad
    corners = apply_h(Hm, [(0, 0), (W, 0), (W, H_IMG), (0, H_IMG)])
    cx, cy = corners.mean(axis=0)
    poly = corners.astype(np.float32)

    def inside(s):
        w2, h2 = W / 2 * s, H_IMG / 2 * s
        return all(
            cv2.pointPolygonTest(poly, p, False) >= 0
            for p in [(cx - w2, cy - h2), (cx + w2, cy - h2),
                      (cx + w2, cy + h2), (cx - w2, cy + h2)]
        )

    lo, hi = 0.5, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if inside(mid) else (lo, mid)
    s = lo * 0.998
    w2, h2 = W / 2 * s, H_IMG / 2 * s
    x0, y0, x1, y1 = cx - w2, cy - h2, cx + w2, cy + h2
    print(f"safe crop keeps {s*100:.1f}% linear")

    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    S = np.diag([W / (x1 - x0), H_IMG / (y1 - y0), 1.0])
    H_total = S @ T @ Hm
    quad = apply_h(np.linalg.inv(H_total), [(0, 0), (W, 0), (0, H_IMG), (W, H_IMG)])
    ff = {f"{k}{i}": round(float(quad[i][j]), 3)
          for i in range(4) for j, k in ((0, "x"), (1, "y"))}
    filt = ("perspective=" + ":".join(f"{k}={v}" for k, v in ff.items())
            + ":interpolation=cubic:sense=source")
    print("ffmpeg filter:\n " + filt)

    with open(args.out, "w") as f:
        json.dump({
            "focal_px": F, "yaw_deg": round(float(yaw), 4),
            "roll_deg": round(float(roll), 4),
            "angles_before": {k: round(v[0], 3) for k, v in before.items()},
            "angles_after": {k: round(v[0], 3) for k, v in after.items()},
            "keep_ratio": round(s, 4), "H_total": H_total.tolist(),
            "ffmpeg_filter": filt,
        }, f, indent=2)
    print(f"wrote {args.out}")

    for path in args.previews:
        img = cv2.imread(path)
        warped = cv2.warpPerspective(img, H_total, (W, H_IMG),
                                     flags=cv2.INTER_CUBIC)
        out = os.path.splitext(path)[0] + "_warp.jpg"
        cv2.imwrite(out, warped, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("preview:", out)


if __name__ == "__main__":
    sys.exit(main())
