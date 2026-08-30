#!/usr/bin/env python3
"""Measure court-line angles in a frame to quantify camera tilt.

Detects white court lines (HSV threshold + HoughLinesP), clusters the
segments into physical lines, and prints each cluster's fitted angle and
extent. Run on a frame BEFORE and AFTER warping to verify the correction.

Usage: 01_measure_tilt.py frame.jpg [frame2.jpg ...]
"""
import sys

import cv2
import numpy as np


def line_clusters(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    # White paint: bright, low saturation. Only look at the court area
    # (lower part of frame) so sky/buildings don't pollute.
    mask = ((hsv[:, :, 2] > 170) & (hsv[:, :, 1] < 60)).astype(np.uint8) * 255
    mask[: int(h * 0.52), :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    segs = cv2.HoughLinesP(
        mask, 1, np.pi / 720, threshold=60, minLineLength=90, maxLineGap=12
    )
    if segs is None:
        return []
    segs = segs.reshape(-1, 4).astype(float)

    feats = []
    for x1, y1, x2, y2 in segs:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if ang > 90:
            ang -= 180
        if ang < -90:
            ang += 180
        length = np.hypot(x2 - x1, y2 - y1)
        feats.append((ang, (x1 + x2) / 2, (y1 + y2) / 2, length, x1, y1, x2, y2))

    # Greedy cluster: same orientation bucket + nearby perpendicular offset.
    used = [False] * len(feats)
    order = sorted(range(len(feats)), key=lambda i: -feats[i][3])
    clusters = []
    for i in order:
        if used[i]:
            continue
        ang_i, mx_i, my_i = feats[i][0], feats[i][1], feats[i][2]
        members = [i]
        used[i] = True
        for j in order:
            if used[j]:
                continue
            ang_j, mx_j, my_j = feats[j][0], feats[j][1], feats[j][2]
            if abs(ang_i - ang_j) > 4:
                continue
            # perpendicular distance of j's midpoint from i's infinite line
            th = np.radians(ang_i)
            d = abs(-np.sin(th) * (mx_j - mx_i) + np.cos(th) * (my_j - my_i))
            if d < 18:
                members.append(j)
                used[j] = True
        pts = []
        for m in members:
            pts.append(feats[m][4:6])
            pts.append(feats[m][6:8])
        pts = np.array(pts)
        vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        ang = np.degrees(np.arctan2(vy, vx))
        if ang > 90:
            ang -= 180
        if ang < -90:
            ang += 180
        t = pts @ np.array([vx, vy]) - (x0 * vx + y0 * vy)
        p_a = (x0 + t.min() * vx, y0 + t.min() * vy)
        p_b = (x0 + t.max() * vx, y0 + t.max() * vy)
        span = t.max() - t.min()
        clusters.append((span, ang, p_a, p_b, len(members)))
    clusters.sort(key=lambda c: -c[0])
    return clusters


def report(path):
    img = cv2.imread(path)
    print(f"\n=== {path}  ({img.shape[1]}x{img.shape[0]}) ===")
    print(f"{'span':>6} {'angle':>8}  {'from':>16} {'to':>16}  segs")
    for span, ang, pa, pb, n in line_clusters(img):
        if span < 130:
            continue
        print(
            f"{span:6.0f} {ang:8.2f}  ({pa[0]:6.1f},{pa[1]:6.1f}) "
            f"({pb[0]:6.1f},{pb[1]:6.1f})  {n}"
        )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
