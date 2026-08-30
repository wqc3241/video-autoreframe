#!/usr/bin/env python3
"""Zoomed crop with a source-coordinate grid for precise point picking.
Usage: zoom_grid.py img x0 y0 w h scale out.png
Grid: minor every 10 src px (thin), major every 50 src px (labeled).
"""
import sys
import cv2

img = cv2.imread(sys.argv[1])
x0, y0, w, h, s = map(int, sys.argv[2:7])
out = sys.argv[7]
crop = img[y0 : y0 + h, x0 : x0 + w]
z = cv2.resize(crop, (w * s, h * s), interpolation=cv2.INTER_LANCZOS4)
for gx in range(x0 - x0 % 10, x0 + w + 1, 10):
    px = (gx - x0) * s
    major = gx % 50 == 0
    col = (0, 0, 255) if major else (0, 200, 255)
    cv2.line(z, (px, 0), (px, h * s), col, 2 if major else 1)
    if major:
        cv2.putText(z, str(gx), (px + 3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
for gy in range(y0 - y0 % 10, y0 + h + 1, 10):
    py = (gy - y0) * s
    major = gy % 50 == 0
    col = (0, 0, 255) if major else (0, 200, 255)
    cv2.line(z, (0, py), (w * s, py), col, 2 if major else 1)
    if major:
        cv2.putText(z, str(gy), (5, py + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imwrite(out, z)
print(out, z.shape)
