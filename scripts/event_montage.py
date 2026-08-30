#!/usr/bin/env python3
"""Tile video frames at given timestamps into one labeled montage.
Usage: event_montage.py video out.jpg t1 t2 t3 ...
Each cell 480x270 with timestamp label; 4 columns.
"""
import subprocess
import sys

import cv2
import numpy as np

video, out = sys.argv[1], sys.argv[2]
times = [float(t) for t in sys.argv[3:]]
cells = []
for t in times:
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(t), "-i", video, "-frames:v", "1",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True,
    )
    arr = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        arr = np.zeros((1080, 1920, 3), np.uint8)
    cell = cv2.resize(arr, (480, 270))
    cv2.putText(cell, f"{t:.2f}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 0), 4)
    cv2.putText(cell, f"{t:.2f}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 255), 2)
    cells.append(cell)
cols = 4
rows = (len(cells) + cols - 1) // cols
while len(cells) < rows * cols:
    cells.append(np.zeros((270, 480, 3), np.uint8))
grid = np.vstack([np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(rows)])
cv2.imwrite(out, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(out, grid.shape)
