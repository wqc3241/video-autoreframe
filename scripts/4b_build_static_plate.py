#!/usr/bin/env python3
"""Stage 4b: build a STATIC clone-shift patch (BGRA png) that covers a
baked-in overlay region for the whole video.

Works because the camera is fixed and the background there is static: one
frame's pixels serve every frame, so the cover is temporally rock-solid
(zero shimmer). Per-frame approaches fail here — ffmpeg delogo's
interpolation streaks get magnified into a smear slab by the vertical
upscale, and cv2.inpaint (Telea) diffuses dark border content (netting,
fixtures) into a dark wedge. Clone-shift from a donor slab in the SAME
frame keeps the right structure (ceiling texture, netting band along the
top edge continue naturally).

Rules learned on the 08-21-2026 reference footage:
- Cover ONE continuous band. Splitting into per-box strips leaves slivers
  of real content between them that no longer connect to anything (a
  floating beam stub).
- Pick the donor slab so it does not contain distinctive one-off features
  (a light fixture, a glare blob directly adjacent) — duplicated generic
  ceiling reads fine, a duplicated fixture does not. Preview and LOOK.
- Feathered alpha (GaussianBlur on the mask) hides the seam; residual
  content discontinuities read as ceiling panel joints once in motion.

Usage:
  4b_build_static_plate.py --src rally_reel.mov --t 34 \
      --cover 1651,0,1920,454 --donor-x 1240 --out corner_plate.png \
      [--feather 41] [--preview corner_preview.jpg]

Apply to the whole reel (scene-space, BEFORE 3_encode for vertical output):
  ffmpeg -i reel.mov -i plate.png -filter_complex "[0:v][1:v]overlay=0:0" \
      -c:v hevc_videotoolbox -b:v 45M -c:a copy reel_clean.mov
"""
import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--t", type=float, default=None,
                    help="timestamp of the frame to clone from")
    ap.add_argument("--cover", required=True,
                    help="x0,y0,x1,y1 region to cover")
    ap.add_argument("--donor-x", type=int, required=True,
                    help="left edge of the donor slab (same rows, same width)")
    ap.add_argument("--out", required=True, help="output BGRA png")
    ap.add_argument("--feather", type=int, default=41,
                    help="odd gaussian kernel for the alpha feather")
    ap.add_argument("--preview", default=None,
                    help="write a composited-corner preview jpg to eyeball")
    args = ap.parse_args()

    x0, y0, x1, y1 = (int(v) for v in args.cover.split(","))
    w = x1 - x0
    cap = cv2.VideoCapture(args.src)
    if args.t is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.t * 1000)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read frame")
    H, W = fr.shape[:2]
    if not (0 <= args.donor_x and args.donor_x + w <= W):
        raise SystemExit(f"donor slab [{args.donor_x},{args.donor_x+w}) outside frame width {W}")
    if args.donor_x < x1 and args.donor_x + w > x0:
        raise SystemExit("donor slab overlaps the cover region — pick another --donor-x")

    plate = fr.copy()
    plate[y0:y1, x0:x1] = fr[y0:y1, args.donor_x:args.donor_x + w]
    mask = np.zeros((H, W), np.uint8)
    mask[y0:y1, x0:x1] = 255
    k = args.feather | 1
    alpha = cv2.GaussianBlur(mask, (k, k), 0)
    cv2.imwrite(args.out, np.dstack([plate, alpha]))
    print(f"wrote {args.out} (cover {w}x{y1-y0} at ({x0},{y0}), donor x={args.donor_x})")

    if args.preview:
        a = alpha.astype(np.float32)[..., None] / 255
        comp = (plate * a + fr * (1 - a)).astype(np.uint8)
        px0 = max(0, x0 - 150)
        py1 = min(H, y1 + 70)
        crop = comp[0:py1, px0:W]
        cv2.imwrite(args.preview, cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2)))
        print(f"preview: {args.preview} — LOOK at it before encoding")


if __name__ == "__main__":
    main()
