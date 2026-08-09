#!/usr/bin/env python3
"""Encode the final cropped + scaled vertical video.

Uses ffmpeg with:
  - sendcmd file driving per-frame crop_x
  - lanczos scale to target
  - libx264 CRF 12 by default (near-visually-lossless)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_aspect(s):
    w, h = s.split(":")
    return int(w), int(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--cmds", required=True, help="sendcmd file from 2_solve_path.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-aspect", default="9:16")
    ap.add_argument("--scale-height", type=int, default=1920,
                    help="Final output height in pixels (width computed from aspect)")
    ap.add_argument("--crf", type=int, default=12,
                    help="H.264 CRF (12 = near-lossless, 18 = visually lossless, 23 = default lossy)")
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--no-upscale", action="store_true",
                    help="Skip the final scale (output at native crop resolution)")
    args = ap.parse_args()

    meta_path = Path(args.cmds + ".meta.json")
    if not meta_path.exists():
        sys.exit(f"missing {meta_path} — did you run 2_solve_path.py?")
    meta = json.loads(meta_path.read_text())
    crop_w = meta["crop_w"]
    crop_h = meta["crop_h"]

    ta_w, ta_h = parse_aspect(args.target_aspect)
    out_h = args.scale_height
    out_w = int(round(out_h * ta_w / ta_h))
    if out_w % 2:
        out_w += 1

    vf = f"sendcmd=f={args.cmds},crop={crop_w}:{crop_h}:0:0"
    if not args.no_upscale:
        vf += f",scale={out_w}:{out_h}:flags=lanczos"

    cmd = [
        "ffmpeg", "-y",
        "-i", args.src,
        "-filter_complex", vf,
        "-c:v", "libx264",
        "-preset", args.preset,
        "-crf", str(args.crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        args.out,
    ]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)
    size_mb = Path(args.out).stat().st_size / (1024 * 1024)
    print(f"Done: {args.out} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
