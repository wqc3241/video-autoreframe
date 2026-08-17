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


def merge_cmd_files(main_path, pip_path, out_path):
    """Merge the main and PiP sendcmd files into one time-sorted file.

    The main solver emits untargeted `crop x N` commands (fine for the
    single-crop graph). With two crop filters in the graph, an untargeted
    `crop` command would be delivered to BOTH instances, so rewrite main
    commands to the named instance `crop@main`. Commands sharing a
    timestamp are grouped into one interval (`t c1, c2, c3;`) because
    sendcmd requires strictly increasing interval start times.
    """
    by_t = {}

    def add(t, cmd):
        by_t.setdefault(t, []).append(cmd)

    for path, retarget in ((main_path, True), (pip_path, False)):
        for line in Path(path).read_text().splitlines():
            line = line.strip().rstrip(";")
            if not line:
                continue
            t_str, rest = line.split(None, 1)
            for cmd in rest.split(","):
                cmd = cmd.strip()
                if retarget and cmd.startswith("crop "):
                    cmd = "crop@main " + cmd[len("crop "):]
                add(float(t_str), cmd)

    lines = [f"{t:.7f} " + ", ".join(cmds) + ";"
             for t, cmds in sorted(by_t.items())]
    Path(out_path).write_text("\n".join(lines) + "\n")


def build_pip_graph(args, crop_w, crop_h, out_w, out_h):
    pip_meta_path = Path(args.pip_cmds + ".meta.json")
    if not pip_meta_path.exists():
        sys.exit(f"missing {pip_meta_path} — did you run 2b_solve_pip.py?")
    pip_meta = json.loads(pip_meta_path.read_text())
    pip_w = pip_meta["pip_w"]
    pip_h = pip_meta["pip_h"]

    merged = args.pip_cmds + ".merged"
    merge_cmd_files(args.cmds, args.pip_cmds, merged)

    disp_w = args.pip_display_w
    disp_h = int(round(disp_w * pip_h / pip_w))
    if disp_w % 2:
        disp_w += 1
    if disp_h % 2:
        disp_h += 1
    b = args.pip_border

    main_scale = ""
    if not args.no_upscale:
        main_scale = f",scale={out_w}:{out_h}:flags=lanczos"
    return (
        f"sendcmd=f={merged},split=2[m][p];"
        f"[m]crop@main={crop_w}:{crop_h}:0:0{main_scale}[main];"
        f"[p]crop@pip={pip_w}:{pip_h}:0:0,"
        f"scale={disp_w}:{disp_h}:flags=lanczos,"
        f"pad={disp_w + 2*b}:{disp_h + 2*b}:{b}:{b}:color=white[pip];"
        f"[main][pip]overlay={args.pip_x}:{args.pip_y}"
    )


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
    ap.add_argument("--pip-cmds", default=None,
                    help="Optional sendcmd file from 2b_solve_pip.py. When "
                    "given, a picture-in-picture tracking the opponent is "
                    "overlaid on the output.")
    ap.add_argument("--pip-display-w", type=int, default=380,
                    help="PiP display width in output pixels (at the default "
                    "1080x1920 output; height follows the PiP aspect). The "
                    "source crop is small, so keep the upscale under ~1.8x")
    ap.add_argument("--pip-x", type=int, default=28,
                    help="PiP left edge in output px")
    ap.add_argument("--pip-y", type=int, default=120,
                    help="PiP top edge in output px (default clears the "
                    "platform UI zone at the very top of vertical players)")
    ap.add_argument("--pip-border", type=int, default=4,
                    help="White border thickness around the PiP, px")
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

    if args.pip_cmds:
        vf = build_pip_graph(args, crop_w, crop_h, out_w, out_h)
    else:
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
