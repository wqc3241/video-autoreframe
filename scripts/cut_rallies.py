#!/usr/bin/env python3
"""Cut curated rallies from the source, optionally applying the warp,
and concat into a rally reel ready for the reframe pipeline.

Input rallies JSON (curate this by hand after adjudication — never feed
raw detector groups straight to the cutter):

  {"pre_pad_s": 1.8, "post_pad_s": 2.5,
   "rallies": [
     {"id": "R1", "first_hit": 40.60, "last_hit": 58.25},
     ...
   ]}

Pads: -1.8s before the first hit keeps the feeder's windup; +2.5s after
the last stroke shows the point resolving. If --warp is given (JSON from
solve_warp.py), its perspective filter is applied in the same pass, so
the whole chain costs ONE resample.

Segments are encoded hevc_videotoolbox 45M (fast hw intermediate, fine
for one more re-encode downstream) and concatenated with -c copy.
Splice offsets land in <outdir>/splices.json — feed the reel to
1_detect + 2_solve_path --cut-snap.

Usage:
  cut_rallies.py --src match.mov --rallies rallies_final.json \
      [--warp warp.json] --outdir work/
"""
import argparse
import json
import os
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--rallies", required=True)
    ap.add_argument("--warp")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.rallies) as f:
        cfg = json.load(f)
    vf = None
    if args.warp:
        with open(args.warp) as f:
            vf = json.load(f)["ffmpeg_filter"]

    seg_paths, splices, t_acc = [], [], 0.0
    for r in cfg["rallies"]:
        t0 = r["first_hit"] - cfg.get("pre_pad_s", 1.8)
        t1 = r["last_hit"] + cfg.get("post_pad_s", 2.5)
        dur = t1 - t0
        out = os.path.join(args.outdir, f"seg_{r['id']}.mov")
        seg_paths.append(out)
        splices.append({"id": r["id"], "reel_start": round(t_acc, 3),
                        "src_start": round(t0, 3), "dur": round(dur, 3)})
        t_acc += dur
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-i", args.src,
               "-t", f"{dur:.3f}"]
        if vf:
            cmd += ["-vf", vf]
        cmd += ["-c:v", "hevc_videotoolbox", "-b:v", "45M", "-tag:v", "hvc1",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                "-video_track_timescale", "60000", "-y", out]
        print(f"{r['id']}: {t0:.2f}..{t1:.2f} ({dur:.1f}s)")
        subprocess.run(cmd, check=True)

    lst = os.path.join(args.outdir, "concat.txt")
    with open(lst, "w") as f:
        for p in seg_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    reel = os.path.join(args.outdir, "rally_reel.mov")
    subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", lst, "-c", "copy", "-y", reel], check=True)
    with open(os.path.join(args.outdir, "splices.json"), "w") as f:
        json.dump(splices, f, indent=1)
    total = sum(s["dur"] for s in splices)
    print(f"reel: {reel}  total {total:.1f}s, splices at "
          + ", ".join(f"{s['reel_start']:.1f}" for s in splices[1:]))


if __name__ == "__main__":
    main()
