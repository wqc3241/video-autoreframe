#!/usr/bin/env python3
"""Stage 4c: paste baked-in UI elements (SwingVision minimap / speed card)
from the UNWARPED source into a finished video at native pixel scale —
straight and undistorted, with presence-gated visibility.

Screen-space vs scene-space:
- The pasted elements are SCREEN-space: fixed position in the output frame.
- The static plate that hides the old baked overlay is SCENE-space: it
  must ride the reel. For VERTICAL output apply the plate to the reel and
  re-run 3_encode first, THEN run this script on that base. For LANDSCAPE
  output base == reel, so --plate folds it into this one pass.

Element syntax: name:x0,y0,x1,y1@X,Y
  (source rect in unwarped-source pixels, destination top-left in base px)
Presence json comes from 4a; elements marked always_on (or absent from the
json) are pasted unconditionally, others get enable='between(...)+...'.

Usage (vertical, base already rebuilt from the plate-cleaned reel):
  4c_transplant_ui.py --base base_clean_9x16.mp4 \
      --src match.mp4 --src-offset 171.0 \
      --element minimap:1633,12,1897,347@794,120 \
      --element card:1633,368,1897,462@794,476 \
      --presence ui_presence.json --out final.mp4

Usage (landscape, one pass):
  4c_transplant_ui.py --base rally_reel.mov --plate corner_plate.png \
      --src match.mp4 --src-offset 171.0 \
      --element minimap:1633,12,1897,347@1633,12 \
      --element card:1633,368,1897,462@1633,368 \
      --presence ui_presence.json --out final_16x9.mp4
"""
import argparse
import json
import subprocess


def probe_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True)
    return float(r.stdout.strip().splitlines()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--src", required=True,
                    help="UNWARPED source the elements are cropped from")
    ap.add_argument("--src-offset", type=float, default=0.0,
                    help="source time that corresponds to base t=0")
    ap.add_argument("--dur", type=float, default=None,
                    help="default: base duration")
    ap.add_argument("--element", action="append", required=True,
                    help="name:x0,y0,x1,y1@X,Y (repeatable)")
    ap.add_argument("--presence", default=None, help="json from 4a")
    ap.add_argument("--plate", default=None,
                    help="static plate png overlaid on the base first "
                         "(landscape one-pass only — for vertical, apply the "
                         "plate to the reel before 3_encode instead)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crf", type=int, default=12)
    ap.add_argument("--preset", default="slow")
    args = ap.parse_args()

    presence = {}
    if args.presence:
        presence = json.load(open(args.presence))
    dur = args.dur if args.dur is not None else probe_duration(args.base)

    elems = []
    for spec in args.element:
        name, rest = spec.split(":")
        rect, pos = rest.split("@")
        x0, y0, x1, y1 = (int(v) for v in rect.split(","))
        X, Y = (int(v) for v in pos.split(","))
        en = ""
        p = presence.get(name)
        if p and not p.get("always_on") and p.get("enable"):
            en = p["enable"]
        elems.append((name, x0, y0, x1 - x0, y1 - y0, X, Y, en))

    inputs = ["-i", args.base]
    n_in = 1
    fc = []
    cur = "0:v"
    if args.plate:
        inputs += ["-i", args.plate]
        fc.append(f"[{cur}][{n_in}:v]overlay=0:0[p0]")
        cur = "p0"
        n_in += 1
    inputs += ["-ss", f"{args.src_offset:.4f}", "-t", f"{dur:.4f}", "-i", args.src]
    src_in = n_in

    outs = [f"s{i}" for i in range(len(elems))]
    fc.append(f"[{src_in}:v]split={len(elems)}" + "".join(f"[{o}]" for o in outs))
    for i, (name, x0, y0, w, h, X, Y, en) in enumerate(elems):
        fc.append(f"[{outs[i]}]crop={w}:{h}:{x0}:{y0}[c{i}]")
    for i, (name, x0, y0, w, h, X, Y, en) in enumerate(elems):
        nxt = "v" if i == len(elems) - 1 else f"t{i}"
        gate = f":enable='{en}'" if en else ""
        fc.append(f"[{cur}][c{i}]overlay={X}:{Y}{gate}[{nxt}]")
        cur = nxt

    cmd = (["ffmpeg", "-v", "error"] + inputs +
           ["-filter_complex", ";".join(fc), "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
            "-y", args.out])
    print(" ".join(cmd[:12]) + " ...")
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
