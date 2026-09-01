#!/usr/bin/env python3
"""Stage 4a: scan a source video and find when baked-in UI elements
(SwingVision speed card, minimap, ...) are actually rendered — they fade
in/out, and pasting a crop of an ABSENT element puts a rectangle of raw
background into the composite.

Presence = mean luma of the element's box region sits in the DARK cluster
(UI boxes are dark panels). The absent-state background is whatever the
scene has there — often NOT bright ceiling (beams/netting read ~100-112 on
the 08-21-2026 reference footage vs card-present 40-47), so thresholds must
come from the measured histogram, never from assumptions. If --thr-on/off
are not given, the scanner requires a clean bimodal split (an empty gap
>= --min-gap-luma between clusters) and places the thresholds inside the
gap; a muddy histogram aborts with instructions to eyeball frames at the
printed candidate timestamps and pass thresholds manually.

Elements on >= 99% of samples are marked always_on (skip enable-gating).
Interval edges are pulled inward (--edge-in) so the composite never shows
a half-faded box; intervals are emitted in OUTPUT time (t_src - t_offset).

Usage:
  4a_detect_ui_presence.py --src match.mp4 --out ui_presence.json \
      --rect card:1633,368,1897,462 --rect minimap:1633,12,1897,347 \
      --t0 171 --t1 323.475 [--t-offset 171] [--step 6] \
      [--thr-on 75 --thr-off 90]
"""
import argparse
import json

import cv2
import numpy as np


def auto_thresholds(v, min_gap):
    s = np.sort(v)
    gaps = s[1:] - s[:-1]
    i = int(np.argmax(gaps))
    lo, hi = s[i], s[i + 1]
    if hi - lo < min_gap:
        return None
    return lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rect", action="append", required=True,
                    help="name:x0,y0,x1,y1 in source pixels (repeatable)")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--t-offset", type=float, default=None,
                    help="subtracted from src time for the emitted intervals "
                         "(default: --t0, i.e. output timeline starts at 0)")
    ap.add_argument("--step", type=int, default=6,
                    help="sample every Nth frame (~10 Hz at 59fps)")
    ap.add_argument("--thr-on", type=float, default=None)
    ap.add_argument("--thr-off", type=float, default=None)
    ap.add_argument("--min-gap-luma", type=float, default=20.0,
                    help="required empty gap between clusters for auto thresholds")
    ap.add_argument("--edge-in", type=float, default=0.12)
    ap.add_argument("--min-len", type=float, default=0.25)
    ap.add_argument("--min-gap", type=float, default=0.35)
    ap.add_argument("--always-on-frac", type=float, default=0.99)
    args = ap.parse_args()

    rects = {}
    for spec in args.rect:
        name, nums = spec.split(":")
        x0, y0, x1, y1 = (int(v) for v in nums.split(","))
        rects[name] = (x0, y0, x1, y1)

    cap = cv2.VideoCapture(args.src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    t1 = args.t1 if args.t1 is not None else cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    t_off = args.t_offset if args.t_offset is not None else args.t0
    f0 = int(args.t0 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    ts, samples = [], {k: [] for k in rects}
    fi = f0
    while True:
        ok = cap.grab()
        if not ok or fi / fps > t1:
            break
        if (fi - f0) % args.step == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            ts.append(fi / fps)
            for k, (x0, y0, x1, y1) in rects.items():
                g = cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                samples[k].append(float(g.mean()))
        fi += 1
    cap.release()

    out = {}
    for k, vals in samples.items():
        v = np.array(vals)
        print(f"{k}: n={len(v)} p5={np.percentile(v,5):.0f} "
              f"p50={np.percentile(v,50):.0f} p95={np.percentile(v,95):.0f}")
        # Unimodal = the element never changes state (an always-on minimap
        # sits at a constant dark luma and has no bimodal gap to threshold).
        if np.percentile(v, 95) - np.percentile(v, 5) < args.min_gap_luma:
            if np.median(v) < 128:
                print("  constant dark — always_on, no gating needed")
                out[k] = {"always_on": True, "intervals": [], "enable": ""}
                continue
            raise SystemExit(
                f"{k}: region is constantly BRIGHT — element never appears; "
                f"check the rect coordinates.")
        if args.thr_on is not None and args.thr_off is not None:
            thr_on, thr_off = args.thr_on, args.thr_off
        else:
            thr = auto_thresholds(v, args.min_gap_luma)
            if thr is None:
                bright = [round(ts[i], 2) for i in np.argsort(v)[-3:]]
                dark = [round(ts[i], 2) for i in np.argsort(v)[:3]]
                raise SystemExit(
                    f"{k}: no clean bimodal gap (largest < {args.min_gap_luma} "
                    f"luma). Eyeball frames near dark={dark} bright={bright} "
                    f"and rerun with explicit --thr-on/--thr-off.")
            thr_on, thr_off = thr
            print(f"  auto thresholds: on<{thr_on:.0f} off>{thr_off:.0f}")

        on_frac = float((v < thr_on).mean())
        if on_frac >= args.always_on_frac:
            print(f"  always_on ({on_frac*100:.1f}% dark) — no gating needed")
            out[k] = {"always_on": True, "intervals": [], "enable": ""}
            continue

        ivs, on, t_on = [], False, 0.0
        for t, val in zip(ts, v):
            if not on and val < thr_on:
                on, t_on = True, t
            elif on and val > thr_off:
                on = False
                ivs.append([t_on, t])
        if on:
            ivs.append([t_on, ts[-1]])
        merged = []
        for a, b in ivs:
            if merged and a - merged[-1][1] < args.min_gap:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        final = [[round(a + args.edge_in - t_off, 3),
                  round(b - args.edge_in - t_off, 3)]
                 for a, b in merged if (b - a) >= args.min_len + 2 * args.edge_in]
        cov = sum(b - a for a, b in final)
        print(f"  {len(final)} intervals, {cov:.1f}s covered "
              f"({cov/(t1-args.t0)*100:.1f}%)")
        out[k] = {"always_on": False, "intervals": final,
                  "enable": "+".join(f"between(t,{a},{b})" for a, b in final)}

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
