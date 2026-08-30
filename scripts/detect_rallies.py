#!/usr/bin/env python3
"""Detect racquet hits from court audio and group them into rallies.

A hit is a broadband transient: band-pass 1.5-7 kHz (ball-strike crack,
suppresses wind/voices/low bounce thud), fast envelope, peak-pick against
a rolling-median noise floor so near (loud) and far (quiet) hits both
register without fixed-threshold tuning.

Qualification uses a DOUBLE GATE — events >= min-hits AND duration >=
min-dur — because far-court hits are missed ~half the time and some
events are bounces; duration anchors the true shot count (rec cadence
~1.2-1.6 s/shot), making the gate robust to both error modes.

Outputs <outdir>/hits.json (all events) and <outdir>/rallies.json
(grouped). The printed gap lists are the adjudication input: see SKILL.md
"Rally extraction" for which holes must be frame-checked before trusting.

Usage:
  detect_rallies.py --src match.mov --outdir work/
  detect_rallies.py --wav work/audio.wav --outdir work/ --min-hits 8
"""
import argparse
import json
import os
import subprocess

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="video file; audio is extracted to outdir")
    ap.add_argument("--wav", help="pre-extracted mono wav (skips extraction)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--band", nargs=2, type=float, default=[1500, 7000])
    ap.add_argument("--env-ms", type=float, default=6.0)
    ap.add_argument("--noise-win-s", type=float, default=2.0)
    ap.add_argument("--snr", type=float, default=5.0)
    ap.add_argument("--min-gap-s", type=float, default=0.45)
    ap.add_argument("--rally-gap-s", type=float, default=2.6,
                    help="max gap within a rally. Do NOT raise past ~2.6 to "
                    "rescue holes — a racket-scoop ball pickup can score as "
                    "a swing and a wide gap then glues two dead rallies.")
    ap.add_argument("--min-hits", type=int, default=8)
    ap.add_argument("--min-dur", type=float, default=10.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    wav = args.wav
    if not wav:
        if not args.src:
            ap.error("need --src or --wav")
        wav = os.path.join(args.outdir, "audio.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-i", args.src, "-map",
                        "0:a:0", "-ac", "1", "-ar", "48000", "-y", wav],
                       check=True)

    sr, x = wavfile.read(wav)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x /= max(1.0, np.abs(x).max())

    sos = butter(4, args.band, btype="band", fs=sr, output="sos")
    y = np.abs(sosfiltfilt(sos, x))
    hop = sr // 1000
    n = len(y) // hop
    env = y[: n * hop].reshape(n, hop).max(axis=1)
    k = max(1, int(args.env_ms))
    env = np.convolve(env, np.ones(k) / k, mode="same")

    dec = 10
    e10 = env[: (n // dec) * dec].reshape(-1, dec).mean(axis=1)
    win = int(args.noise_win_s * 100)
    pad = np.pad(e10, win // 2, mode="edge")
    floor10 = np.array([np.median(pad[i : i + win]) for i in range(len(e10))])
    floor = np.repeat(floor10, dec)
    if len(floor) < n:
        floor = np.pad(floor, (0, n - len(floor)), mode="edge")
    floor = np.maximum(floor[:n], 1e-5)

    snr = env / floor
    min_gap = int(args.min_gap_s * 1000)
    hits = []
    i = 0
    while i < n:
        if snr[i] >= args.snr:
            j = min(n, i + min_gap)
            kpk = i + int(np.argmax(env[i:j]))
            hits.append((kpk / 1000.0, float(snr[kpk]), float(env[kpk])))
            i = kpk + min_gap
        else:
            i += 1

    rallies, cur = [], []
    for h in hits:
        if cur and h[0] - cur[-1][0] > args.rally_gap_s:
            rallies.append(cur)
            cur = []
        cur.append(h)
    if cur:
        rallies.append(cur)

    print(f"hits: {len(hits)}  groups: {len(rallies)}")
    for r in rallies:
        t0, t1 = r[0][0], r[-1][0]
        ok = len(r) >= args.min_hits and (t1 - t0) >= args.min_dur
        gaps = ",".join(f"{r[i+1][0]-r[i][0]:.2f}" for i in range(len(r) - 1))
        print(f"  {t0:7.2f}-{t1:7.2f} ({t1-t0:5.1f}s) hits={len(r):2d}"
              f"{' *QUALIFIES' if ok else ''}  gaps=[{gaps}]")

    with open(os.path.join(args.outdir, "hits.json"), "w") as f:
        json.dump([{"t": round(t, 3), "snr": round(s, 1), "amp": round(a, 5)}
                   for t, s, a in hits], f)
    with open(os.path.join(args.outdir, "rallies.json"), "w") as f:
        json.dump([{"start": round(r[0][0], 3), "end": round(r[-1][0], 3),
                    "hits": len(r),
                    "qualifies": len(r) >= args.min_hits
                    and (r[-1][0] - r[0][0]) >= args.min_dur,
                    "hit_times": [round(h[0], 3) for h in r]}
                   for r in rallies], f, indent=1)
    print(f"wrote {args.outdir}/hits.json, rallies.json")


if __name__ == "__main__":
    main()
