#!/usr/bin/env python3
"""Apply audio-visual verdicts to the hit stream and regroup rallies.

Verdict per event (speeds in heights/s from validate_hits.py):
  confirmed — an assessable player shows a swing (speed >= --swing)
  rejected  — BOTH players assessable and BOTH clearly idle (< --idle):
              neighbour-court sound or a bounce (dropping bounces also
              purifies shot counts)
  kept      — anything else (gray zone / off-frame / poor keypoints):
              insufficient evidence to delete a sound

Regroups confirmed+kept events with the SAME 2.6s gap as detection and
prints the new rally table next to the raw one.

⚠ These verdicts are ADVISORY for boundaries. Two systematic errors mean
any boundary change MUST be dense-frame adjudicated (>=2 fps bursts,
scripts/event_montage.py) before you act on it:
  - false negatives: lunge gets and rushed compact end-of-rally swings
    score only ~1.2-2.4 (a "dead" tail can still be live play);
  - false positives: bending to scoop a ball with the racket scores ~3.2.
Rejections OUTSIDE established rally spans are safe to act on directly.

Usage:
  apply_validation.py --validated work/hits_validated.json \
      [--swing 2.8 --idle 2.0 --rally-gap 2.6 --min-hits 8 --min-dur 10]
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validated", required=True)
    ap.add_argument("--swing", type=float, default=2.8)
    ap.add_argument("--idle", type=float, default=2.0)
    ap.add_argument("--rally-gap", type=float, default=2.6)
    ap.add_argument("--min-hits", type=int, default=8)
    ap.add_argument("--min-dur", type=float, default=10.0)
    args = ap.parse_args()

    with open(args.validated) as f:
        events = json.load(f)

    for e in events:
        n, fa = e["near_speed"], e["far_speed"]
        n_ok, f_ok = e["near_assessable"], e["far_assessable"]
        if (n_ok and n >= args.swing) or (f_ok and fa >= args.swing):
            e["verdict"] = "confirmed"
        elif n_ok and f_ok and n < args.idle and fa < args.idle:
            e["verdict"] = "rejected"
        else:
            e["verdict"] = "kept"

    print("== rejected events (frame-check any near a rally boundary) ==")
    for e in events:
        if e["verdict"] == "rejected":
            print(f"  t={e['t']:7.2f} snr={e['snr']:5.1f} "
                  f"near={e['near_speed']:.2f} far={e['far_speed']:.2f}")

    live = [e for e in events if e["verdict"] != "rejected"]
    rallies, cur = [], []
    for e in live:
        if cur and e["t"] - cur[-1]["t"] > args.rally_gap:
            rallies.append(cur)
            cur = []
        cur.append(e)
    if cur:
        rallies.append(cur)

    print("\n== regrouped rallies (confirmed+kept) ==")
    for r in rallies:
        t0, t1 = r[0]["t"], r[-1]["t"]
        n_c = sum(1 for e in r if e["verdict"] == "confirmed")
        ok = len(r) >= args.min_hits and (t1 - t0) >= args.min_dur
        if len(r) >= 3 or ok:
            print(f"  {t0:7.2f}-{t1:7.2f} ({t1-t0:5.1f}s) events={len(r):2d} "
                  f"(confirmed={n_c}){' *QUALIFIES' if ok else ''}")

    out = args.validated.replace(".json", "_verdicts.json")
    with open(out, "w") as f:
        json.dump(events, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
