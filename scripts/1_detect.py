#!/usr/bin/env python3
"""Detect people per frame using YOLOv8 + ByteTrack, save to JSON.

Usage: 1_detect.py SRC_VIDEO OUT_JSON [--sample-every N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out_json")
    ap.add_argument("--sample-every", type=int, default=6,
                    help="Sample every Nth frame (default 6 = 10Hz at 60fps)")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="YOLO model (yolov8n.pt fastest, yolov8m.pt more accurate)")
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {W}x{H} @ {fps:.2f}fps, {total} frames", flush=True)

    samples = []
    idx = 0
    t0 = time.time()
    last_log = t0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % args.sample_every == 0:
            t_sec = idx / fps
            results = model.track(frame, classes=[0], verbose=False,
                                  conf=args.conf, persist=True,
                                  tracker="bytetrack.yaml")
            people = []
            for r in results:
                if r.boxes is None:
                    continue
                ids = r.boxes.id
                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    h = y2 - y1
                    w = x2 - x1
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    if h < 80:
                        continue
                    tid = int(ids[i].item()) if ids is not None else -1
                    people.append({
                        "tid": tid,
                        "x": cx, "y": cy, "h": h, "w": w,
                        "bbox": [x1, y1, x2, y2],
                        "y2": y2,
                    })
            samples.append({"t": t_sec, "people": people})
            now = time.time()
            if now - last_log > 5:
                pct = idx / total * 100
                rate = idx / (now - t0)
                eta = (total - idx) / rate if rate > 0 else 0
                print(f"  {pct:5.1f}%  frame {idx}/{total}  "
                      f"rate={rate:.0f}fps  ETA={eta:.0f}s", flush=True)
                last_log = now
        idx += 1

    cap.release()
    out = {"fps": fps, "W": W, "H": H, "samples": samples}
    Path(args.out_json).write_text(json.dumps(out))
    n_with = sum(1 for s in samples if s["people"])
    print(f"Done: {len(samples)} samples, {n_with} with >=1 person, "
          f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
