# video-autoreframe

A [Claude Code](https://claude.com/claude-code) skill that turns a landscape video
into a vertical (9:16 by default) clip which smoothly tracks one specific person —
holding steady when they leave frame, and refusing to latch onto bystanders.

Three stages, all in `scripts/`:

| Stage | Script | What it does |
|---|---|---|
| 1 | `1_detect.py` | YOLOv8 + ByteTrack → per-frame people (x, y2, bbox, track id) |
| 2 | `2_solve_path.py` | tid-locked subject picker + static-fixture filter → OSQP-optimized camera trajectory with hard in-frame constraints |
| 3 | `3_encode.py` | ffmpeg `sendcmd` crop + lanczos upscale + H.264 |

## Install

Clone into your Claude Code skills directory:

```bash
git clone https://github.com/wqc3241/video-autoreframe.git ~/.claude/skills/video-autoreframe
```

Then build the venv (~1.1 GB, not in the repo) and check for ffmpeg:

```bash
python3 -m venv ~/.claude/skills/video-autoreframe/venv
~/.claude/skills/video-autoreframe/venv/bin/pip install ultralytics opencv-python-headless numpy scipy osqp
ffmpeg -version || brew install ffmpeg
```

Ask Claude for "a 9:16 vertical of this clip, tracking me" and the skill takes over.
The scripts also run standalone — see the per-video command block in
[`SKILL.md`](SKILL.md).

## Why the odd-looking constants

`SKILL.md` is the real documentation. It carries the failure modes that shaped this
code, each of which cost a debugging session:

- **The 60 fps `sendcmd` float-timing bug.** `1/60` isn't representable in decimal, so
  `%.4f` command times land *after* their frame's PTS and every third frame judders.
  Commands are emitted at `(frame - 0.5) / fps` with `%.7f`.
- **U-shaped camera drift across long invisible gaps.** With zero fit weight while the
  subject is gone, the smoothest path between two boundary velocities bulges sideways —
  often centering a bystander. A weak pull (`0.15`) toward the last known position fixes it.
- **Bystander capture between rallies.** Bbox *height* alone doesn't separate a distant
  walker from the subject at the far baseline; bbox *area* does, cleanly.
- **CRF 12 + lanczos, no unsharp.** CRF 16-20 smears fast pans; frame interpolation does
  not fix per-frame softness.

## License

MIT
