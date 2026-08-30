---
name: video-autoreframe
description: >
  Auto-reframe a landscape video to vertical (9:16) or any aspect ratio,
  keeping a specific person smoothly in view via YOLO detection + ByteTrack
  identity tracking + QP-optimized camera path. Handles off-screen holds,
  bystander rejection, and near/far-court perspective changes. Use when the
  user asks to "crop to 9:16", "make vertical video", "auto reframe",
  "track me in this video", "convert landscape to TikTok/Reels/Shorts", or
  supplies a horizontal video to reshape for social media.
---

# Video Auto-Reframe

Converts a landscape video into a vertical (9:16 by default) clip that smoothly
tracks a target person, handling off-screen gaps and multiple people in frame.

Pipeline (all in `scripts/`):
1. **`1_detect.py`** — YOLOv8 + ByteTrack → per-frame list of people (x, y2, bbox).
2. **`2_solve_path.py`** — temporal-coherent person picker + static-fixture
   filter + QP-optimized camera trajectory with hard safety constraints.
3. **`2b_solve_pip.py`** *(optional)* — opponent (far-court player) picker +
   2-axis QP path → sendcmd for a picture-in-picture overlay.
4. **`3_encode.py`** — emits sendcmd + ffmpeg crop + lanczos upscale + H.264.
   With `--pip-cmds`, overlays a live opponent-tracking PiP (top-left by
   default).

## When to invoke

The user gives you a landscape video and wants it reformatted for vertical
playback while keeping them (or another subject) centered. Typical requests:
- "Turn this into a 9:16 vertical"
- "Track me in this video and crop to portrait"
- "Make a TikTok/Reels version of this"
- "Follow me in this clip"

## Clarifying questions to ask first

Before running, confirm:
1. **Which person is the target?** If multiple people are in the source, ask:
   closer to camera / farther / left / right side. Offer to extract a preview
   frame if they can't tell.
2. **Duration** — full video, or trim to a specific time range.
3. **Aspect ratio** — default 9:16. Confirm if they want 4:5, 1:1, or other.
4. **Output location** — default: same directory as source with `-9x16` suffix.

Also verify source format:
```bash
ffprobe -v error -show_entries stream=width,height,r_frame_rate,codec_name \
  -show_entries format=duration "$SRC"
```

## Running the pipeline

### Setup (once per new machine)

```bash
SKILL_DIR="$HOME/.claude/skills/video-autoreframe"
python3 -m venv "$SKILL_DIR/venv"
"$SKILL_DIR/venv/bin/pip" install --quiet ultralytics opencv-python-headless \
  numpy scipy osqp
```

Verify `ffmpeg`, `ffprobe` are on PATH (`brew install ffmpeg` if not).

### Per-video run

```bash
SRC="/path/to/source.mp4"
OUT="/path/to/output-9x16.mp4"
WORK="/tmp/autoreframe_$$"
mkdir -p "$WORK"

SKILL_DIR="$HOME/.claude/skills/video-autoreframe"
PY="$SKILL_DIR/venv/bin/python"

# Step 1: detect (~2 min for 6-min video on Apple Silicon)
"$PY" "$SKILL_DIR/scripts/1_detect.py" "$SRC" "$WORK/detections.json"

# Step 2: solve camera path (<1s)
# Add --cut-snap for rally compilations (SwingVision "Included Rallies" and
# any fixed-camera auto-edit): the camera then cuts at each splice instead
# of whip-panning across it. See the cut-snap gotcha below.
"$PY" "$SKILL_DIR/scripts/2_solve_path.py" \
  --detections "$WORK/detections.json" \
  --target-aspect 9:16 \
  --out-cmds "$WORK/crop_cmds.txt"

# Step 3: encode (~5 min at CRF 12 slow preset)
"$PY" "$SKILL_DIR/scripts/3_encode.py" \
  --src "$SRC" --cmds "$WORK/crop_cmds.txt" \
  --out "$OUT" --target-aspect 9:16 --crf 12
```

## Opponent picture-in-picture (optional)

Adds a small live "tracking camera" of the opponent (far-court player) in the
top-left of the vertical output while the main crop follows the near player.

```bash
# PiP needs its OWN detection pass — see gotcha below. MPS makes this ~60fps.
"$PY" "$SKILL_DIR/scripts/1_detect.py" "$SRC" "$WORK/opp_detections.json" \
  --model yolov8m.pt --conf 0.15 --min-h 35 --imgsz 1280 --device mps

# Solve the opponent PiP path (x and y)
"$PY" "$SKILL_DIR/scripts/2b_solve_pip.py" \
  --detections "$WORK/opp_detections.json" \
  --out-cmds "$WORK/pip_cmds.txt"

# Encode with the overlay
"$PY" "$SKILL_DIR/scripts/3_encode.py" \
  --src "$SRC" --cmds "$WORK/crop_cmds.txt" \
  --pip-cmds "$WORK/pip_cmds.txt" \
  --out "$OUT" --target-aspect 9:16 --crf 12
```

Key facts learned building this:

- **The default detection pass CANNOT see a far-court opponent.** In 1080p
  indoor-tennis footage the opponent across the net is only ~50-60px tall —
  below the default `--min-h 80` — and at imgsz 640 their confidence is
  marginal. Use `--min-h 35 --imgsz 1280 --conf 0.15` (with yolov8m) for the
  PiP pass. Keep the main-subject pass at defaults; lowering min-h globally
  floods the main picker with tiny bystanders.
- **Do NOT require per-tid motion.** ByteTrack fragments a 50px opponent
  into 10+ short tids per minute, and individual fragments often don't move
  (opponent standing between rallies) — a "tid must have moved ≥100px"
  filter dropped the pick rate to 49% on real footage. Static bystanders are
  instead blacklisted only when a tid is BOTH long-lived (≥6s) and immobile
  (x-range <40px); short still fragments of the real opponent survive.
- **Height-cap the candidates.** When the near player walks deep to collect
  balls they enter the far-court band; at that camera depth the opponent is
  ~50px tall but any near person in the band is 120px+. Cap at 1.8x the
  median in-band height (per-tid median AND per-sample), and the near
  player is cleanly rejected while the opponent (h up to ~72px when serving)
  passes.
- **`sendcmd` targets must be instance-named when the graph has two crops.**
  An untargeted `crop x N` is delivered to EVERY crop filter in the graph.
  2b emits `crop@pip x/y`; 3_encode rewrites main lines to `crop@main` and
  merges the two files (commands sharing a timestamp are grouped into one
  interval because sendcmd needs strictly increasing interval times).
- The opponent search band defaults to x ∈ [0.33W, 0.67W], y2 ∈
  [0.42H, 0.72H] — the far half of the player's own court. Adjacent-court
  players and fence-line walkers sit outside it; tune `--x-min/--x-max/
  --y2-min/--y2-max` per venue if the picker reports low pick rates.
- PiP source-crop height defaults to 2.5× the opponent's median bbox height
  (16:9) with a 5% upward bias, so the framing scales with distance; display
  width 475px (~44% of a 1080-wide output, user-approved size — the ~2x
  lanczos upscale from the small source crop reads fine on phones),
  positioned at (28, 120) — below the platform-UI zone at the top of
  vertical players. Override with `--pip-display-w/--pip-x/--pip-y`. Keep the crop TIGHT: in this camera
  geometry the near player's head/racket reaches into the far-court y-band
  whenever the two players x-align (e.g. mid-court ball pickup), and a loose
  crop fills with the near player's back. Tight crop + up-bias limits the
  intrusion to a head sliver at the PiP's bottom edge.
- **`--tid-memory-s` gates COLD START, so it caps how long a stuck PiP can
  stay stuck.** With the 4.0s default, an opponent standing ~330px from the
  held position (outside the 250px adopt radius) is invisible to all three
  branches until 4s elapse — on the 08-21-2026 footage that produced a
  single 3.55s window with the PiP on empty court, capping coverage at
  96.2%. Dropping to 0.6s lets COLD START re-pick the dominant in-band tid
  (which is the opponent by construction) and lifted coverage to 99.6% /
  100% on the two clips, with **identical** smoothness (mean 0.59, max 17
  px/frame at every value tested). Prefer lowering this over widening
  `--adopt-radius-px`, which reintroduces the phantom-adoption traps below.
- During rally gaps the PiP holds its last position (same
  invisible-fit-weight mechanism as the main path) — better than hiding or
  chasing noise.
- **Three phantom-adoption traps, found by validating opponent-in-PiP
  coverage per sample** (target ≥99%; first pass scored 91%):
  1. A short-lived static phantom (seated spectator, furniture) survives a
     6s-lifetime static blacklist, gets ADOPTed during a brief opponent
     dropout, then anchors the position so the real opponent re-appears
     outside the adopt radius. Blacklist statics from 1.5s of lifetime; the
     opponent's own still fragments being blacklisted too is harmless
     (hold-last-position covers someone who is standing still by
     definition).
  2. SwingVision-style sources are auto-edited rally clips: the opponent
     teleports ~200px across a cut. Adopt radius must be ~250px, not 150.
  3. With the larger radius, a 1-2 sample flicker detection at the band edge
     becomes adoptable during a dropout. Require tids to have ≥5 in-band
     samples (0.5s) before ADOPT/COLD may select them.
  Always re-run the coverage validation after tuning, with
  `scripts/2c_validate_pip.py`. It re-derives the legitimate-opponent
  candidate set using the same band + static/tall/min-count/height-cap
  filters as 2b, parses the emitted `crop@pip` timeline, and reports the
  fraction of samples where a legitimate opponent actually sits inside the
  solved crop, plus the miss timestamps:

  ```bash
  "$PY" "$SKILL_DIR/scripts/2c_validate_pip.py" \
    --detections "$WORK/opp_detections.json" \
    --cmds "$WORK/pip_cmds.txt"
  ```

  Keep its filters in sync with 2b — an over-permissive validator reports
  phantom misses (e.g. omitting the per-sample height cap flags near-player
  detections that 2b correctly rejected).

## Critical gotchas (learned the hard way)

### sendcmd float-timing bug at 60fps

`1/60 = 0.01666...` cannot be exactly represented in decimal. If you emit
sendcmd times as `%.4f`, they round to `0.0167` which is *greater* than the
actual frame PTS. ffmpeg's sendcmd applies a command only when
`pts >= cmd_time`, so every 3rd frame gets "stuck" (command applies a frame
late, and by then the next frame's command overwrites it). This manifests as
visible motion ghosting/judder on playback.

**Fix:** Emit each command at `(frame_index - 0.5) / fps` with `%.7f` precision.
This places the command midway between the previous and target frame's PTS,
guaranteeing `cmd_time < frame_pts` regardless of rounding. Implemented in
`3_encode.py` — do not change.

**Verification:** extract consecutive output frames by index and compute
pixel diffs; should be uniform. Alternating "high, high, low" diffs = the
timing bug is back.

### Don't use bicubic scale for upscaling

`scale=1080:1920` defaults to bicubic which softens edges. Always use
`flags=lanczos` for sharp output. Matters more after a large upscale
(e.g., 608→1080 is a 1.78× upscale).

### Don't over-use unsharp mask

`unsharp` introduces halos on already-sharp content. Only use if the CRF is
forced high (file-size constrained) AND the source is genuinely soft. For
best quality prefer CRF 12 + lanczos with NO unsharp.

### CRF 16-20 is too lossy for fast-motion content

At CRF 20, H.264 softens fast-pan frames to save bits. Use **CRF 12** for
near-visually-lossless output of sports/pans. Expect 500MB-1GB for a
6-minute 1080×1920 video.

### Frame interpolation (RIFE/minterpolate) does NOT fix motion ghosting

Interpolating 60→120fps adds frames but does not remove per-frame softness.
Only useful for slow-motion output or 120Hz displays. If the user reports
"ghosting", investigate timing/compression/scaling before reaching for RIFE.

### Camera drifts in a U-shape during long invisible gaps

Failure mode: when the subject is invisible for >2 seconds (e.g. between
rallies, player walked off-screen), the camera *appears* to track a bystander
mid-gap, even though no detection is being picked. Cause: with
`fit_mask = visible.astype(float)`, the fit term goes to zero during the
gap; the QP then minimizes acceleration freely between the surrounding
visible boundaries. With asymmetric end-velocities, the smoothest path is
NOT a straight line — it bulges (sometimes substantially) toward whichever
direction balances the velocity profile, possibly centering a bystander
who happens to be in that side of the frame.

**Fix:** apply a weak fit pull during invisible frames toward the
forward-filled last-known subject position:
```python
fit_mask = np.where(visible, 1.0, 0.15)
```
0.15 is enough to anchor the camera at the last known position during gaps
while still letting smoothness dominate during normal motion. Implemented
as `--invisible-fit-weight 0.15` in `2_solve_path.py`.

Verify by dumping the crop_x trajectory across an invisible region — it
should be roughly constant, not U-shaped.

### Rally compilations splice without any scene change (`--cut-snap`)

SwingVision "Included Rallies" exports (and any fixed-camera rally
compilation) concatenate rally segments. Because the camera never moves and
the background is identical, **ffmpeg scene detection finds nothing** — even
at `gt(scene,0.06)` — yet the players teleport across the splice.

Symptom: 8-13 bursts per clip where the camera whip-pans the full frame
width over 0.15-0.4s. The first ~0.4s of every rally (often the serve) is
off-frame while the camera catches up.

**Fix:** `--cut-snap` on `2_solve_path.py`. It detects splices from the
subject trajectory (the largest-bbox near-court candidate is a reliable
"where is the main player" proxy), resets the picker at each cut so the
stale lock and jump guard don't fight the new position, and drops the QP
smoothness rows spanning the cut frame so the camera pays no
velocity/acceleration penalty for jumping there. The camera cuts instead of
pans.

Two splice signatures, both needed:
1. **Adjacent samples teleport** — `|dx| >= --cut-jump-px` (400) within
   `--cut-max-dt` (0.25s). Nobody covers 400px in 0.1s, so this is certain.
2. **Subject reappears far away after an absence** — the player was
   undetected for a gap and returns `>= --cut-jump-px` away, within
   `--cut-max-gap-s` (3.0s). A player who genuinely ran that far would have
   been detected the whole way. Catching only signature 1 leaves ~20% of
   splices as whip-pans, because the subject is frequently off-frame right
   before a cut.

Measured on 08-21-2026 SwingVision footage (2 clips, 1080p59): p99 camera
velocity 39.7 -> 17.3 px/frame, `reacquire_jumpy` 35 -> 0, valid picks
83.4% -> 86.1%. Default is OFF; behaviour is byte-identical without the flag.

Residual sub-0.3s fast pans immediately *before* a cut are real — the camera
chasing a wide ball as the rally ends. Don't tune those away.

### Camera tracks the wrong person during between-rally moments

Failure mode: between rallies, the main player briefly walks off-screen or
YOLO drops their detection, and a walking bystander (not motionless, so the
static-fixture filter misses them) gets latched onto.

If this happens, dump per-sample picks around the failure timestamp:
```python
# For each pick, print t, x, y2, h, w, tid
# Compare to known-good regions
```
The wrong-subject pick will have a dramatically smaller bbox **area**
(`h * w`) than the main player even at the far baseline: bystanders typically
have area 2000-2700; the main player has area ≥4500 even when crouched at
the baseline. Use bbox area as the primary discriminator on re-acquire —
relative-h alone is too lenient when the subject was already small.

Note that bbox **height alone** (h) is NOT a clean separator: the main player
at the far baseline drops to h=120-140, overlapping with bystanders' h range.
But the player is always *wider* (w ~ 45-55) than far-edge walkers (w ~ 25-32),
so `h * w` separates cleanly.

### Near-camera pedestrians beat every size filter — gate by position

On courts with a walkway/promenade right behind a fence (08-30-2026 Hudson
River Park footage), a jogger walking close past the fence line reaches
bbox area 12000-16500 and h up to 183 — bigger than the "main player
crouched at baseline" thresholds, so the area floor AND the cold-start
min-h both pass. When the player is briefly off-frame (pulled wide outside
the wide-angle FOV — happens on off-center camera positions), cold start
latches onto the jogger and the camera swings to the fence.

Size cannot fix this; position can. `--x-min/--x-max` on `2_solve_path.py`
reject candidates outside the court region entirely (that footage used
`--x-max 1520`). Look up the fence/court extent in a source frame before
solving. Defaults are ±inf (off).

### Cut-snap's proxy needs its own area floor

The splice detector follows the largest *near-court* candidate with no size
floor. While the player was off-frame for ~3s mid-rally, tiny walkway
detections (area 2500-3700) became the proxy, and their jumps produced 4-6
phantom splices *inside a continuous rally* — each one a hard camera cut.
`--cut-proxy-min-area` (default 4500, matching the main-player minimum)
keeps sub-player-sized detections from ever driving splice detection.
True splices where the player stands near the same spot on both sides
need no cut anyway, so losing tiny-proxy coverage costs nothing.

### tid=-1 is not an identity

ByteTrack emits tid=-1 for untracked boxes. The LOCKED branch treated
"prev pick was -1" + "candidate is -1" as the same person; with the
dt-scaled jump allowance (150 + 1500*dt) a 1.2s gap authorizes a ~1900px
teleport — the lock jumped from the real player (x=80) to a pedestrian
(x=1266) and the camera parked on the fence for 2s. Fixed in
`2_solve_path.py`: tid<0 picks never enter the lock (prev_tid stays None,
position/size memory still updates), so untracked detections can only be
picked through the size-gated re-acquire / cold-start branches.

## Subject-selection strategy

Built around a **ByteTrack-tid-locked state machine** with three states. A
naive "pick highest y2" approach latches onto walking bystanders during
between-rally moments when the main player is briefly off-screen, even though
the bystander has a noticeably smaller bbox. The fix uses tid identity plus
bbox size to distinguish them.

### Baseline filters (apply to every candidate)

1. **y2 threshold 600** (not 700): user at the net has y2 ~625-700. A too-high
   threshold rejects them during net approaches. Coach/bystanders at
   y2 ~640-680 get rejected via the static-fixture filter below.

2. **Static-fixture filter**: bystanders sit motionless at fixed (x, y2)
   positions. Build a 30px-binned histogram across the whole video; bins with
   ≥50 occurrences in the "far-court" band (y2 < 720) are fixture locations;
   reject any candidate in those cells. Skip the filter for y2 ≥ 720 (near
   court = always user). This only catches *motionless* bystanders — walkers
   pass through too many bins to hit the count threshold.

### State machine (per sample)

The picker maintains `prev_tid`, `prev_x`, `prev_y2`, `prev_h`, `prev_t` and
tries three branches in order:

1. **LOCKED — same tid as last pick.** If `prev_tid` appears in the current
   candidates and the x-jump is plausible (`<= 150 + 1500 * dt`), pick it.
   No size check here — ByteTrack tells us this is the same person, even if
   they walked to the far baseline and shrank to h=100. Most picks (>95% in
   well-tracked sequences) come through this branch.

2. **REACQUIRE — similar-sized new tid within 0.8s.** If `prev_tid` is gone
   but we lost it recently, allow switching to a candidate whose:
   - `h >= 0.6 * prev_h` (relative size sanity), AND
   - `h * w >= 3500` (**absolute area floor**)

   The area floor is critical. Relative-h alone is too lenient when the
   subject was at the far baseline (prev_h already small ~120); a narrow
   distant walker with h=85 passes `0.6 * 120 = 72` but their bbox area
   (~2200) is far below the main player's at any baseline distance (~4500+).

3. **COLD START — no recent lock or long absence.** Triggered when
   `prev_tid is None` OR `dt > reacquire_window_s (0.8s)`. Requires
   `h >= cold-start-min-h (170)`. Distant walkers max out around h=140 in
   this video, so they're cleanly rejected. The main player at h=170 is
   typically mid-court — a safe re-entry point.

### tid memory ≫ reacquire window

`prev_tid` is retained for `tid-memory-s (5s)` even though re-acquire by
similarity only fires for 0.8s. This means: if the same player disappears
for 1-4 seconds (gathering balls, brief occlusion) and reappears with the
same ByteTrack id, the LOCKED branch re-engages immediately — even if their
bbox is tiny (h=100 at the far baseline). Without this separation, every
short loss would force cold-start, which requires h≥170, leaving the
main player untracked when they're at the back of the court.

### When *no* branch picks anyone

The QP visibility ramp holds the camera at the last good crop position
(ramp the safety margin 0→90px over 0.4s on visibility transitions).
Better to hold than to chase a wrong subject.

## QP camera-path solver

Minimize `W_VEL·Σ(Δx)² + W_ACC·Σ(Δ²x)² + W_FIT·Σ(x - target)²`
subject to per-frame box constraints keeping the subject in the crop with a
safety margin.

- `W_VEL=5`, `W_ACC=1500`, `W_FIT=1` work well for "indoor sports" footage.
- Safety margin: 90px. Ramp the margin from 0→90 over 0.4s when visibility
  transitions on/off to avoid snap at re-acquisition.
- OSQP solves 22k-variable QP in <1 second.

## Encoding defaults

```
-c:v libx264 -preset slow -crf 12
-pix_fmt yuv420p  (for compatibility)
-c:a aac -b:a 192k
-movflags +faststart
scale=<target>:flags=lanczos
```

Do not set `-bf 0` or `-g 1` unless debugging — those inflate file size 4×
without helping normal playback quality.

## Output verification checklist

Before reporting success to the user:
- [ ] Extract consecutive frames 88-99 (or any ~1.5s into motion) by index;
      compute pair-wise pixel diffs. Should be uniform. If alternating
      high/low, the sendcmd timing bug is back.
- [ ] Confirm output duration matches source.
- [ ] Spot-check a frame at a known-tricky moment (net approach, scene change)
      for subject-in-frame.
- [ ] If the user mentioned specific failure timestamps in past iterations,
      re-verify those regions.
