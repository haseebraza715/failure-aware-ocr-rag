#!/usr/bin/env bash
# render.sh <cast> <outdir> — render gif-preview + mp4 from an asciicast.
# Same pipeline as the shared mkdemo.sh but with an IDLE_LIMIT knob for pacing:
# the shared default (0.12s) collapses long sleeps and can produce an mp4 under
# 15s; raise it (e.g. IDLE_LIMIT=3.0) to preserve the recorded pacing.
set -euo pipefail

CAST="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
OUT="$(cd "$2" && pwd)"
IDLE_LIMIT="${IDLE_LIMIT:-0.12}"

AGG="${AGG:-$HOME/.cargo/bin/agg}"
if [ ! -x "$AGG" ]; then
  AGG="$(command -v agg || true)"
fi
if [ -z "$AGG" ]; then
  echo "ERROR: agg (asciinema gif renderer) not found — install with 'cargo install agg' or set AGG=" >&2
  exit 1
fi

MEDIA_VENV="${MEDIA_VENV:-/var/folders/bf/gc6myzc90cq_vrj0wd351wqw0000gn/T/opencode/media/.venv}"
if [ ! -x "$MEDIA_VENV/bin/python" ]; then
  echo "ERROR: MEDIA_VENV with imageio-ffmpeg not found at $MEDIA_VENV" >&2
  exit 1
fi
FF="$("$MEDIA_VENV/bin/python" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")"

cd "$OUT"
echo "== rendering 30fps source (for mp4) — idle-time-limit $IDLE_LIMIT"
"$AGG" --theme dracula --cols 100 --rows 30 --fps-cap 30 --idle-time-limit "$IDLE_LIMIT" --speed 1.0 --last-frame-duration 1.5 "$CAST" /tmp/raw30.gif
echo "== transcoding to mp4 (h264, yuv420p, faststart)"
"$FF" -y -i /tmp/raw30.gif -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart -an demo.mp4
echo "== rendering preview gif (first 12s, 10fps)"
"$AGG" --theme dracula --cols 100 --rows 30 --fps-cap 10 --idle-time-limit "$IDLE_LIMIT" --speed 1.5 --last-frame-duration 1.5 --select ..12 "$CAST" demo.gif
rm -f /tmp/raw30.gif
echo "== results"
"$FF" -i demo.mp4 2>&1 | grep -E "Duration|Video:" || true
ls -la demo.gif demo.mp4
