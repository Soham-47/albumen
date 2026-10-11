#!/usr/bin/env bash
# Download the Task 4 YouTube video once, then record fps, length, resolution.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p data/video outputs video
URL="https://www.youtube.com/watch?v=Y4JyV94y2_A"
OUT="data/video/source.mp4"

if [[ ! -x .venv/bin/yt-dlp ]]; then
  echo "missing .venv/bin/yt-dlp; create the venv and install requirements first" >&2
  exit 1
fi

# yt-dlp 2026 needs the EJS challenge solver plus a JS runtime (node).
# Pick the best mp4 video and m4a audio and merge them.
.venv/bin/yt-dlp --js-runtimes node \
  -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/mp4" \
  --merge-output-format mp4 \
  -o "$OUT" \
  "$URL"

ln -sfn ../data/video/source.mp4 video/source.mp4

ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration \
  -show_entries format=duration,size \
  -of json "$OUT" | tee outputs/video_probe.json

.venv/bin/python - <<'PY'
import json
from pathlib import Path
probe = json.loads(Path("outputs/video_probe.json").read_text())
stream = probe["streams"][0]
fmt = probe["format"]
num, den = stream.get("avg_frame_rate", "0/1").split("/")
fps = float(num) / float(den) if float(den) else None
info = {
    "path": "data/video/source.mp4",
    "width": int(stream["width"]),
    "height": int(stream["height"]),
    "fps": fps,
    "avg_frame_rate": stream.get("avg_frame_rate"),
    "duration_s": float(fmt.get("duration") or stream.get("duration") or 0),
    "size_bytes": int(fmt.get("size") or 0),
}
Path("outputs/video_info.json").write_text(json.dumps(info, indent=2) + "\n")
print(json.dumps(info, indent=2))
PY
