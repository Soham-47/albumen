#!/usr/bin/env bash
# Phase 1 only: env, Drive ingest, manifest, leakage-safe split, sanity, video.
# Does not run Task 1 SAM.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash scripts/setup_env.sh
# shellcheck disable=SC1091
source .venv/bin/activate
python src/download_data.py --drive-id 1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv --out data/raw
python src/split.py --manifest data/manifest.csv --out data/splits --ratio 0.8 --seed 42
python scripts/phase1_sanity.py
bash scripts/download_video.sh
