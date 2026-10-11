#!/usr/bin/env bash
# One command after Python 3.12, the venv, data/raw, and the SAM checkpoint exist.
# CPU path. Does not retune configs/classical_final.yaml.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -d .venv ]]; then
  bash scripts/setup_env.sh
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python src/task1_sam.py --checkpoint checkpoints/sam_vit_b_01ec64.pth
python scripts/sam_qa_sheets.py
python src/task2_classical.py --cfg configs/classical_final.yaml --save-all --score-dev
python src/task3_eval.py
python src/task4_track.py \
  --src data/video/source.mp4 \
  --full video/tracked_full.mp4 \
  --preview video/tracked_preview.mp4 \
  --frames docs/figures
echo "Pipeline finished. Report source is docs/report.md."
