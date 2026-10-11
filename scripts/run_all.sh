#!/usr/bin/env bash
# Full pipeline is not implemented. Phase 1 stops before Task 1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash scripts/run_phase1.sh
echo "Phase 1 finished. Task 1 SAM is not started."
