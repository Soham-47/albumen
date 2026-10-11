# AI tool disclosure

Grok (xAI), through the Grok Build coding agent, drafted this repository with Soham. No model was trained. Numbers in the report are from scripts run on this machine, not from guessed dataset facts.

## What the agent wrote

- The work plan (sections, trial design, report outline).
- Repository scaffold, `requirements.txt`, and scripts under `scripts/`.
- `src/download_data.py`, `src/split.py`, `src/task1_sam.py`, `src/task2_classical.py`, `src/task3_eval.py`, `src/task4_track.py`, `src/metrics.py`, `src/viz.py`.
- `experiments/log.md` entries after each measured trial.
- `docs/report.md`, this file, `README.md`, and `docs/wellfound_reply.txt`.

## What was not invented by the model

- Image bytes from Google Drive id `1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv`.
- SAM ViT-B weights from `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`. Masks come from the installed `segment-anything` package on CPU.
- The Task 4 source clip from `https://www.youtube.com/watch?v=Y4JyV94y2_A` via yt-dlp.
- Dev and eval scores in `outputs/figures/metrics_summary.json`.
- Tracking counts in `outputs/video/track_summary.json`.

## What a person still has to do

Soham reviews the report, uploads `video/tracked_full.mp4` to Google Drive (anyone with the link), pastes that URL over the placeholder in `docs/report.md` and `docs/video_drive_url.txt`, and rebuilds the PDF if he wants the live link inside it.
