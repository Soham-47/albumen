# AI tool disclosure

I built Albumen with Grok (xAI) in Grok Build. I set the goal, the data, the split, and which recipe to keep. The agent wrote the code and the drafts, and I used it to run the trials and the write-up instead of retyping each step. No model was trained. Every score in the report comes from a script on this machine.

## What I decided

- Use the Roboflow egg photos and the public 1080p clip, and score classical masks against SAM unions because the zip has no boxes.
- Split near-duplicates with perceptual-hash Hamming distance ≤ 8, then a 4:1 cut, seed 42.
- Keep recipe T10 (white HSV, blur, open, close) after the logged trials. Brown, circularity, ellipse fill, and border drop lowered dev mIoU, so they stayed off.
- Report the measured dev and eval numbers, and the tracking counts from the full clip, without retuning on eval.
- Publish the repo as Albumen and state this split of work.

## What the agent drafted

- The repo scaffold, `requirements.txt`, and the scripts under `scripts/`.
- `src/download_data.py`, `src/split.py`, `src/task1_sam.py`, `src/task2_classical.py`, `src/task3_eval.py`, `src/task4_track.py`, `src/metrics.py`, and `src/viz.py`.
- `experiments/log.md` after each measured trial, plus `README.md`, `docs/report.md`, `docs/CODE_WALKTHROUGH.md`, and this file.
- The scoring and tracking runs that filled `outputs/figures/metrics_summary.json` and `outputs/video/track_summary.json`.

## What was not invented

- Image bytes from Google Drive id `1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv`.
- SAM ViT-B weights from `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`. Masks come from the installed `segment-anything` package on CPU.
- The source clip from `https://www.youtube.com/watch?v=Y4JyV94y2_A` via yt-dlp.
- Dev and eval scores, and the tracking counts, which are outputs of those runs.
