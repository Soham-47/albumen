# AI tool disclosure

## Phase 1 (11 Oct 2026)

Grok (Grok Build) wrote the repository scaffold, `src/download_data.py`, `src/split.py`, and the scripts under `scripts/`. The agent then ran those scripts on this machine.

Dataset bytes came from the Google Drive file id in the plan (`1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv`), or from the Roboflow Universe page if that download failed. The SAM ViT-B checkpoint came from `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`. The Task 4 video came from `https://www.youtube.com/watch?v=Y4JyV94y2_A` via yt-dlp.

No SAM masks, classical segmenter, or metrics were produced in this phase.

## Phase 2 (11 Oct 2026)

SAM masks were produced by the installed `segment-anything` package and the ViT-B checkpoint above. No new model was trained. Filter thresholds were chosen after looking at the QA contact sheets.
