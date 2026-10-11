# nosh-egg-cv

Nosh Robotics computer-vision intern assignment. Task 2 is a classical egg segmenter (no neural net). SAM ViT-B supplies pseudo ground truth. Task 4 tracks eggs in one YouTube clip with that same segmenter.

Public repo: https://github.com/Soham-47/nosh-egg-cv

Written report: [docs/report.pdf](docs/report.pdf) (source: `docs/report.md`).

## One command

Once the venv, `data/raw/`, the SAM checkpoint, and `data/video/source.mp4` exist:

```bash
bash scripts/run_all.sh
```

That runs Task 1 (SAM, uses the cache), QA sheets, the frozen classical recipe, Task 3 metrics, and Task 4 tracking. It does not retune `configs/classical_final.yaml`.

First-time setup (Python 3.12):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
bash scripts/setup_env.sh
wget -O checkpoints/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
python src/download_data.py --drive-id 1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv --out data/raw
python src/split.py --manifest data/manifest.csv --out data/splits --ratio 0.8 --seed 42
bash scripts/download_video.sh
bash scripts/run_all.sh
```

`scripts/setup_env.sh` installs the CPU torch wheel from `https://download.pytorch.org/whl/cpu`, then `requirements.txt`.

## Environment measured here

Python 3.12.3, venv `.venv`.

| Package | Version |
|---|---|
| torch | 2.14.1+cpu |
| torchvision | 0.29.1+cpu |
| opencv-python-headless | 5.0.0 |
| pillow | 12.3.0 |
| imagehash | 4.3.2 |
| segment-anything | 1.0 |
| gdown | 6.4.2 |
| yt-dlp | 2026.08.19 |

`torch.cuda.is_available()` is false. `nvidia-smi` failed with a driver/library mismatch (kernel module 595.91, userspace 595.99). All timed work is CPU.

Rough runtimes on this machine:

- SAM ViT-B, 110 images, `points_per_side=16`: mean 46.1 s/image (cache in `outputs/sam_cache/`).
- Classical masks on the same 110 images: seconds, not minutes.
- Tracking: 1440 frames at 1920×1080, one classical pass per frame.

## What each step writes

| Step | Command | Output |
|---|---|---|
| SAM masks | `python src/task1_sam.py` | `outputs/sam_masks/` |
| SAM gallery | `python scripts/sam_qa_sheets.py` | `outputs/sam_qa/page_01.png` … `page_10.png` |
| Classical | `python src/task2_classical.py --cfg configs/classical_final.yaml --save-all --score-dev` | `outputs/classical_masks/`, `outputs/classical_dev_metrics.csv` |
| Metrics | `python src/task3_eval.py` | `outputs/figures/metrics_summary.json`, `iou_hist.png`, `failures_grid.png` |
| Video | `python src/task4_track.py` | `video/tracked_full.mp4`, `video/tracked_preview.mp4`, `docs/figures/track_frame_0*.jpg` |

The live function for a threshold change is `segment_eggs(bgr, cfg)` in `src/task2_classical.py`. The frozen recipe is T10 in `configs/classical_final.yaml` (white HSV plus blur, open, and close). Dev mIoU 0.384 (n=85). Eval mIoU 0.407, Dice 0.546 (n=22). Three images are excluded from scores: `not_damaged_12`, `not_damaged_39`, `not_damaged_145`.

`data/raw/`, `checkpoints/`, `outputs/`, and `video/*.mp4` are gitignored. Figures the report needs are copied under `docs/figures/`.

The full tracked video is not on Google Drive yet. See `docs/video_drive_url.txt`.

AI use is listed in `DISCLOSURE.md`.
