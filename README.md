# Albumen

Public repo: https://github.com/Soham-47/albumen

Albumen segments eggs in still photos and tracks them in video. The segmenter is classical image processing: an HSV white-shell gate, a small blur, and open/close morphology. No network is trained.

SAM ViT-B builds pseudo ground-truth masks. Those masks are the reference for IoU and Dice. A constant-velocity Kalman tracker with Hungarian matching follows eggs in one 1080p clip.

## One command

Once the venv, `data/raw/`, the SAM checkpoint, and `data/video/source.mp4` exist:

```bash
bash scripts/run_all.sh
```

That runs SAM (from cache when present), the SAM contact sheets, the frozen classical recipe, the metrics, and tracking. It does not retune `configs/classical_final.yaml`.

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

Photos are a Roboflow egg export (CC BY 4.0): 110 readable 640×640 JPEGs, folder labels only, no boxes. The video is a public YouTube clip saved as `data/video/source.mp4` (1920×1080, 24 fps, about 60 s). OpenCV cannot decode the AV1 file, so tracking reads frames through ffmpeg.

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
| Panel demo | `python scripts/demo.py` | `outputs/demo_panels.png` |

`segment_eggs(bgr, cfg)` in `src/task2_classical.py` is the function to edit when you change a threshold. The shipped recipe is T10 in `configs/classical_final.yaml` (white HSV, blur 5, ellipse open 5, ellipse close 11). Dev mIoU 0.384 (n=85). Eval mIoU 0.407, Dice 0.546 (n=22). Three images are excluded from scores: `not_damaged_12`, `not_damaged_39`, `not_damaged_145`.

Near-duplicate photos stay on the same side of the split. Groups use perceptual-hash Hamming distance ≤ 8, then a 4:1 cut with seed 42 (87 dev images, 23 eval images).

On the clip, the tracker reports 2 ids, 757 frames with at least one egg, and at most 2 eggs in a frame. A hidden track stays at its last position for up to 30 s and can rejoin a detection up to 1400 px away, so the egg that is moved across the table keeps its id. Video-only gates (area, circularity, aspect, and a lower-frame centroid cut) live at the top of `src/task4_track.py`. They are not copied into the still-image yaml.

`data/raw/`, `checkpoints/`, `outputs/`, and `video/*.mp4` are gitignored. Figures used by the write-up are under `docs/figures/`.

A short code map, including where each knob lives, is in [docs/CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md). The longer write-up is [docs/report.pdf](docs/report.pdf) (source: `docs/report.md`). Notes on model assistance are in `DISCLOSURE.md`.
