# nosh-egg-cv

Nosh Robotics computer-vision intern assignment. This checkout is **Phase 1 only**: repo layout, environment, dataset ingest, sanity checks, and the Task 4 source video. Task 1 SAM has not been run.

## Reproduce Phase 1

```bash
bash scripts/run_phase1.sh
```

That script matches the Phase 1 commands from the plan:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py --drive-id 1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv --out data/raw
python src/split.py --manifest data/manifest.csv --out data/splits --ratio 0.8 --seed 42
```

`scripts/run_all.sh` is a stub. It runs Phase 1 and stops before SAM.

## What is downloaded

- Dataset zip: Google Drive id `1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv` into `data/raw/`.
- SAM ViT-B: `checkpoints/sam_vit_b_01ec64.pth` (~358 MB on disk).
- Video: `https://www.youtube.com/watch?v=Y4JyV94y2_A` to `data/video/source.mp4` (`video/source.mp4` is a symlink).

`data/raw/`, `checkpoints/`, `outputs/`, and the mp4 files are gitignored. `data/manifest.csv` and `data/splits/` are tracked.

## Environment

Python 3.12.3 venv at `.venv`. Measured on this machine after Phase 1:

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

`torch.cuda.is_available()` is false. `nvidia-smi` failed with `Failed to initialize NVML: Driver/library version mismatch` (NVML library 595.99). PyTorch is the CPU wheel from `https://download.pytorch.org/whl/cpu`. `requirements.txt` pins the other packages. Torch is installed by `scripts/setup_env.sh` before that file, because the `+cpu` builds are not on PyPI.

## What Phase 1 found

These numbers come from the Drive zip, not from the Universe page.

- 110 readable JPEGs. Every file is 640×640. No corrupt files. No exact SHA-1 duplicates.
- Folder layout in the zip: `train/egg` 75, `train/Unlabeled` 2, `valid/egg` 22, `test/egg` 11. The Roboflow readme says annotation is folder format. No per-image mask or box file was in the zip.
- `README.roboflow.txt` says the export was 23 May 2025, resize stretch to 640×640, no augmentation, license note in `README.dataset.txt` (CC BY 4.0). Copies are in `data/`.
- Perceptual-hash groups (Hamming ≤ 8, seed 42): 109 groups, dev 87 groups / 87 images, eval 22 groups / 23 images. The only pair inside Hamming 8 is in eval, and it crosses the Roboflow train/test folders: `data/raw/train/egg/not_damaged_27_jpg.rf.9eabbf76d6ac2d4caccb38b11e0d28ad.jpg` and `data/raw/test/egg/not_damaged_78_jpg.rf.d3a6e8624d192abc58ff7df9d5781bb5.jpg`.
- Contact sheet (seed 42, 5 images): `outputs/phase1_samples.png`.
- Video `data/video/source.mp4`: 1920×1080, 24 fps, 60.07 s, 4.25 MB. `video/source.mp4` is a symlink to that file. Format chosen by yt-dlp was av1 1080p (`399`) plus m4a audio (`140`). yt-dlp needed `yt-dlp-ejs` and Node to solve the YouTube JS challenge. The system yt-dlp (2024.04.09) could not download it.

## Assignment PDF

`docs/assignment.pdf` is a copy of `~/Downloads/CV Intern Task Wellfound.pdf` (same bytes as the `(1)` download). The plan path `/workspace/nosh_assignment.pdf` does not exist on this machine.

## Later phases

Do not start Task 1 until asked. The entry point will be:

```bash
python src/task1_sam.py --images data/raw --out data/sam_masks --checkpoint checkpoints/sam_vit_b_01ec64.pth
```
