"""Task 2: classical egg mask. No neural net.

Inputs: a BGR image and configs/classical_final.yaml.
Output: a uint8 mask, 255 on egg pixels.
Run: python src/task2_classical.py --save-all
     python src/task2_classical.py --score-dev
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Excluded from scoring. Masks are still written for every image.
SCORE_SKIP_STEMS = (
    "not_damaged_12_jpg",
    "not_damaged_39_jpg",
    "not_damaged_145_jpg",
)


def load_cfg(path: Path | None = None) -> dict:
    cfg_path = path or (REPO / "configs/classical_final.yaml")
    with cfg_path.open() as handle:
        return yaml.safe_load(handle)


# Same numbers as configs/classical_final.yaml. The file wins when it is present.
# Raise white_v_min to ignore dim paper. Lower white_s_max to ignore skin.
# Set brown_on true to add a brown shell range (this lowered dev mIoU).
# open_k removes specks. close_k fills small holes. blur smooths before the gate.
CONFIG = {
    "mode": "hsv",
    "white_s_max": 60,
    "white_v_min": 170,
    "brown_on": False,
    "brown_h": [5, 30],
    "brown_s": [40, 220],
    "brown_v": [40, 220],
    "blur": 5,
    "open_k": 5,
    "close_k": 11,
    "min_area_frac": 0,
    "max_area_frac": 1,
    "min_circularity": 0,
    "ellipse_fill": False,
    "drop_border_frac": None,
    "fill_holes": False,
}


def to_hsv(bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """Blur in HSV so the colour gate is less speckled."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    kernel = int(cfg.get("blur") or 0)
    if kernel >= 3:
        hsv = cv2.GaussianBlur(hsv, (kernel, kernel), 0)
    return hsv


def white_mask(hsv: np.ndarray, cfg: dict) -> np.ndarray:
    """Eggs in this set are white: low saturation, high value."""
    mask = cv2.inRange(
        hsv,
        (0, 0, int(cfg["white_v_min"])),
        (180, int(cfg["white_s_max"]), 255),
    )
    if cfg.get("brown_on"):
        hue, sat, val = cfg["brown_h"], cfg["brown_s"], cfg["brown_v"]
        brown = cv2.inRange(
            hsv,
            (int(hue[0]), int(sat[0]), int(val[0])),
            (int(hue[1]), int(sat[1]), int(val[1])),
        )
        mask = cv2.bitwise_or(mask, brown)
    return mask


def clean_mask(mask: np.ndarray, cfg: dict) -> np.ndarray:
    """Open drops specks. Close joins broken shell pixels."""
    open_k = int(cfg.get("open_k") or 0)
    close_k = int(cfg.get("close_k") or 0)
    if open_k >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if close_k >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if cfg.get("fill_holes"):
        mask = (ndimage.binary_fill_holes(mask > 0).astype(np.uint8)) * 255
    return mask


def filter_blobs(mask: np.ndarray, cfg: dict) -> np.ndarray:
    """Optional area, circularity, border, and ellipse steps. T10 turns them off."""
    height, width = mask.shape[:2]
    min_frac = float(cfg.get("min_area_frac") or 0)
    max_frac = float(cfg.get("max_area_frac") or 1)
    min_circ = float(cfg.get("min_circularity") or 0)
    border_max = cfg.get("drop_border_frac")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    kept = np.zeros((height, width), np.uint8)
    border = np.zeros((height, width), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    border_count = float(border.sum())
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        frac = area / float(height * width)
        if frac < min_frac or frac > max_frac:
            continue
        component = labels == index
        if min_circ > 0:
            rows, cols = np.nonzero(component)
            if len(cols) < 5:
                continue
            contours, _ = cv2.findContours(
                component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not contours:
                continue
            perimeter = cv2.arcLength(max(contours, key=cv2.contourArea), True)
            circularity = 0.0 if perimeter <= 0 else float(4 * np.pi * area / (perimeter * perimeter))
            if circularity < min_circ:
                continue
        if border_max is not None and float((component & border).sum()) / border_count > float(border_max):
            continue
        if cfg.get("ellipse_fill"):
            contours, _ = cv2.findContours(
                component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contour = max(contours, key=cv2.contourArea) if contours else None
            if contour is not None and len(contour) >= 5:
                (center, axes, angle) = cv2.fitEllipse(contour)
                radians = np.deg2rad(float(angle))
                cos_a, sin_a = np.cos(radians), np.sin(radians)
                yy, xx = np.ogrid[:height, :width]
                x_shift = xx - float(center[0])
                y_shift = yy - float(center[1])
                x_rot = cos_a * x_shift + sin_a * y_shift
                y_rot = -sin_a * x_shift + cos_a * y_shift
                major = max(float(axes[0]), 1.0) / 2.0
                minor = max(float(axes[1]), 1.0) / 2.0
                kept[(x_rot / major) ** 2 + (y_rot / minor) ** 2 <= 1.0] = 255
                continue
        kept[component] = 255
    return kept


def gray_otsu(bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """Trial T01 only. Kept so the experiment script can still call mode=gray_otsu."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = int(cfg.get("blur") or 0)
    if kernel >= 3:
        gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if gray[mask > 0].mean() < gray[mask == 0].mean():
        mask = cv2.bitwise_not(mask)
    return mask


def segment_eggs(bgr: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    """Return a uint8 mask, 255 on egg pixels."""
    if cfg is None:
        cfg = load_cfg()
    if cfg.get("mode") == "gray_otsu":
        mask = gray_otsu(bgr, cfg)
    else:
        mask = white_mask(to_hsv(bgr, cfg), cfg)
    mask = clean_mask(mask, cfg)
    return filter_blobs(mask, cfg)


def _score_split(cfg: dict, split_file: Path) -> dict:
    from src.metrics import iou_dice

    paths = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    rows = []
    ious, dices, empty = [], [], 0
    scored = 0
    for rel in paths:
        stem = Path(rel).stem
        if stem.startswith(SCORE_SKIP_STEMS):
            continue
        gt_path = REPO / "outputs/sam_masks" / f"{stem}_union.png"
        if not gt_path.is_file():
            continue
        bgr = cv2.imread(str(REPO / rel))
        pred = segment_eggs(bgr, cfg) > 0
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 0
        iou, dice = iou_dice(pred, gt)
        ious.append(iou)
        dices.append(dice)
        empty += int(pred.sum() == 0)
        scored += 1
        rows.append((rel, iou, dice, int(pred.sum() == 0)))
    return {
        "n": scored,
        "miou": float(np.mean(ious)) if ious else 0.0,
        "dice": float(np.mean(dices)) if dices else 0.0,
        "empty_pred_rate": (empty / scored) if scored else 0.0,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/classical_final.yaml")
    parser.add_argument("--score-dev", action="store_true")
    parser.add_argument("--save-all", action="store_true")
    args = parser.parse_args()
    cfg = load_cfg(REPO / args.cfg)
    if args.score_dev:
        result = _score_split(cfg, REPO / "data/splits/dev.txt")
        print(
            f"n={result['n']} mIoU={result['miou']:.4f} "
            f"Dice={result['dice']:.4f} empty={result['empty_pred_rate']:.4f}"
        )
    if args.save_all:
        out = REPO / "outputs/classical_masks"
        out.mkdir(parents=True, exist_ok=True)
        with (REPO / "data/manifest.csv").open() as handle:
            images = list(csv.DictReader(handle))
        for row in images:
            bgr = cv2.imread(str(REPO / row["path"]))
            mask = segment_eggs(bgr, cfg)
            cv2.imwrite(str(out / f"{row['image_id']}.png"), mask)
        print(f"wrote {len(images)} masks to {out}")


if __name__ == "__main__":
    main()
