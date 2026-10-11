"""Classical egg mask. One function, knobs in a dict or YAML file.

OpenCV, numpy, and scipy only. No neural net and no grabCut.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]

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


def segment_eggs(bgr: np.ndarray, cfg: dict | None = None) -> np.ndarray:
    """Return a uint8 mask, 255 on egg pixels."""
    if cfg is None:
        cfg = load_cfg()
    h, w = bgr.shape[:2]
    if cfg.get("mode") == "gray_otsu":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        k = int(cfg.get("blur") or 0)
        if k >= 3:
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if gray[mask > 0].mean() < gray[mask == 0].mean():
            mask = cv2.bitwise_not(mask)
    else:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        k = int(cfg.get("blur") or 0)
        if k >= 3:
            hsv = cv2.GaussianBlur(hsv, (k, k), 0)
        white = cv2.inRange(
            hsv,
            (0, 0, int(cfg["white_v_min"])),
            (180, int(cfg["white_s_max"]), 255),
        )
        mask = white
        if cfg.get("brown_on"):
            bh, bs, bv = cfg["brown_h"], cfg["brown_s"], cfg["brown_v"]
            brown = cv2.inRange(
                hsv,
                (int(bh[0]), int(bs[0]), int(bv[0])),
                (int(bh[1]), int(bs[1]), int(bv[1])),
            )
            mask = cv2.bitwise_or(white, brown)
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

    min_frac = float(cfg.get("min_area_frac") or 0)
    max_frac = float(cfg.get("max_area_frac") or 1)
    min_circ = float(cfg.get("min_circularity") or 0)
    border_max = cfg.get("drop_border_frac")
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    kept = np.zeros((h, w), np.uint8)
    border = np.zeros((h, w), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    border_n = float(border.sum())
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        frac = area / float(h * w)
        if frac < min_frac or frac > max_frac:
            continue
        comp = labels == i
        if min_circ > 0:
            ys, xs = np.nonzero(comp)
            if len(xs) < 5:
                continue
            contour = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
            # Perimeter from the component contour, not the raw pixel list.
            cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            per = cv2.arcLength(max(cnts, key=cv2.contourArea), True)
            circ = 0.0 if per <= 0 else float(4 * np.pi * area / (per * per))
            if circ < min_circ:
                continue
        if border_max is not None and float((comp & border).sum()) / border_n > float(border_max):
            continue
        if cfg.get("ellipse_fill"):
            cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt = max(cnts, key=cv2.contourArea) if cnts else None
            if cnt is not None and len(cnt) >= 5:
                (center, axes, angle) = cv2.fitEllipse(cnt)
                ang = np.deg2rad(float(angle))
                cos_a, sin_a = np.cos(ang), np.sin(ang)
                yy, xx = np.ogrid[:h, :w]
                xc = xx - float(center[0])
                yc = yy - float(center[1])
                xr = cos_a * xc + sin_a * yc
                yr = -sin_a * xc + cos_a * yc
                a = max(float(axes[0]), 1.0) / 2.0
                b = max(float(axes[1]), 1.0) / 2.0
                kept[(xr / a) ** 2 + (yr / b) ** 2 <= 1.0] = 255
                continue
        kept[comp] = 255
    return kept


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
