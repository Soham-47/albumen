"""Task 1: SAM ViT-B automatic masks, then an egg filter. CPU only.

Input: images in data/manifest.csv and checkpoints/sam_vit_b_01ec64.pth.
Output: outputs/sam_masks/<stem>_union.png (the mask Task 2 and Task 3 score against).
Run: python src/task1_sam.py
Raw proposals are cached under outputs/sam_cache/ so a rerun skips SAM.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]

# Documented filter. Tuned after looking at the 640x640 egg photos:
# eggs are compact ovals, not the full frame, and usually brighter or more
# saturated than the table.
AREA_MIN = 0.02
AREA_MAX = 0.55
SOLIDITY_MIN = 0.90
ASPECT_MAX = 3.2
ELLIPSE_AREA_RATIO = (0.75, 1.35)
BORDER_TOUCH_MAX = 0.35
VALUE_DELTA_MIN = 12.0
MERGE_IOU = 0.50


def load_manifest() -> list[dict]:
    with (REPO / "data/manifest.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def mask_stats(mask: np.ndarray, bgr: np.ndarray) -> dict | None:
    h, w = mask.shape
    area = int(mask.sum())
    frac = area / float(h * w)
    if frac < AREA_MIN or frac > AREA_MAX:
        return None
    u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull)) or 1.0
    solidity = float(cv2.contourArea(contour)) / hull_area
    if solidity < SOLIDITY_MIN:
        return None
    if len(contour) < 5:
        return None
    try:
        (center, axes, angle) = cv2.fitEllipse(contour)
    except cv2.error:
        return None
    major, minor = float(axes[0]), float(axes[1])
    if min(major, minor) <= 1:
        return None
    aspect = max(major, minor) / min(major, minor)
    if aspect > ASPECT_MAX:
        return None
    ang = np.deg2rad(float(angle))
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    yy, xx = np.ogrid[:h, :w]
    xc = xx - float(center[0])
    yc = yy - float(center[1])
    xr = cos_a * xc + sin_a * yc
    yr = -sin_a * xc + cos_a * yc
    a = max(major, 1.0) / 2.0
    b = max(minor, 1.0) / 2.0
    ell = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
    ell_area = float(ell.sum()) or 1.0
    ratio = area / ell_area
    if not (ELLIPSE_AREA_RATIO[0] <= ratio <= ELLIPSE_AREA_RATIO[1]):
        return None
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    touch = float((mask & border).sum()) / float(border.sum())
    if touch > BORDER_TOUCH_MAX:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    inside = hsv[:, :, 2][mask]
    outside = hsv[:, :, 2][~mask]
    if inside.size == 0 or outside.size == 0:
        return None
    delta = float(abs(inside.mean() - outside.mean()))
    sat = float(hsv[:, :, 1][mask].mean())
    val = float(inside.mean())
    # A white egg on a bright bowl can have a small value gap. Keep it when
    # the interior is clearly white (high V, low saturation).
    white_shell = val >= 165.0 and sat <= 40.0
    if delta < VALUE_DELTA_MIN and not white_shell:
        return None
    return {
        "area": area,
        "frac": round(frac, 4),
        "solidity": round(solidity, 3),
        "aspect": round(aspect, 3),
        "ellipse_ratio": round(ratio, 3),
        "border_touch": round(touch, 3),
        "value_delta": round(delta, 2),
        "saturation": round(sat, 2),
        "value": round(val, 2),
    }


def suppress_halos(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Drop a larger mask that mostly contains a tighter egg mask.

    SAM often returns the egg and a wider outline (shadow, water, or the
    table around it). Intersection-over-union misses that pair because the
    outer mask is much bigger. If the smaller mask sits inside the larger
    one, keep the smaller one.
    """
    drop: set[int] = set()
    order = sorted(range(len(masks)), key=lambda i: int(masks[i].sum()))
    for pos, small_i in enumerate(order):
        small = masks[small_i]
        small_area = int(small.sum()) or 1
        for large_i in order[pos + 1 :]:
            if large_i in drop:
                continue
            large = masks[large_i]
            inter = int(np.logical_and(small, large).sum())
            if inter / small_area >= 0.75 and int(large.sum()) >= 1.35 * small_area:
                drop.add(large_i)
    return [mask for i, mask in enumerate(masks) if i not in drop]


def merge_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    kept: list[np.ndarray] = []
    order = sorted(range(len(masks)), key=lambda i: int(masks[i].sum()), reverse=True)
    for index in order:
        mask = masks[index]
        replaced = False
        for k, prev in enumerate(kept):
            inter = np.logical_and(mask, prev).sum()
            union = np.logical_or(mask, prev).sum()
            iou = inter / union if union else 0
            contained = inter / mask.sum() if mask.sum() else 0
            if iou >= MERGE_IOU or contained >= 0.7:
                if mask.sum() > prev.sum():
                    kept[k] = mask
                replaced = True
                break
        if not replaced:
            kept.append(mask)
    return kept


def sam_proposals(image_rgb: np.ndarray, generator) -> list[np.ndarray]:
    records = generator.generate(image_rgb)
    return [rec["segmentation"].astype(bool) for rec in records]


def colour_seed_boxes(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Light shell or brown/yellow shell against a darker or greener table.
    light = cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))
    warm = cv2.inRange(hsv, (8, 40, 60), (35, 255, 255))
    blob = cv2.bitwise_or(light, warm)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, kernel)
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blob)
    boxes = []
    h, w = blob.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.008 * h * w or area > 0.55 * h * w:
            continue
        boxes.append((int(x), int(y), int(x + bw), int(y + bh)))
    return boxes


def predictor_fallback(image_rgb: np.ndarray, bgr: np.ndarray, predictor) -> list[np.ndarray]:
    boxes = colour_seed_boxes(bgr)
    if not boxes:
        return []
    predictor.set_image(image_rgb)
    masks = []
    for box in boxes:
        box_np = np.array(box, dtype=np.float32)
        pred, scores, _ = predictor.predict(box=box_np, multimask_output=True)
        best = int(np.argmax(scores))
        masks.append(pred[best].astype(bool))
    return masks


def save_outputs(stem: str, masks: list[np.ndarray], out_dir: Path) -> None:
    h, w = masks[0].shape if masks else (640, 640)
    inst = np.zeros((h, w), np.uint16)
    for i, mask in enumerate(masks, start=1):
        inst[mask] = i
    union = (inst > 0).astype(np.uint8) * 255
    Image.fromarray(inst).save(out_dir / f"{stem}_instances.png")
    Image.fromarray(union).save(out_dir / f"{stem}_union.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/sam_vit_b_01ec64.pth")
    args = parser.parse_args()

    import torch
    from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=str(REPO / args.checkpoint))
    sam.to(device)
    # points_per_batch is a memory knob, not a quality knob. 64 OOMs this 16 GB box.
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        points_per_batch=8,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.90,
        min_mask_region_area=400,
    )
    predictor = SamPredictor(sam)

    cache_dir = REPO / "outputs/sam_cache"
    out_dir = REPO / "outputs/sam_masks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest()
    if args.limit:
        rows = rows[: args.limit]
    summary = []
    times = []
    for index, row in enumerate(rows, start=1):
        path = REPO / row["path"]
        stem = row["image_id"]
        cache = cache_dir / f"{stem}.npz"
        bgr = cv2.imread(str(path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        if cache.is_file():
            packed = np.load(cache)["masks"]
            proposals = [packed[i].astype(bool) for i in range(packed.shape[0])] if packed.size else []
            cached = True
        else:
            proposals = sam_proposals(rgb, generator)
            stacked = np.stack(proposals).astype(np.uint8) if proposals else np.zeros((0, *rgb.shape[:2]), np.uint8)
            np.savez_compressed(cache, masks=stacked)
            del stacked
            cached = False
        stats = []
        kept = []
        for mask in proposals:
            info = mask_stats(mask, bgr)
            if info:
                kept.append(mask)
                stats.append(info)
        kept = suppress_halos(merge_masks(kept))
        method = "automatic"
        if not kept:
            fb = predictor_fallback(rgb, bgr, predictor)
            fb_kept = []
            stats = []
            for mask in fb:
                info = mask_stats(mask, bgr)
                if info:
                    fb_kept.append(mask)
                    stats.append(info)
            if fb_kept:
                kept = suppress_halos(merge_masks(fb_kept))
                method = "fallback_box"
            else:
                method = "none"
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if kept:
            save_outputs(stem, kept, out_dir)
        else:
            h, w = bgr.shape[:2]
            Image.fromarray(np.zeros((h, w), np.uint16)).save(out_dir / f"{stem}_instances.png")
            Image.fromarray(np.zeros((h, w), np.uint8)).save(out_dir / f"{stem}_union.png")
        unlabeled = row.get("folder_label") == "Unlabeled"
        summary.append(
            {
                "image": row["path"],
                "image_id": stem,
                "n_eggs": len(kept),
                "areas": ";".join(str(int(m.sum())) for m in kept),
                "method": method,
                "unlabeled": int(unlabeled),
                "n_proposals": len(proposals),
                "cached": int(cached),
                "seconds": round(elapsed, 2),
                "folder_label": row.get("folder_label", ""),
            }
        )
        print(
            f"[{index}/{len(rows)}] {stem} eggs={len(kept)} method={method} "
            f"proposals={len(proposals)} cached={cached} sec={elapsed:.1f} unlabeled={unlabeled}",
            flush=True,
        )
        del proposals, kept, bgr, rgb
        gc.collect()

    fields = list(summary[0].keys()) if summary else []
    with (out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    meta = {
        "n_images": len(summary),
        "mean_seconds": round(float(np.mean(times)), 2) if times else None,
        "median_seconds": round(float(np.median(times)), 2) if times else None,
        "thresholds": {
            "area_frac": [AREA_MIN, AREA_MAX],
            "solidity_min": SOLIDITY_MIN,
            "aspect_max": ASPECT_MAX,
            "ellipse_area_ratio": list(ELLIPSE_AREA_RATIO),
            "border_touch_max": BORDER_TOUCH_MAX,
            "value_delta_min": VALUE_DELTA_MIN,
            "merge_iou": MERGE_IOU,
            "points_per_side": 16,
            "points_per_batch": 8,
        },
    }
    (out_dir / "filter.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
