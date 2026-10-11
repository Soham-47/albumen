"""Four-panel view of the frozen T10 egg mask.

Panels: original | HSV white mask | cleaned mask | final mask on the photo, with SAM in green.
Run: python scripts/demo.py
     python scripts/demo.py --image data/raw/test/egg/not_damaged_101_jpg.rf.479d12a2b41a5ec6d718dcdabd14b0c9.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.task2_classical import clean_mask, load_cfg, segment_eggs, to_hsv, white_mask  # noqa: E402


def _panel(image: np.ndarray, title: str) -> np.ndarray:
    canvas = image.copy()
    cv2.putText(canvas, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def build(image_path: Path, cfg: dict) -> np.ndarray:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise SystemExit(f"cannot read {image_path}")
    raw = white_mask(to_hsv(bgr, cfg), cfg)
    cleaned = clean_mask(raw, cfg)
    final = segment_eggs(bgr, cfg)
    overlay = bgr.copy()
    overlay[final > 0] = (0.45 * overlay[final > 0] + 0.55 * np.array([40, 180, 255])).astype(np.uint8)
    stem = image_path.stem
    sam_path = REPO / "outputs/sam_masks" / f"{stem}_union.png"
    if sam_path.is_file():
        sam = cv2.imread(str(sam_path), cv2.IMREAD_GRAYSCALE)
        if sam is not None:
            contours, _ = cv2.findContours((sam > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (80, 220, 80), 2)
    panels = [
        _panel(bgr, "original"),
        _panel(_to_bgr(raw), "HSV white"),
        _panel(_to_bgr(cleaned), "cleaned"),
        _panel(overlay, "final + SAM"),
    ]
    return np.concatenate(panels, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default="data/raw/test/egg/not_damaged_101_jpg.rf.479d12a2b41a5ec6d718dcdabd14b0c9.jpg",
    )
    parser.add_argument("--out", default="outputs/demo_panels.png")
    args = parser.parse_args()
    sheet = build(REPO / args.image, load_cfg())
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out} shape={sheet.shape[1]}x{sheet.shape[0]}")


if __name__ == "__main__":
    main()
