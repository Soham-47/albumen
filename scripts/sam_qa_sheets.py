"""Draw SAM union outlines on every image, 12 per page."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    summary = list(csv.DictReader((REPO / "outputs/sam_masks/summary.csv").open()))
    out = REPO / "outputs/sam_qa"
    out.mkdir(parents=True, exist_ok=True)
    page = 0
    for start in range(0, len(summary), 12):
        chunk = summary[start : start + 12]
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()
        for axis, row in zip(axes, chunk):
            bgr = cv2.imread(str(REPO / row["image"]))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            stem = row["image_id"]
            union = np.array(cv2.imread(str(REPO / "outputs/sam_masks" / f"{stem}_union.png"), cv2.IMREAD_GRAYSCALE))
            overlay = rgb.copy()
            contours, _ = cv2.findContours((union > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 40, 40), 2)
            axis.imshow(overlay)
            axis.set_title(f"{row['n_eggs']} {row['method']}\n{Path(row['image']).parent.name}", fontsize=8)
            axis.axis("off")
        for axis in axes[len(chunk) :]:
            axis.axis("off")
        page += 1
        fig.tight_layout()
        fig.savefig(out / f"page_{page:02d}.png", dpi=80)
        plt.close(fig)
        print(out / f"page_{page:02d}.png")


if __name__ == "__main__":
    main()
