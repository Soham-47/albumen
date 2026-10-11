"""IoU and Dice for binary masks."""

from __future__ import annotations

import numpy as np


def iou_dice(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    if union == 0:
        return 1.0, 1.0
    iou = inter / union
    dice = (2 * inter) / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 1.0
    return float(iou), float(dice)
