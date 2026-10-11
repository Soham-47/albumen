"""Semantic mask scores: IoU, Dice, precision, recall."""

from __future__ import annotations

import numpy as np


def binary_counts(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, int]:
    p = pred.astype(bool)
    g = gt.astype(bool)
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, np.logical_not(g)).sum())
    fn = int(np.logical_and(np.logical_not(p), g).sum())
    union = int(np.logical_or(p, g).sum())
    return tp, fp, fn, union


def scores(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Per-image semantic scores. Both empty counts as a perfect match."""
    tp, fp, fn, union = binary_counts(pred, gt)
    if union == 0:
        return {"iou": 1.0, "dice": 1.0, "precision": 1.0, "recall": 1.0}
    iou = tp / union
    denom = (2 * tp) + fp + fn
    dice = (2 * tp) / denom if denom else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
    }


def iou_dice(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    row = scores(pred, gt)
    return row["iou"], row["dice"]


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    # Population std matches a closed set of images, not a sample estimate.
    std = float(arr.std(ddof=0)) if arr.size else float("nan")
    return {"mean": float(arr.mean()), "std": std, "n": int(arr.size)}
