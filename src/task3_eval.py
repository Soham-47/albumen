"""Task 3: score the frozen classical masks against SAM union masks.

Eval is the primary split. Dev is reported beside it. The classical recipe
is not retuned here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import mean_std, scores  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# Empty SAM GT or not an egg. Same stems as the Phase 3 scorer.
EXCLUDE_PREFIXES = (
    "not_damaged_12_jpg",
    "not_damaged_39_jpg",
    "not_damaged_145_jpg",
)
MIN_STRATUM = 3


def _stem(path: Path) -> str:
    name = path.name
    if name.endswith("_union.png"):
        name = name[: -len("_union.png")]
    else:
        name = path.stem
    return name


def _excluded(stem: str) -> bool:
    return any(stem.startswith(prefix) for prefix in EXCLUDE_PREFIXES)


def _load_split(path: Path) -> list[Path]:
    lines = path.read_text().splitlines()
    return [REPO / line.strip() for line in lines if line.strip()]


def _gt_hue_sat(bgr: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv[gt > 0]
    if pixels.size == 0:
        return float("nan"), float("nan")
    return float(pixels[:, 0].mean()), float(pixels[:, 1].mean())


def _tertiles(areas: list[float]) -> list[str]:
    if len(areas) < MIN_STRATUM:
        return ["skipped"] * len(areas)
    edges = np.quantile(np.asarray(areas, dtype=np.float64), [1 / 3, 2 / 3])
    labels = []
    for area in areas:
        if area <= edges[0]:
            labels.append("small")
        elif area <= edges[1]:
            labels.append("medium")
        else:
            labels.append("large")
    return labels


def _aggregate(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for key in ("iou", "dice", "precision", "recall"):
        out[key] = mean_std([row[key] for row in rows])
    return out


def _by_label(rows: list[dict], field: str) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row[field], []).append(row)
    result = {}
    for label, group in sorted(grouped.items()):
        if label == "skipped" or len(group) < MIN_STRATUM:
            result[label] = {"n": len(group), "skipped": True}
            continue
        result[label] = _aggregate(group)
    return result


def _failure_panel(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    panel = image.copy()
    fp = np.logical_and(pred > 0, gt == 0)
    fn = np.logical_and(pred == 0, gt > 0)
    panel[fp] = (0.45 * panel[fp] + np.array([0, 0, 180])).astype(np.uint8)
    panel[fn] = (0.45 * panel[fn] + np.array([180, 0, 0])).astype(np.uint8)
    return panel


def _save_hist(eval_rows: list[dict], path: Path) -> None:
    values = [row["iou"] for row in eval_rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=10, range=(0, 1), color="#4c78a8", edgecolor="white")
    ax.set_xlabel("IoU")
    ax.set_ylabel("Eval images")
    ax.set_title(f"Eval IoU (n={len(values)})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_grid(worst: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(len(worst), 4, figsize=(12, 2.2 * len(worst)))
    if len(worst) == 1:
        axes = np.expand_dims(axes, 0)
    headers = ("image", "SAM GT", "classical", "FP red / FN blue")
    for col, title in enumerate(headers):
        axes[0, col].set_title(title, fontsize=9)
    for row_i, row in enumerate(worst):
        bgr = cv2.imread(str(row["image"]))
        gt = cv2.imread(str(row["gt"]), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(str(row["pred"]), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt_rgb = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
        pred_rgb = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
        err = cv2.cvtColor(_failure_panel(bgr, gt, pred), cv2.COLOR_BGR2RGB)
        for col, img in enumerate((rgb, gt_rgb, pred_rgb, err)):
            axes[row_i, col].imshow(img)
            axes[row_i, col].axis("off")
        axes[row_i, 0].set_ylabel(
            f"{row['stem'][:22]}\nIoU {row['iou']:.3f}",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _notes(summary: dict, worst: list[dict]) -> str:
    worst_lines = "\n".join(
        f"- `{row['stem']}` IoU {row['iou']:.3f}, GT area {row['gt_area']}, "
        f"mean hue {row['mean_hue']:.1f}, mean S {row['mean_sat']:.1f}, "
        f"colour label {row['colour']}."
        for row in worst
    )
    eval_iou = summary["eval"]["iou"]
    return f"""# Failure notes (Phase 4 draft)

Scored classical T10 against SAM union masks. Excluded from every score: `not_damaged_12` (empty SAM, egg under herbs), `not_damaged_39` (empty SAM, frame-cut on a printed lid), `not_damaged_145` (foam, not an egg). Masks for those three still exist on disk.

Eval mIoU is {eval_iou['mean']:.4f} (std {eval_iou['std']:.4f}, n={eval_iou['n']}).

Worst 8 eval images by IoU:

{worst_lines}

Themes:

- White clutter: the T10 gate is bright low-saturation pixels. White tables, paper, rice, and plastic pass the same test as a white shell, so precision stays well below recall.
- Missed brown eggs: the brown OR-gate was discarded in Phase 3 because it hurt dev mIoU. Brown shells are mostly false negatives under T10. The colour stratum uses mean saturation inside the SAM mask (S < 40 is white, otherwise brown), not hue alone, because white shells have an unstable hue.
- Touching eggs: Task 3 scores the union mask, so two eggs that touch and stay in one component are not an IoU error by themselves. Instance splits are a Task 4 issue.
- Specular holes: highlights on a shell can fall outside the value gate and leave holes. Close kernel 11 fills some of them and not all of them.
- Scale: T10 has no min-area cut. Tiny eggs are not removed by area. They are missed when they are darker than V 170.
- Domain gap vs the video: the stills are 640x640 Roboflow crops of eggs on tables and packaging. The video is 1920x1080 of eggs moving in a scene that was not in this set. The same white gate will pick up bright backgrounds there too.
- SAM mistakes: some false positives and false negatives versus SAM are SAM's own hands, keycaps, and white plastic. A classical miss on those pixels is not a missed egg.
"""


def evaluate(repo: Path) -> dict:
    sam_dir = repo / "outputs/sam_masks"
    pred_dir = repo / "outputs/classical_masks"
    fig_dir = repo / "outputs/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split: dict[str, list[dict]] = {"dev": [], "eval": []}
    excluded = []
    zero_pred = []
    for split in ("dev", "eval"):
        for image_path in _load_split(repo / "data/splits" / f"{split}.txt"):
            stem = image_path.stem
            gt_path = sam_dir / f"{stem}_union.png"
            pred_path = pred_dir / f"{stem}.png"
            if not gt_path.is_file() or not pred_path.is_file():
                raise FileNotFoundError(f"missing mask for {image_path}")
            if _excluded(stem):
                excluded.append({"stem": stem, "split": split, "image": str(image_path)})
                continue
            bgr = cv2.imread(str(image_path))
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
            if bgr is None or gt is None or pred is None:
                raise RuntimeError(f"unreadable {image_path}")
            if int((pred > 0).sum()) == 0:
                zero_pred.append(stem)
            row = scores(pred, gt)
            hue, sat = _gt_hue_sat(bgr, gt)
            row.update(
                {
                    "stem": stem,
                    "split": split,
                    "image": str(image_path.relative_to(repo)),
                    "gt": str(gt_path.relative_to(repo)),
                    "pred": str(pred_path.relative_to(repo)),
                    "gt_area": int((gt > 0).sum()),
                    "mean_hue": hue,
                    "mean_sat": sat,
                    "colour": "white" if sat < 40 else "brown",
                }
            )
            rows_by_split[split].append(row)

    if len(zero_pred) == len(rows_by_split["dev"]) + len(rows_by_split["eval"]):
        raise RuntimeError("all scored predictions are empty; recipe looks broken")

    for split, rows in rows_by_split.items():
        labels = _tertiles([row["gt_area"] for row in rows])
        for row, label in zip(rows, labels):
            row["size"] = label

    summary = {
        "recipe": "T10",
        "config": "configs/classical_final.yaml",
        "excluded": excluded,
        "exclude_reason": {
            "not_damaged_12_jpg": "empty SAM GT, egg under herbs",
            "not_damaged_39_jpg": "empty SAM GT, frame-cut on a printed lid",
            "not_damaged_145_jpg": "foam, not an egg",
        },
        "zero_pred_stems": zero_pred,
        "colour_rule": "white if mean saturation inside SAM GT < 40, else brown",
        "size_rule": "tertiles of GT area inside each split",
        "min_stratum": MIN_STRATUM,
        "eval": _aggregate(rows_by_split["eval"]),
        "dev": _aggregate(rows_by_split["dev"]),
        "eval_by_size": _by_label(rows_by_split["eval"], "size"),
        "dev_by_size": _by_label(rows_by_split["dev"], "size"),
        "eval_by_colour": _by_label(rows_by_split["eval"], "colour"),
        "dev_by_colour": _by_label(rows_by_split["dev"], "colour"),
    }
    (fig_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    eval_rows = sorted(rows_by_split["eval"], key=lambda row: row["iou"])
    _save_hist(rows_by_split["eval"], fig_dir / "iou_hist.png")
    worst = eval_rows[:8]
    _save_grid(worst, fig_dir / "failures_grid.png")
    (fig_dir / "failure_notes.md").write_text(_notes(summary, worst))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()
    summary = evaluate(args.repo)
    for split in ("eval", "dev"):
        block = summary[split]
        print(
            split,
            "n",
            block["n"],
            "iou",
            f"{block['iou']['mean']:.4f}+-{block['iou']['std']:.4f}",
            "dice",
            f"{block['dice']['mean']:.4f}+-{block['dice']['std']:.4f}",
            "precision",
            f"{block['precision']['mean']:.4f}+-{block['precision']['std']:.4f}",
            "recall",
            f"{block['recall']['mean']:.4f}+-{block['recall']['std']:.4f}",
        )


if __name__ == "__main__":
    main()
