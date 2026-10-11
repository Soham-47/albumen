"""Run the section 5.2 trial ladder on the frozen dev split and print metrics.

Does not write experiments/log.md. The numbers are copied into that log after the run.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.task2_classical import SCORE_SKIP_STEMS, _score_split, load_cfg  # noqa: E402

BASE = load_cfg()


def trial(name: str, **overrides):
    cfg = copy.deepcopy(BASE)
    cfg.update(overrides)
    result = _score_split(cfg, REPO / "data/splits/dev.txt")
    print(
        f"{name} n={result['n']} mIoU={result['miou']:.4f} "
        f"Dice={result['dice']:.4f} empty={result['empty_pred_rate']:.4f}"
    )
    return result


def main() -> None:
    print("skip", SCORE_SKIP_STEMS)
    # Ladder. Each call changes the running recipe relative to the previous keep.
    trial("T01_gray_otsu", mode="gray_otsu", brown_on=False, open_k=0, close_k=0,
          min_area_frac=0, max_area_frac=1, min_circularity=0, ellipse_fill=False,
          drop_border_frac=None, blur=0, fill_holes=False)
    trial("T02_hsv_white", mode="hsv", brown_on=False, white_s_max=60, white_v_min=170,
          open_k=0, close_k=0, min_area_frac=0, max_area_frac=1, min_circularity=0,
          ellipse_fill=False, drop_border_frac=None, blur=0, fill_holes=False)
    trial("T03_white_or_brown", mode="hsv", brown_on=True, white_s_max=60, white_v_min=170,
          open_k=0, close_k=0, min_area_frac=0, max_area_frac=1, min_circularity=0,
          ellipse_fill=False, drop_border_frac=None, blur=0, fill_holes=False)
    trial("T04_morph", mode="hsv", brown_on=True, white_s_max=60, white_v_min=170,
          open_k=5, close_k=11, blur=5, min_area_frac=0, max_area_frac=1, min_circularity=0,
          ellipse_fill=False, drop_border_frac=None, fill_holes=False)
    trial("T05_area_circ", mode="hsv", brown_on=True, white_s_max=60, white_v_min=170,
          open_k=5, close_k=11, blur=5, min_area_frac=0.015, max_area_frac=0.55,
          min_circularity=0.45, ellipse_fill=False, drop_border_frac=None, fill_holes=False)
    trial("T06_ellipse", mode="hsv", brown_on=True, white_s_max=60, white_v_min=170,
          open_k=5, close_k=11, blur=5, min_area_frac=0.015, max_area_frac=0.55,
          min_circularity=0.45, ellipse_fill=True, drop_border_frac=None, fill_holes=False)
    trial("T07_border", mode="hsv", brown_on=True, white_s_max=60, white_v_min=170,
          open_k=5, close_k=11, blur=5, min_area_frac=0.015, max_area_frac=0.55,
          min_circularity=0.45, ellipse_fill=True, drop_border_frac=0.25, fill_holes=False)
    trial("T08_wider_white", mode="hsv", brown_on=True, white_s_max=70, white_v_min=150,
          open_k=5, close_k=11, blur=5, min_area_frac=0.015, max_area_frac=0.55,
          min_circularity=0.45, ellipse_fill=True, drop_border_frac=0.25, fill_holes=False)
    trial("T09_fill_holes", mode="hsv", brown_on=True, white_s_max=70, white_v_min=150,
          open_k=5, close_k=11, blur=5, min_area_frac=0.015, max_area_frac=0.55,
          min_circularity=0.45, ellipse_fill=True, drop_border_frac=0.25, fill_holes=True)


if __name__ == "__main__":
    main()
