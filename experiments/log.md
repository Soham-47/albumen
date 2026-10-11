# Experiment log

Phase 1 does not run classical trials. Append Task 2 trials below this line.

## Task 1 SAM (Phase 2)

Not a classical trial. Generator: ViT-B `sam_vit_b_01ec64.pth`, CPU, `points_per_side=16`, `pred_iou_thresh=0.86`, `stability_score_thresh=0.90`, `min_mask_region_area=400`. The pass that wrote `outputs/sam_cache/` used `points_per_batch=64`. Mean time on that pass: 46.07 s/image, median 42.49 s (110 images).

Egg filter: area fraction `[0.02, 0.55]`, solidity `>= 0.90`, ellipse aspect `<= 3.2`, mask-area / ellipse-area in `[0.75, 1.35]`, border-pixel fraction `<= 0.35`, absolute HSV value gap vs background `>= 12` unless the interior is white (`V >= 165` and `S <= 40`). Overlapping proposals merge at IoU `0.50`. A larger mask that contains a tighter mask (intersection / smaller `>= 0.75` and area ratio `>= 1.35`) is dropped. If nothing survives, a colour-threshold box is sent to `SamPredictor`.

Measured after that filter: 107 automatic, 1 box fallback, 2 empty. Egg-count histogram: 0→2, 1→55, 2→24, 3→17, 4→8, 5→2, 7→2. The two unlabeled images are flagged in `summary.csv`.
