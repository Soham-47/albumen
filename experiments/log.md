# Experiment log

Phase 1 does not run classical trials. Append Task 2 trials below this line.

## Task 1 SAM (Phase 2)

Not a classical trial. Generator: ViT-B `sam_vit_b_01ec64.pth`, CPU, `points_per_side=16`, `pred_iou_thresh=0.86`, `stability_score_thresh=0.90`, `min_mask_region_area=400`. The pass that wrote `outputs/sam_cache/` used `points_per_batch=64`. Mean time on that pass: 46.07 s/image, median 42.49 s (110 images).

Egg filter: area fraction `[0.02, 0.55]`, solidity `>= 0.90`, ellipse aspect `<= 3.2`, mask-area / ellipse-area in `[0.75, 1.35]`, border-pixel fraction `<= 0.35`, absolute HSV value gap vs background `>= 12` unless the interior is white (`V >= 165` and `S <= 40`). Overlapping proposals merge at IoU `0.50`. A larger mask that contains a tighter mask (intersection / smaller `>= 0.75` and area ratio `>= 1.35`) is dropped. If nothing survives, a colour-threshold box is sent to `SamPredictor`.

Measured after that filter: 107 automatic, 1 box fallback, 2 empty. Egg-count histogram: 0→2, 1→55, 2→24, 3→17, 4→8, 5→2, 7→2. The two unlabeled images are flagged in `summary.csv`.

## Task 2 classical (Phase 3)

Scoring uses the frozen `data/splits/dev.txt` only. Three images are excluded from the score, not from mask export: `not_damaged_12` (egg under herbs, empty SAM mask), `not_damaged_39` (egg cut by the frame on a printed lid, empty SAM mask), and `not_damaged_145` (Unlabeled foam, not an egg). That leaves 85 scored dev images. Eval metrics are not computed.

SAM union masks still contain hands, keycaps, and white plastic. A classical miss on those pixels is expected and is not treated as a classical failure by itself.

Leakage check: dev and eval path sets are disjoint (87 and 23 images). Group ids are disjoint. Mean perceptual-hash Hamming distance across the cut is 30.941 over 2001 pairs. Minimum cross-cut distance is 16, which is above the grouping threshold of 8. The split was not changed.

### T01 — gray Otsu
- change: grayscale Otsu, invert if the foreground is darker. No colour, morphology, or shape filter.
- why: simplest bright-blob baseline.
- params: mode=gray_otsu, blur=0
- dev: mIoU=0.3457 Dice=0.4680 empty_pred_rate=0.0000 (n=85)
- decision: discard as the kept recipe. It is the baseline, not the best.
- note: large bright backgrounds (bowls, bags, rice) are kept.

### T02 — HSV white only
- change: replace Otsu with HSV white gate S<=60, V>=170.
- why: eggs in this set are mostly white shells.
- params: mode=hsv, brown_on=false, white_s_max=60, white_v_min=170, no morph, no shape filter
- dev: mIoU=0.3814 Dice=0.5020 empty_pred_rate=0.0000
- decision: keep over T01.
- note: still paints white tables and paper.

### T03 — white OR brown
- change: add brown HSV gate H 5-30, S 40-220, V 40-220, OR-ed with white.
- why: the plan's recipe includes brown shells.
- params: brown_on=true, other gates unchanged from T02
- dev: mIoU=0.3166 Dice=0.4381 empty_pred_rate=0.0000
- decision: discard. Worse than T02.
- note: the brown gate pulls in wood, skin, and bags.

### T04 — morphology on white+brown
- change: Gaussian blur 5, open 5, close 11.
- why: drop speckle and close holes in the colour mask.
- params: blur=5, open_k=5, close_k=11, still white OR brown
- dev: mIoU=0.3186 Dice=0.4390 empty_pred_rate=0.0000
- decision: discard. Tiny gain over T03, still below T02.
- note: morphology cannot undo the brown false positives.

### T05 — area and circularity
- change: keep components with area fraction 0.015-0.55 and circularity >= 0.45.
- why: eggs are compact and not tiny.
- params: min_area_frac=0.015, max_area_frac=0.55, min_circularity=0.45, on top of T04
- dev: mIoU=0.1989 Dice=0.2389 empty_pred_rate=0.5529
- decision: discard.
- note: noisy contours fail the 0.45 circularity test, so more than half the dev images go empty.

### T06 — ellipse fill
- change: fill a fitted ellipse on each surviving component.
- why: shells are oval and the SAM masks are filled ellipses.
- params: ellipse_fill=true, else T05
- dev: mIoU=0.1934 Dice=0.2348 empty_pred_rate=0.5529
- decision: discard.
- note: ellipse fill cannot recover components that T05 already deleted.

### T07 — drop border-touching components
- change: drop a component if more than 25% of the image border pixels lie inside it.
- why: full-frame backgrounds were a T01 failure.
- params: drop_border_frac=0.25, else T06
- dev: mIoU=0.1918 Dice=0.2320 empty_pred_rate=0.5647
- decision: discard.
- note: still stuck behind the circularity cut.

### T08 — wider white gate
- change: white gate S<=70, V>=150, brown still on, same shape filters as T07.
- why: T05-T07 were empty too often; a wider white gate might keep dim shells.
- params: white_s_max=70, white_v_min=150
- dev: mIoU=0.1331 Dice=0.1554 empty_pred_rate=0.7294
- decision: discard.
- note: wider white plus a strict shape filter made more empties, not more eggs.

### T09 — fill holes
- change: binary hole fill before the component filter.
- why: close shell interiors that morphology left open.
- params: fill_holes=true, else T08
- dev: mIoU=0.1323 Dice=0.1548 empty_pred_rate=0.7294
- decision: discard.
- note: no useful change once components are already rejected.

### T10 — white plus morphology
- change: go back to the T02 white gate and add only the T04 blur/open/close. Brown and shape filters off.
- why: T02 was the best colour gate. Morphology helped T03 a little, so try it without brown.
- params: brown_on=false, white_s_max=60, white_v_min=170, blur=5, open_k=5, close_k=11, no area, circularity, ellipse, or border cut
- dev: mIoU=0.3842 Dice=0.5048 empty_pred_rate=0.0000
- decision: keep. Best dev score in this log.
- note: this is `configs/classical_final.yaml`.

### T11 — area cut on T10
- change: area fraction 0.01-0.55 on the T10 mask.
- why: drop tiny white specks without the circularity test that emptied T05.
- params: min_area_frac=0.01, max_area_frac=0.55
- dev: mIoU=0.3737 Dice=0.4869 empty_pred_rate=0.0706
- decision: discard. Below T10.

### T12 — loose circularity and ellipse
- change: circularity >= 0.15 and ellipse fill, area fraction 0.01-0.55.
- why: the plan's ellipse step, with a looser circularity than T05.
- params: min_circularity=0.15, ellipse_fill=true
- dev: mIoU=0.3450 Dice=0.4471 empty_pred_rate=0.1294
- decision: discard. Below T10.

### T13 — loose shape plus border cut
- change: circularity >= 0.2, area fraction 0.008-0.50, ellipse fill, border fraction 0.40.
- why: one more try to cut white backgrounds without emptying the set.
- params: min_circularity=0.2, min_area_frac=0.008, max_area_frac=0.50, drop_border_frac=0.40
- dev: mIoU=0.3331 Dice=0.4258 empty_pred_rate=0.1882
- decision: discard.

## Final pick

T10 white HSV plus morphology. Dev n=85, mIoU=0.3842, mean Dice=0.5048, empty_pred_rate=0. Masks for every image, including eval and the three excluded images, are in `outputs/classical_masks/`. Per-image dev scores are in `outputs/classical_dev_metrics.csv`. Eval scores are not in that file.
