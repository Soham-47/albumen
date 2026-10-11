# Code walkthrough

One file per task. The shipped numbers come from the yaml and the tracking knobs below. Do not retune them unless the interviewer asks for a live edit.

Knobs for still images: `configs/classical_final.yaml` (this file wins) and the same keys in `CONFIG` in `src/task2_classical.py`.
Knobs for the video: the block at the top of `src/task4_track.py` (`min_area`, `max_distance`, `max_missed_frames`, and the neighbours).

## Task 1 — `src/task1_sam.py`

Run: `python src/task1_sam.py`

- Loads SAM ViT-B from `checkpoints/sam_vit_b_01ec64.pth`.
- `SamAutomaticMaskGenerator` proposes masks. Raw proposals are cached in `outputs/sam_cache/` so a second run skips the network.
- An egg filter keeps compact ovals (area, solidity, aspect, ellipse fit, border touch, brightness).
- Overlapping proposals are merged when IoU is at least 0.5 or one mask is 70% inside the other.
- The union mask is `outputs/sam_masks/<stem>_union.png`. Task 2 and Task 3 treat that file as ground truth.
- Three stems are not eggs or have an empty SAM mask: `not_damaged_12`, `not_damaged_39`, `not_damaged_145`. Scores skip them. Masks are still written.
- CPU only on this machine. A full uncached pass is about 47 seconds per image.
- SAM mistakes (hands, keycaps, plastic) count as classical errors, because there are no human masks in the zip.

## Task 2 — `src/task2_classical.py`

Run: `python src/task2_classical.py --cfg configs/classical_final.yaml --save-all --score-dev`

- `segment_eggs(img, cfg)` is the only function a live edit needs to call.
- `to_hsv` converts BGR to HSV and blurs. Blur runs before the colour gate so specks do not become blobs.
- `white_mask` keeps low saturation and high value. White shells sit there. Optional brown is an OR, and it is off.
- `clean_mask` opens (drop specks) then closes (join broken shell). Hole fill is off in T10.
- `filter_blobs` can drop by area, circularity, or border touch, or replace a blob with an ellipse. T10 sets those gates so they do nothing.
- `gray_otsu` is trial T01 only. `mode: hsv` never calls it.
- Dev score (n=85): mIoU 0.3842, Dice 0.5048. Masks land in `outputs/classical_masks/`.
- Changing the yaml and re-running `--score-dev` is the whole experiment loop. The log of discarded trials is `experiments/log.md`.

## Task 3 — `src/task3_eval.py`

Run: `python src/task3_eval.py`

- Reads the masks Task 1 and Task 2 already wrote. It does not refit the recipe.
- Primary number is the eval split (`data/splits/eval.txt`). Dev is printed beside it.
- Metrics are IoU, Dice, precision, and recall against the SAM union. The summary uses the population standard deviation.
- Eval (n=22): mIoU 0.4074, Dice 0.5462.
- Writes `outputs/figures/metrics_summary.json`, `docs/figures/iou_hist.png`, and `docs/figures/failures_grid.png`.
- Groups were split by perceptual-hash Hamming distance, seed 42, so near-duplicate frames do not cross the cut. See `src/split.py`.
- Strata in the summary (size, hue) are descriptive. They did not choose T10.
- Empty predictions and the three excluded stems are counted in the JSON, not hidden.

## Task 4 — `src/task4_track.py`

Run: `python src/task4_track.py`

- Frames come from ffmpeg. OpenCV cannot decode the AV1 source.
- `detect_eggs` runs `segment_eggs`, then keeps blobs by area, circularity, box aspect, and a minimum centroid y. Those gates are video-only. The still-image yaml stays loose because the same gates deleted real eggs in photos.
- `match_to_tracks` builds a cost for every track-detection pair and solves it with the Hungarian algorithm. Cost is `(1 - iou_weight) * (distance / max_distance) + iou_weight * (1 - box IoU)`. Pairs farther than `max_distance` are forbidden.
- Kalman state on each track is `[x, y, vx, vy]`: position in pixels and speed in pixels per frame. `predict` moves x and y by one frame of speed. `update` corrects x and y from the detection. Speed is inferred.
- `update_tracks` applies matches, increments `miss` on unmatched ids, drops an id after `max_missed_frames`, and births a new id for a detection nobody claimed.
- `draw` paints a fitted ellipse, a centroid, and `id N`.
- On 1440 frames the frozen gates give 4 unique ids, 757 frames with at least one egg, and at most 2 eggs in a frame.
- Outputs: `video/tracked_full.mp4`, `video/tracked_preview.mp4`, `docs/figures/track_frame_01.jpg` through `track_frame_04.jpg`.

## If they ask X, edit Y

1. Brown eggs. Set `brown_on: true` in `configs/classical_final.yaml` (line 7). The hue, saturation, and value ranges are the next three lines. Re-run Task 2. This lowered dev mIoU on the white-egg set.
2. Morphology kernel. `open_k` is line 12 and `close_k` is line 13 of the same yaml. Odd integers. Larger open removes more specks. Larger close fills bigger gaps and can glue the egg to the table.
3. Fill holes or draw an ellipse. `fill_holes` is line 19. `ellipse_fill` is line 17. Set either to `true`. Both run inside `clean_mask` / `filter_blobs` in `src/task2_classical.py`.
4. Circularity. `min_circularity` in the yaml (line 16) filters still images. For the video, edit `min_circularity` near line 36 of `src/task4_track.py`. `0` turns the still-image check off. About `0.7` keeps rounder blobs.
5. Fewer ID switches. Raise `max_missed_frames` (line 40 of `src/task4_track.py`) so an occluded egg keeps its id. To also use appearance, add a colour term in `match_to_tracks` (the cost assignment near the `iou_weight` line) and give it a weight that still sums with the distance and IoU terms.
6. A new metric. Add the function in `src/metrics.py`, call it from `src/task3_eval.py` where IoU and Dice are aggregated, and add the key to `metrics_summary.json`. Do not refit T10 on eval.
7. A different video. Pass `--src path/to/video.mp4` to `python src/task4_track.py`. The decoder is ffmpeg, so the file must be a format ffmpeg can read. Gates assume 1080p eggs; on a smaller frame, lower `min_area` and `max_distance`.
8. Speed. In `run` (`src/task4_track.py`), resize the frame before `segment_eggs`, then scale centroids and boxes back before `draw`. A half-size frame is the first thing to try. Do not resize the still-image eval set if you still want the published mIoU.

Demo of the four stages: `python scripts/demo.py` writes `outputs/demo_panels.png` (original, HSV white mask, cleaned mask, final mask plus the SAM outline).
