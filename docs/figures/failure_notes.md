# Failure notes (Phase 4 draft)

Scored classical T10 against SAM union masks. Excluded from every score: `not_damaged_12` (empty SAM, egg under herbs), `not_damaged_39` (empty SAM, frame-cut on a printed lid), `not_damaged_145` (foam, not an egg). Masks for those three still exist on disk.

Eval mIoU is 0.4074 (std 0.2142, n=22).

Worst 8 eval images by IoU:

- `not_damaged_155_jpg.rf.a65395f65d93bcd6cbf6bdab6d94e551` IoU 0.079, GT area 27968, mean hue 108.0, mean S 12.1, colour label white.
- `not_damaged_104_jpg.rf.3bf8cb51d5ad8934390c8e2c3b79b32f` IoU 0.094, GT area 33077, mean hue 24.0, mean S 8.9, colour label white.
- `not_damaged_113_jpg.rf.75e3389c0809c7c0f42fbfd25219e65e` IoU 0.147, GT area 48812, mean hue 26.8, mean S 13.3, colour label white.
- `not_damaged_78_jpg.rf.d3a6e8624d192abc58ff7df9d5781bb5` IoU 0.169, GT area 58026, mean hue 36.3, mean S 9.3, colour label white.
- `not_damaged_27_jpg.rf.9eabbf76d6ac2d4caccb38b11e0d28ad` IoU 0.170, GT area 59100, mean hue 27.3, mean S 10.8, colour label white.
- `not_damaged_110_jpg.rf.2de10ae949146248df94e52f9065bdd8` IoU 0.248, GT area 107239, mean hue 58.0, mean S 23.7, colour label white.
- `not_damaged_41_jpg.rf.630408063dc3fdfa9a9a7f49e1c1d93c` IoU 0.258, GT area 90183, mean hue 38.3, mean S 11.0, colour label white.
- `not_damaged_157_jpg.rf.b244a82199f2158510ccf485e54d98a2` IoU 0.304, GT area 147508, mean hue 76.2, mean S 71.3, colour label brown.

Themes:

- White clutter: the T10 gate is bright low-saturation pixels. White tables, paper, rice, and plastic pass the same test as a white shell, so precision stays well below recall.
- Missed brown eggs: the brown OR-gate was discarded in Phase 3 because it hurt dev mIoU. Brown shells are mostly false negatives under T10. The colour stratum uses mean saturation inside the SAM mask (S < 40 is white, otherwise brown), not hue alone, because white shells have an unstable hue.
- Touching eggs: Task 3 scores the union mask, so two eggs that touch and stay in one component are not an IoU error by themselves. Instance splits are a Task 4 issue.
- Specular holes: highlights on a shell can fall outside the value gate and leave holes. Close kernel 11 fills some of them and not all of them.
- Scale: T10 has no min-area cut. Tiny eggs are not removed by area. They are missed when they are darker than V 170.
- Domain gap vs the video: the stills are 640x640 Roboflow crops of eggs on tables and packaging. The video is 1920x1080 of eggs moving in a scene that was not in this set. The same white gate will pick up bright backgrounds there too.
- SAM mistakes: some false positives and false negatives versus SAM are SAM's own hands, keycaps, and white plastic. A classical miss on those pixels is not a missed egg.
