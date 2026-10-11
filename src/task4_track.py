"""Task 4: track eggs in the source video with the frozen T10 mask.

configs/classical_final.yaml is not changed. OpenCV cannot decode the AV1
source, so frames go through ffmpeg. Instance gates below are tracking-only.
They drop shirt, hand, and text blobs that the white HSV mask also lights up.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.task2_classical import load_cfg, segment_eggs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# Tracking-only. Not written back into the still-image recipe.
MIN_AREA = 8000
MAX_AREA = 90000
MIN_CIRC = 0.70
MIN_ASPECT = 0.68
MIN_CY_FRAC = 0.48
MAX_DIST = 400.0
MAX_MISS = 48
IOU_W = 0.65
Q_POS = 25.0
Q_VEL = 40.0
R_MEAS = 36.0

PALETTE = [
    (40, 180, 255),
    (80, 220, 80),
    (255, 120, 40),
    (200, 80, 255),
    (60, 220, 220),
    (80, 80, 255),
    (180, 255, 80),
    (255, 180, 200),
]


def _ffmpeg_decode(path: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ],
        stdout=subprocess.PIPE,
    )


def _ffmpeg_encode(path: Path, w: int, h: int, fps: float) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "23", "-movflags", "+faststart", str(path),
        ],
        stdin=subprocess.PIPE,
    )


def instances(mask: np.ndarray) -> list[dict]:
    height = mask.shape[0]
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    found = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_AREA or area > MAX_AREA:
            continue
        if float(cents[i][1]) < MIN_CY_FRAC * height:
            continue
        comp = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        per = cv2.arcLength(cnt, True)
        circ = 0.0 if per <= 0 else float(4 * np.pi * area / (per * per))
        if circ < MIN_CIRC:
            continue
        x, y, bw, bh = (int(v) for v in stats[i, :4])
        if min(bw, bh) / float(max(bw, bh)) < MIN_ASPECT:
            continue
        found.append(
            {
                "centroid": np.array([float(cents[i][0]), float(cents[i][1])]),
                "bbox": (x, y, bw, bh),
                "contour": cnt,
                "area": area,
            }
        )
    return found


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter) / float(union) if union else 0.0


class Track:
    def __init__(self, tid: int, det: dict) -> None:
        self.tid = tid
        self.miss = 0
        self.hits = 1
        self.bbox = det["bbox"]
        c = det["centroid"]
        self.x = np.array([c[0], c[1], 0.0, 0.0], dtype=np.float64)
        self.p = np.diag([R_MEAS, R_MEAS, 100.0, 100.0])

    def predict(self) -> np.ndarray:
        f = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + np.diag([Q_POS, Q_POS, Q_VEL, Q_VEL])
        return self.x[:2].copy()

    def update(self, det: dict) -> None:
        z = det["centroid"]
        h = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        s = h @ self.p @ h.T + np.eye(2) * R_MEAS
        k = self.p @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ (z - h @ self.x)
        self.p = (np.eye(4) - k @ h) @ self.p
        self.bbox = det["bbox"]
        self.miss = 0
        self.hits += 1


def associate(tracks: list[Track], dets: list[dict]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not tracks or not dets:
        for tr in tracks:
            tr.predict()
        return [], list(range(len(tracks))), list(range(len(dets)))
    pred = [tr.predict() for tr in tracks]
    cost = np.full((len(tracks), len(dets)), 1e3, dtype=np.float64)
    for i, tr in enumerate(tracks):
        for j, det in enumerate(dets):
            dist = float(np.linalg.norm(pred[i] - det["centroid"]))
            if dist > MAX_DIST:
                continue
            overlap = _bbox_iou(tr.bbox, det["bbox"])
            cost[i, j] = (1.0 - IOU_W) * (dist / MAX_DIST) + IOU_W * (1.0 - overlap)
    rows, cols = linear_sum_assignment(cost)
    pairs, used_t, used_d = [], set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] >= 1e3:
            continue
        pairs.append((int(r), int(c)))
        used_t.add(int(r))
        used_d.add(int(c))
    free_t = [i for i in range(len(tracks)) if i not in used_t]
    free_d = [j for j in range(len(dets)) if j not in used_d]
    return pairs, free_t, free_d


def draw(frame: np.ndarray, dets: list[dict], ids: list[int]) -> None:
    for det, tid in zip(dets, ids):
        color = PALETTE[tid % len(PALETTE)]
        cnt = det["contour"]
        if len(cnt) >= 5:
            cv2.ellipse(frame, cv2.fitEllipse(cnt), color, 3)
        else:
            cv2.drawContours(frame, [cnt], -1, color, 3)
        x, y = det["centroid"]
        cv2.circle(frame, (int(x), int(y)), 6, color, -1)
        cv2.putText(
            frame, f"id {tid}", (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA,
        )


def _probe(path: Path) -> dict:
    raw = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration,size",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames,codec_type",
            "-of", "json", str(path),
        ]
    )
    return json.loads(raw)


def run(src: Path, full_out: Path, preview_out: Path, frames_dir: Path) -> dict:
    cfg = load_cfg()
    probe = _probe(src)
    stream = next(s for s in probe["streams"] if s.get("codec_type") == "video")
    w, h = int(stream["width"]), int(stream["height"])
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)

    decoder = _ffmpeg_decode(src)
    encoder = _ffmpeg_encode(full_out, w, h, fps)
    assert decoder.stdout is not None and encoder.stdin is not None
    frame_bytes = w * h * 3

    tracks: list[Track] = []
    next_id = 1
    counts: list[int] = []
    frame_i = 0
    while True:
        buf = decoder.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
        dets = instances(segment_eggs(frame, cfg))
        pairs, free_t, free_d = associate(tracks, dets)
        ids = [0] * len(dets)
        for ti, di in pairs:
            tracks[ti].update(dets[di])
            ids[di] = tracks[ti].tid
        for ti in free_t:
            tracks[ti].miss += 1
        tracks = [tr for tr in tracks if tr.miss <= MAX_MISS]
        for di in free_d:
            tracks.append(Track(next_id, dets[di]))
            ids[di] = next_id
            next_id += 1
        draw(frame, dets, ids)
        encoder.stdin.write(frame.tobytes())
        counts.append(len(dets))
        frame_i += 1
        if frame_i % 200 == 0:
            print(f"frame {frame_i} dets={counts[-1]} ids={next_id-1}", flush=True)

    decoder.stdout.close()
    encoder.stdin.close()
    decoder.wait()
    encoder.wait()
    if decoder.returncode != 0 or encoder.returncode != 0:
        raise SystemExit(f"ffmpeg failed decode={decoder.returncode} encode={encoder.returncode}")

    window = int(min(30 * fps, len(counts))) or 1
    best_s, best_sum, running = 0, sum(counts[:window]), sum(counts[:window])
    for s in range(1, len(counts) - window + 1):
        running += counts[s + window - 1] - counts[s - 1]
        if running > best_sum:
            best_sum, best_s = running, s
    start_s = best_s / fps
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-v", "error", "-ss", f"{start_s:.3f}",
            "-i", str(full_out), "-t", "30", "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
            str(preview_out),
        ]
    )

    frames_dir.mkdir(parents=True, exist_ok=True)
    det_frames = [i for i, c in enumerate(counts) if c > 0]
    saved = []
    picks = []
    if det_frames:
        for frac in (0.15, 0.40, 0.65, 0.90):
            picks.append(det_frames[min(len(det_frames) - 1, int(frac * (len(det_frames) - 1)))])
    for k, fi in enumerate(picks, start=1):
        out_p = frames_dir / f"track_frame_{k:02d}.jpg"
        subprocess.check_call(
            [
                "ffmpeg", "-y", "-v", "error", "-ss", f"{fi / fps:.3f}",
                "-i", str(full_out), "-frames:v", "1", str(out_p),
            ]
        )
        saved.append(str(out_p.relative_to(REPO)))

    full_meta = _probe(full_out)
    prev_meta = _probe(preview_out)
    present = [c for c in counts if c > 0]
    summary = {
        "frames": frame_i,
        "fps": fps,
        "width": w,
        "height": h,
        "unique_ids": next_id - 1,
        "frames_with_eggs": len(present),
        "max_eggs_in_frame": int(max(counts) if counts else 0),
        "mean_eggs_when_present": float(np.mean(present)) if present else 0.0,
        "preview_start_s": start_s,
        "preview_window_detections": int(best_sum),
        "gates": {
            "min_area": MIN_AREA,
            "max_area": MAX_AREA,
            "min_circularity": MIN_CIRC,
            "min_aspect": MIN_ASPECT,
            "min_cy_frac": MIN_CY_FRAC,
            "max_dist_px": MAX_DIST,
            "max_miss": MAX_MISS,
            "kalman": True,
            "ellipse_draw": True,
        },
        "screenshots": saved,
        "screenshot_frames": picks,
        "full": {
            "path": str(full_out.relative_to(REPO)),
            "bytes": full_out.stat().st_size,
            "duration": float(full_meta["format"]["duration"]),
        },
        "preview": {
            "path": str(preview_out.relative_to(REPO)),
            "bytes": preview_out.stat().st_size,
            "duration": float(prev_meta["format"]["duration"]),
        },
    }
    note_path = REPO / "outputs/video/track_summary.json"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(json.dumps(summary, indent=2) + "\n")
    (REPO / "docs/video_drive_url.txt").write_text(
        "Local file ready for Google Drive upload. Set sharing to anyone with the link.\n"
        f"path: {full_out}\n"
        f"bytes: {full_out.stat().st_size}\n"
        f"duration_s: {summary['full']['duration']}\n"
        "drive_url: PLACEHOLDER not uploaded\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/video/source.mp4")
    parser.add_argument("--full", default="video/tracked_full.mp4")
    parser.add_argument("--preview", default="video/tracked_preview.mp4")
    parser.add_argument("--frames", default="docs/figures")
    args = parser.parse_args()
    summary = run(REPO / args.src, REPO / args.full, REPO / args.preview, REPO / args.frames)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
