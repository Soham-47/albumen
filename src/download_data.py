"""Download the egg dataset and write data/manifest.csv.

Preferred source is the Google Drive zip from the plan (no Roboflow API key).
Columns follow the plan (image_id, path, width, height, sha1) and also record
filename, source, and any annotation file found next to the image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import zipfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
LABEL_SUFFIXES = {".txt", ".xml", ".json"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_drive_zip(drive_id: str, zip_path: Path) -> None:
    import gdown

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    # gdown 6.x takes the file id directly. `--fuzzy` is not in this release.
    gdown.download(id=drive_id, output=str(zip_path), quiet=False)


def unzip_into(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(out_dir)


def find_annotation(image_path: Path, raw_root: Path) -> str:
    """Return a repo-relative annotation path if one sits beside the image."""
    stem = image_path.stem
    parent = image_path.parent
    for suffix in LABEL_SUFFIXES:
        candidate = parent / f"{stem}{suffix}"
        if candidate.is_file() and candidate.resolve() != image_path.resolve():
            try:
                return str(candidate.resolve().relative_to(REPO_ROOT))
            except ValueError:
                return str(candidate)
    # Common Roboflow/YOLO dataset-level files in the same folder.
    for name in ("_annotations.coco.json", "annotations.json", "labels.csv"):
        candidate = parent / name
        if candidate.is_file():
            try:
                return str(candidate.resolve().relative_to(REPO_ROOT))
            except ValueError:
                return str(candidate)
    return ""


def build_manifest(raw_root: Path, manifest_path: Path, source: str) -> list[dict]:
    rows: list[dict] = []
    images = sorted(
        p for p in raw_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    for image_path in images:
        rel = image_path.resolve().relative_to(REPO_ROOT)
        rel_raw = image_path.resolve().relative_to(raw_root.resolve())
        roboflow_split = rel_raw.parts[0] if len(rel_raw.parts) > 1 else ""
        folder_label = rel_raw.parts[1] if len(rel_raw.parts) > 2 else ""
        record = {
            "image_id": image_path.stem,
            "filename": image_path.name,
            "path": str(rel),
            "width": "",
            "height": "",
            "sha1": sha1_file(image_path),
            "source": source,
            "roboflow_split": roboflow_split,
            "folder_label": folder_label,
            "annotation": find_annotation(image_path, raw_root),
            "read_error": "",
        }
        try:
            with Image.open(image_path) as img:
                img.verify()
            with Image.open(image_path) as img:
                record["width"] = str(img.size[0])
                record["height"] = str(img.size[1])
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            record["read_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(record)

    # Disambiguate duplicate stems so image_id is unique.
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["image_id"]] = counts.get(row["image_id"], 0) + 1
    seen: dict[str, int] = {}
    for row in rows:
        stem = row["image_id"]
        if counts[stem] > 1:
            seen[stem] = seen.get(stem, 0) + 1
            row["image_id"] = f"{stem}__{seen[stem]}"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "filename",
        "path",
        "width",
        "height",
        "sha1",
        "source",
        "roboflow_split",
        "folder_label",
        "annotation",
        "read_error",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-id", required=True)
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--zip-path", default="data/egg_drive.zip")
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.out).resolve()
    manifest_path = (REPO_ROOT / args.manifest).resolve()
    zip_path = (REPO_ROOT / args.zip_path).resolve()

    print(f"Downloading Drive file {args.drive_id} -> {zip_path}")
    download_drive_zip(args.drive_id, zip_path)
    print(f"Unzipping into {out_dir}")
    unzip_into(zip_path, out_dir)
    source = f"gdrive:{args.drive_id}"
    for name in ("README.roboflow.txt", "README.dataset.txt"):
        src = out_dir / name
        if src.is_file():
            (REPO_ROOT / "data" / name).write_bytes(src.read_bytes())
            print(f"copied {name} -> data/{name}")
    rows = build_manifest(out_dir, manifest_path, source)
    readable = sum(1 for row in rows if not row["read_error"])
    print(f"images={len(rows)} readable={readable} manifest={manifest_path}")
    if len(rows) < 100 or len(rows) > 120:
        print(
            "STOP: image count is not about 110. Universe page claimed 110. Inspect data/raw/."
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
