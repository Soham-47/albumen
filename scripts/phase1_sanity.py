"""Count images, list resolutions, flag corrupt and exact-hash duplicates,
and save a 5-image contact sheet.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--out", default="outputs/phase1_samples.png")
    parser.add_argument("--report", default="outputs/phase1_sanity.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    manifest_path = REPO_ROOT / args.manifest
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    resolutions = Counter()
    by_hash: dict[str, list[str]] = defaultdict(list)
    corrupt = []
    for row in rows:
        key = f"{row['width']}x{row['height']}" if row["width"] else "unreadable"
        resolutions[key] += 1
        by_hash[row["sha1"]].append(row["path"])
        if row.get("read_error"):
            corrupt.append({"path": row["path"], "error": row["read_error"]})
            continue
        image_path = REPO_ROOT / row["path"]
        try:
            with Image.open(image_path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            corrupt.append({"path": row["path"], "error": f"{type(exc).__name__}: {exc}"})

    duplicates = [
        {"sha1": digest, "paths": paths}
        for digest, paths in sorted(by_hash.items())
        if len(paths) > 1
    ]

    readable = [row for row in rows if row["width"] and row["path"] not in {c["path"] for c in corrupt}]
    rng = random.Random(args.seed)
    sample = rng.sample(readable, k=min(args.n, len(readable))) if readable else []

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if sample:
        fig, axes = plt.subplots(1, len(sample), figsize=(3 * len(sample), 3.4))
        if len(sample) == 1:
            axes = [axes]
        for axis, row in zip(axes, sample):
            with Image.open(REPO_ROOT / row["path"]) as img:
                axis.imshow(img.convert("RGB"))
            axis.set_title(f"{row['filename']}\n{row['width']}x{row['height']}", fontsize=8)
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

    report = {
        "image_count": len(rows),
        "readable_count": len(readable),
        "resolutions": dict(resolutions.most_common()),
        "corrupt": corrupt,
        "exact_sha1_duplicate_groups": duplicates,
        "contact_sheet": str(out_path.relative_to(REPO_ROOT)) if sample else "",
        "contact_sheet_files": [row["path"] for row in sample],
        "sample_seed": args.seed,
    }
    report_path = REPO_ROOT / args.report
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
