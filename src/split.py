"""Group near-duplicate images and split groups 4:1 into dev and eval.

Two images share a group when their perceptual-hash Hamming distance is <= 8.
The split is over groups, seed 42, so near-duplicates do not cross the cut.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        self.parent[self.find(left)] = self.find(right)


def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--out", default="data/splits")
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hamming", type=int, default=8)
    args = parser.parse_args()

    manifest_path = (REPO_ROOT / args.manifest).resolve()
    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in load_manifest(manifest_path) if not row.get("read_error")]
    hashes = []
    for row in rows:
        image_path = REPO_ROOT / row["path"]
        with Image.open(image_path) as img:
            hashes.append(imagehash.phash(img.convert("RGB")))

    groups = UnionFind(len(rows))
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if hashes[i] - hashes[j] <= args.hamming:
                groups.union(i, j)

    members: dict[int, list[int]] = {}
    for index in range(len(rows)):
        members.setdefault(groups.find(index), []).append(index)

    root_ids = sorted(members)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(root_ids))
    n_dev = int(round(len(root_ids) * args.ratio))
    # Keep at least one eval group when there is more than one group.
    if len(root_ids) > 1:
        n_dev = min(max(n_dev, 1), len(root_ids) - 1)
    dev_roots = {root_ids[i] for i in order[:n_dev]}

    dev_paths = []
    eval_paths = []
    group_payload = {}
    for number, root in enumerate(sorted(root_ids, key=lambda r: min(members[r]))):
        group_id = f"g{number:04d}"
        paths = []
        for index in sorted(members[root]):
            rel = rows[index]["path"]
            paths.append(rel)
            if root in dev_roots:
                dev_paths.append(rel)
            else:
                eval_paths.append(rel)
        group_payload[group_id] = {
            "split": "dev" if root in dev_roots else "eval",
            "paths": paths,
            "phash": str(hashes[members[root][0]]),
        }

    dev_set = set(dev_paths)
    eval_set = set(eval_paths)
    if dev_set & eval_set:
        raise SystemExit("group overlap: the same path is in dev and eval")
    dev_groups = {k for k, v in group_payload.items() if v["split"] == "dev"}
    eval_groups = {k for k, v in group_payload.items() if v["split"] == "eval"}
    if dev_groups & eval_groups:
        raise SystemExit("group id overlap")

    (out_dir / "group_ids.json").write_text(json.dumps(group_payload, indent=2) + "\n")
    (out_dir / "dev.txt").write_text("".join(p + "\n" for p in dev_paths))
    (out_dir / "eval.txt").write_text("".join(p + "\n" for p in eval_paths))
    print(
        f"groups={len(group_payload)} dev_groups={len(dev_groups)} "
        f"eval_groups={len(eval_groups)} dev_images={len(dev_paths)} "
        f"eval_images={len(eval_paths)} hamming<={args.hamming} seed={args.seed}"
    )


if __name__ == "__main__":
    main()
