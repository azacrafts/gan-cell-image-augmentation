"""Compute shsy5y instance counts per image on sparse train (scenario B) for procedural placement sampling."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.coco_filters import resolve_shsy5y_category_id  # noqa: E402
from src.data.coco_livecell import load_split_manifest  # noqa: E402
from src.data.procedural_shsy5y import load_sparse_manifest, sparse_train_allowed_image_ids  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description="Count shsy5y instances per train image in sparse scenario B; write percentiles for procedural masks."
    )
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument(
        "--sparse-manifest",
        type=Path,
        default=ROOT / "manifests" / "shsy5y_sparse_train_seed42.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "manifests" / "sparse_train_shsy5y_placement_stats.json",
    )
    args = p.parse_args()

    repo_root = args.repo_root.resolve()
    official = load_split_manifest(repo_root)
    train_path = repo_root / official["train_annotations"]

    with train_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    with args.sparse_manifest.open("r", encoding="utf-8") as f:
        sparse_man = json.load(f)

    shsy_id = resolve_shsy5y_category_id(coco["categories"])
    allowed = sparse_train_allowed_image_ids(coco, sparse_man)

    by_image: Counter[int] = Counter()
    for ann in coco["annotations"]:
        iid = int(ann["image_id"])
        if iid not in allowed:
            continue
        if int(ann["category_id"]) != shsy_id:
            continue
        by_image[iid] += 1

    counts = sorted(by_image.values())
    arr = np.array(counts, dtype=np.float64) if counts else np.array([0.0])

    def pct(p: float) -> float:
        return float(np.percentile(arr, p)) if len(arr) else 0.0

    try:
        sm_rel = str(args.sparse_manifest.resolve().relative_to(repo_root))
    except ValueError:
        sm_rel = str(args.sparse_manifest)
    try:
        ta_rel = str(train_path.resolve().relative_to(repo_root))
    except ValueError:
        ta_rel = str(train_path)

    out = {
        "sparse_manifest": sm_rel,
        "train_annotations": ta_rel,
        "shsy5y_category_id": shsy_id,
        "n_sparse_train_images": len(allowed),
        "n_images_with_at_least_one_shsy5y": len(counts),
        "shsy5y_instances_per_image": {
            "description": "One value per image that has >=1 shsy5y instance in sparse train B.",
            "n_images": len(counts),
            "min": int(arr.min()) if len(arr) else 0,
            "p05": round(pct(5)),
            "p25": round(pct(25)),
            "p50": round(pct(50)),
            "p75": round(pct(75)),
            "p95": round(pct(95)),
            "max": int(arr.max()) if len(arr) else 0,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out} ({len(counts)} images with shsy5y, count range {out['shsy5y_instances_per_image']['min']}..{out['shsy5y_instances_per_image']['max']})")


if __name__ == "__main__":
    main()
