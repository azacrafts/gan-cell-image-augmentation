"""Build shsy5y 5% sparse train manifest; verify non-shsy5y counts match full train."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.coco_filters import (  # noqa: E402
    build_ann_index,
    is_shsy5y_image,
    resolve_shsy5y_category_id,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_instances_by_category(
    annotations: list[dict],
    image_ids: set[int],
) -> Counter[int]:
    c = Counter()
    for a in annotations:
        if int(a["image_id"]) in image_ids:
            c[int(a["category_id"])] += 1
    return c


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LIVECell shsy5y sparse train manifest (5%%)")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=ROOT / "manifests" / "shsy5y_sparse_train_seed42.json")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()

    official_path = repo_root / "manifests" / "split_official.json"
    with official_path.open("r", encoding="utf-8") as f:
        official = json.load(f)
    train_ann_path = repo_root / official["train_annotations"]

    with train_ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    shsy5y_id = resolve_shsy5y_category_id(coco["categories"])
    ann_by_image = build_ann_index(coco["annotations"])

    shsy5y_images: list[dict] = []
    other_images: list[dict] = []
    for img in coco["images"]:
        iid = int(img["id"])
        if is_shsy5y_image(iid, ann_by_image, shsy5y_id):
            shsy5y_images.append(img)
        else:
            other_images.append(img)

    n_shsy = len(shsy5y_images)
    n_keep = max(1, int(round(n_shsy * args.fraction))) if n_shsy else 0

    shsy_ids_sorted = sorted(int(x["id"]) for x in shsy5y_images)
    rng = random.Random(args.seed)
    kept_ids = rng.sample(shsy_ids_sorted, n_keep) if n_keep else []
    kept_set = set(kept_ids)
    excluded_ids = [i for i in shsy_ids_sorted if i not in kept_set]

    source_hash = sha256_file(train_ann_path)

    # Sparse train images = all other images + kept shsy5y
    sparse_image_ids = {int(x["id"]) for x in other_images} | set(kept_ids)

    # Count instances per category for full train vs sparse train
    full_ids = {int(x["id"]) for x in coco["images"]}
    cat_full = count_instances_by_category(coco["annotations"], full_ids)
    cat_sparse = count_instances_by_category(coco["annotations"], sparse_image_ids)

    id_to_name = {int(c["id"]): c["name"] for c in coco["categories"]}

    manifest = {
        "seed": args.seed,
        "fraction": args.fraction,
        "selection_rule": "n_keep = max(1, int(round(n_shsy5y_images * fraction))) if n_shsy5y_images > 0 else 0",
        "shsy5y_category_id": shsy5y_id,
        "shsy5y_category_name": "shsy5y",
        "rule": "is_shsy5y_image iff image has >=1 annotation and every annotation.category_id == shsy5y_id",
        "source_annotation_file": str(train_ann_path.relative_to(repo_root)),
        "source_annotation_sha256": source_hash,
        "kept_shsy5y_image_ids": sorted(kept_ids),
        "excluded_shsy5y_image_ids": excluded_ids,
        "kept_shsy5y_file_names": [
            next(x["file_name"] for x in shsy5y_images if int(x["id"]) == k) for k in sorted(kept_ids)
        ],
        "counts": {
            "full_train": {
                "images": len(coco["images"]),
                "shsy5y_images": n_shsy,
                "non_shsy5y_images": len(other_images),
                "instances_total": len(coco["annotations"]),
                "instances_by_category": {id_to_name[k]: c for k, c in sorted(cat_full.items())},
            },
            "sparse_train_b": {
                "images": len(sparse_image_ids),
                "shsy5y_images": len(kept_ids),
                "non_shsy5y_images": len(other_images),
                "instances_total": sum(cat_sparse.values()),
                "instances_by_category": {id_to_name[k]: c for k, c in sorted(cat_sparse.items())},
            },
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Verification: non-shsy5y image count == full train non-shsy5y
    ok_images = len(other_images) == manifest["counts"]["sparse_train_b"]["non_shsy5y_images"]
    ok_inst_other = all(
        cat_sparse[cid] == cat_full[cid]
        for cid in id_to_name
        if cid != shsy5y_id
    )

    print("=== LIVECell sparse split (shsy5y) ===")
    print(f"Train annotations: {train_ann_path}")
    print(f"SHA256: {source_hash[:16]}...")
    print(f"shsy5y train images (full): {n_shsy} -> kept {n_keep} ({100*args.fraction:.1f}%)")
    print(f"Non-shsy5y train images: {len(other_images)} (unchanged in B)")
    print(f"Sparse train total images: {len(sparse_image_ids)}")
    print()
    print("Instances per category (full train vs sparse train):")
    for cid in sorted(id_to_name):
        name = id_to_name[cid]
        cf, cs = cat_full[cid], cat_sparse[cid]
        mark = "OK" if (cid == shsy5y_id or cf == cs) else "MISMATCH"
        print(f"  {name:8s}  full={cf:8d}  sparse={cs:8d}  {mark}")
    print()
    print(f"VERIFY non-shsy5y image count unchanged: {ok_images} (must be True)")
    print(f"VERIFY non-shsy5y instance counts unchanged: {ok_inst_other} (must be True)")
    print(f"Wrote manifest: {args.out}")


if __name__ == "__main__":
    main()
