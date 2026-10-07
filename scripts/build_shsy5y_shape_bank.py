"""Build a disk bank of shsy5y instance crops from LIVECell train (sparse scenario B)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.coco_filters import resolve_shsy5y_category_id  # noqa: E402
from src.data.coco_livecell import load_split_manifest, rasterize_instance_binary_mask  # noqa: E402
from src.data.procedural_shsy5y import crop_binary_mask, mask_area, sparse_train_allowed_image_ids  # noqa: E402


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build shsy5y instance shape bank from LIVECell train (sparse scenario B: all train images except excluded pure-shsy5y FOVs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Uses manifests/shsy5y_sparse_train_seed42.json by default (excluded_shsy5y_image_ids define sparse train).",
    )
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument(
        "--sparse-manifest",
        type=Path,
        default=ROOT / "manifests" / "shsy5y_sparse_train_seed42.json",
    )
    p.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "shsy5y_shape_bank_sparse")
    p.add_argument("--padding", type=int, default=2, help="Crop bbox padding in pixels")
    p.add_argument("--force", action="store_true", help="Overwrite existing bank_meta.json")
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

    out_root = args.output_dir.resolve()
    shapes_dir = out_root / "shapes"
    meta_path = out_root / "bank_meta.json"

    if meta_path.is_file() and not args.force:
        print(f"Refusing to overwrite {meta_path} (use --force)", file=sys.stderr)
        sys.exit(1)

    shapes_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    for ann in tqdm(coco["annotations"], desc="annotations"):
        iid = int(ann["image_id"])
        if iid not in allowed:
            continue
        if int(ann["category_id"]) != shsy_id:
            continue
        aid = int(ann["id"])
        im_info = next((im for im in coco["images"] if int(im["id"]) == iid), None)
        if im_info is None:
            continue
        h, w = int(im_info["height"]), int(im_info["width"])
        full = rasterize_instance_binary_mask(h, w, ann, fill=1)
        if mask_area(full) < 1:
            continue
        cropped, bbox = crop_binary_mask(full, pad=args.padding)
        area_px = mask_area(cropped)
        ch, cw = cropped.shape
        rel = f"shapes/{aid:08d}.png"
        Image.fromarray((cropped > 0).astype(np.uint8) * 255, mode="L").save(out_root / rel, format="PNG")
        entries.append(
            {
                "annotation_id": aid,
                "image_id": iid,
                "origin_image_width": w,
                "origin_image_height": h,
                "path": rel,
                "area_px": area_px,
                "bbox_wh": [cw, ch],
                "bbox_xywh": list(bbox),
            }
        )

    bank = {
        "sparse_manifest": str(args.sparse_manifest.resolve()),
        "sparse_manifest_sha256": _sha256_file(args.sparse_manifest),
        "train_annotations": str(train_path.resolve()),
        "train_annotations_sha256": _sha256_file(train_path),
        "shsy5y_category_id": shsy_id,
        "crop_padding_px": args.padding,
        "n_entries": len(entries),
        "entries": entries,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(bank, f, indent=2)
    print(f"Wrote {len(entries)} shapes to {out_root}")


if __name__ == "__main__":
    main()
