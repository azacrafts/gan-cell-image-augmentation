"""Generate N procedural shsy5y semantic masks from a shape bank (layout JSON + CLI)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.coco_livecell import build_cat_id_to_semantic_label, load_split_manifest  # noqa: E402
from src.data.procedural_shsy5y import (  # noqa: E402
    LayoutConfig,
    generate_one_semantic_mask,
    load_layout_config,
    load_placement_stats,
    load_shape_bank,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sample procedural semantic masks from a shsy5y shape bank (sparse train).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Layout JSON (optional --layout) can override LayoutConfig fields such as "
        "scale_mode, placement_source (sparse_train_stats = sample counts from manifests/sparse_train_shsy5y_placement_stats.json; "
        "fixed = use n_clusters / cluster_size_* / n_isolated), n_instances_cap, "
        "occupancy, n_clusters, cluster_size_min, cluster_size_max, n_isolated, "
        "mean_instance_area_px, target_touching_pairs, max_attempts.\n"
        "Refresh stats: python scripts/compute_sparse_train_shsy5y_placement_stats.py\n"
        "Build the shape bank first: python scripts/build_shsy5y_shape_bank.py\n"
        "Note: saved PNGs are semantic labels (small uint8 values, e.g. 6); they look "
        "almost black in viewers but have nonzero pixels — see stats.foreground_pixels in the manifest.",
    )
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--bank-dir", type=Path, default=ROOT / "runs" / "shsy5y_shape_bank_sparse")
    p.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "procedural_masks_shsy5y")
    p.add_argument("--train-annotations", type=Path, default=None, help="COCO train JSON (default: split_official)")
    p.add_argument("--num-masks", type=int, default=783)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument(
        "--layout",
        type=Path,
        default=None,
        help="JSON file overriding LayoutConfig fields (occupancy, n_clusters, ...)",
    )
    p.add_argument(
        "--max-gen-retries",
        type=int,
        default=80,
        help="If generation fails (e.g. layout too strict), retry this many times per mask before aborting.",
    )
    p.add_argument(
        "--scale-mode",
        choices=("training_match", "occupancy"),
        default=None,
        help="Override layout JSON: training_match = cell size as in 512 training (default); "
        "occupancy = legacy global scaling to target occupancy.",
    )
    p.add_argument(
        "--placement-source",
        choices=("sparse_train_stats", "fixed"),
        default=None,
        help="Override layout: sparse_train_stats = sample instance counts from sparse-train percentiles (default); "
        "fixed = use n_clusters, cluster_size_*, n_isolated from config.",
    )
    args = p.parse_args()

    repo_root = args.repo_root.resolve()
    official = load_split_manifest(repo_root)
    train_path = args.train_annotations or (repo_root / official["train_annotations"])
    with train_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    cat_id_to_label = build_cat_id_to_semantic_label(coco["categories"])
    shsy_id = next(int(c["id"]) for c in coco["categories"] if c.get("name") == "shsy5y")
    shsy_label = cat_id_to_label[shsy_id]

    bank_dir = args.bank_dir.resolve()
    shapes, areas, _entries, origin_hw, has_origin = load_shape_bank(bank_dir)
    cfg = load_layout_config(args.layout)
    if args.scale_mode is not None:
        cfg.scale_mode = args.scale_mode
    if args.placement_source is not None:
        cfg.placement_source = args.placement_source

    placement_stats: dict | None = None
    if cfg.placement_source == "sparse_train_stats":
        placement_stats = load_placement_stats(repo_root, cfg.placement_stats_path)

    if cfg.scale_mode == "training_match" and not has_origin:
        print(
            "This shape bank has no origin_image_width/height per entry. "
            "Rebuild with: python scripts/build_shsy5y_shape_bank.py --output-dir <same> --force\n"
            "Or set scale_mode to \"occupancy\" in --layout JSON (legacy; distorts cell sizes).",
            file=sys.stderr,
        )
        sys.exit(1)

    out_root = args.output_dir.resolve()
    msk_dir = out_root / "masks"
    msk_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    samples: list[dict] = []

    for k in tqdm(range(args.num_masks), desc="procedural"):
        sem: np.ndarray | None = None
        stats: dict | None = None
        last_err: BaseException | None = None
        for _retry in range(max(1, args.max_gen_retries)):
            try:
                sem_t, stats_t = generate_one_semantic_mask(
                    shapes,
                    areas,
                    origin_hw,
                    args.image_size,
                    args.image_size,
                    shsy_label,
                    cfg,
                    rng,
                    placement_stats=placement_stats,
                )
            except RuntimeError as e:
                last_err = e
                continue
            if not np.any(sem_t > 0):
                last_err = RuntimeError("mask has zero foreground pixels")
                continue
            sem, stats = sem_t, stats_t
            break
        if sem is None or stats is None:
            msg = f"mask index {k}: could not produce non-empty semantic mask after {args.max_gen_retries} tries."
            if last_err is not None:
                msg += f" Last error: {last_err}"
            print(msg, file=sys.stderr)
            raise RuntimeError(msg)

        stem = f"proc_{k:05d}"
        mpath = msk_dir / f"{stem}.png"
        Image.fromarray(sem, mode="L").save(mpath, format="PNG")
        samples.append(
            {
                "mask": f"masks/{stem}.png",
                "synthetic_id": -(k + 1),
                "stats": stats,
            }
        )

    manifest = {
        "policy": "procedural_masks_from_shape_bank",
        "bank_dir": str(bank_dir),
        "placement_stats": str(repo_root / cfg.placement_stats_path)
        if cfg.placement_source == "sparse_train_stats"
        else None,
        "layout_config": asdict(cfg),
        "layout_json": str(args.layout.resolve()) if args.layout is not None else None,
        "seed": args.seed,
        "image_size": args.image_size,
        "shsy5y_semantic_label": shsy_label,
        "num_samples": len(samples),
        "samples": samples,
    }
    with (out_root / "procedural_mask_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(samples)} masks under {out_root}")


if __name__ == "__main__":
    main()
