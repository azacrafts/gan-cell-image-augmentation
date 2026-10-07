"""
Generate synthetic shsy5y phase images using a trained Pix2Pix generator.

- ``--mask-source excluded_real`` (default): rasterize real train masks from pure-shsy5y images
  excluded by the sparse manifest (evaluation-style holdout).
- ``--mask-source procedural``: read semantic masks produced by ``generate_procedural_masks.py``.

Saves paired PNGs + synthetic_manifest.json for experiment C (``images/``, ``masks/``, and for
``excluded_real`` also ``gt/`` ground-truth phase). Semantic masks are uint8 ``0 .. num_classes-1``;
use ``--save-mask-preview`` for visible mask previews.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synthesis.pix2pix_synthetic import SyntheticGenerateConfig, generate_synthetic_dataset  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pix2Pix mask-to-phase synthesis for experiment C.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example pipeline (procedural masks):\n"
        "  python scripts/build_shsy5y_shape_bank.py --output-dir runs/shsy5y_shape_bank_sparse\n"
        "  python scripts/generate_procedural_masks.py --bank-dir runs/shsy5y_shape_bank_sparse "
        "--output-dir runs/procedural_masks_shsy5y --num-masks 100\n"
        "  python scripts/generate_synthetic_shsy5y.py --mask-source procedural "
        "--procedural-mask-dir runs/procedural_masks_shsy5y --output-dir runs/synthetic_shsy5y\n",
    )
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "cgan_shsy5y_pix2pix" / "pix2pix_shsy5y.pt")
    p.add_argument(
        "--sparse-manifest",
        type=Path,
        default=ROOT / "manifests" / "shsy5y_sparse_train_seed42.json",
    )
    p.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "synthetic_shsy5y")
    p.add_argument("--image-size", type=int, default=512, help="Must match cGAN training size (>=512)")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=0, help="if >0, only first N excluded images")
    p.add_argument(
        "--save-mask-preview",
        action="store_true",
        help="Also write masks_preview/*.png scaled 0-255 for quick visual checks (not used by training).",
    )
    p.add_argument(
        "--mask-source",
        choices=["excluded_real", "procedural"],
        default="excluded_real",
        help="excluded_real: COCO masks from excluded shsy5y train images; procedural: pre-built mask folder.",
    )
    p.add_argument(
        "--procedural-mask-dir",
        type=Path,
        default=None,
        help="Directory with procedural_mask_manifest.json + masks/ (required for --mask-source procedural).",
    )
    p.add_argument(
        "--procedural-manifest",
        type=Path,
        default=None,
        help="Defaults to <procedural-mask-dir>/procedural_mask_manifest.json",
    )
    args = p.parse_args()

    if args.mask_source == "procedural" and args.procedural_mask_dir is None:
        print("--procedural-mask-dir is required for --mask-source procedural", file=sys.stderr)
        sys.exit(2)

    cfg = SyntheticGenerateConfig(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        checkpoint=args.checkpoint.resolve(),
        device=torch.device(args.device),
        image_size=args.image_size,
        mask_source=args.mask_source,
        sparse_manifest=args.sparse_manifest.resolve(),
        procedural_mask_dir=args.procedural_mask_dir.resolve() if args.procedural_mask_dir else None,
        procedural_manifest=args.procedural_manifest.resolve() if args.procedural_manifest else None,
        limit=args.limit,
        save_mask_preview=args.save_mask_preview,
    )
    generate_synthetic_dataset(cfg)


if __name__ == "__main__":
    main()
