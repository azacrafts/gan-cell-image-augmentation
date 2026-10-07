"""
Generate Pix2Pix synthetic phase images (same behavior as generate_synthetic_shsy5y.py), then:

- ``excluded_real``: writes ``images/``, ``masks/``, and ``gt/`` (LIVECell phase resized like cGAN) at the same
  level; paired **PSNR** and **LPIPS** use those saved GT files when present.

- ``procedural``: **Fréchet distance** on DINOv2 CLS features (``dinov2_vits14`` by default) between generated
  images and a reference pool of pure-shsy5y official **train** FOVs (subsampled with ``--fid-real-limit``).
  No per-sample GT (manifest lists ``gt``: null).

Writes ``synthetic_metrics.json`` next to ``synthetic_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.synthetic_metrics import (  # noqa: E402
    evaluate_paired_psnr_lpips,
    run_procedural_frechet,
    write_metrics_json,
)
from src.synthesis.pix2pix_synthetic import SyntheticGenerateConfig, generate_synthetic_dataset  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate synthetic shsy5y phase images and run mask-source-specific metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "cgan_shsy5y_pix2pix" / "pix2pix_shsy5y.pt")
    p.add_argument(
        "--sparse-manifest",
        type=Path,
        default=ROOT / "manifests" / "shsy5y_sparse_train_seed42.json",
    )
    p.add_argument("--output-dir", type=Path, required=True, help="Dedicated output directory for this run.")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=0, help="Cap number of masks processed (0 = all).")
    p.add_argument("--save-mask-preview", action="store_true")
    p.add_argument(
        "--mask-source",
        choices=["excluded_real", "procedural"],
        required=True,
        help="excluded_real: PSNR+LPIPS; procedural: DINOv2 Fréchet vs real train pool.",
    )
    p.add_argument("--procedural-mask-dir", type=Path, default=None)
    p.add_argument("--procedural-manifest", type=Path, default=None)
    p.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Only run synthesis; skip PSNR/LPIPS or Fréchet.",
    )
    # Procedural / Fréchet
    p.add_argument("--fid-real-limit", type=int, default=0, help="Max real train FOVs for FD (0 = use all).")
    p.add_argument("--fid-seed", type=int, default=42)
    p.add_argument(
        "--dinov2-backbone",
        default="dinov2_vits14",
        help="torch.hub DINOv2 model name (e.g. dinov2_vits14, dinov2_vitb14).",
    )
    p.add_argument("--metrics-batch-size", type=int, default=16)
    # Paired metrics
    p.add_argument("--lpips-net", default="squeeze", choices=["alex", "vgg", "squeeze"])

    args = p.parse_args()

    if args.mask_source == "procedural" and args.procedural_mask_dir is None:
        print("--procedural-mask-dir is required for --mask-source procedural", file=sys.stderr)
        sys.exit(2)

    repo_root = args.repo_root.resolve()
    out_dir = args.output_dir.resolve()
    device = torch.device(args.device)

    cfg = SyntheticGenerateConfig(
        repo_root=repo_root,
        output_dir=out_dir,
        checkpoint=args.checkpoint.resolve(),
        device=device,
        image_size=args.image_size,
        mask_source=args.mask_source,
        sparse_manifest=args.sparse_manifest.resolve(),
        procedural_mask_dir=args.procedural_mask_dir.resolve() if args.procedural_mask_dir else None,
        procedural_manifest=args.procedural_manifest.resolve() if args.procedural_manifest else None,
        limit=args.limit,
        save_mask_preview=args.save_mask_preview,
    )
    manifest = generate_synthetic_dataset(cfg)

    if args.skip_metrics:
        return

    metrics_payload: dict = {
        "output_dir": str(out_dir),
        "mask_source": args.mask_source,
    }

    if args.mask_source == "excluded_real":
        m = evaluate_paired_psnr_lpips(out_dir, manifest, repo_root, device=device, lpips_net=args.lpips_net)
        metrics_payload["paired_reconstruction"] = m
    else:
        m = run_procedural_frechet(
            out_dir,
            manifest,
            repo_root,
            device=device,
            fid_real_limit=args.fid_real_limit,
            fid_seed=args.fid_seed,
            dinov2_backbone=args.dinov2_backbone,
            metrics_batch_size=args.metrics_batch_size,
        )
        metrics_payload["distribution_fd"] = m

    out_json = out_dir / "synthetic_metrics.json"
    write_metrics_json(out_json, metrics_payload)
    print(f"Wrote metrics: {out_json}")
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
