"""
Train mask-to-phase Pix2Pix generator on full LIVECell **shsy5y** train images (cGAN-Seg–style).
Uses semantic one-hot masks as conditioning; real phase in [0,1] RGB.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.coco_filters import image_has_category  # noqa: E402
from src.data.coco_livecell import LiveCellSemanticDataset  # noqa: E402
from src.models.pix2pix import NLayerDiscriminator, UNetGenerator, init_weights  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def mask_to_onehot(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(B,H,W) int64 -> (B,C,H,W) float."""
    b, h, w = mask.shape
    out = torch.zeros(b, num_classes, h, w, device=mask.device, dtype=torch.float32)
    for c in range(num_classes):
        out[:, c] = (mask == c).float()
    return out


def collate_cgan(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    return {"image": images, "mask": masks}


def main() -> None:
    p = argparse.ArgumentParser(description="Train Pix2Pix cGAN on shsy5y LIVECell train")
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda-l1", type=float, default=100.0)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument(
        "--train-mode",
        choices=("target", "non_target"),
        default="target",
        help=(
            "target: shsy5y-only FOVs (optionally sparse via --sparse-train-manifest). "
            "non_target: all train FOVs with ZERO shsy5y instances (pretrain stage)."
        ),
    )
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional Pix2Pix checkpoint to initialize generator/discriminator weights from.",
    )
    p.add_argument(
        "--sparse-train-manifest",
        type=Path,
        default=None,
        help=(
            "Optional sparse shsy5y manifest JSON (e.g. manifests/shsy5y_sparse_train_seed42.json). "
            "When set, only the kept shsy5y image IDs are used (with only_shsy5y=True)."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.image_size < 512:
        print("image_size must be >= 512 (8-level U-Net requires H,W divisible by 256; bottleneck stays >= 2x2).", file=sys.stderr)
        sys.exit(2)

    repo_root = args.repo_root.resolve()
    out_dir = args.output_dir or (repo_root / "runs" / "cgan_shsy5y_pix2pix")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    set_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    init_ckpt: Path | None = None
    if args.init_checkpoint is not None:
        ip = Path(args.init_checkpoint)
        if not ip.is_file():
            ip = repo_root / ip
        if not ip.is_file():
            raise FileNotFoundError(f"Init checkpoint not found: {args.init_checkpoint}")
        init_ckpt = ip.resolve()

    sparse_manifest: Path | None = None
    if args.sparse_train_manifest is not None:
        mp = Path(args.sparse_train_manifest)
        if not mp.is_file():
            mp = repo_root / "manifests" / mp
        if not mp.is_file():
            raise FileNotFoundError(f"Sparse manifest not found: {args.sparse_train_manifest}")
        sparse_manifest = mp.resolve()

    if args.train_mode == "target":
        ds: torch.utils.data.Dataset = LiveCellSemanticDataset(
            repo_root,
            "train",
            image_size=args.image_size,
            augment_train=True,
            only_shsy5y=True,
            sparse_train_manifest=sparse_manifest,
            raw_image=True,
        )
    else:
        # Pretrain on all official train images that contain ZERO shsy5y instances.
        base = LiveCellSemanticDataset(
            repo_root,
            "train",
            image_size=args.image_size,
            augment_train=True,
            only_shsy5y=False,
            sparse_train_manifest=None,
            raw_image=True,
        )
        indices = [
            i
            for i in range(len(base))
            if not image_has_category(int(base.images[i]["id"]), base._ann_by_image, base.shsy5y_id)
        ]
        ds = torch.utils.data.Subset(base, indices)

    if args.limit_train > 0:
        n = min(args.limit_train, len(ds))
        ds = torch.utils.data.Subset(ds, list(range(n)))

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_cgan,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )

    base_ds = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    num_classes = base_ds.num_classes

    net_g = UNetGenerator(mask_channels=num_classes, base=64).to(device)
    net_d = NLayerDiscriminator(mask_channels=num_classes, image_channels=3, ndf=64, n_layers=3).to(device)
    net_g.apply(init_weights)
    net_d.apply(init_weights)

    if init_ckpt is not None:
        try:
            ckpt = torch.load(init_ckpt, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(init_ckpt, map_location=device)
        if not isinstance(ckpt, dict) or "generator" not in ckpt:
            raise ValueError(f"Init checkpoint does not look like a Pix2Pix checkpoint: {init_ckpt}")
        net_g.load_state_dict(ckpt["generator"], strict=True)
        if "discriminator" in ckpt:
            net_d.load_state_dict(ckpt["discriminator"], strict=True)

    opt_g = torch.optim.Adam(net_g.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        net_g.train()
        net_d.train()
        ep_d = 0.0
        ep_g = 0.0
        n_batches = 0
        for batch in tqdm(loader, desc=f"epoch {epoch}/{args.epochs}"):
            real = batch["image"].to(device)
            mask = batch["mask"].to(device)
            oh = mask_to_onehot(mask, num_classes)

            # --- D ---
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake = net_g(oh)
            pred_real = net_d(oh, real)
            pred_fake = net_d(oh, fake.detach())
            loss_d_real = bce(pred_real, torch.ones_like(pred_real))
            loss_d_fake = bce(pred_fake, torch.zeros_like(pred_fake))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
            loss_d.backward()
            opt_d.step()

            # --- G ---
            opt_g.zero_grad(set_to_none=True)
            fake = net_g(oh)
            pred_fake = net_d(oh, fake)
            loss_g_adv = bce(pred_fake, torch.ones_like(pred_fake))
            loss_g_l1 = l1(fake, real) * args.lambda_l1
            loss_g = loss_g_adv + loss_g_l1
            loss_g.backward()
            opt_g.step()

            ep_d += loss_d.item()
            ep_g += loss_g.item()
            n_batches += 1

        row = {
            "epoch": epoch,
            "loss_d": ep_d / max(1, n_batches),
            "loss_g": ep_g / max(1, n_batches),
        }
        history.append(row)
        print(f"Epoch {epoch}: loss_d={row['loss_d']:.4f} loss_g={row['loss_g']:.4f}")

    ckpt = {
        "generator": net_g.state_dict(),
        "discriminator": net_d.state_dict(),
        "num_classes": num_classes,
        "image_size": args.image_size,
        "mask_channels": num_classes,
    }
    torch.save(ckpt, out_dir / "pix2pix_shsy5y.pt")
    with (out_dir / "train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    meta = {
        "checkpoint": str((out_dir / "pix2pix_shsy5y.pt").relative_to(repo_root)),
        "train_mode": args.train_mode,
        "train_subset": (
            "shsy5y_only_sparse_manifest"
            if (args.train_mode == "target" and sparse_manifest is not None)
            else ("shsy5y_only_full_train" if args.train_mode == "target" else "no_shsy5y_instances_full_train")
        ),
        "sparse_train_manifest": str(sparse_manifest.relative_to(repo_root)) if sparse_manifest else None,
        "init_checkpoint": str(init_ckpt.relative_to(repo_root)) if init_ckpt else None,
        "image_size": args.image_size,
        "num_classes": num_classes,
    }
    with (out_dir / "cgan_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved generator to {out_dir / 'pix2pix_shsy5y.pt'}")


if __name__ == "__main__":
    main()
