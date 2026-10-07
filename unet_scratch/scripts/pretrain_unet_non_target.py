from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
SCRATCH_ROOT = SCRIPT_DIR.parent
if str(SCRATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRATCH_ROOT))

from data.datasets import build_pretrain_datasets
from models.unet import UNet
from train_utils import (
    append_epoch_record,
    evaluate_model,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain U-Net on official train split excluding SHSY5Y.")
    parser.add_argument("--repo-root", type=Path, default=SCRATCH_ROOT.parent, help="External repo root with manifests and LIVECell.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRATCH_ROOT / "runs" / "pretrain_non_target",
        help="Output directory under unet_scratch by default.",
    )
    parser.add_argument("--target-category", type=str, default="shsy5y")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_last.pt if present.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_ds, val_ds = build_pretrain_datasets(
        args.repo_root,
        target_category=args.target_category,
        image_size=args.image_size,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_dice = -1.0
    start_epoch = 1
    history = []

    last_ckpt = args.output_dir / "checkpoint_last.pt"
    if args.resume and last_ckpt.exists():
        ckpt = load_checkpoint(last_ckpt, model, optimizer)
        best_val_dice = float(ckpt.get("best_val_dice", -1.0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        history = list(ckpt.get("extra", {}).get("history", []))

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            progress_desc=f"Train epoch {epoch}/{args.epochs}",
            progress_leave=True,
        )
        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            progress_desc=f"Val epoch {epoch}/{args.epochs}",
            progress_leave=True,
        )
        append_epoch_record(history, epoch, train_metrics, val_metrics)

        save_checkpoint(
            last_ckpt,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_dice=best_val_dice,
            extra={"history": history},
        )
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            save_checkpoint(
                args.output_dir / "checkpoint_best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_dice=best_val_dice,
                extra={"history": history},
            )
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_dice={train_metrics['dice']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"best_val={best_val_dice:.4f}"
        )

    payload = {
        "stage": "pretrain_non_target",
        "target_category": args.target_category,
        "repo_root": str(args.repo_root),
        "output_dir": str(args.output_dir),
        "dataset_sizes": {"train": len(train_ds), "val": len(val_ds)},
        "best_val_dice": best_val_dice,
        "history": history,
    }
    write_json(args.output_dir / "metrics_pretrain.json", payload)


if __name__ == "__main__":
    main()
