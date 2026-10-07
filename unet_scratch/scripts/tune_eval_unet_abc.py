from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
SCRATCH_ROOT = SCRIPT_DIR.parent
if str(SCRATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRATCH_ROOT))

from data.datasets import build_shsy5y_test_dataset, build_tuning_datasets
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
    parser = argparse.ArgumentParser(description="Tune pretrained U-Net in A/B/C SHSY5Y paradigm and test on SHSY5Y test subset.")
    parser.add_argument("--experiment", type=str, required=True, choices=["A", "B", "C"])
    parser.add_argument("--repo-root", type=Path, default=SCRATCH_ROOT.parent, help="External repo root with manifests and LIVECell.")
    parser.add_argument("--output-root", type=Path, default=SCRATCH_ROOT / "runs", help="Root for A/B/C outputs under unet_scratch.")
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=SCRATCH_ROOT / "runs" / "pretrain_non_target" / "checkpoint_best.pt",
    )
    parser.add_argument(
        "--sparse-manifest",
        type=Path,
        default=SCRATCH_ROOT.parent / "manifests" / "shsy5y_sparse_train_seed42.json",
    )
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=SCRATCH_ROOT.parent / "runs" / "synthetic_shsy5y",
        help="Used only for experiment C.",
    )
    parser.add_argument("--target-category", type=str, default="shsy5y")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--limit-synth", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from experiment checkpoint_last.pt.")
    return parser.parse_args()


def _experiment_dir(output_root: Path, experiment: str) -> Path:
    return output_root / f"exp_{experiment.lower()}"


def _write_summary(output_root: Path, experiment: str, result_payload: Dict[str, object]) -> None:
    summary_path = output_root / "summary_abc.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {"experiments": {}}
    summary["experiments"][experiment.lower()] = result_payload
    write_json(summary_path, summary)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    exp_dir = _experiment_dir(args.output_root, args.experiment)
    exp_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if not args.pretrained_checkpoint.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_checkpoint}")
    load_checkpoint(args.pretrained_checkpoint, model, optimizer=None)

    train_ds, val_ds = build_tuning_datasets(
        args.repo_root,
        target_category=args.target_category,
        image_size=args.image_size,
        experiment=args.experiment,
        sparse_manifest_path=args.sparse_manifest,
        synthetic_dir=args.synthetic_dir if args.experiment == "C" else None,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        limit_synth=args.limit_synth,
    )
    test_ds = build_shsy5y_test_dataset(
        args.repo_root,
        target_category=args.target_category,
        image_size=args.image_size,
        limit_test=args.limit_test,
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
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    history = []
    best_val_dice = -1.0
    start_epoch = 1

    last_ckpt = exp_dir / "checkpoint_last.pt"
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
            progress_desc=f"Train {args.experiment} epoch {epoch}/{args.epochs}",
            progress_leave=True,
        )
        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            progress_desc=f"Val {args.experiment} epoch {epoch}/{args.epochs}",
            progress_leave=True,
        )
        append_epoch_record(history, epoch, train_metrics, val_metrics)

        save_checkpoint(
            last_ckpt,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_dice=best_val_dice,
            extra={"history": history, "experiment": args.experiment},
        )
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            save_checkpoint(
                exp_dir / "checkpoint_best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_dice=best_val_dice,
                extra={"history": history, "experiment": args.experiment},
            )
        print(
            f"Epoch {epoch}/{args.epochs} [{args.experiment}] | "
            f"train_dice={train_metrics['dice']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"best_val={best_val_dice:.4f}"
        )

    load_checkpoint(exp_dir / "checkpoint_best.pt", model, optimizer=None)
    test_metrics = evaluate_model(
        model,
        test_loader,
        device,
        progress_desc=f"Test {args.experiment} on SHSY5Y",
        progress_leave=True,
    )
    result_payload = {
        "experiment": args.experiment,
        "target_category": args.target_category,
        "repo_root": str(args.repo_root),
        "output_dir": str(exp_dir),
        "dataset_sizes": {"train": len(train_ds), "val": len(val_ds), "test_shsy5y_only": len(test_ds)},
        "best_val_dice": best_val_dice,
        "test_metrics": test_metrics,
        "history": history,
        "synthetic_dir": str(args.synthetic_dir) if args.experiment == "C" else None,
        "sparse_manifest": str(args.sparse_manifest),
    }
    write_json(exp_dir / "results_test_shsy5y.json", result_payload)
    _write_summary(args.output_root, args.experiment, result_payload)


if __name__ == "__main__":
    main()
