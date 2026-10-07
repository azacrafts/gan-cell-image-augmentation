"""
Phase 1: Mask R-CNN on LIVECell **official train** with target-line polygons removed —
foreground = all non-target cells (one class). Does **not** use sparse manifests unless
``--sparse-train-manifest`` is set.

Saves ``maskrcnn_pretrain_non_target.pt`` for ``train_baseline.py --downstream instance_binary_pretrained``.
Evaluates mask AP on official val (non-target instances only).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.livecell_instance import LiveCellInstanceDataset, collate_detection_fn  # noqa: E402
from src.eval.coco_instance_map import evaluate_mask_rcnn_mask_ap  # noqa: E402
from src.models.mask_rcnn_binary import build_mask_rcnn_binary  # noqa: E402


def assert_torchvision_cuda_detection_ops(device: torch.device) -> None:
    if device.type != "cuda":
        return
    try:
        import torchvision.ops as tv_ops

        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], device=device)
        scores = torch.tensor([0.9], device=device)
        _ = tv_ops.nms(boxes, scores, 0.5)
    except NotImplementedError as err:
        print(
            "torchvision CUDA ops (e.g. nms) are not available; use --device cpu or reinstall "
            "matching torch+torchvision with CUDA from https://pytorch.org/get-started/locally/",
            file=sys.stderr,
        )
        raise SystemExit(1) from err


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    running = 0.0
    n = 0
    for images, targets, _ in tqdm(loader, desc="train", leave=False):
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        if loss.dim() != 0:
            loss = loss.mean()
        loss.backward()
        optimizer.step()
        running += float(loss.item()) * len(images)
        n += len(images)
    return running / max(1, n)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def main() -> None:
    p = argparse.ArgumentParser(description="Pretrain Mask R-CNN: background vs non-target cells (target polygons dropped).")
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--target-category", type=str, default="shsy5y")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--sparse-train-manifest",
        type=Path,
        default=None,
        help="Optional: same sparse manifest as experiments B/C (non-shsy5y full ∪ 5%% shsy5y).",
    )
    p.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
        help="Train from ImageNet backbone only (random detection heads).",
    )
    args = p.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = args.output_dir or (repo_root / "runs" / "pretrain_maskrcnn_non_target")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    assert_torchvision_cuda_detection_ops(device)

    set_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_ds = LiveCellInstanceDataset(
        repo_root,
        "train",
        target_category_name=args.target_category,
        mode="non_target",
        sparse_train_manifest=args.sparse_train_manifest,
        image_size=args.image_size,
        raw_image=False,
    )
    val_ds = LiveCellInstanceDataset(
        repo_root,
        "val",
        target_category_name=args.target_category,
        mode="non_target",
        sparse_train_manifest=None,
        image_size=args.image_size,
        raw_image=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_detection_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_detection_fn,
        pin_memory=device.type == "cuda",
    )

    model = build_mask_rcnn_binary(num_classes=2, pretrained_backbone=not args.no_pretrained_backbone)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, device, optimizer)
        history.append({"epoch": epoch, "train_loss": loss})
        print(f"Epoch {epoch}/{args.epochs}  train_loss={loss:.4f}")

    print("Evaluating mask AP on official val (non-target instances only)...")
    val_metrics = evaluate_mask_rcnn_mask_ap(
        model,
        val_loader,
        device,
        image_size=args.image_size,
        score_thresh=0.5,
        category_id=1,
        desc="val mask AP",
    )

    ckpt_path = out_dir / "maskrcnn_pretrain_non_target.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)

    results = {
        "phase": "pretrain_non_target_cells",
        "target_category": args.target_category,
        "train": "official_train_non_target_instances",
        "val_eval": "official_val_non_target_instances",
        "sparse_train_manifest": str(args.sparse_train_manifest) if args.sparse_train_manifest else None,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "metrics_val": val_metrics,
        "history": history,
        "checkpoint": str(ckpt_path),
    }
    out_json = out_dir / "pretrain_non_target_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Val mask AP @0.50: {val_metrics['mask_ap50']:.4f}")
    print(f"Val mask AP @0.50:0.95: {val_metrics['mask_ap50_95']:.4f}")
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved metrics: {out_json}")


if __name__ == "__main__":
    main()
