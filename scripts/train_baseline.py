"""
Experiment A/B/C: semantic segmentation (DeepLabV3) or binary instance (Mask R-CNN) on LIVECell.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.coco_livecell import LiveCellSemanticDataset, SyntheticPairedSemanticDataset  # noqa: E402
from src.data.livecell_instance import (  # noqa: E402
    LiveCellInstanceDataset,
    SyntheticPairedInstanceDataset,
    collate_detection_fn,
)
from src.eval.coco_instance_map import (  # noqa: E402
    evaluate_mask_rcnn_mask_ap,
    evaluate_pseudo_instance_ap_semantic,
)
from src.eval.instance_binary_metrics import evaluate_mask_rcnn_pixel_metrics  # noqa: E402
from src.models.mask_rcnn_binary import build_mask_rcnn_binary  # noqa: E402


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    return {"image": images, "mask": masks, "image_id": [b["image_id"] for b in batch]}


def _semantic_target_label(meta_ds: LiveCellSemanticDataset, target_category_name: str) -> int:
    tid = next(int(c["id"]) for c in meta_ds.categories if c.get("name") == target_category_name)
    return int(meta_ds.cat_id_to_label[tid])


@torch.no_grad()
def evaluate_semantic(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    category_names: list[str],
    *,
    use_amp: bool = False,
) -> dict[str, Any]:
    model.eval()
    total_correct = 0
    total_pixels = 0
    inter = torch.zeros(num_classes, dtype=torch.float64, device=device)
    union = torch.zeros(num_classes, dtype=torch.float64, device=device)
    nb = device.type == "cuda"

    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["image"].to(device, non_blocking=nb)
        y = batch["mask"].to(device, non_blocking=nb)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(x)["out"]
            pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_pixels += y.numel()

        for c in range(num_classes):
            p_c = pred == c
            t_c = y == c
            inter[c] += (p_c & t_c).sum().double()
            union[c] += (p_c | t_c).sum().double()

    iou_per_class: dict[str, float] = {}
    for c in range(num_classes):
        name = "background" if c == 0 else category_names[c - 1]
        u = union[c].item()
        if u <= 0:
            iou_per_class[name] = float("nan")
        else:
            iou_per_class[name] = float((inter[c] / union[c]).item())

    valid = [v for v in iou_per_class.values() if not math.isnan(v)]
    miou = sum(valid) / len(valid) if valid else float("nan")

    fg = [iou_per_class[n] for n in category_names if not math.isnan(iou_per_class[n])]
    miou_fg = sum(fg) / len(fg) if fg else float("nan")

    return {
        "pixel_accuracy": total_correct / max(1, total_pixels),
        "miou_all_classes_including_bg": miou,
        "miou_foreground_8_lines": miou_fg,
        "iou_per_class": iou_per_class,
    }


def train_one_epoch_semantic(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: Optional[torch.amp.GradScaler],
    *,
    use_amp: bool,
) -> float:
    model.train()
    running = 0.0
    n = 0
    nb = device.type == "cuda"
    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["image"].to(device, non_blocking=nb)
        y = batch["mask"].to(device, non_blocking=nb)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            o = model(x)
            loss = criterion(o["out"], y)
            if "aux" in o:
                loss = loss + 0.5 * criterion(o["aux"], y)
            if loss.dim() != 0:
                loss = loss.mean()
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += float(loss.detach().item()) * x.size(0)
        n += x.size(0)
    return running / max(1, n)


def train_one_epoch_mask_rcnn(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    *,
    use_amp: bool,
) -> float:
    model.train()
    running = 0.0
    n = 0
    nb = device.type == "cuda"
    for images, targets, _ in tqdm(loader, desc="train", leave=False):
        images = [im.to(device, non_blocking=nb) for im in images]
        targets = [{k: v.to(device, non_blocking=nb) for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if loss.dim() != 0:
                loss = loss.mean()
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += float(loss.detach().item()) * len(images)
        n += len(images)
    return running / max(1, n)


@torch.no_grad()
def eval_mask_rcnn_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool = False,
) -> float:
    model.train()
    running = 0.0
    n = 0
    nb = device.type == "cuda"
    for images, targets, _ in tqdm(loader, desc="val", leave=False):
        images = [im.to(device, non_blocking=nb) for im in images]
        targets = [{k: v.to(device, non_blocking=nb) for k, v in t.items()} for t in targets]
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if loss.dim() != 0:
                loss = loss.mean()
        running += float(loss.detach().item()) * len(images)
        n += len(images)
    return running / max(1, n)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def clear_training_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def configure_cuda_performance(device: torch.device, *, deterministic: bool) -> None:
    """Fast path: cuDNN autotune + TF32. Deterministic path: reproducible but slower."""
    if device.type != "cuda":
        return
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def dataloader_kwargs(device: torch.device, workers: int) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = max(2, min(8, workers * 2))
    return kw


def train_loader_drop_last_for_batchnorm(n_samples: int, batch_size: int) -> bool:
    """
    Training BatchNorm (DeepLab / ResNet) fails on batches of size 1. When the dataset
    would yield a final batch of exactly one sample, drop it (only if there is at least
    one full batch before it).
    """
    if n_samples <= 1:
        return False
    if n_samples < batch_size:
        return False
    return (n_samples % batch_size) == 1


def pseudo_instance_ap_dataloader_kwargs(workers: int) -> dict[str, Any]:
    """
    Pseudo-instance AP runs after semantic eval; pin_memory + many workers on CUDA
    can exhaust device memory (esp. Windows). No pin_memory, half workers; pair with
    batch_size = 2 * training batch for fewer GPU round-trips.
    """
    nw = max(0, workers // 2)
    kw: dict[str, Any] = {
        "num_workers": nw,
        "pin_memory": False,
    }
    if nw > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return kw


def assert_torchvision_cuda_detection_ops(device: torch.device) -> None:
    """
    Mask R-CNN needs torchvision CUDA ops (e.g. nms). A mismatched pip/conda torch+torchvision
    pair often exposes CPU-only torchvision while tensors live on GPU → NotImplementedError.
    """
    if device.type != "cuda":
        return
    try:
        import torchvision.ops as tv_ops

        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], device=device)
        scores = torch.tensor([0.9], device=device)
        _ = tv_ops.nms(boxes, scores, 0.5)
    except NotImplementedError as err:
        print(
            "torchvision CUDA ops (e.g. nms) are not available for this install, but --device is CUDA.\n"
            "Typical cause: PyTorch was built with CUDA while torchvision is CPU-only or from a different build.\n"
            "Fix: reinstall matching wheels from https://pytorch.org/get-started/locally/ (same CUDA version),\n"
            "     or run on CPU:  --device cpu",
            file=sys.stderr,
        )
        raise SystemExit(1) from err


def load_maskrcnn_checkpoint(model: nn.Module, path: Path, device: torch.device) -> None:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="LIVECell downstream: semantic (DeepLab) or binary instance (Mask R-CNN), experiments A/B/C."
    )
    p.add_argument(
        "--downstream",
        choices=("semantic", "instance_binary", "instance_binary_pretrained"),
        default="semantic",
        help="semantic=DeepLab 8-line; instance_*=Mask R-CNN target-only instances.",
    )
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Required for instance_binary_pretrained: checkpoint from pretrain_maskrcnn_non_target.py.",
    )
    p.add_argument(
        "--target-category",
        type=str,
        default="shsy5y",
        help="COCO category name for the target line (instance paths and sparse manifest logic).",
    )
    p.add_argument(
        "--pseudo-instance-ap",
        action="store_true",
        help="For semantic only: add pseudo mask AP50/AP50-95 on final test (argmax vs instance GT). "
        "Redundant if --semantic-target-only (that mode always runs pseudo AP on test and periodic evals).",
    )
    p.add_argument(
        "--semantic-target-only",
        action="store_true",
        help="DeepLab only: train/val/test on images with ≥1 target line (--target-category), "
        "like instance_binary image coverage; masks stay multi-class. Enables pseudo mask AP on periodic and final test.",
    )
    p.add_argument(
        "--experiment",
        choices=("A", "B", "C"),
        default="A",
        help="A=full train; B=sparse shsy5y (5%%); C=sparse + synthetic shsy5y (see --synthetic-dir).",
    )
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader workers (pseudo-instance AP eval uses half this, no pin_memory).",
    )
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Single GPU, e.g. cuda:0 or cuda:1. Mask R-CNN on CUDA needs torchvision CUDA ops; if nms fails, use cpu.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--limit-train", type=int, default=0, help="if >0, use only first N train images")
    p.add_argument("--limit-test", type=int, default=0, help="if >0, use only first N test images")
    p.add_argument("--val-eval", action="store_true", help="also run eval on val after each epoch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--sparse-train-manifest",
        type=Path,
        default=None,
        help="For B/C: JSON with kept_shsy5y_image_ids (default: manifests/shsy5y_sparse_train_seed42.json).",
    )
    p.add_argument(
        "--synthetic-dir",
        type=Path,
        default=None,
        help="For C: folder with synthetic_manifest.json + images/ + masks/.",
    )
    p.add_argument(
        "--mask-rcnn-pretrained-backbone",
        action="store_true",
        default=True,
        help="Use COCO Mask R-CNN weights + 2-class heads (instance_binary scratch).",
    )
    p.add_argument(
        "--no-mask-rcnn-pretrained-backbone",
        action="store_false",
        dest="mask_rcnn_pretrained_backbone",
        help="Train Mask R-CNN from ImageNet backbone only (random detection heads).",
    )
    p.add_argument(
        "--instance-test-full-official",
        action="store_true",
        help="Instance path: evaluate on full official test split. Default: only test images with ≥1 target instance (same subset for A/B/C).",
    )
    p.add_argument(
        "--checkpoint-best-filename",
        type=str,
        default="checkpoint_best_train.pt",
        help="In --output-dir: overwritten whenever epoch train loss improves (best checkpoint so far).",
    )
    p.add_argument(
        "--test-every-n-checkpoint-saves",
        type=int,
        default=5,
        help="After every N train-loss improvements (checkpoint overwrites), run full test eval. 0 disables.",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Slower reproducible mode (cuDNN deterministic, no autotune). Default is fast non-deterministic training.",
    )
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision on CUDA (default: AMP enabled on GPU for speed).",
    )
    args = p.parse_args()

    if (
        args.downstream == "semantic"
        and args.semantic_target_only
        and args.checkpoint_best_filename == "checkpoint_best_train.pt"
    ):
        args.checkpoint_best_filename = f"checkpoint_best_deeplab_{args.experiment.lower()}_target.pt"

    if args.downstream == "instance_binary_pretrained":
        if args.init_checkpoint is None or not Path(args.init_checkpoint).is_file():
            print("--init-checkpoint must point to an existing file for instance_binary_pretrained.", file=sys.stderr)
            sys.exit(1)

    if args.batch_size is None:
        # Slightly larger default batch for instance path improves GPU utilization vs 4.
        args.batch_size = 6 if args.downstream != "semantic" else 16

    repo_root = args.repo_root.resolve()
    if args.output_dir is None:
        sub = {"A": "exp_a_baseline", "B": "exp_b_sparse", "C": "exp_c_sparse_cgan"}[args.experiment]
        args.output_dir = repo_root / "runs" / sub
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    set_seed(args.seed)
    configure_cuda_performance(device, deterministic=args.deterministic)

    sparse_manifest: Path | None = None
    if args.experiment in ("B", "C"):
        sparse_manifest = args.sparse_train_manifest or (
            repo_root / "manifests" / "shsy5y_sparse_train_seed42.json"
        )
        if not sparse_manifest.is_file():
            print(f"Sparse manifest not found: {sparse_manifest}", file=sys.stderr)
            sys.exit(1)

    if args.downstream == "semantic":
        _run_semantic(args, repo_root, device, sparse_manifest)
    else:
        _run_instance(args, repo_root, device, sparse_manifest)


def _run_semantic(
    args: argparse.Namespace,
    repo_root: Path,
    device: torch.device,
    sparse_manifest: Path | None,
) -> None:
    st_only = args.semantic_target_only
    sem_kw: dict[str, Any] = {
        "only_images_with_target_category": st_only,
        "target_category_name": args.target_category,
    }
    train_ds: LiveCellSemanticDataset | ConcatDataset
    train_ds = LiveCellSemanticDataset(
        repo_root,
        "train",
        image_size=args.image_size,
        augment_train=True,
        sparse_train_manifest=sparse_manifest,
        **sem_kw,
    )
    if args.experiment == "C":
        syn_root = args.synthetic_dir or (repo_root / "runs" / "synthetic_shsy5y")
        syn_root = syn_root.resolve()
        if not (syn_root / "synthetic_manifest.json").is_file():
            print(f"Experiment C requires synthetic data at {syn_root / 'synthetic_manifest.json'}", file=sys.stderr)
            sys.exit(1)
        synth_ds = SyntheticPairedSemanticDataset(
            syn_root,
            manifest_name="synthetic_manifest.json",
            image_size=args.image_size,
            raw_image=False,
        )
        train_ds = ConcatDataset([train_ds, synth_ds])
        print(f"Experiment C: concat sparse train ({len(train_ds.datasets[0])}) + synthetic ({len(train_ds.datasets[1])})")

    val_ds = LiveCellSemanticDataset(
        repo_root,
        "val",
        image_size=args.image_size,
        augment_train=False,
        **sem_kw,
    )
    test_ds = LiveCellSemanticDataset(
        repo_root,
        "test",
        image_size=args.image_size,
        augment_train=False,
        **sem_kw,
    )

    meta_ds: LiveCellSemanticDataset = (
        train_ds.datasets[0] if isinstance(train_ds, ConcatDataset) else train_ds
    )
    category_names = [c["name"] for c in sorted(meta_ds.categories, key=lambda x: int(x["id"]))]
    num_classes = meta_ds.num_classes

    if args.limit_train > 0:
        train_ds = Subset(train_ds, list(range(min(args.limit_train, len(train_ds)))))
    if args.limit_test > 0:
        test_ds = Subset(test_ds, list(range(min(args.limit_test, len(test_ds)))))

    if len(train_ds) == 0:
        print("Train dataset is empty after filtering (--semantic-target-only or --limit-train?).", file=sys.stderr)
        sys.exit(1)
    if len(train_ds) == 1:
        print(
            "Train dataset has only 1 image; DeepLab BatchNorm cannot train with a single-sample batch. "
            "Relax --semantic-target-only / --limit-train or use a smaller --batch-size with more data.",
            file=sys.stderr,
        )
        sys.exit(1)

    train_gen = torch.Generator()
    train_gen.manual_seed(args.seed)
    dl_common = dataloader_kwargs(device, args.workers)
    train_drop_last = train_loader_drop_last_for_batchnorm(len(train_ds), args.batch_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        generator=train_gen,
        drop_last=train_drop_last,
        **dl_common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        **dl_common,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        **dl_common,
    )

    inst_test_loader_for_pseudo: DataLoader | None = None
    if st_only:
        inst_test = LiveCellInstanceDataset(
            repo_root,
            "test",
            target_category_name=args.target_category,
            mode="target",
            sparse_train_manifest=None,
            image_size=args.image_size,
            raw_image=False,
            test_only_images_with_target=True,
        )
        if args.limit_test > 0:
            inst_test = Subset(inst_test, list(range(min(args.limit_test, len(inst_test)))))
        pseudo_ap_bs = max(1, min(args.batch_size * 2, 32))
        inst_test_loader_for_pseudo = DataLoader(
            inst_test,
            batch_size=pseudo_ap_bs,
            shuffle=False,
            collate_fn=collate_detection_fn,
            **pseudo_instance_ap_dataloader_kwargs(args.workers),
        )

    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights, num_classes=21)
    last = model.classifier[-1]
    aux_last = model.aux_classifier[-1]
    assert isinstance(last, nn.Conv2d) and isinstance(aux_last, nn.Conv2d)
    model.classifier[-1] = nn.Conv2d(last.in_channels, num_classes, 1, bias=last.bias is not None)
    model.aux_classifier[-1] = nn.Conv2d(
        aux_last.in_channels, num_classes, 1, bias=aux_last.bias is not None
    )
    model.to(device)

    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_path = args.output_dir / args.checkpoint_best_filename
    best_train_loss = float("inf")
    checkpoint_saves = 0
    periodic_test_evals: list[dict[str, Any]] = []

    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch_semantic(
            model, train_loader, device, optimizer, criterion, scaler, use_amp=use_amp
        )
        row: dict[str, Any] = {"epoch": epoch, "train_loss": loss}
        if args.val_eval:
            val_metrics = evaluate_semantic(
                model, val_loader, device, num_classes, category_names, use_amp=use_amp
            )
            row["val"] = val_metrics

        if loss < best_train_loss:
            best_train_loss = loss
            checkpoint_saves += 1
            torch.save(model.state_dict(), best_path)
            row["best_train_loss_checkpoint"] = True
            row["checkpoint_save_index"] = checkpoint_saves
            print(
                f"Epoch {epoch}/{args.epochs}  train_loss={loss:.4f}  "
                f"→ best checkpoint #{checkpoint_saves} → {best_path.name}"
            )
            if (
                args.test_every_n_checkpoint_saves > 0
                and checkpoint_saves % args.test_every_n_checkpoint_saves == 0
            ):
                clear_training_memory(device)
                model.eval()
                pt = evaluate_semantic(
                    model, test_loader, device, num_classes, category_names, use_amp=use_amp
                )
                if inst_test_loader_for_pseudo is not None:
                    tgt_label = _semantic_target_label(meta_ds, args.target_category)
                    clear_training_memory(device)
                    pseudo_p = evaluate_pseudo_instance_ap_semantic(
                        model,
                        inst_test_loader_for_pseudo,
                        device,
                        target_semantic_label=tgt_label,
                        image_size=args.image_size,
                        desc="periodic pseudo mask AP",
                    )
                    pt = {
                        **pt,
                        **pseudo_p,
                        "pseudo_instance_ap_note": (
                            "CC on argmax target channel vs COCO target instances (approximate)."
                        ),
                    }
                clear_training_memory(device)
                model.train()
                periodic_test_evals.append(
                    {
                        "epoch": epoch,
                        "train_loss_at_save": loss,
                        "checkpoint_save_index": checkpoint_saves,
                        "metrics_test": pt,
                    }
                )
                if inst_test_loader_for_pseudo is not None:
                    print(
                        f"  periodic test (after save #{checkpoint_saves}): "
                        f"pix_acc={pt['pixel_accuracy']:.4f}  "
                        f"miou_fg={pt['miou_foreground_8_lines']:.4f}  "
                        f"pseudo_AP50={pt.get('pseudo_mask_ap50', float('nan')):.4f}  "
                        f"pseudo_AP50_95={pt.get('pseudo_mask_ap50_95', float('nan')):.4f}"
                    )
                else:
                    print(
                        f"  periodic test (after save #{checkpoint_saves}): "
                        f"pix_acc={pt['pixel_accuracy']:.4f}  "
                        f"miou_fg={pt['miou_foreground_8_lines']:.4f}"
                    )
        else:
            print(f"Epoch {epoch}/{args.epochs}  train_loss={loss:.4f}")

        history.append(row)

    print("Evaluating on official test split...")
    clear_training_memory(device)
    test_metrics = evaluate_semantic(
        model, test_loader, device, num_classes, category_names, use_amp=use_amp
    )
    clear_training_memory(device)

    if st_only:
        if inst_test_loader_for_pseudo is None:
            raise RuntimeError("inst_test_loader_for_pseudo must be set when --semantic-target-only")
        tgt_label = _semantic_target_label(meta_ds, args.target_category)
        clear_training_memory(device)
        pseudo = evaluate_pseudo_instance_ap_semantic(
            model,
            inst_test_loader_for_pseudo,
            device,
            target_semantic_label=tgt_label,
            image_size=args.image_size,
            desc="pseudo instance AP (test)",
        )
        clear_training_memory(device)
        test_metrics = {**test_metrics, **pseudo}
        test_metrics["pseudo_instance_ap_note"] = (
            "CC on argmax target channel vs COCO target instances (approximate)."
        )
    elif args.pseudo_instance_ap:
        inst_test = LiveCellInstanceDataset(
            repo_root,
            "test",
            target_category_name=args.target_category,
            mode="target",
            sparse_train_manifest=None,
            image_size=args.image_size,
            raw_image=False,
        )
        if args.limit_test > 0:
            inst_test = Subset(inst_test, list(range(min(args.limit_test, len(inst_test)))))
        pseudo_ap_bs_alt = max(1, min(args.batch_size * 2, 32))
        inst_loader = DataLoader(
            inst_test,
            batch_size=pseudo_ap_bs_alt,
            shuffle=False,
            collate_fn=collate_detection_fn,
            **pseudo_instance_ap_dataloader_kwargs(args.workers),
        )
        tgt_label = _semantic_target_label(meta_ds, args.target_category)
        clear_training_memory(device)
        pseudo = evaluate_pseudo_instance_ap_semantic(
            model,
            inst_loader,
            device,
            target_semantic_label=tgt_label,
            image_size=args.image_size,
            desc="pseudo instance AP",
        )
        clear_training_memory(device)
        test_metrics = {**test_metrics, **pseudo}
        test_metrics["pseudo_instance_ap_note"] = (
            "CC on argmax target channel vs COCO target instances (approximate)."
        )

    exp_key = {
        "A": "A_baseline_full_train",
        "B": "B_sparse_shsy5y",
        "C": "C_sparse_shsy5y_plus_cgan_synth",
    }[args.experiment]
    train_desc = "full_official_train"
    if args.experiment == "A":
        train_desc = "full_official_train"
    elif args.experiment == "B":
        train_desc = str(sparse_manifest)
    else:
        train_desc = f"{sparse_manifest} + synthetic:{args.synthetic_dir or (repo_root / 'runs' / 'synthetic_shsy5y')}"
    if st_only:
        train_desc = f"{train_desc} | semantic_target_only (images with ≥1 {args.target_category} instance)"

    out: dict[str, Any] = {
        "experiment": exp_key,
        "experiment_letter": args.experiment,
        "downstream": "semantic",
        "semantic_target_only": st_only,
        "init_checkpoint": None,
        "target_category": args.target_category,
        "train": train_desc,
        "test": (
            f"official_test_images_with_{args.target_category}"
            if st_only
            else "official_test"
        ),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "limit_train": args.limit_train or None,
        "limit_test": args.limit_test or None,
        "metrics_test": test_metrics,
        "history": history,
        "best_train_loss": best_train_loss,
        "checkpoint_best_path": str(best_path),
        "checkpoint_saves_on_improve": checkpoint_saves,
        "periodic_test_eval": periodic_test_evals,
        "test_every_n_checkpoint_saves": args.test_every_n_checkpoint_saves,
    }
    res_suffix = "_target" if st_only else ""
    out_path = args.output_dir / f"results_exp_{args.experiment.lower()}{res_suffix}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    torch.save(
        model.state_dict(),
        args.output_dir / f"deeplabv3_exp_{args.experiment.lower()}{res_suffix}.pt",
    )

    print()
    label = {
        "A": "Experiment A - baseline (full train)",
        "B": "Experiment B - sparse shsy5y train",
        "C": "Experiment C - sparse + cGAN synthetic shsy5y",
    }[args.experiment]
    if st_only:
        label = f"{label} (target-line images only)"
    print(f"=== Test metrics ({label}, semantic) ===")
    print(f"Pixel accuracy: {test_metrics['pixel_accuracy']:.4f}")
    print(f"mIoU (all 9 incl. bg): {test_metrics['miou_all_classes_including_bg']:.4f}")
    print(f"mIoU (8 foreground lines): {test_metrics['miou_foreground_8_lines']:.4f}")
    print("Per-class IoU:")
    for name, v in test_metrics["iou_per_class"].items():
        if name == "background":
            continue
        mark = "  <-- shsy5y" if name == "shsy5y" else ""
        print(f"  {name:8s}  {v:.4f}{mark}")
    if "pseudo_mask_ap50" in test_metrics:
        print(
            f"Pseudo mask AP50 / AP50-95: {test_metrics.get('pseudo_mask_ap50', float('nan')):.4f} / "
            f"{test_metrics.get('pseudo_mask_ap50_95', float('nan')):.4f}"
        )
    print(f"\nSaved: {out_path}")


def _run_instance(
    args: argparse.Namespace,
    repo_root: Path,
    device: torch.device,
    sparse_manifest: Path | None,
) -> None:
    assert_torchvision_cuda_detection_ops(device)

    train_ds = LiveCellInstanceDataset(
        repo_root,
        "train",
        target_category_name=args.target_category,
        mode="target",
        sparse_train_manifest=sparse_manifest,
        image_size=args.image_size,
        raw_image=False,
    )
    if args.experiment == "C":
        syn_root = args.synthetic_dir or (repo_root / "runs" / "synthetic_shsy5y")
        syn_root = syn_root.resolve()
        if not (syn_root / "synthetic_manifest.json").is_file():
            print(f"Experiment C requires synthetic data at {syn_root / 'synthetic_manifest.json'}", file=sys.stderr)
            sys.exit(1)
        ref_sem = LiveCellSemanticDataset(
            repo_root,
            "train",
            image_size=args.image_size,
            augment_train=False,
            sparse_train_manifest=sparse_manifest,
        )
        shsy_label = int(ref_sem.cat_id_to_label[ref_sem.shsy5y_id])
        synth_inst = SyntheticPairedInstanceDataset(
            syn_root,
            shsy5y_semantic_label=shsy_label,
            manifest_name="synthetic_manifest.json",
            image_size=args.image_size,
            raw_image=False,
        )
        train_ds = ConcatDataset([train_ds, synth_inst])
        print(f"Experiment C: concat sparse instance train ({len(train_ds.datasets[0])}) + synthetic ({len(train_ds.datasets[1])})")

    val_ds = LiveCellInstanceDataset(
        repo_root,
        "val",
        target_category_name=args.target_category,
        mode="target",
        sparse_train_manifest=None,
        image_size=args.image_size,
        raw_image=False,
    )
    test_target_only = not args.instance_test_full_official
    test_ds = LiveCellInstanceDataset(
        repo_root,
        "test",
        target_category_name=args.target_category,
        mode="target",
        sparse_train_manifest=None,
        image_size=args.image_size,
        raw_image=False,
        test_only_images_with_target=test_target_only,
    )
    if len(test_ds) == 0:
        print(
            "Test set is empty after filtering (no images with target instances?). "
            "Use --instance-test-full-official or check annotations.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.limit_train > 0:
        train_ds = Subset(train_ds, list(range(min(args.limit_train, len(train_ds)))))
    if args.limit_test > 0:
        test_ds = Subset(test_ds, list(range(min(args.limit_test, len(test_ds)))))

    if len(train_ds) == 1:
        print(
            "Train dataset has only 1 image; BatchNorm cannot train with a single-sample batch.",
            file=sys.stderr,
        )
        sys.exit(1)

    train_gen = torch.Generator()
    train_gen.manual_seed(args.seed)
    dl_common = dataloader_kwargs(device, args.workers)
    train_drop_last = train_loader_drop_last_for_batchnorm(len(train_ds), args.batch_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_detection_fn,
        generator=train_gen,
        drop_last=train_drop_last,
        **dl_common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_detection_fn,
        **dl_common,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_detection_fn,
        **dl_common,
    )

    if args.downstream == "instance_binary_pretrained":
        model = build_mask_rcnn_binary(num_classes=2, pretrained_backbone=False)
        model.to(device)
        load_maskrcnn_checkpoint(model, args.init_checkpoint.resolve(), device)
    else:
        model = build_mask_rcnn_binary(num_classes=2, pretrained_backbone=args.mask_rcnn_pretrained_backbone)
        model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_path = args.output_dir / args.checkpoint_best_filename
    best_train_loss = float("inf")
    checkpoint_saves = 0
    periodic_test_evals: list[dict[str, Any]] = []
    test_desc = (
        "official_test_full" if args.instance_test_full_official else "official_test_min_1_target_instance"
    )

    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch_mask_rcnn(
            model, train_loader, device, optimizer, scaler, use_amp=use_amp
        )
        row: dict[str, Any] = {"epoch": epoch, "train_loss": loss}
        if args.val_eval:
            row["val_loss"] = eval_mask_rcnn_loss(model, val_loader, device, use_amp=use_amp)

        if loss < best_train_loss:
            best_train_loss = loss
            checkpoint_saves += 1
            torch.save({"model": model.state_dict()}, best_path)
            row["best_train_loss_checkpoint"] = True
            row["checkpoint_save_index"] = checkpoint_saves
            print(
                f"Epoch {epoch}/{args.epochs}  train_loss={loss:.4f}  "
                f"→ best checkpoint #{checkpoint_saves} → {best_path.name}"
            )
            if (
                args.test_every_n_checkpoint_saves > 0
                and checkpoint_saves % args.test_every_n_checkpoint_saves == 0
            ):
                clear_training_memory(device)
                model.eval()
                ap_m = evaluate_mask_rcnn_mask_ap(
                    model,
                    test_loader,
                    device,
                    image_size=args.image_size,
                    score_thresh=0.5,
                    category_id=1,
                    desc="periodic test mask AP",
                )
                pix_m = evaluate_mask_rcnn_pixel_metrics(
                    model,
                    test_loader,
                    device,
                    score_thresh=0.5,
                    category_id=1,
                    desc="periodic test dice/px",
                )
                clear_training_memory(device)
                model.train()
                merged = {**ap_m, **pix_m}
                periodic_test_evals.append(
                    {
                        "epoch": epoch,
                        "train_loss_at_save": loss,
                        "checkpoint_save_index": checkpoint_saves,
                        "test_split": test_desc,
                        "metrics_test": merged,
                    }
                )
                print(
                    f"  periodic test (after save #{checkpoint_saves}): "
                    f"mask_AP50={ap_m['mask_ap50']:.4f}  dice_mean={pix_m.get('dice_mean', float('nan')):.4f}"
                )
        else:
            print(f"Epoch {epoch}/{args.epochs}  train_loss={loss:.4f}")

        history.append(row)

    print(f"Evaluating mask AP on test split ({test_desc})...")
    clear_training_memory(device)
    test_metrics = evaluate_mask_rcnn_mask_ap(
        model,
        test_loader,
        device,
        image_size=args.image_size,
        score_thresh=0.5,
        category_id=1,
        desc="test mask AP",
    )
    clear_training_memory(device)

    exp_key = {
        "A": "A_baseline_full_train",
        "B": "B_sparse_shsy5y",
        "C": "C_sparse_shsy5y_plus_cgan_synth",
    }[args.experiment]
    train_desc = "full_official_train"
    if args.experiment == "A":
        train_desc = "full_official_train"
    elif args.experiment == "B":
        train_desc = str(sparse_manifest)
    else:
        train_desc = f"{sparse_manifest} + synthetic:{args.synthetic_dir or (repo_root / 'runs' / 'synthetic_shsy5y')}"

    out: dict[str, Any] = {
        "experiment": exp_key,
        "experiment_letter": args.experiment,
        "downstream": args.downstream,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "target_category": args.target_category,
        "train": train_desc,
        "test": test_desc,
        "instance_test_full_official": bool(args.instance_test_full_official),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "limit_train": args.limit_train or None,
        "limit_test": args.limit_test or None,
        "metrics_test": test_metrics,
        "history": history,
        "best_train_loss": best_train_loss,
        "checkpoint_best_path": str(best_path),
        "checkpoint_saves_on_improve": checkpoint_saves,
        "periodic_test_eval": periodic_test_evals,
        "test_every_n_checkpoint_saves": args.test_every_n_checkpoint_saves,
    }
    out_path = args.output_dir / f"results_exp_{args.experiment.lower()}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    torch.save({"model": model.state_dict()}, args.output_dir / f"maskrcnn_exp_{args.experiment.lower()}.pt")

    label = {
        "A": "Experiment A - baseline (full train)",
        "B": "Experiment B - sparse shsy5y train",
        "C": "Experiment C - sparse + cGAN synthetic shsy5y",
    }[args.experiment]
    print(f"\n=== Test metrics ({label}, {args.downstream}) ===")
    print(f"Mask AP @0.50: {test_metrics['mask_ap50']:.4f}")
    print(f"Mask AP @0.50:0.95: {test_metrics['mask_ap50_95']:.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
