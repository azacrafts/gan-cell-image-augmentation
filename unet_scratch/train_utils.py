from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    dloss = dice_loss(logits, targets)
    return bce_weight * bce + (1.0 - bce_weight) * dloss


def _batch_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = targets.float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return {"dice": float(dice), "iou": float(iou)}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    progress_desc: Optional[str] = None,
    use_tqdm: bool = True,
    progress_leave: bool = True,
) -> Dict[str, float]:
    model.train()
    loss_sum = 0.0
    dice_sum = 0.0
    iou_sum = 0.0
    n = 0

    use_bar = bool(use_tqdm and tqdm is not None)
    iterator = (
        tqdm(loader, desc=progress_desc, total=len(loader), leave=progress_leave, dynamic_ncols=True)
        if use_bar
        else loader
    )
    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = bce_dice_loss(logits, masks)
        loss.backward()
        optimizer.step()

        metrics = _batch_metrics(logits.detach(), masks.detach())
        batch_size = int(images.shape[0])
        loss_sum += float(loss.item()) * batch_size
        dice_sum += metrics["dice"] * batch_size
        iou_sum += metrics["iou"] * batch_size
        n += batch_size
        if use_bar:
            iterator.set_postfix(loss=f"{loss.item():.4f}", dice=f"{metrics['dice']:.4f}", iou=f"{metrics['iou']:.4f}")

    if n == 0:
        return {"loss": 0.0, "dice": 0.0, "iou": 0.0}
    return {"loss": loss_sum / n, "dice": dice_sum / n, "iou": iou_sum / n}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    progress_desc: Optional[str] = None,
    use_tqdm: bool = True,
    progress_leave: bool = True,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    dice_sum = 0.0
    iou_sum = 0.0
    n = 0
    use_bar = bool(use_tqdm and tqdm is not None)
    iterator = (
        tqdm(loader, desc=progress_desc, total=len(loader), leave=progress_leave, dynamic_ncols=True)
        if use_bar
        else loader
    )
    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        loss = bce_dice_loss(logits, masks)
        metrics = _batch_metrics(logits, masks)
        batch_size = int(images.shape[0])
        loss_sum += float(loss.item()) * batch_size
        dice_sum += metrics["dice"] * batch_size
        iou_sum += metrics["iou"] * batch_size
        n += batch_size
        if use_bar:
            iterator.set_postfix(loss=f"{loss.item():.4f}", dice=f"{metrics['dice']:.4f}", iou=f"{metrics['iou']:.4f}")

    if n == 0:
        return {"loss": 0.0, "dice": 0.0, "iou": 0.0}
    return {"loss": loss_sum / n, "dice": dice_sum / n, "iou": iou_sum / n}


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_dice: float,
    extra: Dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_dice": best_val_dice,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_epoch_record(history: List[Dict[str, Any]], epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
    history.append(
        {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
        }
    )
