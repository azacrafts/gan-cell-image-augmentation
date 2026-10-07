"""Pixel-wise Dice and accuracy for binary target masks (merged instance masks)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def merge_instance_masks(masks: torch.Tensor) -> np.ndarray:
    """``masks`` (N,H,W) float in [0,1] → (H,W) uint8 {0,1}."""
    if masks.dim() != 3 or masks.shape[0] == 0:
        if masks.dim() == 3:
            h, w = int(masks.shape[1]), int(masks.shape[2])
        else:
            h, w = 512, 512
        return np.zeros((h, w), dtype=np.uint8)
    m = (masks.detach().cpu().float().numpy() > 0.5).any(axis=0)
    return m.astype(np.uint8)


def merge_pred_masks_from_maskrcnn(
    out: dict[str, torch.Tensor],
    *,
    score_thresh: float,
    category_id: int = 1,
    fallback_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Single-image Mask R-CNN output → binary (H,W) uint8."""
    scores = out["scores"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy()
    masks = out["masks"].detach().cpu().squeeze(1).numpy()
    if masks.size == 0:
        hw = fallback_hw if fallback_hw is not None else (512, 512)
        return np.zeros((hw[0], hw[1]), dtype=np.uint8)
    h, w = masks.shape[1], masks.shape[2]
    acc = np.zeros((h, w), dtype=np.float32)
    for i in range(len(scores)):
        if float(scores[i]) < score_thresh:
            continue
        if int(labels[i]) != int(category_id):
            continue
        acc = np.maximum(acc, masks[i])
    return (acc > 0.5).astype(np.uint8)


def dice_binary(pred: np.ndarray, gt: np.ndarray, *, eps: float = 1e-6) -> float:
    """Dice on foreground (1). Both (H,W) uint8/binary."""
    p = pred.astype(np.float64).ravel()
    g = gt.astype(np.float64).ravel()
    inter = float((p * g).sum())
    s = float(p.sum() + g.sum())
    if s < eps and inter < eps:
        return 1.0
    if s < eps:
        return 0.0
    return float((2.0 * inter + eps) / (s + eps))


def pixel_accuracy_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    """Accuracy treating class 1 as foreground, 0 as background."""
    p = pred.astype(np.uint8).ravel()
    g = gt.astype(np.uint8).ravel()
    return float((p == g).mean())


@torch.no_grad()
def evaluate_mask_rcnn_pixel_metrics(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    *,
    score_thresh: float = 0.5,
    category_id: int = 1,
    desc: str = "pixel metrics",
) -> dict[str, Any]:
    """
    Mean Dice and mean pixel accuracy over the loader (binary target vs background).
    Skips images with empty GT with a count (should not happen if test set is target-filtered).
    """
    model.eval()
    dices: list[float] = []
    accs: list[float] = []
    n_empty_gt = 0

    for images, targets, _ in tqdm(loader, desc=desc, leave=False):
        imgs_dev = [im.to(device) for im in images]
        outputs = model(imgs_dev)

        for out, tgt in zip(outputs, targets):
            gt_bin = merge_instance_masks(tgt["masks"])
            if gt_bin.sum() == 0:
                n_empty_gt += 1
                continue
            pred_bin = merge_pred_masks_from_maskrcnn(
                out,
                score_thresh=score_thresh,
                category_id=category_id,
                fallback_hw=(gt_bin.shape[0], gt_bin.shape[1]),
            )
            if pred_bin.shape != gt_bin.shape:
                raise ValueError(f"Shape mismatch pred {pred_bin.shape} vs gt {gt_bin.shape}")
            dices.append(dice_binary(pred_bin, gt_bin))
            accs.append(pixel_accuracy_binary(pred_bin, gt_bin))

    n_used = len(dices)
    out: dict[str, Any] = {
        "dice_mean": float(np.mean(dices)) if n_used else float("nan"),
        "pixel_accuracy_mean": float(np.mean(accs)) if n_used else float("nan"),
        "n_images_evaluated": n_used,
        "n_images_empty_gt_skipped": n_empty_gt,
    }
    return out


def safe_mask_ap_dict(
    ap50: float,
    ap5095: float,
    *,
    n_gt_anns: int,
    n_pred: int,
    n_images: int,
) -> dict[str, Any]:
    """Attach diagnostics when AP is nan or suspicious."""
    reasons: list[str] = []
    if n_images == 0:
        reasons.append("no_images_in_loader")
    if n_gt_anns == 0:
        reasons.append("no_gt_instances_in_loader")
    if ap50 != ap50 or ap5095 != ap5095:  # NaN check without numpy
        reasons.append("coco_eval_returned_nan")
    return {
        "mask_ap50": ap50,
        "mask_ap50_95": ap5095,
        "diagnostics": {
            "n_test_images": n_images,
            "n_gt_instances": n_gt_anns,
            "n_pred_instances_above_thresh": n_pred,
            "nan_guard_reasons": reasons,
        },
    }
