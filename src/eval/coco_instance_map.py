"""COCO-style mask AP (AP50, AP50–95) via pycocotools; optional pseudo-instance AP from semantic argmax + CC."""

from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

import pycocotools.mask as mask_util

from src.data.livecell_instance import _semantic_mask_to_instances


def _binary_mask_to_rle(mask_hw: np.ndarray, *, prob_thresh: float = 0.0) -> dict[str, Any]:
    """``mask_hw`` binary uint8 or float; if values in [0,1] use ``prob_thresh`` (e.g. 0.5)."""
    m = np.asarray(mask_hw)
    if m.max() <= 1.0 + 1e-6 and prob_thresh > 0:
        binary = (m > prob_thresh).astype(np.uint8)
    else:
        binary = (m > 0).astype(np.uint8)
    if binary.sum() == 0:
        h, w = binary.shape
        rle = {"size": [h, w], "counts": ""}
        return rle
    rle = mask_util.encode(np.asfortranarray(binary))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _bbox_area_from_binary(mask_hw: np.ndarray) -> tuple[list[float], float]:
    binary = (np.asarray(mask_hw) > 0).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return [0.0, 0.0, 1.0, 1.0], 1.0
    x0, x1 = float(xs.min()), float(xs.max()) + 1.0
    y0, y1 = float(ys.min()), float(ys.max()) + 1.0
    w, h = x1 - x0, y1 - y0
    return [x0, y0, w, h], float(binary.sum())


def _build_coco_gt(
    images_info: list[dict[str, Any]],
    gt_anns: list[dict[str, Any]],
    category_id: int = 1,
    category_name: str = "fg",
) -> COCO:
    ds: dict[str, Any] = {
        "images": images_info,
        "categories": [{"id": category_id, "name": category_name}],
        "annotations": gt_anns,
    }
    coco = COCO()
    coco.dataset = ds
    coco.createIndex()
    return coco


def _gt_anns_from_target_masks(
    image_id: int,
    masks: torch.Tensor,
    ann_id_start: int,
    category_id: int = 1,
) -> tuple[list[dict[str, Any]], int]:
    """``masks`` float (N,H,W) in [0,1] or binary."""
    out: list[dict[str, Any]] = []
    aid = ann_id_start
    m = masks.detach().cpu().float().numpy()
    for i in range(m.shape[0]):
        hw = m[i]
        if hw.max() < 0.01:
            continue
        rle = _binary_mask_to_rle(hw)
        bbox, area = _bbox_area_from_binary(hw)
        out.append(
            {
                "id": aid,
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": rle,
            }
        )
        aid += 1
    return out, aid


def _pred_results_from_mask_rcnn(
    out: dict[str, torch.Tensor],
    image_id: int,
    score_thresh: float,
    category_id: int = 1,
) -> list[dict[str, Any]]:
    """Torchvision Mask R-CNN single-image output dict."""
    results: list[dict[str, Any]] = []
    boxes = out["boxes"].detach().cpu()
    scores = out["scores"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy()
    masks = out["masks"].detach().cpu().squeeze(1).numpy()  # N,H,W sigmoid

    for i in range(len(scores)):
        if float(scores[i]) < score_thresh:
            continue
        if int(labels[i]) != int(category_id):
            continue
        hw = masks[i]
        rle = _binary_mask_to_rle(hw, prob_thresh=0.5)
        bbox, area = _bbox_area_from_binary(hw)
        results.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "segmentation": rle,
                "score": float(scores[i]),
                "bbox": bbox,
                "area": area,
            }
        )
    return results


def _pred_results_from_pseudo_masks(
    masks_hw: np.ndarray,
    image_id: int,
    category_id: int = 1,
    pseudo_score: float = 1.0,
) -> list[dict[str, Any]]:
    """``masks_hw`` (N,H,W) uint8 or float per instance."""
    results: list[dict[str, Any]] = []
    for i in range(masks_hw.shape[0]):
        hw = masks_hw[i]
        if np.asarray(hw).max() < 0.01:
            continue
        rle = _binary_mask_to_rle(hw)
        bbox, area = _bbox_area_from_binary(hw)
        results.append(
            {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "segmentation": rle,
                "score": float(pseudo_score),
                "bbox": bbox,
                "area": area,
            }
        )
    return results


def evaluate_mask_rcnn_mask_ap(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    *,
    image_size: int,
    score_thresh: float = 0.5,
    category_id: int = 1,
    desc: str = "mask AP",
) -> dict[str, float]:
    """
    Run Mask R-CNN on ``loader`` batches ``(images, targets, image_ids)``.
    GT is taken from ``targets`` (instance masks). Returns mask AP50 and AP50–95.
    """
    model.eval()
    images_info: list[dict[str, Any]] = []
    gt_anns: list[dict[str, Any]] = []
    pred_results: list[dict[str, Any]] = []
    ann_id = 1

    with torch.no_grad():
        for images, targets, image_ids in tqdm(loader, desc=desc, leave=False):
            for img_t, tgt, iid in zip(images, targets, image_ids):
                iid = int(iid)
                images_info.append({"id": iid, "width": image_size, "height": image_size})
                glist, ann_id = _gt_anns_from_target_masks(
                    iid, tgt["masks"], ann_id, category_id=category_id
                )
                gt_anns.extend(glist)

            imgs_dev = [im.to(device) for im in images]
            outputs = model(imgs_dev)

            for out, iid in zip(outputs, image_ids):
                pred_results.extend(
                    _pred_results_from_mask_rcnn(
                        out, int(iid), score_thresh, category_id=category_id
                    )
                )

    coco_gt = _build_coco_gt(images_info, gt_anns, category_id=category_id)
    if len(gt_anns) == 0:
        return {"mask_ap50": float("nan"), "mask_ap50_95": float("nan")}
    if len(pred_results) == 0:
        return {"mask_ap50": 0.0, "mask_ap50_95": 0.0}

    coco_dt = coco_gt.loadRes(pred_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()
    # stats[0]: AP @[0.5:0.95], stats[1]: AP@0.5 for segm
    stats = coco_eval.stats
    ap5095 = float(stats[0]) if stats is not None and len(stats) > 0 else float("nan")
    ap50 = float(stats[1]) if stats is not None and len(stats) > 1 else float("nan")
    return {"mask_ap50": ap50, "mask_ap50_95": ap5095}


def evaluate_pseudo_instance_ap_semantic(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    *,
    target_semantic_label: int,
    image_size: int,
    desc: str = "pseudo AP",
) -> dict[str, float]:
    """
    Semantic segmentation model: argmax → CC on predicted target channel → COCO mask AP vs instance GT.
    ``target_semantic_label`` is dense label index (1..K) matching ``LiveCellSemanticDataset`` masks.
    """
    model.eval()
    images_info: list[dict[str, Any]] = []
    gt_anns: list[dict[str, Any]] = []
    pred_results: list[dict[str, Any]] = []
    ann_id = 1
    category_id = 1

    with torch.no_grad():
        for images, targets, image_ids in tqdm(loader, desc=desc, leave=False):
            for img_t, tgt, iid in zip(images, targets, image_ids):
                iid = int(iid)
                images_info.append({"id": iid, "width": image_size, "height": image_size})
                glist, ann_id = _gt_anns_from_target_masks(iid, tgt["masks"], ann_id, category_id=category_id)
                gt_anns.extend(glist)

            imgs_dev = [im.to(device) for im in images]
            x = torch.stack(imgs_dev, dim=0)
            logits = model(x)["out"]
            pred = logits.argmax(dim=1)

            for p, iid in zip(pred, image_ids):
                p_hw = p.detach().cpu().numpy()
                bin_m = (p_hw == int(target_semantic_label)).astype(np.uint8)
                masks_np, _ = _semantic_mask_to_instances(bin_m, fg_label=-1)
                pred_results.extend(
                    _pred_results_from_pseudo_masks(masks_np, int(iid), category_id=category_id)
                )

    coco_gt = _build_coco_gt(images_info, gt_anns, category_id=category_id)
    if len(gt_anns) == 0:
        return {"pseudo_mask_ap50": float("nan"), "pseudo_mask_ap50_95": float("nan")}
    if len(pred_results) == 0:
        return {"pseudo_mask_ap50": 0.0, "pseudo_mask_ap50_95": 0.0}

    coco_dt = coco_gt.loadRes(pred_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()
    stats = coco_eval.stats
    ap5095 = float(stats[0]) if stats is not None and len(stats) > 0 else float("nan")
    ap50 = float(stats[1]) if stats is not None and len(stats) > 1 else float("nan")
    return {"pseudo_mask_ap50": ap50, "pseudo_mask_ap50_95": ap5095}
