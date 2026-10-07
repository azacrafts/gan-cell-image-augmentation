"""
Evaluate a saved segmentation checkpoint on the LIVECell **test** split with the same
target-instance filtering as ``train_baseline.py`` (default: images with ≥1 target cell).

Supports:
  - **Mask R-CNN** (``maskrcnn_exp_*.pt``, ``checkpoint_best_train.pt`` with ``{"model": ...}``):
    mask AP50 / AP50–95 (with nan guards), mean Dice, mean pixel accuracy on merged binary target masks.
  - **DeepLab** (``deeplabv3_exp_*.pt``, ``checkpoint_best_train.pt`` raw state dict): pixel accuracy + mean Dice
    on the semantic target line, plus **pseudo-instance** mask AP50 / AP50–95 (argmax → CC vs COCO instance GT,
    same as ``train_baseline.py --pseudo-instance-ap``).

Architecture is inferred when ``--model auto`` (default).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.coco_filters import image_has_category  # noqa: E402
from src.data.coco_livecell import LiveCellSemanticDataset  # noqa: E402
from src.data.livecell_instance import LiveCellInstanceDataset, collate_detection_fn  # noqa: E402
from src.eval.coco_instance_map import (  # noqa: E402
    evaluate_mask_rcnn_mask_ap,
    evaluate_pseudo_instance_ap_semantic,
)
from src.eval.instance_binary_metrics import (  # noqa: E402
    evaluate_mask_rcnn_pixel_metrics,
    safe_mask_ap_dict,
)
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
            "torchvision CUDA ops missing; use --device cpu or reinstall torch+torchvision.\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from err


def load_checkpoint_raw(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def guess_architecture(state: dict[str, Any]) -> str:
    keys = list(state.keys())
    # Prefer Mask R-CNN over DeepLab; match DDP keys like ``module.roi_heads...``.
    if any("roi_heads" in k for k in keys):
        return "mask_rcnn"
    if any("aux_classifier" in k for k in keys):
        return "deeplab"
    raise ValueError(
        "Could not infer architecture from checkpoint keys. Pass --model mask_rcnn or --model deeplab."
    )


def count_mask_rcnn_loader_stats(loader: DataLoader) -> tuple[int, int]:
    """Approximate (n_images, n_gt_instances)."""
    n_img = 0
    n_inst = 0
    for _, targets, _ in loader:
        for t in targets:
            n_img += 1
            n_inst += int(t["masks"].shape[0])
    return n_img, n_inst


@torch.no_grad()
def evaluate_deeplab(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    num_classes: int,
    target_semantic_label: int,
) -> dict[str, float]:
    model.eval()
    total_correct = 0
    total_pixels = 0
    dice_sum = 0.0
    n_imgs = 0
    for batch in tqdm(loader, desc="deeplab eval", leave=False):
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        logits = model(x)["out"]
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_pixels += y.numel()

        p_t = (pred == target_semantic_label).float()
        g_t = (y == target_semantic_label).float()
        inter = (p_t * g_t).sum(dim=(1, 2))
        s = p_t.sum(dim=(1, 2)) + g_t.sum(dim=(1, 2))
        dice_b = torch.where(
            s > 0,
            2.0 * inter / (s + 1e-6),
            torch.where(inter == 0, torch.ones_like(inter), torch.zeros_like(inter)),
        )
        dice_sum += dice_b.sum().item()
        n_imgs += pred.shape[0]

    return {
        "pixel_accuracy": total_correct / max(1, total_pixels),
        "dice_target_mean": dice_sum / max(1, n_imgs),
        "n_images": n_imgs,
    }


def build_test_loader_instance(
    repo_root: Path,
    args: argparse.Namespace,
) -> tuple[DataLoader, str]:
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
        raise SystemExit("Test dataset is empty (check annotations and filters).")
    test_desc = (
        "official_test_full" if args.instance_test_full_official else "official_test_min_1_target_instance"
    )
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_detection_fn,
        pin_memory=torch.device(args.device).type == "cuda",
    )
    return loader, test_desc


def build_test_loader_deeplab(
    repo_root: Path,
    args: argparse.Namespace,
    target_only: bool,
) -> tuple[DataLoader, str]:
    base = LiveCellSemanticDataset(
        repo_root,
        "test",
        image_size=args.image_size,
        augment_train=False,
    )
    if target_only:
        indices = [
            i
            for i in range(len(base))
            if image_has_category(int(base.images[i]["id"]), base._ann_by_image, base.shsy5y_id)
        ]
        if not indices:
            raise SystemExit("No test images with ≥1 shsy5y instance.")
        test_ds = Subset(base, indices)
        test_desc = "official_test_min_1_target_instance (semantic loader)"
    else:
        test_ds = base
        test_desc = "official_test_full (semantic loader)"
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=_collate_semantic,
        pin_memory=torch.device(args.device).type == "cuda",
    )
    return loader, test_desc


def sanitize_json_floats(obj: Any) -> Any:
    """Replace NaN/Inf so ``json.dump`` produces valid JSON (null)."""
    if isinstance(obj, dict):
        return {k: sanitize_json_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json_floats(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.floating):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return obj


def _collate_semantic(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    return {"image": images, "mask": masks, "image_id": [b["image_id"] for b in batch]}


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate LIVECell segmentation checkpoint (Mask R-CNN or DeepLab).")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--model", choices=("auto", "mask_rcnn", "deeplab"), default="auto")
    p.add_argument("--target-category", type=str, default="shsy5y")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument(
        "--instance-test-full-official",
        action="store_true",
        help="Use full official test. Default: same as train_baseline (≥1 target instance).",
    )
    p.add_argument("--output-json", type=Path, default=None, help="Write metrics JSON here.")
    args = p.parse_args()

    repo_root = args.repo_root.resolve()
    device = torch.device(args.device)
    assert_torchvision_cuda_detection_ops(device)

    state = load_checkpoint_raw(args.checkpoint.resolve(), device)
    arch = args.model if args.model != "auto" else guess_architecture(state)

    out: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "architecture": arch,
        "repo_root": str(repo_root),
    }

    if arch == "mask_rcnn":
        loader, test_desc = build_test_loader_instance(repo_root, args)
        n_img, n_gt_inst = count_mask_rcnn_loader_stats(loader)

        model = build_mask_rcnn_binary(num_classes=2, pretrained_backbone=False)
        model.to(device)
        model.load_state_dict(state, strict=True)

        ap = evaluate_mask_rcnn_mask_ap(
            model,
            loader,
            device,
            image_size=args.image_size,
            score_thresh=args.score_thresh,
            category_id=1,
            desc="mask AP",
        )
        n_pred = 0
        model.eval()
        with torch.no_grad():
            for images, _, _ in loader:
                imgs = [im.to(device) for im in images]
                for o in model(imgs):
                    scores = o["scores"].detach().cpu().numpy()
                    labels = o["labels"].detach().cpu().numpy()
                    for i in range(len(scores)):
                        if float(scores[i]) < args.score_thresh:
                            continue
                        if int(labels[i]) == 1:
                            n_pred += 1

        ap_wrapped = safe_mask_ap_dict(
            ap["mask_ap50"],
            ap["mask_ap50_95"],
            n_gt_anns=n_gt_inst,
            n_pred=n_pred,
            n_images=n_img,
        )
        pix = evaluate_mask_rcnn_pixel_metrics(
            model,
            loader,
            device,
            score_thresh=args.score_thresh,
            category_id=1,
            desc="dice / pix acc",
        )

        out["test_split"] = test_desc
        out["metrics"] = {**ap_wrapped, **pix}
        if math.isnan(ap_wrapped["mask_ap50"]) and n_gt_inst > 0:
            out["metrics"]["nan_guard_note"] = (
                "If AP is nan despite GT, try lowering --score-thresh or check torchvision/pycocotools."
            )

    else:
        target_only = not args.instance_test_full_official
        loader, test_desc = build_test_loader_deeplab(repo_root, args, target_only=target_only)
        meta = LiveCellSemanticDataset(repo_root, "train", image_size=args.image_size, augment_train=False)
        num_classes = meta.num_classes
        target_label = int(meta.cat_id_to_label[meta.shsy5y_id])

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
        model.load_state_dict(state, strict=True)

        m = evaluate_deeplab(
            model,
            loader,
            device,
            num_classes=num_classes,
            target_semantic_label=target_label,
        )

        inst_test_only = not args.instance_test_full_official
        inst_ds = LiveCellInstanceDataset(
            repo_root,
            "test",
            target_category_name=args.target_category,
            mode="target",
            sparse_train_manifest=None,
            image_size=args.image_size,
            raw_image=False,
            test_only_images_with_target=inst_test_only,
        )
        if len(inst_ds) == 0:
            raise SystemExit(
                "Instance test set is empty for pseudo mask AP (check annotations and filters)."
            )
        inst_loader = DataLoader(
            inst_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_detection_fn,
            pin_memory=torch.device(args.device).type == "cuda",
        )
        n_img, n_gt_inst = count_mask_rcnn_loader_stats(inst_loader)
        pseudo = evaluate_pseudo_instance_ap_semantic(
            model,
            inst_loader,
            device,
            target_semantic_label=target_label,
            image_size=args.image_size,
            desc="pseudo mask AP (test)",
        )
        ap_wrapped = safe_mask_ap_dict(
            pseudo["pseudo_mask_ap50"],
            pseudo["pseudo_mask_ap50_95"],
            n_gt_anns=n_gt_inst,
            n_pred=0,
            n_images=n_img,
        )
        ap_wrapped["diagnostics"]["pseudo_instance_ap"] = True
        ap_wrapped["diagnostics"]["n_pred_instances_above_thresh"] = None

        out["test_split"] = test_desc
        out["metrics"] = {
            **ap_wrapped,
            "mask_ap_note": (
                "Pseudo-instance mask AP (semantic argmax → connected components on target channel vs COCO instance GT); "
                "same definition as train_baseline.py --pseudo-instance-ap. "
                "n_pred_instances_above_thresh is not applicable here."
            ),
            "pixel_accuracy": m["pixel_accuracy"],
            "dice_target_mean": m["dice_target_mean"],
            "n_images": m["n_images"],
            "target_semantic_label": target_label,
        }
        if math.isnan(ap_wrapped["mask_ap50"]) and n_gt_inst > 0:
            out["metrics"]["nan_guard_note"] = (
                "If pseudo AP is nan despite GT, check torchvision/pycocotools and the semantic checkpoint."
            )

    print(json.dumps(sanitize_json_floats(out["metrics"]), indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(sanitize_json_floats(out), f, indent=2)
        print(f"Wrote {args.output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
