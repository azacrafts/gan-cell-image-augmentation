"""Paired PSNR/LPIPS and DINOv2 Fréchet distance for synthetic phase images."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torchvision import transforms

from src.data.coco_filters import (
    build_ann_index,
    is_shsy5y_image,
    resolve_shsy5y_category_id,
)
from src.data.coco_livecell import load_split_manifest


def load_gray_bilinear_float01(path: Path, image_size: int) -> np.ndarray:
    """Grayscale LIVECell phase: bilinear resize to (image_size, image_size), float [0,1] (H,W)."""
    im = Image.open(path).convert("L")
    im = im.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def psnr_float(a_hw: np.ndarray, b_hw: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((a_hw - b_hw) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / mse))


def list_pure_shsy5y_train_paths(repo_root: Path) -> list[Path]:
    """All pure-shsy5y FOVs in official train split (paths under images_root)."""
    repo_root = repo_root.resolve()
    official = load_split_manifest(repo_root)
    train_path = repo_root / official["train_annotations"]
    images_root = repo_root / official["images_root"]
    with train_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    ann_by_image = build_ann_index(coco["annotations"])
    shsy5y_id = resolve_shsy5y_category_id(coco["categories"])
    out: list[Path] = []
    for im in coco["images"]:
        iid = int(im["id"])
        if is_shsy5y_image(iid, ann_by_image, shsy5y_id):
            out.append(images_root / im["file_name"])
    return sorted(out)


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet distance between two Gaussians (e.g. Heusel et al. FID recipe)."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def _load_dinov2(backbone: str, device: torch.device) -> torch.nn.Module:
    """backbone e.g. dinov2_vits14, dinov2_vitb14."""
    return torch.hub.load("facebookresearch/dinov2", backbone, pretrained=True, trust_repo=True).to(device).eval()


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def _dinov2_features_from_gray_paths(
    paths: list[Path],
    image_size: int,
    dinov2: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Stack of L2-normalized CLS features (N, D), float32."""
    to_t = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    feats_list: list[torch.Tensor] = []
    mean = _IMAGENET_MEAN.to(device)
    std = _IMAGENET_STD.to(device)
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        xs: list[torch.Tensor] = []
        for p in batch_paths:
            rgb = Image.merge("RGB", (Image.open(p).convert("L"),) * 3)
            t = to_t(rgb)
            xs.append(t)
        x = torch.stack(xs, dim=0).to(device)
        x = (x - mean) / std
        out = dinov2.forward_features(x)
        tok = out["x_norm_clstoken"]
        tok = F.normalize(tok, dim=1)
        feats_list.append(tok.cpu())
    return torch.cat(feats_list, dim=0).numpy().astype(np.float64)


def frechet_dinov2_cls(
    paths_real: list[Path],
    paths_gen: list[Path],
    *,
    image_size: int,
    device: torch.device,
    backbone: str = "dinov2_vits14",
    batch_size: int = 16,
) -> dict[str, Any]:
    """
    Fréchet distance between DINOv2 CLS features (L2-normalized) of two image sets.
    Grayscale microscopy: channels repeated ×3, ImageNet normalization (matches training pipeline style).
    """
    if len(paths_real) < 2 or len(paths_gen) < 2:
        return {
            "error": "Fréchet distance needs at least 2 images in each of real and generated sets.",
            "n_reference_real": len(paths_real),
            "n_generated": len(paths_gen),
        }
    dinov2 = _load_dinov2(backbone, device)
    fr = _dinov2_features_from_gray_paths(paths_real, image_size, dinov2, device, batch_size)
    fg = _dinov2_features_from_gray_paths(paths_gen, image_size, dinov2, device, batch_size)
    mu_r = np.mean(fr, axis=0)
    mu_g = np.mean(fg, axis=0)
    sigma_r = np.cov(fr, rowvar=False)
    sigma_g = np.cov(fg, rowvar=False)
    d = feature_dim = int(mu_r.shape[0])
    fd = _frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
    note = None
    n_r, n_g = len(paths_real), len(paths_gen)
    if min(n_r, n_g) < d:
        note = (
            f"Fréchet covariance uses {d}-dim features with n_real={n_r}, n_gen={n_g}; "
            "estimates can be high-variance when n < dim."
        )
    return {
        "frechet_dinov2": fd,
        "FD_dinov2": fd,
        "fid_backbone": backbone,
        "fid_layer": "x_norm_clstoken",
        "feature_dim": d,
        "n_reference_real": n_r,
        "n_generated": n_g,
        "note": note,
    }


def evaluate_paired_psnr_lpips(
    synthetic_root: Path,
    manifest: dict[str, Any],
    repo_root: Path,
    *,
    device: torch.device,
    lpips_net: str = "alex",
) -> dict[str, Any]:
    """
    For mask_source excluded_real: PSNR and LPIPS vs ground truth for each sample with source_image_id.
    If the manifest lists a ``gt`` path (saved under output_dir by synthesis), that PNG is used;
    otherwise LIVECell is loaded and resized the same way (grayscale, bilinear to image_size).
    """
    import lpips  # lazy import

    repo_root = repo_root.resolve()
    synthetic_root = synthetic_root.resolve()
    official = load_split_manifest(repo_root)
    train_path = repo_root / official["train_annotations"]
    images_root = repo_root / official["images_root"]
    with train_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    id_to_img = {int(im["id"]): im for im in coco["images"]}

    image_size = int(manifest["image_size"])
    loss_fn = lpips.LPIPS(net=lpips_net).to(device)
    loss_fn.eval()

    per_image: list[dict[str, Any]] = []
    psnrs: list[float] = []
    lpips_vals: list[float] = []

    for s in manifest["samples"]:
        iid = int(s["source_image_id"])
        if iid < 0:
            continue
        info = id_to_img[iid]
        syn_path = synthetic_root / s["image"]
        rel_gt = s.get("gt")
        if rel_gt:
            gt_path = synthetic_root / rel_gt
        else:
            gt_path = images_root / info["file_name"]
        gt = load_gray_bilinear_float01(gt_path, image_size)
        pred = load_gray_bilinear_float01(syn_path, image_size)
        p = psnr_float(gt, pred, data_range=1.0)
        psnrs.append(p)

        # LPIPS: (1,3,H,W) in [-1, 1]
        gt_t = torch.from_numpy(np.stack([gt, gt, gt], axis=0)).unsqueeze(0).to(device)
        pr_t = torch.from_numpy(np.stack([pred, pred, pred], axis=0)).unsqueeze(0).to(device)
        gt_t = gt_t * 2.0 - 1.0
        pr_t = pr_t * 2.0 - 1.0
        with torch.no_grad():
            l = loss_fn(gt_t, pr_t).item()
        lpips_vals.append(l)
        row: dict[str, Any] = {
            "source_image_id": iid,
            "synthetic_image": s["image"],
            "psnr": p,
            "lpips": l,
        }
        if rel_gt:
            row["gt_image"] = rel_gt
        per_image.append(row)

    if not psnrs:
        return {
            "error": "no paired samples (need excluded_real with source_image_id >= 0)",
            "lpips_backbone": lpips_net,
        }

    return {
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_std": float(np.std(psnrs)),
        "lpips_mean": float(np.mean(lpips_vals)),
        "lpips_std": float(np.std(lpips_vals)),
        "lpips_backbone": lpips_net,
        "n_paired": len(psnrs),
        "per_image": per_image,
    }


def subsample_paths(paths: list[Path], limit: int, seed: int) -> list[Path]:
    if limit <= 0 or limit >= len(paths):
        return list(paths)
    rng = random.Random(seed)
    return sorted(rng.sample(paths, limit), key=lambda p: str(p))


def run_procedural_frechet(
    synthetic_root: Path,
    manifest: dict[str, Any],
    repo_root: Path,
    *,
    device: torch.device,
    fid_real_limit: int,
    fid_seed: int,
    dinov2_backbone: str,
    metrics_batch_size: int,
) -> dict[str, Any]:
    synthetic_root = synthetic_root.resolve()
    image_size = int(manifest["image_size"])
    gen_paths = sorted((synthetic_root / "images").glob("*.png"))
    if not gen_paths:
        return {"error": "no generated PNGs in images/"}

    all_real = list_pure_shsy5y_train_paths(repo_root)
    real_paths = subsample_paths(all_real, fid_real_limit, fid_seed)

    out = frechet_dinov2_cls(
        real_paths,
        gen_paths,
        image_size=image_size,
        device=device,
        backbone=dinov2_backbone,
        batch_size=metrics_batch_size,
    )
    out["fid_real_pool"] = "pure_shsy5y_official_train"
    out["n_reference_real_available"] = len(all_real)
    out["fid_real_limit"] = fid_real_limit
    out["fid_seed"] = fid_seed
    return out


def write_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
