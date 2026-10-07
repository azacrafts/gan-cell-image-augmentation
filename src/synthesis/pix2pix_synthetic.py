"""Shared Pix2Pix mask-to-phase synthesis and manifest writing (used by CLI scripts)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.data.coco_filters import build_ann_index
from src.data.coco_livecell import (
    build_cat_id_to_semantic_label,
    load_split_manifest,
    rasterize_semantic_mask,
)
from src.models.pix2pix import UNetGenerator


def mask_to_onehot_np(mask_hw: np.ndarray, num_classes: int) -> np.ndarray:
    h, w = mask_hw.shape
    oh = np.zeros((num_classes, h, w), dtype=np.float32)
    for c in range(num_classes):
        oh[c] = (mask_hw == c).astype(np.float32)
    return oh


def save_semantic_mask_png(mask_u8_hw: np.ndarray, path: Path) -> None:
    if mask_u8_hw.dtype != np.uint8:
        mask_u8_hw = mask_u8_hw.astype(np.uint8, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_u8_hw, mode="L").save(path, format="PNG")


@torch.no_grad()
def synthesize_phase_from_mask(
    net_g: torch.nn.Module,
    mask_rs: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    """Semantic mask (H,W) uint8 -> grayscale uint8 phase (H,W)."""
    oh = mask_to_onehot_np(mask_rs, num_classes)
    x = torch.from_numpy(oh).unsqueeze(0).to(device)
    fake = net_g(x).squeeze(0).cpu().clamp(0, 1).numpy()
    fake_gray = 0.299 * fake[0] + 0.587 * fake[1] + 0.114 * fake[2]
    return (fake_gray * 255.0).astype(np.uint8)


def save_gt_phase_resized(
    livecell_image_path: Path,
    out_path: Path,
    image_size: int,
) -> None:
    """Grayscale phase from LIVECell, bilinear resize to (image_size, image_size), uint8 PNG."""
    im = Image.open(livecell_image_path).convert("L")
    im = im.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, format="PNG")


def verify_mask_png(path: Path, expect_max_label: int) -> tuple[int, int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"mask PNG not written: {path}")
    n = path.stat().st_size
    if n == 0:
        raise RuntimeError(f"mask PNG is 0 bytes: {path}")
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    mx = int(arr.max())
    fg = int(np.sum(arr > 0))
    if mx > expect_max_label:
        raise ValueError(f"mask {path} has max label {mx} > expected {expect_max_label}")
    return n, mx, fg


def load_generator_checkpoint(checkpoint: Path, device: torch.device) -> tuple[UNetGenerator, int]:
    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)
    num_classes = int(ckpt["num_classes"])
    net_g = UNetGenerator(mask_channels=num_classes, base=64).to(device)
    net_g.load_state_dict(ckpt["generator"])
    net_g.eval()
    return net_g, num_classes


MaskSource = Literal["excluded_real", "procedural"]


@dataclass
class SyntheticGenerateConfig:
    repo_root: Path
    output_dir: Path
    checkpoint: Path
    device: torch.device
    image_size: int
    mask_source: MaskSource
    sparse_manifest: Path
    procedural_mask_dir: Path | None
    procedural_manifest: Path | None
    limit: int
    save_mask_preview: bool


@torch.no_grad()
def generate_synthetic_dataset(cfg: SyntheticGenerateConfig) -> dict[str, Any]:
    """
    Write images/, masks/, synthetic_manifest.json under cfg.output_dir.
    For ``excluded_real``, also writes gt/ (ground-truth phase PNGs, same resize as cGAN training).
    Returns the manifest dict (also written to disk).
    """
    repo_root = cfg.repo_root.resolve()
    net_g, num_classes = load_generator_checkpoint(cfg.checkpoint, cfg.device)
    device = cfg.device

    official = load_split_manifest(repo_root)
    train_path = repo_root / official["train_annotations"]
    images_root = repo_root / official["images_root"]

    out_root = cfg.output_dir.resolve()
    img_dir = out_root / "images"
    msk_dir = out_root / "masks"
    gt_dir = out_root / "gt"
    preview_dir = out_root / "masks_preview"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    expect_max_label = num_classes - 1

    def _rel(p: Path) -> str:
        p = p.resolve()
        try:
            return str(p.relative_to(repo_root))
        except ValueError:
            return str(p)

    samples: list[dict] = []
    zero_fg = 0

    if cfg.mask_source == "excluded_real":
        with train_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)
        id_to_img = {int(im["id"]): im for im in coco["images"]}
        ann_by_image = build_ann_index(coco["annotations"])
        cat_id_to_label = build_cat_id_to_semantic_label(coco["categories"])

        with cfg.sparse_manifest.open("r", encoding="utf-8") as f:
            sparse = json.load(f)
        excluded = [int(x) for x in sparse["excluded_shsy5y_image_ids"]]
        if cfg.limit > 0:
            excluded = excluded[: cfg.limit]

        for k, iid in enumerate(tqdm(excluded, desc="synthesize")):
            info = id_to_img[iid]
            h, w = int(info["height"]), int(info["width"])
            anns = ann_by_image.get(iid, [])
            mask_np = rasterize_semantic_mask(h, w, anns, cat_id_to_label)
            mask_pil = Image.fromarray(mask_np, mode="L")
            mask_pil = mask_pil.resize((cfg.image_size, cfg.image_size), resample=Image.Resampling.NEAREST)
            mask_rs = np.array(mask_pil, dtype=np.uint8, copy=True)
            fake_u8 = synthesize_phase_from_mask(net_g, mask_rs, num_classes, device)
            stem = f"synth_{k:05d}_src{iid}"
            img_path = img_dir / f"{stem}.png"
            msk_path = msk_dir / f"{stem}.png"
            gt_path = gt_dir / f"{stem}.png"
            Image.fromarray(fake_u8, mode="L").save(img_path, format="PNG")
            save_semantic_mask_png(mask_rs, msk_path)
            verify_mask_png(msk_path, expect_max_label=expect_max_label)
            save_gt_phase_resized(images_root / info["file_name"], gt_path, cfg.image_size)
            fg = int(np.sum(mask_rs > 0))
            if fg == 0:
                zero_fg += 1
                print(
                    f"WARNING: mask has no foreground pixels (all background): {msk_path} "
                    f"(source_image_id={iid}, n_annots={len(anns)})",
                    file=sys.stderr,
                )
            if cfg.save_mask_preview:
                prev = (mask_rs.astype(np.float32) / max(1.0, float(mask_rs.max())) * 255.0).astype(np.uint8)
                preview_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(prev, mode="L").save(preview_dir / f"{stem}.png", format="PNG")
            samples.append(
                {
                    "image": f"images/{stem}.png",
                    "mask": f"masks/{stem}.png",
                    "gt": f"gt/{stem}.png",
                    "source_image_id": iid,
                    "synthetic_id": -(k + 1),
                }
            )
        policy = "one_synthetic_per_excluded_shsy5y_train_image"
        extra_meta: dict = {"sparse_manifest": _rel(cfg.sparse_manifest)}
    else:
        proc_root = cfg.procedural_mask_dir
        if proc_root is None:
            raise ValueError("--procedural-mask-dir is required for mask_source=procedural")
        proc_root = proc_root.resolve()
        man_path = cfg.procedural_manifest or (proc_root / "procedural_mask_manifest.json")
        if not man_path.is_file():
            raise FileNotFoundError(f"Procedural manifest not found: {man_path}")
        with man_path.open("r", encoding="utf-8") as f:
            proc_meta = json.load(f)
        proc_samples = proc_meta["samples"]
        if cfg.limit > 0:
            proc_samples = proc_samples[: cfg.limit]

        for k, s in enumerate(tqdm(proc_samples, desc="synthesize")):
            rel = s["mask"]
            src_m = proc_root / rel
            if not src_m.is_file():
                raise FileNotFoundError(src_m)
            mask_pil = Image.open(src_m).convert("L")
            if mask_pil.size != (cfg.image_size, cfg.image_size):
                mask_pil = mask_pil.resize((cfg.image_size, cfg.image_size), resample=Image.Resampling.NEAREST)
            mask_rs = np.array(mask_pil, dtype=np.uint8, copy=True)
            fake_u8 = synthesize_phase_from_mask(net_g, mask_rs, num_classes, device)
            sid = int(s.get("synthetic_id", -(k + 1)))
            stem = f"synth_{k:05d}_proc{sid}"
            img_path = img_dir / f"{stem}.png"
            msk_path = msk_dir / f"{stem}.png"
            Image.fromarray(fake_u8, mode="L").save(img_path, format="PNG")
            save_semantic_mask_png(mask_rs, msk_path)
            verify_mask_png(msk_path, expect_max_label=expect_max_label)
            fg = int(np.sum(mask_rs > 0))
            if fg == 0:
                zero_fg += 1
                print(f"WARNING: procedural mask has no foreground: {src_m}", file=sys.stderr)
            if cfg.save_mask_preview:
                prev = (mask_rs.astype(np.float32) / max(1.0, float(mask_rs.max())) * 255.0).astype(np.uint8)
                preview_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(prev, mode="L").save(preview_dir / f"{stem}.png", format="PNG")
            samples.append(
                {
                    "image": f"images/{stem}.png",
                    "mask": f"masks/{stem}.png",
                    "gt": None,
                    "source_image_id": -1,
                    "synthetic_id": sid,
                    "procedural_source_mask": rel,
                }
            )
        policy = "procedural_masks_from_bank"
        extra_meta = {
            "procedural_mask_dir": _rel(proc_root),
            "procedural_manifest": _rel(man_path),
            "shape_bank_dir": proc_meta.get("bank_dir"),
        }

    manifest: dict[str, Any] = {
        "policy": policy,
        "mask_source": cfg.mask_source,
        "cgan_checkpoint": _rel(cfg.checkpoint),
        "image_size": cfg.image_size,
        "num_samples": len(samples),
        "gt_dir": "gt" if cfg.mask_source == "excluded_real" else None,
        "samples": samples,
        **extra_meta,
    }
    with (out_root / "synthetic_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(samples)} pairs under {out_root}")
    print(
        f"Mask check: semantic labels in 0..{expect_max_label}; "
        f"masks with zero foreground (all bg): {zero_fg} / {len(samples)} "
        f"(viewers show semantic masks as nearly black; use --save-mask-preview to add masks_preview/)"
    )
    return manifest
