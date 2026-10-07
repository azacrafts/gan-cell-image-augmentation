"""Load results_exp_{a,b,c}.json from runs and print a comparison table (mIoU, IoU, mask AP when present)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_metrics(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=ROOT)
    args = p.parse_args()
    root = args.repo_root.resolve()
    paths = {
        "A": root / "runs" / "exp_a_baseline" / "results_exp_a.json",
        "B": root / "runs" / "exp_b_sparse" / "results_exp_b.json",
        "C": root / "runs" / "exp_c_sparse_cgan" / "results_exp_c.json",
    }
    rows = []
    for name, path in paths.items():
        data = load_metrics(path)
        if data is None:
            rows.append(
                {
                    "name": name,
                    "path": str(path),
                    "missing": True,
                }
            )
            continue
        m = data.get("metrics_test", {})
        downstream = data.get("downstream", "semantic")
        iou = m.get("iou_per_class", {})
        sh = iou.get("shsy5y", float("nan"))
        miou_fg = m.get("miou_foreground_8_lines", float("nan"))
        pacc = m.get("pixel_accuracy", float("nan"))
        map50 = m.get("mask_ap50", float("nan"))
        map5095 = m.get("mask_ap50_95", float("nan"))
        pmap50 = m.get("pseudo_mask_ap50", float("nan"))
        pmap5095 = m.get("pseudo_mask_ap50_95", float("nan"))
        rows.append(
            {
                "name": name,
                "path": str(path),
                "missing": False,
                "downstream": downstream,
                "miou_fg": miou_fg,
                "shsy5y_iou": sh,
                "pixel_accuracy": pacc,
                "mask_ap50": map50,
                "mask_ap50_95": map5095,
                "pseudo_mask_ap50": pmap50,
                "pseudo_mask_ap50_95": pmap5095,
            }
        )

    print("=== LIVECell downstream experiments (official test) ===\n")
    print(
        f"{'Exp':>4}  {'downstream':>28}  {'mIoU_fg':>10}  {'IoU_shsy5y':>12}  "
        f"{'mask_AP50':>10}  {'mask_AP50-95':>12}  {'pseudo50':>10}  results file"
    )
    print("-" * 120)
    for r in rows:
        if r["missing"]:
            print(f"  {r['name']}  {'(missing)':>28}  {'':>10}  {'':>12}  {'':>10}  {'':>12}  {'':>10}  {r['path']}")
        else:
            ds = str(r["downstream"])
            mf = r["miou_fg"]
            sh = r["shsy5y_iou"]
            m50 = r["mask_ap50"]
            m95 = r["mask_ap50_95"]
            p50 = r["pseudo_mask_ap50"]
            print(
                f"  {r['name']}  {ds:>28}  {mf:10.4f}  {sh:12.4f}  "
                f"{m50:10.4f}  {m95:12.4f}  {p50:10.4f}  {r['path']}"
            )
    print()
    out_txt = root / "runs" / "summary_table.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8") as f:
        f.write(
            "exp\tdownstream\tmiou_fg\tiou_shsy5y\tpixel_accuracy\t"
            "mask_ap50\tmask_ap50_95\tpseudo_mask_ap50\tpseudo_mask_ap50_95\n"
        )
        for r in rows:
            if r["missing"]:
                f.write(f"{r['name']}\t\t\t\t\t\t\t\n")
            else:
                f.write(
                    f"{r['name']}\t{r['downstream']}\t{r['miou_fg']}\t{r['shsy5y_iou']}\t{r['pixel_accuracy']}\t"
                    f"{r['mask_ap50']}\t{r['mask_ap50_95']}\t{r['pseudo_mask_ap50']}\t{r['pseudo_mask_ap50_95']}\n"
                )
    print(f"Wrote TSV summary: {out_txt}")


if __name__ == "__main__":
    main()
