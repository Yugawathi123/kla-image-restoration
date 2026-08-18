"""
generate_report.py

Answers the key question: "is the trained model actually better than doing
nothing?" -- compares your trained model against a naive bicubic-upsample
baseline (zero learning, just interpolation) on the SAME held-out validation
split used during training (same seed, same val_split, so this is a fair,
un-contaminated comparison -- these images were never trained on).

Also saves side-by-side comparison images (degraded / bicubic baseline /
model output / ground truth) for your results slide.

Usage:
    python generate_report.py --train_dir /path/to/train --stats stats.json \
        --checkpoint checkpoints/best.pt --out report/ --num_samples_to_save 8

Must be run with the SAME --val_split and --seed used in train.py (defaults
match train.py's defaults: val_split=0.1, seed=42) so the split lines up
with what the model actually never saw during training.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import split_pairs, load_norm_stats
from model import RestorationNet
from tta import predict_with_tta


def compute_psnr(pred, gt, data_range):
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 10 * np.log10((data_range ** 2) / mse)


def compute_ssim(pred, gt, data_range):
    from skimage.metrics import structural_similarity as sk_ssim
    return float(sk_ssim(pred, gt, data_range=data_range))


def to_uint8_png(arr, lo, hi):
    a = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (a * 255).astype(np.uint8)


def to_uint8_png_autocontrast(arr, low_pct=1.0, high_pct=99.0):
    """Per-panel contrast stretch for viewing only (does not affect metrics).
    See predict_single.py for the full explanation."""
    lo = np.percentile(arr, low_pct)
    hi = np.percentile(arr, high_pct)
    a = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (a * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, required=True)
    ap.add_argument("--stats", type=str, default="stats.json")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    ap.add_argument("--out", type=str, default="report")
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_samples_to_save", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = load_norm_stats(args.stats)

    # Same deterministic split used by train.py (see dataset.py::split_pairs) --
    # these are the images the model never saw during training.
    _, val_pairs = split_pairs(args.train_dir, val_split=args.val_split, seed=args.seed)
    print(f"Evaluating on {len(val_pairs)} held-out validation pairs "
          f"(never seen during training)")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt.get("args", {})
    model = RestorationNet(
        num_features=model_args.get("num_features", 64),
        num_blocks=model_args.get("num_blocks", 12),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    data_range = stats["gt_true_max"] - stats["gt_true_min"]

    model_psnrs, model_ssims = [], []
    tta_psnrs, tta_ssims = [], []
    bicubic_psnrs, bicubic_ssims = [], []

    for i, (gt_path, lr_path) in enumerate(val_pairs):
        gt = np.load(gt_path).astype(np.float32)
        lr = np.load(lr_path).astype(np.float32)
        if gt.ndim == 3:
            gt = gt[..., 0]
        if lr.ndim == 3:
            lr = lr[..., 0]

        lr_norm = (lr - stats["offset"]) / stats["scale"]

        lr_t = torch.from_numpy(lr_norm).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred_t = model(lr_t)
            tta_t = predict_with_tta(model, lr_t)
            bicubic_t = F.interpolate(lr_t, scale_factor=2, mode="bicubic", align_corners=False)

        pred_np = pred_t.squeeze().cpu().numpy()
        tta_np = tta_t.squeeze().cpu().numpy()
        bicubic_np = bicubic_t.squeeze().cpu().numpy()

        # denormalize back to true GT scale for fair metric comparison
        pred_denorm = pred_np * stats["scale"] + stats["offset"]
        tta_denorm = tta_np * stats["scale"] + stats["offset"]
        bicubic_denorm = bicubic_np * stats["scale"] + stats["offset"]

        model_psnrs.append(compute_psnr(pred_denorm, gt, data_range))
        model_ssims.append(compute_ssim(pred_denorm, gt, data_range))
        tta_psnrs.append(compute_psnr(tta_denorm, gt, data_range))
        tta_ssims.append(compute_ssim(tta_denorm, gt, data_range))
        bicubic_psnrs.append(compute_psnr(bicubic_denorm, gt, data_range))
        bicubic_ssims.append(compute_ssim(bicubic_denorm, gt, data_range))

        if i < args.num_samples_to_save:
            lr_png = to_uint8_png_autocontrast(lr)
            bicubic_png = to_uint8_png_autocontrast(bicubic_denorm)
            pred_png = to_uint8_png_autocontrast(pred_denorm)
            tta_png = to_uint8_png_autocontrast(tta_denorm)
            gt_png = to_uint8_png_autocontrast(gt)

            # side-by-side: degraded (upscaled for display) | bicubic | model | model+TTA | GT
            h, w = gt_png.shape
            lr_disp = np.array(Image.fromarray(lr_png).resize((w, h), Image.NEAREST))
            composite = np.concatenate([lr_disp, bicubic_png, pred_png, tta_png, gt_png], axis=1)
            Image.fromarray(composite).save(out_dir / f"compare_{i:03d}.png")

    summary = {
        "num_val_samples": len(val_pairs),
        "bicubic_baseline_avg_psnr": float(np.mean(bicubic_psnrs)),
        "bicubic_baseline_avg_ssim": float(np.mean(bicubic_ssims)),
        "model_avg_psnr": float(np.mean(model_psnrs)),
        "model_avg_ssim": float(np.mean(model_ssims)),
        "model_tta_avg_psnr": float(np.mean(tta_psnrs)),
        "model_tta_avg_ssim": float(np.mean(tta_ssims)),
        "psnr_improvement_over_bicubic_db": float(np.mean(model_psnrs) - np.mean(bicubic_psnrs)),
        "ssim_improvement_over_bicubic": float(np.mean(model_ssims) - np.mean(bicubic_ssims)),
        "psnr_gain_from_tta_db": float(np.mean(tta_psnrs) - np.mean(model_psnrs)),
        "ssim_gain_from_tta": float(np.mean(tta_ssims) - np.mean(model_ssims)),
    }

    with open(out_dir / "validation_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS (held-out validation split, never trained on)")
    print("=" * 70)
    print(f"{'Metric':<20}{'Bicubic':<15}{'Model':<15}{'Model + TTA':<15}")
    print(f"{'PSNR (dB)':<20}{summary['bicubic_baseline_avg_psnr']:<15.2f}"
          f"{summary['model_avg_psnr']:<15.2f}{summary['model_tta_avg_psnr']:<15.2f}")
    print(f"{'SSIM':<20}{summary['bicubic_baseline_avg_ssim']:<15.4f}"
          f"{summary['model_avg_ssim']:<15.4f}{summary['model_tta_avg_ssim']:<15.4f}")
    print(f"\nModel vs bicubic:  {summary['psnr_improvement_over_bicubic_db']:+.2f} dB PSNR, "
          f"{summary['ssim_improvement_over_bicubic']:+.4f} SSIM")
    print(f"TTA vs plain model: {summary['psnr_gain_from_tta_db']:+.2f} dB PSNR, "
          f"{summary['ssim_gain_from_tta']:+.4f} SSIM  (at ~4x the inference cost)")
    print(f"\nSaved {min(args.num_samples_to_save, len(val_pairs))} comparison images to {out_dir}")
    print("Panel order: degraded | bicubic | model | model+TTA | ground truth")
    print(f"Full report: {out_dir / 'validation_report.json'}")


if __name__ == "__main__":
    main()