r"""
predict_single.py

Test the model on ONE image, the way you'd actually use it: give it a
degraded .npy file, get back a restored .npy file -- plus a PNG you can
actually look at (this repo's other scripts only ever produced .npy or
processed folders in bulk; this is the "just show me one image" version).

Usage (no ground truth -- e.g. a real test image):
    python predict_single.py --input path\to\000005.npy --checkpoint checkpoints\best.pt --stats stats.json --out demo_output

Usage (with ground truth -- e.g. one of your training pairs, for a sanity check):
    python predict_single.py --input path\to\train\NoisyLR\000005.npy --gt path\to\train\GT\000005.npy --checkpoint checkpoints\best.pt --stats stats.json --out demo_output

Produces in --out:
    restored.npy         -- the restored image, raw values, same scale as GT
    comparison.png        -- a viewable image: degraded | restored | (GT if given)
    (prints PSNR/SSIM to console if --gt is given)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import RestorationNet
from evaluate import load_stats, normalize, denormalize, compute_psnr, compute_ssim
from tta import predict_with_tta


def to_uint8_png(arr, lo, hi):
    a = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (a * 255).astype(np.uint8)


def to_uint8_png_autocontrast(arr, low_pct=1.0, high_pct=99.0):
    """Stretch THIS panel's own value range to fill 0-255, for viewing only.
    Real inspection images can be naturally dark/low-contrast (small mean,
    small std) -- displaying them on a fixed global [0,1] scale crushes
    everything toward black even though the underlying data and the model's
    output are both fine. This is purely cosmetic: it never touches the
    saved restored.npy values or any PSNR/SSIM computation, only how the
    PNG looks on screen."""
    lo = np.percentile(arr, low_pct)
    hi = np.percentile(arr, high_pct)
    a = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (a * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Path to one degraded .npy file")
    ap.add_argument("--gt", type=str, default=None, help="Optional: path to matching ground truth .npy")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    ap.add_argument("--stats", type=str, default="stats.json")
    ap.add_argument("--out", type=str, default="demo_output")
    ap.add_argument("--tta", action="store_true",
                     help="Enable test-time flip ensembling (4x forward passes, averaged)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    stats = load_stats(args.stats)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt.get("args", {})
    model = RestorationNet(
        num_features=model_args.get("num_features", 64),
        num_blocks=model_args.get("num_blocks", 12),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    arr = np.load(args.input).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    print(f"Input: {args.input}  shape={arr.shape}  min={arr.min():.4f}  max={arr.max():.4f}")

    norm = normalize(arr, stats)
    tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        if args.tta:
            pred = predict_with_tta(model, tensor)
            print("TTA (flip ensembling): ON")
        else:
            pred = model(tensor)

    pred_np = pred.squeeze().float().cpu().numpy()
    restored = denormalize(pred_np, stats)

    np.save(out_dir / "restored.npy", restored.astype(np.float32))
    print(f"Restored: shape={restored.shape}  min={restored.min():.4f}  max={restored.max():.4f}")

    lo, hi = stats["gt_true_min"], stats["gt_true_max"]

    input_png = to_uint8_png_autocontrast(arr)
    restored_png = to_uint8_png_autocontrast(restored)

    # upscale the degraded input for side-by-side display (nearest neighbor,
    # so it visibly looks blocky/low-res next to the restored output --
    # this is just for viewing, not used in any metric)
    h, w = restored_png.shape
    input_disp = np.array(Image.fromarray(input_png).resize((w, h), Image.NEAREST))

    panels = [input_disp, restored_png]
    labels = ["Degraded Input (upscaled for display)", "Restored Output"]

    if args.gt:
        gt = np.load(args.gt).astype(np.float32)
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt_png = to_uint8_png_autocontrast(gt)
        panels.append(gt_png)
        labels.append("Ground Truth")

        data_range = hi - lo
        psnr = compute_psnr(restored, gt, data_range)
        ssim = compute_ssim(restored, gt, data_range)
        print(f"\nPSNR vs ground truth: {psnr:.2f} dB")
        print(f"SSIM vs ground truth: {ssim:.4f}")

    composite = np.concatenate(panels, axis=1)
    Image.fromarray(composite).save(out_dir / "comparison.png")

    print(f"\nSaved: {out_dir / 'restored.npy'}")
    print(f"Saved: {out_dir / 'comparison.png'}  ({' | '.join(labels)})")


if __name__ == "__main__":
    main()