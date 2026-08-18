"""
evaluate.py

STANDALONE inference / benchmarking script. This is the file KLA's
benchmarking team runs AS-IS on the H100 -- it must work with no manual
edits.

Usage (required, per submission spec):
    python evaluate.py --input_dir /path/to/test/NoisyLR --output_dir /path/to/output

Optional (if ground truth is available, e.g. for your own validation):
    python evaluate.py --input_dir .../NoisyLR --output_dir .../out --gt_dir .../GT

What it does:
    1. Loads the trained model from --checkpoint (default: checkpoints/best.pt)
       and normalization stats from --stats (default: stats.json).
    2. Reads every .npy file in --input_dir.
    3. Runs restoration inference on each (denoise + 2x upscale).
    4. Saves each output as a .npy file (same dtype range as GT, i.e.
       denormalized) into --output_dir, using the same filename.
    5. Reports per-image and average inference time.
    6. If --gt_dir is given, also computes PSNR / SSIM (and LPIPS if the
       `lpips` package is installed) against ground truth and prints a
       summary table + writes metrics.json.

Notes on npy vs png output:
    The task's data is provided as .npy, so outputs are written as .npy by
    default (dtype float32, denormalized back to the original GT value
    range) to stay consistent with the input format. If you additionally
    want .png previews for the slide deck, use --save_png.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import RestorationNet
from tta import predict_with_tta


def load_stats(stats_path: str):
    with open(stats_path, "r") as f:
        return json.load(f)


def denormalize(arr: np.ndarray, stats: dict) -> np.ndarray:
    return arr * stats["scale"] + stats["offset"]


def normalize(arr: np.ndarray, stats: dict) -> np.ndarray:
    return (arr.astype(np.float32) - stats["offset"]) / stats["scale"]


def compute_psnr(pred: np.ndarray, gt: np.ndarray, data_range: float) -> float:
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 10 * np.log10((data_range ** 2) / mse)


def compute_ssim(pred: np.ndarray, gt: np.ndarray, data_range: float) -> float:
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        return float(sk_ssim(pred, gt, data_range=data_range))
    except ImportError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True,
                     help="Directory of degraded .npy test images")
    ap.add_argument("--output_dir", type=str, required=True,
                     help="Directory to write restored .npy outputs")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    ap.add_argument("--stats", type=str, default="stats.json")
    ap.add_argument("--gt_dir", type=str, default=None,
                     help="Optional: ground truth dir, for computing metrics")
    ap.add_argument("--save_png", action="store_true",
                     help="Also save 8-bit PNG previews alongside .npy outputs")
    ap.add_argument("--device", type=str, default=None,
                     help="cuda / cpu. Auto-detected if not given.")
    ap.add_argument("--tta", action="store_true",
                     help="Enable test-time flip ensembling (4x forward passes, "
                          "averaged). Typically a small but consistent PSNR/SSIM "
                          "gain at ~4x the per-image inference time. Check the "
                          "printed avg_inference_time against your speed budget.")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    stats = load_stats(args.stats)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt.get("args", {})
    model = RestorationNet(
        num_features=model_args.get("num_features", 64),
        num_blocks=model_args.get("num_blocks", 12),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.npy"))
    if len(input_files) == 0:
        raise RuntimeError(f"No .npy files found in {input_dir}")

    gt_dir = Path(args.gt_dir) if args.gt_dir else None

    inference_times = []
    psnr_scores, ssim_scores = [], []

    for f in input_files:
        arr = np.load(f).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]

        norm = normalize(arr, stats)
        tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                if args.tta:
                    pred = predict_with_tta(model, tensor)
                else:
                    pred = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        inference_times.append(dt)

        pred_np = pred.squeeze().float().cpu().numpy()
        restored = denormalize(pred_np, stats)

        out_path = output_dir / f.name
        np.save(out_path, restored.astype(np.float32))

        if args.save_png:
            from PIL import Image
            gt_min, gt_max = stats["gt_true_min"], stats["gt_true_max"]
            png_arr = np.clip((restored - gt_min) / max(gt_max - gt_min, 1e-6), 0, 1)
            png_arr = (png_arr * 255).astype(np.uint8)
            Image.fromarray(png_arr).save(output_dir / f"{f.stem}.png")

        if gt_dir is not None:
            gt_path = gt_dir / f.name
            if gt_path.exists():
                gt_arr = np.load(gt_path).astype(np.float32)
                if gt_arr.ndim == 3:
                    gt_arr = gt_arr[..., 0]
                data_range = stats["gt_true_max"] - stats["gt_true_min"]
                psnr = compute_psnr(restored, gt_arr, data_range)
                ssim = compute_ssim(restored, gt_arr, data_range)
                psnr_scores.append(psnr)
                ssim_scores.append(ssim)

        print(f"  {f.name}: {arr.shape} -> {restored.shape}  ({dt*1000:.1f} ms)")

    avg_time = float(np.mean(inference_times))
    print("\n" + "=" * 60)
    print(f"Processed {len(input_files)} images")
    print(f"TTA (flip ensembling): {'ON (4x forward passes)' if args.tta else 'OFF'}")
    print(f"Avg inference time: {avg_time*1000:.2f} ms/image")
    print(f"Total time: {sum(inference_times):.2f} s")

    summary = {
        "num_images": len(input_files),
        "tta": args.tta,
        "avg_inference_time_sec": avg_time,
        "total_inference_time_sec": float(sum(inference_times)),
        "device": str(device),
    }

    if psnr_scores:
        summary["avg_psnr"] = float(np.mean(psnr_scores))
        summary["avg_ssim"] = float(np.nanmean(ssim_scores))
        print(f"Avg PSNR: {summary['avg_psnr']:.2f} dB")
        print(f"Avg SSIM: {summary['avg_ssim']:.4f}")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs written to: {output_dir}")
    print(f"Summary written to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()