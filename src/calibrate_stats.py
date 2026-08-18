"""
calibrate_stats.py

Run this AFTER data_audit.py confirms your file structure, and BEFORE
training. It computes normalization stats from the GT training images only
(never from NoisyLR -- we don't want the "exceeds range" behavior of noisy
images to affect what "1.0" means) and saves them to stats.json.

Usage:
    python calibrate_stats.py --train_dir /path/to/train --out stats.json

Both dataset.py (training) and evaluate.py (inference) load this same
stats.json, so normalization is guaranteed consistent between train and test.

Why offset/scale instead of a fixed /255:
    We don't yet know if your .npy files are uint8, uint16, or float, or
    what range they occupy. Using data-driven percentiles instead of a
    hardcoded assumption avoids silently mis-normalizing everything, which
    would be a swift and effective way to sink model quality.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dataset import match_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="stats.json")
    ap.add_argument("--low_pct", type=float, default=0.5,
                     help="lower percentile clip for offset")
    ap.add_argument("--high_pct", type=float, default=99.5,
                     help="upper percentile clip for scale")
    args = ap.parse_args()

    root = Path(args.train_dir)
    gt_dir = root / "GT"
    lr_dir = root / "NoisyLR"
    pairs = match_pairs(gt_dir, lr_dir)
    print(f"Computing stats from {len(pairs)} GT images...")

    all_vals = []
    for gt_path, _ in pairs:
        arr = np.load(gt_path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        all_vals.append(arr.flatten())

    all_vals = np.concatenate(all_vals)

    low = float(np.percentile(all_vals, args.low_pct))
    high = float(np.percentile(all_vals, args.high_pct))
    true_min = float(all_vals.min())
    true_max = float(all_vals.max())

    offset = low
    scale = max(high - low, 1e-6)

    stats = {
        "offset": offset,
        "scale": scale,
        "gt_true_min": true_min,
        "gt_true_max": true_max,
        "low_percentile": args.low_pct,
        "high_percentile": args.high_pct,
        "note": "normalized = (raw - offset) / scale. NoisyLR is normalized "
                "with the SAME offset/scale (not clipped), so overshoot from "
                "speckle noise remains visible to the model as values >1 or <0.",
    }

    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
