"""
Run this FIRST, before anything else.

python data_audit.py --train_dir /path/to/train

Expected structure (edit --train_dir to point at the folder that directly
contains GT/ and NoisyLR/):

    train/
      GT/
        xxxx.npy
      NoisyLR/
        xxxx.npy

It will:
  1. List file counts in GT and NoisyLR.
  2. Try to pair files by filename (with a few fallback heuristics).
  3. Report shape / dtype / min / max / mean for a sample of pairs.
  4. Report how many distinct (GT_shape, Noisy_shape) combinations exist,
     since the task mixes 512->256 and 256->128 downsampling.

Paste the full printed output back to Claude so normalization + the
dataset loader can be calibrated to the real data instead of assumptions.
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np


def find_pairs(gt_dir: Path, lr_dir: Path):
    gt_files = {f.stem: f for f in gt_dir.glob("*.npy")}
    lr_files = {f.stem: f for f in lr_dir.glob("*.npy")}

    pairs = []
    unmatched_gt = []

    # 1) exact stem match
    for stem, gt_path in gt_files.items():
        if stem in lr_files:
            pairs.append((gt_path, lr_files[stem]))
            continue

        # 2) strip common suffixes/prefixes and retry
        candidates = [
            re.sub(r"(_gt|_GT|_clean|_hr|_HR)$", "", stem),
        ]
        found = False
        for cand in candidates:
            for lr_stem in lr_files:
                lr_clean = re.sub(r"(_lr|_LR|_noisy|_ns|_degraded)$", "", lr_stem)
                if lr_clean == cand or lr_stem == stem:
                    pairs.append((gt_path, lr_files[lr_stem]))
                    found = True
                    break
            if found:
                break
        if not found:
            unmatched_gt.append(stem)

    # 3) fallback: if nothing matched by name, try sorted positional pairing
    if len(pairs) == 0 and len(gt_files) == len(lr_files):
        print(">>> No filename matches found. Falling back to SORTED ORDER pairing.")
        print(">>> This is risky -- verify manually that order is meaningful!")
        gt_sorted = sorted(gt_files.values())
        lr_sorted = sorted(lr_files.values())
        pairs = list(zip(gt_sorted, lr_sorted))
        unmatched_gt = []

    return pairs, unmatched_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, required=True,
                     help="Folder containing GT/ and NoisyLR/ subfolders")
    ap.add_argument("--sample_n", type=int, default=15,
                     help="How many pairs to print detailed stats for")
    args = ap.parse_args()

    root = Path(args.train_dir)
    gt_dir = root / "GT"
    lr_dir = root / "NoisyLR"

    assert gt_dir.exists(), f"Missing GT dir at {gt_dir}"
    assert lr_dir.exists(), f"Missing NoisyLR dir at {lr_dir}"

    gt_list = sorted(gt_dir.glob("*.npy"))
    lr_list = sorted(lr_dir.glob("*.npy"))

    print("=" * 70)
    print(f"GT files:      {len(gt_list)}")
    print(f"NoisyLR files: {len(lr_list)}")
    print("=" * 70)

    print("\nFirst 10 GT filenames:")
    for f in gt_list[:10]:
        print(" ", f.name)
    print("\nFirst 10 NoisyLR filenames:")
    for f in lr_list[:10]:
        print(" ", f.name)

    pairs, unmatched = find_pairs(gt_dir, lr_dir)
    print(f"\nPaired: {len(pairs)}  |  Unmatched GT: {len(unmatched)}")
    if unmatched:
        print("  Sample unmatched GT stems:", unmatched[:10])

    print("\n" + "=" * 70)
    print(f"DETAILED STATS (sampling up to {args.sample_n} pairs)")
    print("=" * 70)

    shape_combos = {}
    dtypes_seen = set()

    for i, (gt_path, lr_path) in enumerate(pairs):
        gt = np.load(gt_path)
        lr = np.load(lr_path)

        combo = (gt.shape, lr.shape)
        shape_combos[combo] = shape_combos.get(combo, 0) + 1
        dtypes_seen.add((str(gt.dtype), str(lr.dtype)))

        if i < args.sample_n:
            print(f"\nPair {i}: {gt_path.name}  <->  {lr_path.name}")
            print(f"  GT     shape={gt.shape} dtype={gt.dtype} "
                  f"min={gt.min():.4f} max={gt.max():.4f} mean={gt.mean():.4f} std={gt.std():.4f}")
            print(f"  Noisy  shape={lr.shape} dtype={lr.dtype} "
                  f"min={lr.min():.4f} max={lr.max():.4f} mean={lr.mean():.4f} std={lr.std():.4f}")
            ratio = gt.shape[0] / lr.shape[0] if lr.shape[0] else None
            print(f"  Downsample ratio (GT/Noisy, dim0): {ratio}")

    print("\n" + "=" * 70)
    print("Shape combinations across ALL pairs (GT_shape -> Noisy_shape : count):")
    for combo, count in sorted(shape_combos.items(), key=lambda x: -x[1]):
        print(f"  {combo[0]} -> {combo[1]} : {count}")

    print("\nDtypes seen (GT_dtype, Noisy_dtype):", dtypes_seen)

    # Global min/max across a larger sample (up to 100) to catch outliers
    print("\n" + "=" * 70)
    print("GLOBAL RANGE CHECK (up to 100 pairs) -- do Noisy values exceed GT range?")
    print("=" * 70)
    gt_mins, gt_maxs, lr_mins, lr_maxs = [], [], [], []
    for gt_path, lr_path in pairs[:100]:
        gt = np.load(gt_path)
        lr = np.load(lr_path)
        gt_mins.append(gt.min()); gt_maxs.append(gt.max())
        lr_mins.append(lr.min()); lr_maxs.append(lr.max())

    print(f"GT    global min/max: {min(gt_mins):.4f} / {max(gt_maxs):.4f}")
    print(f"Noisy global min/max: {min(lr_mins):.4f} / {max(lr_maxs):.4f}")
    print("\nDone. Paste this entire output back to Claude.")


if __name__ == "__main__":
    main()
