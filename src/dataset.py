"""
dataset.py

Paired dataset loader for the KLA restoration task.

Expected directory layout (matches what you described):

    train_dir/
      GT/
        xxxx.npy      <- clean, full resolution (512x512 or 256x256)
      NoisyLR/
        xxxx.npy      <- degraded, low-res (256x256 or 128x128)

Key design decisions:

  1. NORMALIZATION: We do NOT hardcode a fixed scale like /255. Instead we
     read normalization stats from a JSON file (produced by calibrate_stats.py)
     computed on the GT training set only. This matters because:
       - We don't yet know if these are uint8, uint16, or float arrays.
       - The degraded images can legitimately exceed the GT's max value
         (speckle noise pushing pixels out of range) -- if we clip based on
         GT stats, we'd be throwing away real signal we specifically want
         the model to learn to recognize as "noise, not detail." So we
         normalize both by the SAME scale (derived from GT), WITHOUT
         clipping the noisy input, and let the network learn to pull
         overshoot pixels back into range as part of denoising.

  2. PATCH-BASED TRAINING: with only ~200 pairs and laptop GPU VRAM, we
     train on random cropped patches rather than full images. This both
     fits memory and multiplies the effective number of training samples
     each epoch via random crop location + flips/rotations.

  3. PAIRING: filenames are matched by stem with a few fallback suffix
     rules. If your actual filenames differ from these patterns, this is
     the file to edit (see `match_pairs`) -- send me the audit script
     output and I'll adjust this precisely instead of guessing.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def match_pairs(gt_dir: Path, lr_dir: Path) -> List[Tuple[Path, Path]]:
    gt_files = {f.stem: f for f in sorted(gt_dir.glob("*.npy"))}
    lr_files = {f.stem: f for f in sorted(lr_dir.glob("*.npy"))}

    pairs = []
    for stem, gt_path in gt_files.items():
        if stem in lr_files:
            pairs.append((gt_path, lr_files[stem]))
            continue
        cand = re.sub(r"(_gt|_GT|_clean|_hr|_HR)$", "", stem)
        found = False
        for lr_stem, lr_path in lr_files.items():
            lr_clean = re.sub(r"(_lr|_LR|_noisy|_ns|_degraded)$", "", lr_stem)
            if lr_clean == cand:
                pairs.append((gt_path, lr_path))
                found = True
                break
        if not found:
            pass  # silently skip; calibrate_stats.py / data_audit.py surfaces mismatches loudly

    if len(pairs) == 0 and len(gt_files) == len(lr_files):
        gt_sorted = sorted(gt_files.values())
        lr_sorted = sorted(lr_files.values())
        pairs = list(zip(gt_sorted, lr_sorted))

    return pairs


def load_norm_stats(stats_path: str):
    with open(stats_path, "r") as f:
        stats = json.load(f)
    return stats  # expects {"scale": float, "offset": float, ...}


def split_pairs(train_dir: str, val_split: float = 0.1, seed: int = 42):
    """Split the full pair list into train/val pair lists ONCE, using a
    fixed seed, so the same split can be reconstructed deterministically by
    any script (train.py, generate_report.py, etc). This replaces using
    torch's random_split directly on a Dataset instance, which silently
    shared one Dataset object (and its augment/crop settings) between both
    the train and val subsets -- meaning validation was being randomly
    cropped and flipped exactly like training data, adding noise to the
    per-epoch validation metric during training.
    """
    root = Path(train_dir)
    gt_dir = root / "GT"
    lr_dir = root / "NoisyLR"
    pairs = match_pairs(gt_dir, lr_dir)
    if len(pairs) == 0:
        raise RuntimeError(
            f"No pairs found under {train_dir}. Check GT/NoisyLR subfolder "
            f"names and filename conventions (see match_pairs in dataset.py)."
        )

    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)

    val_len = max(1, int(len(pairs) * val_split))
    val_indices = set(indices[:val_len])

    train_pairs = [pairs[i] for i in range(len(pairs)) if i not in val_indices]
    val_pairs = [pairs[i] for i in range(len(pairs)) if i in val_indices]
    return train_pairs, val_pairs


class PairedRestorationDataset(Dataset):
    def __init__(self, train_dir: str = None, stats_path: str = "stats.json",
                 patch_size: int = 96, augment: bool = True,
                 scale_factor: int = 2, pairs: List[Tuple[Path, Path]] = None):
        """
        patch_size: crop size taken from the LOW-RES image. The
                    corresponding GT crop is patch_size * scale_factor.
                    Pass None for NO cropping -- returns full images as-is,
                    used for validation/evaluation (never for training,
                    since full-image batches of varying use cases don't
                    benefit from the memory savings patches give).
        scale_factor: fixed at 2 per the problem statement (both
                    512->256 and 256->128 are 2x downsamples).
        pairs: optional explicit list of (gt_path, lr_path) tuples. If not
                    given, scans train_dir/GT and train_dir/NoisyLR directly
                    (original behavior). Pass this when you want a specific
                    train or val subset, e.g. from split_pairs().
        """
        if pairs is not None:
            self.pairs = pairs
        else:
            root = Path(train_dir)
            self.gt_dir = root / "GT"
            self.lr_dir = root / "NoisyLR"
            self.pairs = match_pairs(self.gt_dir, self.lr_dir)

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No pairs found under {train_dir}. Check GT/NoisyLR subfolder "
                f"names and filename conventions (see match_pairs in dataset.py)."
            )

        stats = load_norm_stats(stats_path)
        self.scale = stats["scale"]
        self.offset = stats.get("offset", 0.0)

        self.patch_size = patch_size
        self.augment = augment
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.pairs)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        return (arr.astype(np.float32) - self.offset) / self.scale

    def __getitem__(self, idx):
        gt_path, lr_path = self.pairs[idx]
        gt = np.load(gt_path).astype(np.float32)
        lr = np.load(lr_path).astype(np.float32)

        # Defensive: some npy files may have a trailing channel dim (H,W,1).
        if gt.ndim == 3:
            gt = gt[..., 0]
        if lr.ndim == 3:
            lr = lr[..., 0]

        gt = self._normalize(gt)
        lr = self._normalize(lr)

        if self.patch_size is None:
            # Full-image mode: no cropping, no augmentation. Used for
            # validation so the metric reflects the real, stable image
            # rather than a randomly-cropped-and-flipped patch.
            lr_t = torch.from_numpy(lr.copy()).unsqueeze(0).float()
            gt_t = torch.from_numpy(gt.copy()).unsqueeze(0).float()
            return lr_t, gt_t

        h_lr, w_lr = lr.shape
        ps = self.patch_size

        if h_lr < ps or w_lr < ps:
            # image smaller than requested patch -> use whole image (rare edge case)
            ps_h, ps_w = h_lr, w_lr
            top, left = 0, 0
        else:
            ps_h = ps_w = ps
            top = random.randint(0, h_lr - ps_h)
            left = random.randint(0, w_lr - ps_w)

        lr_patch = lr[top:top + ps_h, left:left + ps_w]
        gt_top, gt_left = top * self.scale_factor, left * self.scale_factor
        gt_patch = gt[gt_top:gt_top + ps_h * self.scale_factor,
                       gt_left:gt_left + ps_w * self.scale_factor]

        if self.augment:
            if random.random() < 0.5:
                lr_patch = np.fliplr(lr_patch).copy()
                gt_patch = np.fliplr(gt_patch).copy()
            if random.random() < 0.5:
                lr_patch = np.flipud(lr_patch).copy()
                gt_patch = np.flipud(gt_patch).copy()
            k = random.randint(0, 3)
            if k:
                lr_patch = np.rot90(lr_patch, k).copy()
                gt_patch = np.rot90(gt_patch, k).copy()

        lr_t = torch.from_numpy(lr_patch).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt_patch).unsqueeze(0).float()
        return lr_t, gt_t


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python dataset.py <train_dir> <stats_json>")
        sys.exit(0)
    ds = PairedRestorationDataset(sys.argv[1], sys.argv[2], patch_size=64)
    print(f"Found {len(ds)} pairs")
    lr, gt = ds[0]
    print("LR patch:", lr.shape, lr.min().item(), lr.max().item())
    print("GT patch:", gt.shape, gt.min().item(), gt.max().item())