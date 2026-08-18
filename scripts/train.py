"""
train.py

Reproducible training script for the KLA restoration model.

Usage:
    python train.py --train_dir /path/to/train --stats stats.json \
        --epochs 200 --batch_size 16 --patch_size 96 --out checkpoints/

Designed for a single laptop GPU (e.g. RTX 4050/4060 mobile, ~6-8GB VRAM):
  - Small patch-based crops (default 96x96 on the LR side) keep memory low.
  - Mixed precision (torch.amp) roughly halves memory use and speeds up
    training on modern NVIDIA GPUs.
  - Small model (~1M params, see model.py) trains quickly even on CPU-grade
    epoch counts if needed.

With ~200 image pairs and patch-based random cropping, each epoch sees many
different crops per image, so 150-300 epochs is a reasonable starting range
-- adjust based on the validation loss curve.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import PairedRestorationDataset, split_pairs
from model import RestorationNet, count_parameters
from losses import CombinedLoss


def psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    if mse <= 1e-12:
        return 99.0
    import math
    return 10 * math.log10((data_range ** 2) / mse)


def evaluate_epoch(model, loader, device):
    model.eval()
    total_mse = 0.0
    n = 0
    with torch.no_grad():
        for lr, gt in loader:
            lr, gt = lr.to(device), gt.to(device)
            pred = model(lr)
            mse = torch.mean((pred - gt) ** 2).item()
            total_mse += mse * lr.size(0)
            n += lr.size(0)
    return total_mse / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, required=True)
    ap.add_argument("--stats", type=str, default="stats.json")
    ap.add_argument("--out", type=str, default="checkpoints")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--patch_size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--num_blocks", type=int, default=12)
    ap.add_argument("--num_features", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--ssim_weight", type=float, default=0.3)
    ap.add_argument("--edge_weight", type=float, default=0.15,
                     help="Weight for the Sobel edge-sharpness loss term. "
                          "Higher values push harder for sharp edges at some "
                          "risk of amplifying noise; 0 disables it entirely.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split ONCE into train/val pair lists (deterministic, seeded), then
    # build two SEPARATE Dataset instances:
    #   - train_ds: patch-cropped + augmented (flips/rotations), as before.
    #   - val_ds:   full images, NO cropping, NO augmentation.
    # Previously both shared one Dataset object via torch's random_split,
    # which meant validation images were being randomly cropped and flipped
    # exactly like training data -- adding noise to the per-epoch val metric
    # and occasionally letting a "lucky crop" look better than the model
    # actually was on the full image. This version reports the real,
    # stable, full-image validation PSNR every epoch.
    train_pairs, val_pairs = split_pairs(args.train_dir, val_split=args.val_split, seed=args.seed)
    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

    train_ds = PairedRestorationDataset(
        stats_path=args.stats, patch_size=args.patch_size, augment=True, pairs=train_pairs
    )
    val_ds = PairedRestorationDataset(
        stats_path=args.stats, patch_size=None, augment=False, pairs=val_pairs
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    model = RestorationNet(num_features=args.num_features, num_blocks=args.num_blocks).to(device)
    print(f"Model params: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = CombinedLoss(ssim_weight=args.ssim_weight, edge_weight=args.edge_weight).to(device)

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 0
    best_val_mse = float("inf")

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_mse = ckpt.get("best_val_mse", float("inf"))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    log_path = out_dir / "train_log.jsonl"

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_n = 0
        skipped_steps = 0

        for batch_idx, (lr_img, gt_img) in enumerate(train_loader):
            lr_img, gt_img = lr_img.to(device, non_blocking=True), gt_img.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                pred = model(lr_img)

            # Compute the loss in fp32 even though the forward pass above ran in fp16.
            # ROOT CAUSE of the stuck-training symptom (train_loss frozen for 20+
            # epochs, "lr_scheduler.step() before optimizer.step()" warning on epoch 1):
            # SSIM's variance computation (E[X^2] - E[X]^2) is numerically fragile in
            # fp16 and, once GradScaler's loss-scaling factor is applied, was
            # overflowing to inf almost every batch -- causing GradScaler to silently
            # skip the optimizer step nearly 100% of the time, leaving the weights
            # frozen near initialization while the loop kept printing numbers.
            pred_fp32 = pred.float()
            gt_fp32 = gt_img.float()
            loss, parts = loss_fn(pred_fp32, gt_fp32)

            if not torch.isfinite(loss):
                # Fail loudly and immediately instead of silently plateauing for
                # dozens of epochs -- GradScaler would otherwise skip
                # optimizer.step() here with no visible error, leaving weights
                # frozen while the loop keeps printing numbers.
                raise RuntimeError(
                    f"Non-finite loss ({loss.item()}) at epoch {epoch+1}, batch {batch_idx}. "
                    f"Loss parts: {parts}. This usually means a NaN/Inf appeared somewhere "
                    f"in the forward pass (check model.py / losses.py fp16 handling) or the "
                    f"learning rate is too high. Aborting rather than continuing silently."
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            pre_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            post_scale = scaler.get_scale()
            if post_scale < pre_scale:
                # GradScaler detected inf/nan internally and skipped this step.
                # A handful of these right at the very start of training (while
                # the scaler calibrates) is normal. If this keeps firing every
                # batch past the first ~10-20 steps, something upstream is
                # producing NaNs and needs investigating.
                skipped_steps += 1

            running_loss += loss.item() * lr_img.size(0)
            running_n += lr_img.size(0)

        if skipped_steps > 0:
            print(f"  [warning] GradScaler skipped {skipped_steps} step(s) this epoch "
                  f"due to inf/nan gradients.")

        scheduler.step()
        train_loss = running_loss / max(running_n, 1)
        val_mse = evaluate_epoch(model, val_loader, device)
        val_psnr = psnr_from_mse(val_mse)
        dt = time.time() - t0

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.4f} "
              f"| val_mse={val_mse:.6f} | val_psnr={val_psnr:.2f}dB "
              f"| lr={scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s")

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "epoch": epoch + 1, "train_loss": train_loss,
                "val_mse": val_mse, "val_psnr": val_psnr,
                "lr": scheduler.get_last_lr()[0], "time_sec": dt,
            }) + "\n")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_mse": best_val_mse,
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "last.pt")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            ckpt["best_val_mse"] = best_val_mse
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  -> New best model saved (val_mse={val_mse:.6f})")

    print("Training complete.")
    print(f"Best val MSE: {best_val_mse:.6f}")
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main()