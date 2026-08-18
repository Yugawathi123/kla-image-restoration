# Model Development Report — KLA Image Restoration

This documents the full development journey: every architecture change, bug found and
fixed, and the measured result at each step. Companion to `README.md` (which covers
setup/usage) — this file is the "how we got here and why" record.

---

## 1. Task recap

Single model that takes a degraded semiconductor inspection image (speckle noise +
Gaussian-like softness + 2x downsampling) and restores it: denoised, sharpened, and
upscaled back to full resolution — in one forward pass, fast enough for real-time
inspection use.

**Dataset actually provided:** 3,200 paired samples, `.npy`, float32, single resolution
tier — GT `256×256` (normalized to exactly `[0, 1]`) / NoisyLR `128×128` (legitimately
exceeds that range due to speckle overshoot — observed global range `[-0.10, 1.71]`).
Test set: 400 NoisyLR-only images (blind, no ground truth).

**Hardware:** single laptop GPU, NVIDIA RTX 3050 Laptop (6GB VRAM), Windows, conda env.

---

## 2. Base architecture (`RestorationNet`, all versions)

Fully convolutional, resolution-agnostic network — same weights handle any input size,
always upscaling 2x. Structure:

```
Input (1×H×W)
  → head conv (1 → num_features channels)
  → N residual blocks (the "body" — denoising + detail work happens here, at LOW
    resolution, which is cheaper and avoids amplifying noise before upscaling)
  → body_tail conv
  → feature-level residual add (head output + body output)
  → PixelShuffle upsample block (2x, sub-pixel convolution — sharper than
    transpose-conv, fewer checkerboard artifacts)
  → tail conv (features → 1 channel)
  → + bicubic-upsampled copy of the raw input (global residual — the network only
    has to predict a CORRECTION on top of a smooth upsample, not reconstruct the
    image from scratch; converges faster and more stably on a modest dataset)
Output (1×2H×2W)
```

**Deliberately excluded, and why:**
- **No BatchNorm** — BN statistics learned on noisy training images generalize poorly
  to out-of-distribution data, and are known to hurt restoration quality (EDSR paper).
  Plain conv + activation residual blocks used instead.
- **No adversarial (GAN) or perceptual loss** — these produce sharper-looking output by
  learning to hallucinate plausible texture, which is directly risky for a defect
  inspection tool: a fabricated "sharp edge" could mask or fake a real defect. The
  problem statement explicitly warns against exactly this ("without introducing
  artificial patterns"). Every version below stays clear of this.
- **No diffusion / transformer backbone** — multi-step or attention-heavy inference
  would risk the speed benchmark, and neither is realistically trainable from scratch
  on ~2,880 training images in the time available.

The block used *inside* the body has changed across versions — that's the main lever
we experimented with (see §4).

---

## 3. Bugs found and fixed along the way

These weren't quality tweaks — they were correctness bugs that silently produced wrong
results before being caught.

### 3.1 — Mixed precision (AMP) silently skipping every optimizer step
**Symptom:** `train_loss` frozen at the same value for 20+ epochs; a
`lr_scheduler.step() before optimizer.step()` warning on epoch 1.
**Root cause:** the SSIM loss computes variance as `E[X²] − E[X]²`, which is
numerically fragile under fp16. Once GradScaler's loss-scaling factor was applied, this
occasionally overflowed to `inf`, and GradScaler's safety mechanism silently skipped the
weight update on nearly every batch — the model never actually learned anything.
**Fix:** run the (expensive) conv forward pass in fp16 as intended, but cast to fp32
specifically for the loss computation. Same fix applied to the bicubic base upsample
inside the model, which has a known separate fp16/NaN issue on CUDA.

### 3.2 — Validation metric was noisy/unreliable
**Symptom:** per-epoch `val_psnr` bounced around instead of improving smoothly.
**Root cause:** the validation subset was created via `torch.utils.data.random_split`
on the *same* `Dataset` object used for training — meaning validation images were being
randomly cropped and flip/rotate-augmented exactly like training data, every epoch.
**Fix:** `dataset.py::split_pairs()` now does a clean, deterministic train/val split
up front; validation uses a separate `Dataset` instance with `patch_size=None` (full
image, no crop) and `augment=False`. Validation numbers are now stable and trustworthy.

### 3.3 — Real images displaying as solid black
**Symptom:** an uploaded comparison PNG of a real (non-synthetic) validation image was
almost entirely black, looking broken.
**Root cause:** real inspection images can be naturally low-contrast (observed mean
~0.27, std ~0.06) — displaying them on a scale fixed to the dataset's global `[0,1]`
range crushes everything toward black even though the underlying data and model output
were both fine.
**Fix:** added `to_uint8_png_autocontrast()` — stretches each panel's *own* 1st–99th
percentile range to fill the visible range, for display only. Never touches saved
`.npy` data or any PSNR/SSIM computation.

### 3.4 — CUDA/CPU device mismatch when the edge loss was added
**Symptom:** `RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
(torch.FloatTensor) should be the same`.
**Root cause:** the new Sobel edge loss stores its filter kernels as registered buffers,
which default to CPU and don't move to GPU automatically.
**Fix:** `loss_fn = CombinedLoss(...).to(device)` in `train.py`.

---

## 4. Version-by-version changes and results

All PSNR/SSIM figures below are from `generate_report.py` on the same held-out
validation split (320 images, never trained on) — apples-to-apples across versions.
Bicubic baseline (no learning at all) on this split: **~22.9 dB PSNR / ~0.55 SSIM.**

| Version | Block type | Params | Edge loss | Val PSNR | Val SSIM | Inference speed | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| v1 | Plain ResBlock | 1.07M | — | 28.17 dB | 0.7682 | ~10 ms/img | First working model. Fixed the AMP bug (§3.1) to get here. |
| v2 | Plain ResBlock | 1.07M | — | 28.46 dB | 0.7784 | ~10 ms/img | Same architecture; fixed validation bug (§3.2). Near-identical result to v1 confirms the model itself is stable — the fix just gave a cleaner ruler. |
| v3 | Plain ResBlock | 1.07M | 0.15 | 28.53 dB | 0.7787 | ~10 ms/img | Added Sobel edge loss. Small, real, edge-concentrated improvement (confirmed via pixel-diff heatmap) but modest in magnitude. |
| v4 | Plain ResBlock (wider/deeper) | 3.07M (2.87x) | 0.35 | **28.91 dB** | **0.7870** | 17.3 ms/img | Scaled capacity + stronger edge weight together. Clearly the biggest single jump so far (+0.38 dB vs v3, ~5x the v2→v3 gain), confirmed via visual diff. |
| v5 | RCAB (channel attention) | 1.08M (+0.65%) | — | *not trained* | — | — | Implemented and tested (see `model.py`), not run — RRDB chosen instead as the next experiment. |
| v6 | RRDB (dense blocks) | 4.50M | 0.35 | *in progress* | — | *TBD* | Genuinely different design (ESRGAN backbone, no adversarial training) rather than a scaled variant. Training restarted at 100 epochs (from 150) after per-epoch time came in at ~160s instead of the ~90-120s estimated — full 100-epoch cosine schedule completes properly rather than truncating a longer one. |

**Test-time augmentation (flip ensembling, `tta.py`):** evaluated at every version from
v3 onward. Consistently gives a very small gain (+0.03 to +0.04 dB PSNR) at ~4x the
inference cost. **Decision: not used for final submission** — given the stated priority
on both speed and accuracy together, the cost doesn't justify the gain here.

---

## 5. Loss function

`losses.py` — `CombinedLoss = Charbonnier + ssim_weight × (1 − SSIM) + edge_weight × SobelEdgeLoss`

- **Charbonnier** (smooth L1): robust to the occasional wild outlier pixel from speckle
  noise, unlike MSE which over-penalizes outliers and pushes the network toward blur.
- **SSIM**: directly optimizes structural similarity — one of the metrics actually
  being scored on, and rewards preserving structure over just matching raw intensities.
- **Sobel edge loss** (added v3 onward): compares gradient maps of prediction vs.
  ground truth. Added specifically because v1/v2 output was visibly over-smoothing fine
  high-frequency detail (a spiky/hair-like edge texture was blurred into a soft halo) —
  confirmed via a real uploaded validation image, not just guessed at. Directly targets
  the problem statement's explicit requirement to restore edge sharpness without
  introducing artificial patterns.

---

## 6. Architecture variants implemented (`model.py`)

Three interchangeable block types, selected via CLI flag, same surrounding network:

- **`ResBlock`** (default): conv → ReLU → conv, residual add, scaled by 0.2 (EDSR-style
  stability trick).
- **`RCAB`** (`--use_attention`): same shape as `ResBlock`, plus a squeeze-and-excitation
  channel-attention gate before the residual add — lets the network learn which feature
  channels matter most for the current input. Adds <1% more parameters; a proven,
  cheap quality lever from super-resolution research (RCAN).
- **`RRDB`** (`--use_rrdb`): three `ResidualDenseBlock`s per RRDB, each with 5 conv
  layers where every layer sees the concatenation of all earlier layers' outputs within
  that block (dense feature reuse) — the actual ESRGAN backbone, used here purely as a
  feature extractor with our existing losses, deliberately without any GAN training.
  Much more compute per block (~15 conv layers vs. 2), so far fewer blocks are needed
  for a comparable parameter budget.

All three are reconstructed correctly from a saved checkpoint automatically —
`evaluate.py`, `generate_report.py`, and `predict_single.py` all read the architecture
type back out of the checkpoint's saved training args, so the right model is always
rebuilt regardless of which variant was trained.

---

## 7. Tooling built alongside the model

| Script | Purpose |
|---|---|
| `data_audit.py` | Inspect raw `.npy` shapes/dtypes/ranges/pairing before trusting any of it. |
| `calibrate_stats.py` | Computes normalization from GT training data only → `stats.json`. |
| `dataset.py` | Paired loader; patch-crop + augment for training, clean full-image mode for validation. |
| `model.py` | `RestorationNet` — all three block variants, see §6. |
| `losses.py` | `CombinedLoss` — Charbonnier + SSIM + Sobel edge, see §5. |
| `tta.py` | Test-time flip ensembling helper, used optionally by the scripts below. |
| `train.py` | Training loop — AMP, cosine LR schedule, checkpointing, all architecture/loss options exposed as CLI flags. |
| `evaluate.py` | **Standalone benchmarking script** — runs inference on a folder of test images, reports timing, optional `--tta`. This is what KLA's benchmarking script uses as-is. |
| `generate_report.py` | Held-out validation comparison: bicubic baseline vs. model vs. model+TTA, PSNR/SSIM, saved comparison images. |
| `predict_single.py` | Single-image inference + viewable PNG comparison, for spot-checks. |

---

## 8. Current status / next steps

- v6 (RRDB) training restarted with `--epochs 100` after the timing correction; result
  pending.
- Once v6 finishes: run `generate_report.py` and `evaluate.py` on it exactly as done for
  v1–v4, compare against v4 (current best: 28.91 dB / 0.7870 / 17.3 ms), and do the same
  visual spiky-edge check used to judge every previous version honestly rather than by
  the aggregate metric alone.
- Final decision on which checkpoint to submit will be based on: PSNR/SSIM on held-out
  validation, a real visual check for artifacts/over-smoothing, and inference speed —
  the same three criteria used throughout, matching the problem statement's own stated
  priorities (accuracy + speed, no hallucinated detail).
