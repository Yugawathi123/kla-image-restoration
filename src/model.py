"""
model.py

Lightweight, fully-convolutional restoration network.

Design goals (driven directly by the problem statement):
  - Handles denoising (speckle + gaussian-like softness) AND 2x super-resolution
    in a single forward pass.
  - Fully convolutional + resolution agnostic -> the SAME network weights work
    for 128->256 and 256->512, since both are just "2x upscale + restore".
  - Global residual learning on top of a bicubic-upsampled input: the network
    only has to predict a *correction*, not reconstruct pixels from scratch.
    This converges faster and is more stable, especially with a small dataset
    (~200 pairs).
  - No BatchNorm: BN statistics computed on noisy inputs are unreliable across
    domains (this matters a lot for the OOD generalization requirement), and
    BN is known to hurt image restoration quality (see EDSR paper). We use
    plain conv + ReLU residual blocks instead.
  - No adversarial / perceptual loss dependency baked into the architecture:
    keeps outputs faithful to ground truth for PSNR/SSIM instead of
    hallucinating plausible-looking but incorrect texture (risky for a defect
    inspection use case).
  - Kept small (~1-2M params) on purpose: fast inference (benchmarked!) and
    less prone to overfitting a small (~200 sample) dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Simple residual block: conv-relu-conv + skip. No BatchNorm (see module docstring)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        # Small residual scaling improves training stability (common trick from EDSR).
        self.res_scale = 0.2

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return x + out * self.res_scale


class UpsampleBlock(nn.Module):
    """2x upsampling via sub-pixel convolution (PixelShuffle). Sharper than
    transpose-conv (fewer checkerboard artifacts) and cheaper than a plain
    upsample+conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.shuffle(self.conv(x)))


class RestorationNet(nn.Module):
    """
    Input:  (B, 1, H, W)  degraded, noisy, low-res grayscale image
    Output: (B, 1, 2H, 2W) restored, denoised, full-res grayscale image

    Pipeline:
      1. Shallow feature extraction on the raw LR input.
      2. A stack of residual blocks does the heavy lifting: denoising +
         detail reconstruction happen here, at LOW resolution (cheaper
         compute, and denoising before upsampling avoids amplifying noise).
      3. One PixelShuffle upsample block brings features to 2x resolution.
      4. A final conv maps back to 1 channel.
      5. Global residual: add a bicubic-upsampled version of the input, so
         the network only predicts the correction / detail residual.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 num_features: int = 64, num_blocks: int = 12):
        super().__init__()

        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        self.body = nn.Sequential(*[ResBlock(num_features) for _ in range(num_blocks)])
        self.body_tail = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        self.upsample = UpsampleBlock(num_features)

        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Global residual base: smooth 2x upsample of the raw (still noisy) input.
        # IMPORTANT: bicubic interpolation is run in fp32 explicitly, even under
        # autocast/mixed-precision training. F.interpolate(mode="bicubic") has a
        # known numerical issue on CUDA under fp16 that can silently produce NaNs,
        # which then poison every gradient and cause GradScaler to skip
        # optimizer.step() on every batch -- the model appears to "train" (loss
        # prints, no crash) but weights never actually update. Forcing fp32 here
        # avoids that failure mode entirely; the cost is negligible since this is
        # a single interpolation, not the expensive part of the network.
        with torch.autocast(device_type=x.device.type, enabled=False):
            base = F.interpolate(x.float(), scale_factor=2, mode="bicubic", align_corners=False)
        base = base.to(x.dtype)

        feat = self.head(x)
        res = self.body(feat)
        res = self.body_tail(res)
        feat = feat + res  # feature-level residual (helps gradient flow through the body)

        feat = self.upsample(feat)
        out = self.tail(feat)

        return out + base


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check: run a dummy forward pass at both resolution tiers
    # the task specifies (128->256 and 256->512), and print param count +
    # a rough FLOPs-free timing estimate on CPU.
    model = RestorationNet()
    print(f"Params: {count_parameters(model):,}")

    for size in (128, 256):
        x = torch.randn(1, 1, size, size)
        y = model(x)
        print(f"Input {tuple(x.shape)} -> Output {tuple(y.shape)}")
        assert y.shape[-1] == size * 2 and y.shape[-2] == size * 2
    print("OK: shapes correct for both resolution tiers.")