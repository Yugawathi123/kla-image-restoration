"""
losses.py

Loss design rationale (tied directly to the problem statement):

  - Charbonnier loss (smooth L1) instead of plain MSE: MSE over-penalizes the
    occasional wild outlier pixel that speckle noise produces, which pushes
    the network toward blurring to minimize squared error on those outliers.
    Charbonnier is more robust to those outliers while still being
    differentiable everywhere (unlike plain L1 at 0).

  - SSIM loss: directly optimizes structural similarity, which is one of
    the three metrics KLA scores on (PSNR / SSIM / LPIPS), and encourages
    preserving real structure over just matching raw pixel intensities.

  - No adversarial or perceptual (VGG) loss: these encourage
    "plausible-looking" texture that may not be faithful to the real
    underlying structure -- risky for a defect-inspection pipeline where a
    hallucinated detail could hide or fabricate a defect. Skipped by design.

  - Sobel edge loss: Charbonnier + SSIM alone tend to over-smooth fine,
    ambiguous high-frequency detail (observed directly on real validation
    images -- hair-like edges were blurred into a soft halo instead of
    resolved). This term compares Sobel gradient maps of prediction vs
    target, directly rewarding sharp, correctly-placed edges instead of
    letting the network "hedge" by averaging detail away.

Combined loss = charbonnier + ssim_weight * (1 - SSIM) + edge_weight * sobel_edge_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(0) * g.unsqueeze(1)
    return window_2d.unsqueeze(0).unsqueeze(0)  # (1,1,W,W)


class SSIMLoss(nn.Module):
    """Differentiable SSIM, single-channel, computed on images assumed to be
    normalized to [0, 1] (data_range=1.0)."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self._window_cache = {}

    def _get_window(self, channels, device, dtype):
        key = (channels, device, dtype)
        if key not in self._window_cache:
            w = _gaussian_window(self.window_size, self.sigma, device, dtype)
            w = w.expand(channels, 1, self.window_size, self.window_size).contiguous()
            self._window_cache[key] = w
        return self._window_cache[key]

    def forward(self, pred, target, data_range: float = 1.0):
        # Force fp32 for SSIM regardless of caller's autocast context.
        # The variance terms below (E[x^2] - E[x]^2) involve subtracting two
        # close numbers, which is prone to precision-cancellation noise in
        # fp16 -- occasionally producing spurious negative variances and
        # propagating NaNs into the gradient. SSIM is computed on a handful
        # of conv2d calls, not the network's expensive part, so fp32 here
        # costs essentially nothing.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred = pred.float()
            target = target.float()

            channels = pred.shape[1]
            window = self._get_window(channels, pred.device, pred.dtype)
            pad = self.window_size // 2

            mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
            mu2 = F.conv2d(target, window, padding=pad, groups=channels)

            mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

            sigma1_sq = (F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq).clamp_min(0)
            sigma2_sq = (F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq).clamp_min(0)
            sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu1_mu2

        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        ssim_val = ssim_map.mean()
        return 1.0 - ssim_val  # loss: minimize (1 - SSIM)


class SobelEdgeLoss(nn.Module):
    """Penalizes the difference between the predicted and target image's
    EDGE MAPS (via fixed Sobel gradient filters), rather than raw pixel
    values. Charbonnier + SSIM both reward getting the average right; on
    fine, ambiguous high-frequency detail (hair-like edges, sharp texture),
    the "safe" way to minimize those losses is to blur the detail into an
    average rather than commit to a sharp (possibly slightly-off) edge --
    which is exactly the over-smoothing behavior visible on real validation
    images. This loss directly rewards matching edge magnitude, giving the
    network a reason to keep gradients sharp instead of averaging them away.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred, target):
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred = pred.float()
            target = target.float()

            gx_pred = F.conv2d(pred, self.sobel_x, padding=1)
            gy_pred = F.conv2d(pred, self.sobel_y, padding=1)
            gx_tgt = F.conv2d(target, self.sobel_x, padding=1)
            gy_tgt = F.conv2d(target, self.sobel_y, padding=1)

            loss = torch.mean(torch.abs(gx_pred - gx_tgt)) + torch.mean(torch.abs(gy_pred - gy_tgt))
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, ssim_weight: float = 0.3, edge_weight: float = 0.15,
                 charbonnier_eps: float = 1e-3):
        super().__init__()
        self.charbonnier = CharbonnierLoss(eps=charbonnier_eps)
        self.ssim = SSIMLoss()
        self.edge = SobelEdgeLoss()
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight

    def forward(self, pred, target, data_range: float = 1.0):
        l_pix = self.charbonnier(pred, target)
        l_ssim = self.ssim(pred, target, data_range=data_range)
        l_edge = self.edge(pred, target)
        total = l_pix + self.ssim_weight * l_ssim + self.edge_weight * l_edge
        return total, {
            "charbonnier": l_pix.item(),
            "ssim_loss": l_ssim.item(),
            "edge_loss": l_edge.item(),
        }


if __name__ == "__main__":
    pred = torch.rand(2, 1, 64, 64)
    target = torch.rand(2, 1, 64, 64)
    loss_fn = CombinedLoss()
    total, parts = loss_fn(pred, target)
    print("Total loss:", total.item(), "| parts:", parts)

    # identical images -> ssim loss should be ~0
    same_loss, same_parts = loss_fn(pred, pred)
    print("Identical-image loss (sanity check, should be near 0):", same_loss.item(), same_parts)