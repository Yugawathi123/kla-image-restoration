"""
tta.py

Test-time augmentation via flip ensembling.

Idea: run the same input through the model 4 times -- as-is, horizontally
flipped, vertically flipped, and both -- undo each flip on the output, and
average the 4 results. Since the model has no reason to treat "flipped"
differently from "not flipped" (noise and structure look the same either
way), averaging predictions from slightly different views of the same input
smooths out some of the model's individual prediction noise, typically
worth a small but consistent PSNR/SSIM gain.

Cost: exactly 4x the compute of a single forward pass. On this model
(~1M params, ~10ms/image on an RTX 3050 laptop), that's still only ~35-45ms
per image -- decide with real numbers whether that trade-off is worth it
for your speed budget by comparing evaluate.py's reported avg_inference_time
with and without --tta.
"""

import torch


@torch.no_grad()
def predict_with_tta(model, lr_tensor: torch.Tensor) -> torch.Tensor:
    """
    lr_tensor: (B, 1, H, W) already normalized, on the correct device.
    Returns: (B, 1, 2H, 2W) averaged prediction.
    """
    preds = []

    # 1. original
    preds.append(model(lr_tensor))

    # 2. horizontal flip
    flipped = torch.flip(lr_tensor, dims=[3])
    out = model(flipped)
    preds.append(torch.flip(out, dims=[3]))

    # 3. vertical flip
    flipped = torch.flip(lr_tensor, dims=[2])
    out = model(flipped)
    preds.append(torch.flip(out, dims=[2]))

    # 4. both (180 degree rotation)
    flipped = torch.flip(lr_tensor, dims=[2, 3])
    out = model(flipped)
    preds.append(torch.flip(out, dims=[2, 3]))

    return torch.stack(preds, dim=0).mean(dim=0)
