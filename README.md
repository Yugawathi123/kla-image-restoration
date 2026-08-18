# kla-image-restoration
Deep learning based image restoration for noisy low-resolution semiconductor inspection images.
KLA Image Restoration
Deep Learning-Based Restoration of Noisy Low-Resolution Semiconductor Inspection Images

A deep learning image restoration system designed to recover high-quality images from noisy and low-resolution semiconductor inspection images.

The project focuses on improving image quality while preserving fine structural details that are important for semiconductor inspection and defect analysis.

🔬 Problem Statement

Semiconductor manufacturing relies on microscopic inspection images to identify defects and verify chip quality.

However, inspection images can suffer from:

Speckle/noise contamination
Low spatial resolution
Loss of fine structural details
Reduced image contrast and clarity

These degradations can make small defects difficult to identify and may affect downstream inspection and analysis.

The objective of this project is to develop a deep learning-based restoration pipeline that converts degraded low-resolution images into cleaner and higher-quality images while preserving important structural details.

🎯 Project Objectives
Remove noise from degraded semiconductor inspection images.
Restore lost image details.
Improve image reconstruction quality.
Preserve structural information during restoration.
Evaluate restoration using PSNR and SSIM.
Provide reproducible training and inference scripts.
Generate restored images for visual inspection and comparison.
🧠 Approach

The restoration pipeline follows:

Degraded / Noisy Low-Resolution Image
                │
                ▼
        Data Preprocessing
                │
                ▼
      Deep Learning Restoration
                │
                ▼
       Restored High-Quality Image
                │
        ┌───────┴────────┐
        ▼                ▼
   Quantitative       Visual
    Evaluation       Comparison
        │                │
     PSNR/SSIM        PNG Outputs

The model is trained using paired:

NoisyLR  →  Ground Truth

images stored in .npy format.

📊 V4 Model Performance

The final V4 model was evaluated on a held-out validation set of 320 image pairs that were never used during training.

Method	PSNR (dB)	SSIM
Bicubic Baseline	22.93	0.5508
V4 Model	28.91	0.7870
V4 + TTA	28.94	0.7879
Improvement over Bicubic
PSNR improvement : +5.98 dB
SSIM improvement : +0.2362

The model therefore provides a substantial improvement over conventional bicubic upscaling.

Test-set inference

The trained model was also processed on 400 test images.

Number of test images : 400
Average inference time: 17.32 ms/image
Total inference time  : 6.93 s
TTA                    : OFF
📈 Evaluation Metrics
PSNR — Peak Signal-to-Noise Ratio

PSNR measures the similarity between the restored image and the ground-truth image based on pixel-level reconstruction error.

Higher PSNR is generally better.

Higher PSNR → Lower reconstruction error

The V4 model achieved:

28.91 dB PSNR

compared with:

22.93 dB for bicubic interpolation.

SSIM — Structural Similarity Index

SSIM measures how structurally similar the restored image is to the ground-truth image.

It considers characteristics such as:

Luminance
Contrast
Structural information

SSIM generally ranges from 0 to 1, with values closer to 1 indicating greater structural similarity.

The V4 model achieved:

0.7870 SSIM

compared with:

0.5508 SSIM for bicubic interpolation.

🔄 Test-Time Augmentation

The pipeline also supports Test-Time Augmentation (TTA) using flip-based ensembling.

V4 results:

Plain Model:
PSNR = 28.91 dB
SSIM = 0.7870


Model + TTA:
PSNR = 28.94 dB
SSIM = 0.7879

TTA provides a small improvement:

PSNR : +0.03 dB
SSIM : +0.0009

at approximately 4× the inference cost.

Therefore, the plain model is preferable when inference speed is important.
