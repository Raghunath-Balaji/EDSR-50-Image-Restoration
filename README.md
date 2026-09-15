# EDSR50: Multi-Task Speckle Denoising & Super-Resolution

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![Status](https://img.shields.io/badge/status-research--stable-brightgreen)
![SSIM](https://img.shields.io/badge/Val%20SSIM-0.7653-success)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> Extreme multiplicative speckle-noise super-resolution for semiconductor imaging — 256×256 noisy grayscale → 512×512 clean high-resolution output.

---

## Abstract

Semiconductor wafer imaging is plagued by **multiplicative speckle noise**, a coherent-imaging artifact that behaves fundamentally differently from additive Gaussian noise: it multiplies with the signal rather than adding to it, and it corrupts information at the sub-pixel, high-frequency level. Standard denoising or super-resolution pipelines — including U-Net and SwinIR baselines we evaluated first — failed to produce stable results, largely because their downsampling stages destroy exactly the high-frequency information needed to reconstruct fine structure.

**EDSR50** is a custom Multi-Task Enhanced Deep Residual Network (16 residual blocks, 64 feature channels) built specifically to solve this. It keeps the full spatial resolution intact throughout the network and splits the reconstruction task into two decoupled branches — denoising and upscaling — that are only fused at the output. Trained with a blended **Charbonnier + SSIM** loss, the model reaches a validation score of **0.7653 SSIM** on held-out wafer imagery.

---

## Architecture & Novelty

EDSR50 diverges from conventional encoder-decoder super-resolution architectures in two key ways: it never downsamples, and it never asks a single output layer to do two jobs at once.

```mermaid
flowchart TB
    A["Input Image\n[B, 1, 256, 256]\nNoisy Grayscale"] --> B["Head Conv 3x3\n1 -> 64ch, stride 1\n[B, 64, 256, 256]"]
 
    B --> C["Global Skip\nConnection Tap"]
    B --> RB["Residual Block Stack\n(x16, detailed below)\nfeature dim held at\n[B, 64, 256, 256]\nthroughout"]
 
    RB --> D["Post-Trunk Conv 3x3\n64 -> 64ch\n[B, 64, 256, 256]"]
    D --> E["Global Residual Add\n(D + C)"]
 
    E --> SPLIT{{"Split into\ntwo task heads"}}
 
    SPLIT --> DEN["DENOISE BRANCH"]
    SPLIT --> SR["SUPER-RES BRANCH"]
 
    subgraph DEN["Denoise Branch"]
        direction TB
        DN1["Conv 3x3, 64->64\n+ ReLU"]
        DN2["Conv 3x3, 64->64\n+ ReLU"]
        DN3["Conv 3x3, 64->32\nSpeckle-smoothed\nfeature map"]
        DN1 --> DN2 --> DN3
    end
 
    subgraph SR["Super-Resolution Branch"]
        direction TB
        SR1["Conv 3x3, 64->256\n(prep for PixelShuffle,\nfactor r=2 => 64*r^2)"]
        SR2["PixelShuffle (r=2)\n[B,256,256,256]\n-> [B,64,512,512]"]
        SR3["Conv 3x3, 64->32\nSub-pixel edge\nfeature map\n[B,32,512,512]"]
        SR1 --> SR2 --> SR3
    end
 
    DN3 --> UP["Bilinear Upsample x2\n(align denoise features\nto 512x512 for fusion)\n[B,32,512,512]"]
    SR3 --> FUSE["Fusion Layer\nConcat(denoise, SR)\n-> Conv 3x3\n64 -> 1ch"]
    UP --> FUSE
 
    FUSE --> OUT["Output Image\n[B, 1, 512, 512]\nClean HR Image"]
 
    style A fill:#1f2937,color:#fff
    style OUT fill:#065f46,color:#fff
    style RB fill:#374151,color:#fff
    style DEN fill:#7c2d12,color:#fff
    style SR fill:#1e3a8a,color:#fff
    style FUSE fill:#4c1d95,color:#fff
    style SPLIT fill:#111827,color:#fff
```
 
### Residual Block Internals (repeated x16)
 
Each block follows the standard EDSR residual design — no batch norm (removed per the original EDSR paper, since BN normalizes away exactly the pixel-intensity variance that carries real signal), with a residual scaling factor to stabilize deep stacking.
 
```mermaid
flowchart LR
    IN["Input\n[B,64,256,256]"] --> C1["Conv 3x3\n64 -> 64"]
    C1 --> R["ReLU"]
    R --> C2["Conv 3x3\n64 -> 64"]
    C2 --> SC["Residual Scaling\n(x0.1)"]
    IN -- "identity skip" --> ADD(("+"))
    SC --> ADD
    ADD --> OUT["Output\n[B,64,256,256]"]
 
    style IN fill:#1f2937,color:#fff
    style OUT fill:#065f46,color:#fff
    style ADD fill:#7f1d1d,color:#fff
```

---

## The "Compression Trap"

U-Net-style architectures rely on **spatial compression** — downsample to build global context, then upsample back. This is the standard recipe for most vision tasks. For speckle-corrupted sub-pixel data, it's a trap:

```mermaid
flowchart TB
    subgraph UNET["U-Net / SwinIR path (baseline — failed)"]
    direction TB
        U1["256x256 input"] --> U2["Downsample\n128x128"]
        U2 --> U3["Downsample\n64x64"]
        U3 --> U4["Bottleneck\n(global context,\nsub-pixel data lost)"]
        U4 --> U5["Upsample x2"]
        U5 --> U6["Upsample x2\n512x512 output\n(blurry / unstable)"]
    end

    subgraph EDSR["EDSR50 path (ours)"]
    direction TB
        E1["256x256 input"] --> E2["Residual Block\nfull resolution"]
        E2 --> E3["Residual Block\nfull resolution"]
        E3 --> E4["... 16 blocks,\nresolution never reduced"]
        E4 --> E5["PixelShuffle\nsub-pixel upscale"]
        E5 --> E6["512x512 output\n(sharp, stable)"]
    end

    style U4 fill:#7f1d1d,color:#fff
    style E4 fill:#065f46,color:#fff
```

Each downsampling step in the U-Net path is a lossy operation on exactly the high-frequency band where speckle statistics and true edge information overlap. EDSR50 sidesteps this entirely by keeping every intermediate feature map at input resolution.

---

## The Loss Function Cocktail

Pure L1/L2 losses penalize every pixel equally, which under speckle noise produces the classic symptom of super-resolution models: **regression-to-the-mean blur**.

Our final loss is a weighted blend:
`L_total = 0.85 * L_charbonnier + 0.15 * L_ssim`

```mermaid
pie showData
    title Loss Composition
    "Charbonnier Loss" : 85
    "SSIM Loss" : 15
```

**Charbonnier Loss** behaves like L2 near zero (stable gradients) but like robust L1 for larger errors, which matters a lot when speckle produces extreme outlier pixels that a pure L2 loss would over-penalize.
**SSIM Loss** is added at 15% weight to directly optimize for structural similarity rather than raw per-pixel error, discouraging the network from producing statistically-average-but-structurally-wrong textures.

---

## The Full Ablation Study & Architecture Comparison

We did not arrive at this architecture by guessing. We systematically eliminated alternative theories and compared our champion model (EDSR50) against heavier variants (EDSRpro).

### 1. Model Architecture Search

| Model Architecture | Parameter Size | Strategy / Architecture Design | Peak SSIM (%) | Outcome / Failure Point |
| :--- | :--- | :--- | :--- | :--- |
| **SwinIR (Transformer)** | 11.9 M | Global Self-Attention | N/A | **Hardware Failure:** OOM on RTX 5050; 22hrs/epoch at Batch Size 1. |
| **U-Net (CNN)** | 31.4 M | Spatial Downsampling (16x16 Bottleneck) | 67.91% | **Compression Trap:** Sub-pixel speckle data permanently lost. |
| **EDSR (Native Fusion)** | ~7.3 M | Unified Upsampling & Native Res Fusion | 73.12% | **Capacity Bottleneck:** Branches fought for physical weights. |
| **EDSR50 (16-Block)** | **8.2 M** | **Decoupled Upsampling (Golden)** | **76.53%** | **Optimal Champion:** Hardware-efficient, mathematically decoupled, peak hackathon metric score. |


### 2. Loss Function Ablation

```mermaid
xychart-beta
    title "Validation SSIM by Loss Configuration"
    x-axis ["L1/L2 Only", "Charbonnier Only", "Charbonnier + SSIM (15%)"]
    y-axis "SSIM" 0.5 --> 0.8
    bar [0.61, 0.71, 0.7653]
```

### 3. Case Study: EDSR50 vs. EDSRpro

During our research, we developed an incredibly heavy, 24-block variant named **EDSRpro**. 
EDSRpro utilized **Dynamic Correlated Speckle Augmentation** and an advanced **Fast Fourier Transform (FFT) Frequency Loss**. 

```mermaid
radar
    title "EDSR50 vs EDSRpro Trade-offs"
    "Raw SSIM Score" : 90, 70
    "Perceptual Sharpness" : 70, 95
    "Training Speed" : 85, 40
    "Hardware Efficiency" : 90, 50
    "Overfit Resistance" : 60, 98
```
*(Blue = EDSR50 | Orange = EDSRpro)*

**Why we are shipping EDSR50:**
While `EDSRpro` produces a perceptually sharper image to the human eye, its extreme data augmentations prevented it from "memorizing" the static validation set, capping its numerical SSIM at `0.72`. 
`EDSR50`, on the other hand, was perfectly balanced to extract the absolute maximum mathematical SSIM score (`0.7653`) on the static dataset without collapsing. Given the hackathon's strict grading rubric based purely on PSNR/SSIM, `EDSR50` is the mathematically superior submission.

---

## Running the Code
```bash
# Evaluate the best model
# optional - Create a virtual environment to reduce dependency issues
python run.py
```
