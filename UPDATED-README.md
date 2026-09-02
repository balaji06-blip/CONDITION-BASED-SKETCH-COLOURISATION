# Condition-Based Sketch Colourisation

  Balaji Periyadurai 

**MSc Dissertation Project — University of Surrey**  
**Programme:** MSc Computer Vision, Robotics and Machine Learning  
**Supervision:** Prof. Yi-Zhe Song and Chaitat Utintu  
**Year:** 2026

## Overview

This project investigates controllable anime sketch colourisation using sparse colour guidance.

Given an RGB sketch, a sparse RGB colour-hint map, and a binary hint mask, the system predicts a full RGB colour image while preserving sketch structure and following the supplied colour guidance.

Three systems were implemented and evaluated under a common experimental setting:

1. **Condition-based Pix2Pix cGAN baseline**
2. **conditional diffusion model**
3. **Hybrid Pix2Pix–Diffusion fusion model**

All systems use the same paired dataset, **256 × 256** image resolution, a **5% sparse colour-hint budget**, and the same held-out test partition.

## Project Objectives

1. Build a reproducible sparse-hint anime sketch-colourisation pipeline.
2. Implement and evaluate a strong Pix2Pix cGAN baseline.
3. Design and train a conditional diffusion model with multi-scale conditioning.
4. Compare reconstruction quality, perceptual quality, and inference speed.
5. Analyse successes, failure cases, limitations, and future GAN–diffusion fusion strategies.

## Dataset

The project uses the **Anime Sketch Colorization Pair** dataset from Kaggle:

https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair

The paired images contain:

```text
[ COLOUR IMAGE | SKETCH ]
       left        right
```

| Partition | Images |
|---|---:|
| Training | 14,224 |
| Validation | 1,772 |
| Test | 1,773 |

The supplied evaluation set is shuffled using **seed 42** and split into validation and test sets.

> The dataset itself is not included in this repository. Download it from Kaggle and place it in the expected local data directory.

## Sparse Colour Guidance

```text
Sketch RGB       = 3 channels
Colour hints RGB = 3 channels
Hint mask        = 1 channel
--------------------------------
Total            = 7 channels
```

The default hint budget is **5% of image pixels**.

## Model 1 — Pix2Pix cGAN Baseline

- 8-level U-Net generator
- 7-channel conditional input
- PatchGAN discriminator
- adversarial BCE + L1 reconstruction
- reconstruction coefficient: λ = 100
- 100 training epochs
- batch size 8
- single-pass inference

## Model 2 — Conditional Diffusion

The proposed model is a custom **Palette-inspired conditional diffusion model** trained from scratch.

Main components:

- 7-channel sketch + colour-hint condition
- noisy RGB concatenated with condition → 10 input channels
- conditional U-Net
- multi-scale condition encoder
- sinusoidal timestep embedding + MLP
- FiLM-style timestep modulation
- Group Normalisation
- self-attention at low resolutions
- cosine diffusion schedule
- Min-SNR weighting
- EMA model
- deterministic DDIM sampling

### Diffusion configuration

| Setting | Value |
|---|---:|
| Image size | 256 × 256 |
| Training diffusion timesteps | 1,000 |
| Training epochs | 100 |
| Hint rate | 5% |
| Physical batch | 4 |
| Gradient accumulation | 2 |
| Effective batch | 8 |
| Base channels | 64 |
| Attention heads | 4 |
| DDIM test steps | 100 |
| DDIM eta | 0 |
| EMA decay | 0.9999 |

### Training objective

```text
L_total = L_noise + 0.10 L_x0 + 0.50 L_hint
```

Hard hint projection is applied during diffusion sampling.

## Model 3 — Hybrid Pix2Pix–Diffusion Fusion

The trained Pix2Pix generator is frozen and used to provide a coarse RGB colour prior to the diffusion model.

```text
Sketch + sparse hints
        |
        v
  Frozen Pix2Pix
        |
        v
Coarse RGB prediction
        |
        v
Sketch + hints + mask + Pix2Pix prediction
        |
        v
Conditional diffusion refiner
        |
        v
Final colour image
```

The fusion condition has **10 channels**, and together with the noisy RGB image the diffusion U-Net receives **13 channels**.

## Final Results

| Metric | Pix2Pix cGAN | Conditional Diffusion | Hybrid Fusion |
|---|---:|---:|---:|
| **PSNR (dB) ↑** | 24.2793 ± 2.0089 | **26.2284 ± 2.5634** | 26.0240 ± 2.4550 |
| **SSIM ↑** | 0.8783 ± 0.0450 | **0.8958 ± 0.0711** | 0.8733 ± 0.0836 |
| **LPIPS ↓** | 0.0993 ± 0.0325 | **0.0675 ± 0.0576** | 0.0763 ± 0.0663 |
| **Inference time (ms/image) ↓** | **1.0922** | 2300.82 | 1934.28 |

### Main finding

- **Conditional diffusion** achieved the best reconstruction and perceptual-quality metrics.
- **Pix2Pix** was by far the fastest and is most suitable for real-time interaction.
- **Hybrid fusion** retained much of the diffusion PSNR/LPIPS advantage and was about **15.9% faster than standalone diffusion**, but its SSIM was lower.

## Evaluation Metrics

- **PSNR** — higher is better.
- **SSIM** — higher is better.
- **LPIPS** — lower is better.
- **Inference time** — lower is better.

## Reproducibility

Important controls include:

- fixed random seed: **42**
- same held-out test partition for all models
- same image resolution
- same 5% hint budget
- deterministic validation/test hint generation
- saved checkpoints
- CSV training logs
- per-image test metrics
- JSON/text result summaries
- generated and ground-truth outputs stored with corresponding filenames

## Recommended Repository Structure

```text
CONDITION-BASED-SKETCH-COLOURISATION/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── baseline_model/
│   └── Baseline-model.py
│
├── diffusion_model/
│   └── Diffusion-model.py
│
├── fusion_model/
│   └── Fusion-model.py
│
├── results/
│   ├── baseline/
│   ├── diffusion/
│   └── fusion/
│
├── figures/
│   ├── training/
│   ├── qualitative/
│   └── architecture/
│
└── docs/
    └── dissertation.pdf   # optional
```

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies:

- PyTorch
- torchvision
- NumPy
- Pillow
- matplotlib
- tqdm
- scikit-image
- LPIPS

## Data Setup

A typical local layout is:

```text
archive/
└── data/
    ├── train/
    └── val/
```

Do **not** commit the full dataset to GitHub.

## Running the Experiments

```bash
python baseline_model/Baseline-model.py
python diffusion_model/Diffusion-model.py
python fusion_model/Fusion-model.py
```

Check the dataset and output paths in each script before running.

## Important Limitations

1. Colour hints are synthetically sampled from ground-truth images.
2. The comparison is system-level rather than perfectly component-matched.
3. Hard hint projection can improve paired diffusion metrics at hinted pixels.
4. Diffusion uses a smaller fixed validation subset because sampling is expensive.
5. Evaluation is limited to one anime dataset and 256 × 256 resolution.
6. Diffusion inference is far slower than Pix2Pix.
7. Formal statistical significance testing was not included.

## Future Work

- confidence-guided GAN–diffusion fusion
- residual or region-selective refinement
- fewer DDIM sampling steps
- semantic or brush-like colour hints
- reference-image conditioning
- higher-resolution and cross-dataset evaluation
- user studies
- statistical significance testing

## Project Contribution

The project contribution is the **controlled implementation and evaluation of three working colourisation systems** under the same sparse-guidance setting, including a seven-channel conditioning formulation, a GPU-feasible conditional diffusion system, deterministic evaluation, and an implemented GAN–diffusion fusion experiment.

## Hardware

Main experiments were designed for Linux using an **NVIDIA RTX A4000 16 GB GPU** with mixed-precision training when CUDA is available.

## Citation

```text
Balaji Periyadurai,
"Condition-Based Sketch Colourisation,"
MSc Dissertation,
University of Surrey,
2026.
```

## Author

**Balaji Periyadurai**  
MSc Computer Vision, Robotics and Machine Learning  
University of Surrey
