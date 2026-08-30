# ================================================================
# EXPERIMENT 3: HYBRID PIX2PIX + CONDITIONAL DIFFUSION FUSION
# Frozen pretrained Pix2Pix coarse colouriser + conditional DDPM refinement
# Designed for Linux / RTX A4000 16 GB
#
# Dataset format (same as the Pix2Pix baseline):
#   archive/data/train/*.jpg|png   -> [ COLOUR | SKETCH ]
#   archive/data/val/*.jpg|png     -> [ COLOUR | SKETCH ]
#
# Main fair-comparison configuration:
#   IMAGE_SIZE       = 256
#   HINT_RATE        = 0.05   (same as baseline)
#   TRAIN_EPOCHS     = 100
#   DIFFUSION_STEPS  = 1000
#   TEST_DDIM_STEPS  = 100
#
# Stronger components:
# - multi-scale conditioning encoder
# - residual U-Net with time FiLM
# - self-attention at 32x32 and 16x16
# - cosine diffusion schedule
# - Min-SNR-gamma loss weighting
# - auxiliary x0 reconstruction + hint-consistency losses
# - EMA weights for validation/test
# - deterministic DDIM sampling
# - BF16-first mixed precision + FP32 loss/DDIM arithmetic
# - zero-initialised residual/attention outputs for stability
# - high-noise auxiliary-loss masking + finite-value guards
# - gradient accumulation
# - checkpoint/resume
# - PSNR / SSIM / optional LPIPS / inference timing
# ================================================================

import os
import csv
import math
import time
import json
import glob
import random
import copy
from contextlib import nullcontext

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision.transforms.functional as TF

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

try:
    import lpips
    LPIPS_AVAILABLE = True
except Exception:
    LPIPS_AVAILABLE = False


# ================================================================
# 1. PATHS
# ================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "archive", "data")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_SOURCE_DIR = os.path.join(DATA_ROOT, "val")

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "fusion_pix2pix_diffusion_100epochs")
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_ROOT, "logs")
SAMPLE_DIR = os.path.join(OUTPUT_ROOT, "samples")
FIGURE_DIR = os.path.join(OUTPUT_ROOT, "figures")
EVAL_DIR = os.path.join(OUTPUT_ROOT, "evaluation")
GENERATED_DIR = os.path.join(EVAL_DIR, "generated")
REAL_DIR = os.path.join(EVAL_DIR, "ground_truth")
COARSE_DIR = os.path.join(EVAL_DIR, "pix2pix_coarse")

for folder in [
    OUTPUT_ROOT,
    CHECKPOINT_DIR,
    LOG_DIR,
    SAMPLE_DIR,
    FIGURE_DIR,
    EVAL_DIR,
    GENERATED_DIR,
    REAL_DIR,
    COARSE_DIR,
]:
    os.makedirs(folder, exist_ok=True)


# ================================================================
# 2. CONFIGURATION
# ================================================================

IMAGE_SIZE = 256
HINT_RATE = 0.05              # same 5% sparse colour-hint setting

# Pretrained Experiment-1 Pix2Pix checkpoint.
# This model is FROZEN and used only to produce a coarse colour prediction.
BASELINE_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "baseline_100epochs_linux", "checkpoints", "best_model.pth"
)

TRAIN_EPOCHS = 100
DIFFUSION_STEPS = 1000        # training/noise schedule length

# DDIM sampling: quality vs speed
VAL_DDIM_STEPS = 30
SAMPLE_DDIM_STEPS = 50
TEST_DDIM_STEPS = 100
DDIM_ETA = 0.0                # deterministic sampling

# RTX A4000 16 GB safe starting point
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2          # effective batch size = 8
VAL_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
NUM_WORKERS = 4

BASE_CHANNELS = 64
DROPOUT = 0.10
ATTENTION_HEADS = 4

LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
BETA1 = 0.9
BETA2 = 0.999
WARMUP_STEPS = 2000
MAX_GRAD_NORM = 1.0

# Min-SNR gamma (epsilon prediction)
MIN_SNR_GAMMA = 5.0

# Extra reconstruction terms to help PSNR/SSIM + hint adherence
LAMBDA_X0_L1 = 0.05
LAMBDA_HINT = 0.25

# Auxiliary x0/hint losses are only used when the signal is not vanishingly small.
# Noise prediction still trains on ALL 1000 timesteps. This avoids unstable x0
# reconstruction near t=999 where alpha_bar is extremely close to zero.
AUX_MIN_SNR = 0.05

EMA_DECAY = 0.9999
EMA_WARMUP_STEPS = 1000

VAL_EVERY = 5
MONITOR_FIRST_N_EPOCHS = 10
VAL_MAX_IMAGES = 64           # validation is expensive for diffusion
SAVE_EVERY = 5
SAMPLE_EVERY = 5
NUM_SAMPLE_IMAGES = 4

SEED = 42
RUN_FINAL_TEST = True

# If CUDA OOM occurs, change only these two lines:
# BATCH_SIZE = 2
# GRAD_ACCUM_STEPS = 4


# ================================================================
# 3. DEVICE + REPRODUCIBILITY
# ================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
USE_BF16 = bool(
    USE_AMP
    and hasattr(torch.cuda, "is_bf16_supported")
    and torch.cuda.is_bf16_supported()
)
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
USE_GRAD_SCALER = bool(USE_AMP and AMP_DTYPE == torch.float16)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    # RTX A4000 (Ampere) benefits from TF32 for FP32 matmuls/convolutions.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def autocast_context():
    if USE_AMP:
        return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)
    return nullcontext()


def make_grad_scaler():
    # BF16 has FP32-like exponent range and normally does not need GradScaler.
    enabled = USE_GRAD_SCALER
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ================================================================
# 4. HELPERS
# ================================================================

def get_image_files(folder):
    files = []
    for extension in ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]:
        files.extend(glob.glob(os.path.join(folder, extension)))
    return sorted(list(set(files)))


def check_dataset():
    print("\nChecking dataset...")

    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Training folder not found:\n{TRAIN_DIR}")
    if not os.path.isdir(VAL_SOURCE_DIR):
        raise FileNotFoundError(f"Validation folder not found:\n{VAL_SOURCE_DIR}")

    train_files = get_image_files(TRAIN_DIR)
    val_files = get_image_files(VAL_SOURCE_DIR)

    if len(train_files) == 0:
        raise RuntimeError("No training images found.")
    if len(val_files) < 4:
        raise RuntimeError("Not enough images in val folder.")

    print("Training images   :", len(train_files))
    print("Validation source :", len(val_files))

    with Image.open(train_files[0]) as img:
        w, h = img.size
        print("Example size      :", img.size)
        if w != 2 * h:
            print("WARNING: expected paired [COLOUR | SKETCH] image, usually width = 2*height.")

    return train_files, val_files


def np_to_norm_tensor(arr):
    # uint8/float [0,255] -> float tensor [-1,1], CHW
    arr = arr.astype(np.float32) / 127.5 - 1.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr.copy()).float()


def tensor_to_uint8(tensor):
    x = tensor.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1.0) * 127.5).round().clamp(0, 255).byte()
    if x.ndim == 3:
        x = x.permute(1, 2, 0)
    elif x.ndim == 4:
        x = x.permute(0, 2, 3, 1)
    return x.numpy()


def resize_lanczos(image, size):
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    return image.resize((size, size), resample)


# ================================================================
# 5. DATASET
# ================================================================

class AnimeSketchColourDataset(Dataset):
    """
    Each file is [COLOUR | SKETCH].

    Returns:
      target         : 3xHxW colour image in [-1,1]
      sketch         : 3xHxW sketch in [-1,1]
      hint_rgb       : 3xHxW diffusion hint tensor; background = 0
      hint_mask      : 1xHxW; 1 at hint pixels, 0 elsewhere
      base_condition : 7xHxW = sketch(3) + diffusion hints(3) + mask(1)
      gan_input      : 7xHxW input matching the ORIGINAL baseline preprocessing;
                       its empty hint background is -1, not 0.

    Fusion condition is created on the GPU as:
      sketch(3) + hint_rgb(3) + mask(1) + frozen Pix2Pix output(3) = 10 channels.
    """

    def __init__(self, paths, image_size=256, hint_rate=0.05,
                 augment=False, deterministic_hints=False):
        self.paths = paths
        self.image_size = image_size
        self.hint_rate = hint_rate
        self.augment = augment
        self.deterministic_hints = deterministic_hints

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        paired = Image.open(path).convert("RGB")
        width, height = paired.size
        midpoint = width // 2

        colour = paired.crop((0, 0, midpoint, height))
        sketch = paired.crop((midpoint, 0, width, height))
        paired.close()

        colour = resize_lanczos(colour, self.image_size)
        sketch = resize_lanczos(sketch, self.image_size)

        if self.augment and random.random() > 0.5:
            colour = TF.hflip(colour)
            sketch = TF.hflip(sketch)

        colour_np = np.asarray(colour, dtype=np.float32)
        sketch_np = np.asarray(sketch, dtype=np.float32)

        target = np_to_norm_tensor(colour_np)
        sketch_tensor = np_to_norm_tensor(sketch_np)

        h, w, _ = colour_np.shape
        n_pixels = h * w
        n_hints = max(1, int(n_pixels * self.hint_rate))

        if self.deterministic_hints:
            rng = np.random.default_rng(SEED + index)
            idx = rng.choice(n_pixels, size=n_hints, replace=False)
        else:
            idx = np.random.choice(n_pixels, size=n_hints, replace=False)

        rows = idx // w
        cols = idx % w

        colour_norm_np = colour_np / 127.5 - 1.0
        hint_mask_np = np.zeros((h, w), dtype=np.float32)
        hint_mask_np[rows, cols] = 1.0

        # Diffusion representation: empty background = 0.
        diffusion_hint_np = np.zeros_like(colour_norm_np, dtype=np.float32)
        diffusion_hint_np[rows, cols, :] = colour_norm_np[rows, cols, :]

        # Baseline representation: its original code builds a uint/float hint map
        # with zeros and THEN normalises to [-1,1], so empty background = -1.
        # Keeping this exact convention is important when loading the pretrained GAN.
        gan_hint_np = np.full_like(colour_norm_np, -1.0, dtype=np.float32)
        gan_hint_np[rows, cols, :] = colour_norm_np[rows, cols, :]

        hint_rgb = torch.from_numpy(
            np.transpose(diffusion_hint_np, (2, 0, 1)).copy()
        ).float()
        gan_hint_rgb = torch.from_numpy(
            np.transpose(gan_hint_np, (2, 0, 1)).copy()
        ).float()
        hint_mask = torch.from_numpy(hint_mask_np[None, :, :].copy()).float()

        base_condition = torch.cat(
            [sketch_tensor, hint_rgb, hint_mask], dim=0
        )
        gan_input = torch.cat(
            [sketch_tensor, gan_hint_rgb, hint_mask], dim=0
        )

        return {
            "target": target,
            "sketch": sketch_tensor,
            "hint_rgb": hint_rgb,
            "hint_mask": hint_mask,
            "base_condition": base_condition,
            "gan_input": gan_input,
            "name": os.path.basename(path),
        }


# ================================================================
# 6. FROZEN PIX2PIX GENERATOR (EXPERIMENT 1)
# ================================================================

class Pix2PixEncoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels, use_batchnorm=True):
        super().__init__()
        layers = [
            nn.Conv2d(
                input_channels, output_channels,
                kernel_size=4, stride=2, padding=1,
                bias=not use_batchnorm,
            )
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(output_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Pix2PixDecoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                input_channels, output_channels,
                kernel_size=4, stride=2, padding=1, bias=False,
            ),
            nn.BatchNorm2d(output_channels),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Pix2PixGenerator(nn.Module):
    """
    Exact generator architecture from baseline-model.py.
    It is loaded from Experiment 1 and kept frozen throughout Experiment 3.
    """
    def __init__(self, in_channels=7, out_channels=3, ngf=64):
        super().__init__()

        self.enc1 = Pix2PixEncoderBlock(in_channels, ngf, use_batchnorm=False)
        self.enc2 = Pix2PixEncoderBlock(ngf, ngf * 2)
        self.enc3 = Pix2PixEncoderBlock(ngf * 2, ngf * 4)
        self.enc4 = Pix2PixEncoderBlock(ngf * 4, ngf * 8)
        self.enc5 = Pix2PixEncoderBlock(ngf * 8, ngf * 8)
        self.enc6 = Pix2PixEncoderBlock(ngf * 8, ngf * 8)
        self.enc7 = Pix2PixEncoderBlock(ngf * 8, ngf * 8)
        self.enc8 = Pix2PixEncoderBlock(ngf * 8, ngf * 8)

        self.dec1 = Pix2PixDecoderBlock(ngf * 8, ngf * 8, dropout=True)
        self.dec2 = Pix2PixDecoderBlock(ngf * 16, ngf * 8, dropout=True)
        self.dec3 = Pix2PixDecoderBlock(ngf * 16, ngf * 8, dropout=True)
        self.dec4 = Pix2PixDecoderBlock(ngf * 16, ngf * 8)
        self.dec5 = Pix2PixDecoderBlock(ngf * 16, ngf * 4)
        self.dec6 = Pix2PixDecoderBlock(ngf * 8, ngf * 2)
        self.dec7 = Pix2PixDecoderBlock(ngf * 4, ngf)

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(
                ngf * 2, out_channels,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        d1 = self.dec1(e8)
        d2 = self.dec2(torch.cat([d1, e7], dim=1))
        d3 = self.dec3(torch.cat([d2, e6], dim=1))
        d4 = self.dec4(torch.cat([d3, e5], dim=1))
        d5 = self.dec5(torch.cat([d4, e4], dim=1))
        d6 = self.dec6(torch.cat([d5, e3], dim=1))
        d7 = self.dec7(torch.cat([d6, e2], dim=1))

        return self.output_layer(torch.cat([d7, e1], dim=1))


def load_frozen_pix2pix():
    if not os.path.isfile(BASELINE_CHECKPOINT):
        raise FileNotFoundError(
            "Pretrained Pix2Pix baseline checkpoint was not found:\n"
            f"{BASELINE_CHECKPOINT}\n\n"
            "Run the baseline first or edit BASELINE_CHECKPOINT near the top "
            "of this fusion script."
        )

    generator = Pix2PixGenerator(
        in_channels=7,
        out_channels=3,
        ngf=64,
    ).to(DEVICE)

    checkpoint = torch.load(BASELINE_CHECKPOINT, map_location=DEVICE)
    if "generator" not in checkpoint:
        raise KeyError(
            f"{BASELINE_CHECKPOINT} does not contain a 'generator' state_dict."
        )

    generator.load_state_dict(checkpoint["generator"], strict=True)
    generator.eval()

    for parameter in generator.parameters():
        parameter.requires_grad_(False)

    print("Loaded frozen Pix2Pix baseline:", BASELINE_CHECKPOINT)
    print("Pix2Pix checkpoint epoch       :", checkpoint.get("epoch", "unknown"))
    print("Pix2Pix parameters             :", f"{sum(p.numel() for p in generator.parameters()):,}")
    return generator


@torch.no_grad()
def build_fusion_condition(gan_generator, batch):
    """
    10-channel diffusion condition:
      sketch 3 + sparse hints 3 + hint mask 1 + Pix2Pix coarse RGB 3.
    """
    base_condition = batch["base_condition"].to(DEVICE, non_blocking=True)
    gan_input = batch["gan_input"].to(DEVICE, non_blocking=True)

    with autocast_context():
        coarse = gan_generator(gan_input)

    coarse = coarse.float().clamp(-1.0, 1.0)
    condition = torch.cat([base_condition, coarse], dim=1)
    return condition, coarse


# ================================================================
# 7. DIFFUSION MODEL COMPONENTS
# ================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        device = t.device
        emb = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.0):
        super().__init__()

        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2),
        )

        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv1(F.silu(self.norm1(x)))

        scale, shift = self.time_mlp(time_emb).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by attention heads")

        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w

        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b, 3, self.heads, self.head_dim, n)
        q, k, v = qkv.unbind(dim=1)

        # B, heads, tokens, head_dim
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            scale = self.head_dim ** -0.5
            attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
            out = torch.matmul(attn, v)

        out = out.transpose(-2, -1).contiguous().reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class ConditionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionEncoder(nn.Module):
    """Injects sketch + hints at every U-Net scale."""

    def __init__(self, cond_ch=10, base=64):
        super().__init__()
        self.c0 = ConditionBlock(cond_ch, base, downsample=False)       # 256
        self.c1 = ConditionBlock(base, base * 2, downsample=True)      # 128
        self.c2 = ConditionBlock(base * 2, base * 4, downsample=True)  # 64
        self.c3 = ConditionBlock(base * 4, base * 4, downsample=True)  # 32
        self.c4 = ConditionBlock(base * 4, base * 8, downsample=True)  # 16

    def forward(self, cond):
        c0 = self.c0(cond)
        c1 = self.c1(c0)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        return [c0, c1, c2, c3, c4]


class StrongConditionalUNet(nn.Module):
    def __init__(self, base=64, cond_ch=10, out_ch=3,
                 dropout=0.1, attention_heads=4):
        super().__init__()

        time_dim = base * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base),
            nn.Linear(base, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.cond_encoder = ConditionEncoder(cond_ch=cond_ch, base=base)

        # x_t (3) + fusion condition (10) = 13 channels
        self.in_conv = nn.Conv2d(3 + cond_ch, base, 3, padding=1)

        # 256x256
        self.d1a = ResBlock(base, base, time_dim, dropout)
        self.d1b = ResBlock(base, base, time_dim, dropout)
        self.down1 = Downsample(base, base * 2)

        # 128x128
        self.d2a = ResBlock(base * 2, base * 2, time_dim, dropout)
        self.d2b = ResBlock(base * 2, base * 2, time_dim, dropout)
        self.down2 = Downsample(base * 2, base * 4)

        # 64x64
        self.d3a = ResBlock(base * 4, base * 4, time_dim, dropout)
        self.d3b = ResBlock(base * 4, base * 4, time_dim, dropout)
        self.down3 = Downsample(base * 4, base * 4)

        # 32x32 + attention
        self.d4a = ResBlock(base * 4, base * 4, time_dim, dropout)
        self.d4attn = AttentionBlock(base * 4, attention_heads)
        self.d4b = ResBlock(base * 4, base * 4, time_dim, dropout)
        self.down4 = Downsample(base * 4, base * 8)

        # 16x16 bottleneck + attention
        self.mid1 = ResBlock(base * 8, base * 8, time_dim, dropout)
        self.midattn = AttentionBlock(base * 8, attention_heads)
        self.mid2 = ResBlock(base * 8, base * 8, time_dim, dropout)

        # 16 -> 32
        self.up4 = Upsample(base * 8, base * 4)
        self.u4a = ResBlock(base * 8, base * 4, time_dim, dropout)
        self.u4attn = AttentionBlock(base * 4, attention_heads)
        self.u4b = ResBlock(base * 4, base * 4, time_dim, dropout)

        # 32 -> 64
        self.up3 = Upsample(base * 4, base * 4)
        self.u3a = ResBlock(base * 8, base * 4, time_dim, dropout)
        self.u3b = ResBlock(base * 4, base * 4, time_dim, dropout)

        # 64 -> 128
        self.up2 = Upsample(base * 4, base * 2)
        self.u2a = ResBlock(base * 4, base * 2, time_dim, dropout)
        self.u2b = ResBlock(base * 2, base * 2, time_dim, dropout)

        # 128 -> 256
        self.up1 = Upsample(base * 2, base)
        self.u1a = ResBlock(base * 2, base, time_dim, dropout)
        self.u1b = ResBlock(base, base, time_dim, dropout)

        final_conv = nn.Conv2d(base, out_ch, 3, padding=1)
        nn.init.zeros_(final_conv.weight)
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)
        self.out = nn.Sequential(
            nn.GroupNorm(32, base),
            nn.SiLU(),
            final_conv,
        )

    def forward(self, x_t, t, cond):
        time_emb = self.time_mlp(t)
        c0, c1, c2, c3, c4 = self.cond_encoder(cond)

        h = self.in_conv(torch.cat([x_t, cond], dim=1)) + c0

        h = self.d1a(h, time_emb)
        h = self.d1b(h, time_emb)
        s1 = h
        h = self.down1(h) + c1

        h = self.d2a(h, time_emb)
        h = self.d2b(h, time_emb)
        s2 = h
        h = self.down2(h) + c2

        h = self.d3a(h, time_emb)
        h = self.d3b(h, time_emb)
        s3 = h
        h = self.down3(h) + c3

        h = self.d4a(h, time_emb)
        h = self.d4attn(h)
        h = self.d4b(h, time_emb)
        s4 = h
        h = self.down4(h) + c4

        h = self.mid1(h, time_emb)
        h = self.midattn(h)
        h = self.mid2(h, time_emb)

        h = self.up4(h)
        h = torch.cat([h, s4], dim=1)
        h = self.u4a(h, time_emb)
        h = self.u4attn(h)
        h = self.u4b(h, time_emb)

        h = self.up3(h)
        h = torch.cat([h, s3], dim=1)
        h = self.u3a(h, time_emb)
        h = self.u3b(h, time_emb)

        h = self.up2(h)
        h = torch.cat([h, s2], dim=1)
        h = self.u2a(h, time_emb)
        h = self.u2b(h, time_emb)

        h = self.up1(h)
        h = torch.cat([h, s1], dim=1)
        h = self.u1a(h, time_emb)
        h = self.u1b(h, time_emb)

        return self.out(h)


# ================================================================
# 7. COSINE DIFFUSION SCHEDULE
# ================================================================

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-8, 0.999).float()


class DiffusionSchedule:
    def __init__(self, timesteps, device):
        self.timesteps = timesteps
        self.device = device

        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    @staticmethod
    def extract(values, t, x_shape):
        out = values.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x0, t, noise):
        a = self.extract(self.sqrt_alpha_bar, t, x0.shape)
        b = self.extract(self.sqrt_one_minus_alpha_bar, t, x0.shape)
        return a * x0 + b * noise

    def predict_x0_from_eps(self, x_t, t, eps):
        a_bar = self.extract(self.alpha_bar, t, x_t.shape)
        return (x_t - torch.sqrt(1.0 - a_bar) * eps) / torch.sqrt(a_bar.clamp_min(1e-8))

    def snr(self, t, x_shape):
        a_bar = self.extract(self.alpha_bar, t, x_shape)
        return a_bar / (1.0 - a_bar).clamp_min(1e-8)


# ================================================================
# 8. EMA
# ================================================================

@torch.no_grad()
def update_ema(ema_model, model, step):
    if step < EMA_WARMUP_STEPS:
        ema_model.load_state_dict(model.state_dict())
        return

    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(EMA_DECAY).add_(p.data, alpha=1.0 - EMA_DECAY)

    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)


# ================================================================
# 9. DDIM SAMPLER
# ================================================================

@torch.no_grad()
def ddim_sample(model, schedule, condition, hint_rgb, hint_mask,
                steps=100, eta=0.0, seed=None):
    model.eval()

    b, _, h, w = condition.shape

    if seed is not None:
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(seed)
        x = torch.randn((b, 3, h, w), device=DEVICE, generator=generator)
    else:
        x = torch.randn((b, 3, h, w), device=DEVICE)

    times = torch.linspace(
        schedule.timesteps - 1,
        0,
        steps,
        device=DEVICE
    ).long()

    time_pairs = list(zip(times.tolist(), times[1:].tolist() + [-1]))

    for current_t, next_t in time_pairs:
        t = torch.full((b,), current_t, device=DEVICE, dtype=torch.long)

        with autocast_context():
            pred_eps = model(x, t, condition)

        # Always perform DDIM arithmetic in FP32. This is especially important
        # near t=999, where division by sqrt(alpha_bar) can overflow FP16.
        pred_eps = pred_eps.float()
        x = x.float()
        alpha_t = schedule.alpha_bar[current_t].float()
        pred_x0 = (x - torch.sqrt(1.0 - alpha_t) * pred_eps) / torch.sqrt(alpha_t.clamp_min(1e-8))
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        # Hard projection: exact adherence to the sparse colour hints.
        pred_x0 = pred_x0 * (1.0 - hint_mask) + hint_rgb * hint_mask

        if next_t < 0:
            x = pred_x0
            break

        alpha_next = schedule.alpha_bar[next_t].float()

        sigma = eta * torch.sqrt(
            ((1.0 - alpha_next) / (1.0 - alpha_t)).clamp_min(0.0)
            * (1.0 - alpha_t / alpha_next).clamp_min(0.0)
        )

        c = torch.sqrt((1.0 - alpha_next - sigma ** 2).clamp_min(0.0))

        if eta > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = torch.sqrt(alpha_next) * pred_x0 + c * pred_eps + sigma * noise

    return x.clamp(-1.0, 1.0)


# ================================================================
# 10. TRAINING LOSS
# ================================================================

def diffusion_training_loss(model, gan_generator, schedule, batch):
    target = batch["target"].to(DEVICE, non_blocking=True)
    condition, _ = build_fusion_condition(gan_generator, batch)
    hint_mask = batch["hint_mask"].to(DEVICE, non_blocking=True)

    b = target.size(0)
    t = torch.randint(0, schedule.timesteps, (b,), device=DEVICE, dtype=torch.long)
    noise = torch.randn_like(target)
    x_t = schedule.q_sample(target, t, noise)

    with autocast_context():
        pred_noise = model(x_t, t, condition)

    pred_noise_f = pred_noise.float()
    target_f = target.float()
    noise_f = noise.float()
    x_t_f = x_t.float()
    hint_mask_f = hint_mask.float()

    if not torch.isfinite(pred_noise_f).all():
        raise FloatingPointError(
            "Non-finite network output detected before loss computation."
        )

    per_sample_mse = (pred_noise_f - noise_f).pow(2).mean(dim=(1, 2, 3))
    snr = schedule.snr(t, target_f.shape).reshape(b).float()
    weights = torch.minimum(
        snr, torch.full_like(snr, MIN_SNR_GAMMA)
    ) / snr.clamp_min(1e-8)
    noise_loss = (weights * per_sample_mse).mean()

    pred_x0 = schedule.predict_x0_from_eps(
        x_t_f, t, pred_noise_f
    ).clamp(-1.0, 1.0)

    aux_mask = (snr >= AUX_MIN_SNR).float()
    aux_count = aux_mask.sum().clamp_min(1.0)

    x0_per_sample = (pred_x0 - target_f).abs().mean(dim=(1, 2, 3))
    x0_l1 = (x0_per_sample * aux_mask).sum() / aux_count

    hint_abs = (pred_x0 - target_f).abs() * hint_mask_f
    hint_denom = (
        hint_mask_f.sum(dim=(1, 2, 3)) * target_f.shape[1]
    ).clamp_min(1.0)
    hint_per_sample = hint_abs.sum(dim=(1, 2, 3)) / hint_denom
    hint_loss = (hint_per_sample * aux_mask).sum() / aux_count

    total_loss = (
        noise_loss
        + LAMBDA_X0_L1 * x0_l1
        + LAMBDA_HINT * hint_loss
    )

    if not torch.isfinite(total_loss):
        raise FloatingPointError(
            f"Non-finite loss detected: noise={noise_loss.item()}, "
            f"x0={x0_l1.item()}, hint={hint_loss.item()}."
        )

    return (
        total_loss,
        noise_loss.detach(),
        x0_l1.detach(),
        hint_loss.detach(),
    )


# ================================================================
# 11. LEARNING-RATE SCHEDULER
# ================================================================

def make_lr_scheduler(optimizer, total_optimizer_steps):
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return max(1e-6, step / max(1, WARMUP_STEPS))

        progress = (step - WARMUP_STEPS) / max(1, total_optimizer_steps - WARMUP_STEPS)
        progress = min(max(progress, 0.0), 1.0)

        min_ratio = MIN_LEARNING_RATE / LEARNING_RATE
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ================================================================
# 12. TRAIN ONE EPOCH
# ================================================================

def train_one_epoch(model, ema_model, gan_generator, schedule, loader, optimizer,
                    lr_scheduler, scaler, epoch, global_step):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    running_total = 0.0
    running_noise = 0.0
    running_x0 = 0.0
    running_hint = 0.0
    skipped_steps = 0
    consecutive_bad_steps = 0

    progress = tqdm(loader, desc=f"Training epoch {epoch}", leave=True)

    for batch_index, batch in enumerate(progress, start=1):
        total_loss, noise_loss, x0_l1, hint_loss = diffusion_training_loss(
            model, gan_generator, schedule, batch
        )

        scaled_loss = total_loss / GRAD_ACCUM_STEPS
        scaler.scale(scaled_loss).backward()

        should_step = (
            batch_index % GRAD_ACCUM_STEPS == 0
            or batch_index == len(loader)
        )

        if should_step:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), MAX_GRAD_NORM
            )

            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                lr_scheduler.step()
                global_step += 1
                update_ema(ema_model, model, global_step)
                consecutive_bad_steps = 0
            else:
                skipped_steps += 1
                consecutive_bad_steps += 1
                optimizer.zero_grad(set_to_none=True)
                # For FP16 GradScaler, update lets it lower the scale.
                if hasattr(scaler, "is_enabled") and scaler.is_enabled():
                    scaler.update()
                print(
                    f"\nWARNING: skipped non-finite gradient step "
                    f"at batch {batch_index}."
                )
                if consecutive_bad_steps >= 3:
                    raise FloatingPointError(
                        "Three consecutive non-finite gradient steps detected. "
                        "Run stopped to protect checkpoints."
                    )

        running_total += float(total_loss.detach().item())
        running_noise += float(noise_loss.item())
        running_x0 += float(x0_l1.item())
        running_hint += float(hint_loss.item())

        progress.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "noise": f"{noise_loss.item():.4f}",
            "x0": f"{x0_l1.item():.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            "skip": skipped_steps,
        })

    n = len(loader)
    return {
        "train_loss": running_total / n,
        "noise_loss": running_noise / n,
        "x0_l1": running_x0 / n,
        "hint_loss": running_hint / n,
        "global_step": global_step,
        "skipped_steps": skipped_steps,
    }


# ================================================================
# 13. VALIDATION
# ================================================================

@torch.no_grad()
def validate_model(model, gan_generator, schedule, loader, ddim_steps):
    model.eval()
    gan_generator.eval()

    psnr_scores = []
    ssim_scores = []

    for batch_index, batch in enumerate(tqdm(loader, desc="Validation", leave=False)):
        target = batch["target"].to(DEVICE)
        condition, _ = build_fusion_condition(gan_generator, batch)
        hint_rgb = batch["hint_rgb"].to(DEVICE)
        hint_mask = batch["hint_mask"].to(DEVICE)

        generated = ddim_sample(
            model,
            schedule,
            condition,
            hint_rgb,
            hint_mask,
            steps=ddim_steps,
            eta=0.0,
            seed=SEED + batch_index,
        )

        generated_np = tensor_to_uint8(generated)
        target_np = tensor_to_uint8(target)

        for i in range(generated_np.shape[0]):
            psnr_scores.append(psnr_fn(target_np[i], generated_np[i], data_range=255))
            ssim_scores.append(ssim_fn(
                target_np[i], generated_np[i], data_range=255, channel_axis=2
            ))

    return float(np.mean(psnr_scores)), float(np.mean(ssim_scores))


# ================================================================
# 14. SAMPLE GRID
# ================================================================

@torch.no_grad()
def save_sample_grid(model, gan_generator, schedule, loader, epoch):
    model.eval()
    gan_generator.eval()
    batch = next(iter(loader))

    target = batch["target"].to(DEVICE)
    condition, coarse = build_fusion_condition(gan_generator, batch)
    hint_rgb = batch["hint_rgb"].to(DEVICE)
    hint_mask = batch["hint_mask"].to(DEVICE)

    generated = ddim_sample(
        model,
        schedule,
        condition,
        hint_rgb,
        hint_mask,
        steps=SAMPLE_DDIM_STEPS,
        eta=0.0,
        seed=SEED,
    )

    n = min(NUM_SAMPLE_IMAGES, generated.size(0))
    rows = []

    for i in range(n):
        sketch_img = tensor_to_uint8(batch["sketch"][i])

        hint_disp = torch.ones_like(batch["hint_rgb"][i])
        m = batch["hint_mask"][i]
        hint_disp = hint_disp * (1.0 - m) + batch["hint_rgb"][i] * m
        hint_img = tensor_to_uint8(hint_disp)

        coarse_img = tensor_to_uint8(coarse[i])
        generated_img = tensor_to_uint8(generated[i])
        target_img = tensor_to_uint8(target[i])

        row = np.concatenate(
            [sketch_img, hint_img, coarse_img, generated_img, target_img],
            axis=1,
        )
        rows.append(row)

    grid = np.concatenate(rows, axis=0)
    path = os.path.join(SAMPLE_DIR, f"epoch_{epoch:03d}.png")
    Image.fromarray(grid).save(path)
    return path


# ================================================================
# 15. CHECKPOINTS
# ================================================================

def save_checkpoint(model, ema_model, optimizer, lr_scheduler, scaler,
                    epoch, global_step, history, best_psnr, best_ssim,
                    best_epoch, filename):
    path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "history": history,
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "best_epoch": best_epoch,
        "config": {
            "image_size": IMAGE_SIZE,
            "hint_rate": HINT_RATE,
            "diffusion_steps": DIFFUSION_STEPS,
            "base_channels": BASE_CHANNELS,
            "condition_channels": 10,
            "fusion": "sketch + hints + mask + frozen_pix2pix_output",
            "baseline_checkpoint": BASELINE_CHECKPOINT,
        },
    }, path)
    return path


# ================================================================
# 16. TRAINING CURVES
# ================================================================

def save_training_curves(history):
    epochs = [x["epoch"] for x in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [x["train_loss"] for x in history], label="Total")
    plt.plot(epochs, [x["noise_loss"] for x in history], label="Noise")
    plt.plot(epochs, [x["x0_l1"] for x in history], label="x0 L1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Hybrid Pix2Pix-Diffusion Training Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    loss_path = os.path.join(FIGURE_DIR, "training_losses.png")
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()

    val_epochs = [x["epoch"] for x in history if x.get("val_psnr") is not None]
    val_psnr = [x["val_psnr"] for x in history if x.get("val_psnr") is not None]
    val_ssim = [x["val_ssim"] for x in history if x.get("val_ssim") is not None]

    psnr_path = None
    ssim_path = None

    if val_epochs:
        plt.figure(figsize=(10, 6))
        plt.plot(val_epochs, val_psnr, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Validation PSNR (dB)")
        plt.title("Fusion Validation PSNR")
        plt.grid(alpha=0.3)
        psnr_path = os.path.join(FIGURE_DIR, "validation_psnr.png")
        plt.savefig(psnr_path, dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(val_epochs, val_ssim, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Validation SSIM")
        plt.title("Fusion Validation SSIM")
        plt.grid(alpha=0.3)
        ssim_path = os.path.join(FIGURE_DIR, "validation_ssim.png")
        plt.savefig(ssim_path, dpi=150, bbox_inches="tight")
        plt.close()

    return loss_path, psnr_path, ssim_path


# ================================================================
# 17. FINAL TEST EVALUATION
# ================================================================

@torch.no_grad()
def evaluate_test(model, gan_generator, schedule, loader):
    model.eval()
    gan_generator.eval()

    perceptual_metric = None
    if LPIPS_AVAILABLE:
        try:
            perceptual_metric = lpips.LPIPS(net="alex").to(DEVICE)
            perceptual_metric.eval()
            print("LPIPS enabled.")
        except Exception as error:
            print("LPIPS could not be initialized:", error)
            perceptual_metric = None
    else:
        print("LPIPS package not installed; LPIPS will be NA.")

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    inference_times = []
    records = []
    visual_results = []

    for batch_index, batch in enumerate(tqdm(loader, desc="Final fusion test evaluation")):
        target = batch["target"].to(DEVICE)
        hint_rgb = batch["hint_rgb"].to(DEVICE)
        hint_mask = batch["hint_mask"].to(DEVICE)

        # Fusion inference timing includes BOTH stages:
        # Pix2Pix coarse prediction + DDIM refinement.
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        condition, coarse = build_fusion_condition(gan_generator, batch)

        generated = ddim_sample(
            model,
            schedule,
            condition,
            hint_rgb,
            hint_mask,
            steps=TEST_DDIM_STEPS,
            eta=DDIM_ETA,
            seed=SEED + batch_index,
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        inference_times.append(elapsed / generated.size(0))

        generated_np_batch = tensor_to_uint8(generated)
        coarse_np_batch = tensor_to_uint8(coarse)
        target_np_batch = tensor_to_uint8(target)

        for i in range(generated.size(0)):
            generated_np = generated_np_batch[i]
            coarse_np = coarse_np_batch[i]
            target_np = target_np_batch[i]

            p = psnr_fn(target_np, generated_np, data_range=255)
            s = ssim_fn(target_np, generated_np, data_range=255, channel_axis=2)

            lp = None
            if perceptual_metric is not None:
                lp = float(perceptual_metric(
                    generated[i:i+1].float(),
                    target[i:i+1].float()
                ).item())
                lpips_scores.append(lp)

            psnr_scores.append(p)
            ssim_scores.append(s)

            name = batch["name"][i]
            Image.fromarray(generated_np).save(os.path.join(GENERATED_DIR, name))
            Image.fromarray(coarse_np).save(os.path.join(COARSE_DIR, name))
            Image.fromarray(target_np).save(os.path.join(REAL_DIR, name))

            records.append({"name": name, "psnr": p, "ssim": s, "lpips": lp})

            if len(visual_results) < 40:
                hint_disp = torch.ones_like(batch["hint_rgb"][i])
                m = batch["hint_mask"][i]
                hint_disp = hint_disp * (1.0 - m) + batch["hint_rgb"][i] * m

                visual_results.append({
                    "name": name,
                    "sketch": tensor_to_uint8(batch["sketch"][i]),
                    "hint": tensor_to_uint8(hint_disp),
                    "coarse": coarse_np,
                    "generated": generated_np,
                    "target": target_np,
                    "psnr": p,
                    "ssim": s,
                })

    mean_psnr = float(np.mean(psnr_scores))
    std_psnr = float(np.std(psnr_scores))
    mean_ssim = float(np.mean(ssim_scores))
    std_ssim = float(np.std(ssim_scores))

    if lpips_scores:
        mean_lpips = float(np.mean(lpips_scores))
        std_lpips = float(np.std(lpips_scores))
    else:
        mean_lpips = None
        std_lpips = None

    average_inference_ms = float(np.mean(inference_times) * 1000.0)

    metrics_path = os.path.join(EVAL_DIR, "metrics_test.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "psnr", "ssim", "lpips"])
        for r in records:
            writer.writerow([
                r["name"],
                f"{r['psnr']:.4f}",
                f"{r['ssim']:.4f}",
                f"{r['lpips']:.4f}" if r["lpips"] is not None else "NA",
            ])

    summary = {
        "model": "Hybrid Pix2Pix + Conditional Diffusion",
        "epochs": TRAIN_EPOCHS,
        "hint_rate": HINT_RATE,
        "fusion_condition_channels": 10,
        "diffusion_unet_total_input_channels": 13,
        "diffusion_steps_train": DIFFUSION_STEPS,
        "ddim_steps_test": TEST_DDIM_STEPS,
        "psnr_mean": mean_psnr,
        "psnr_std": std_psnr,
        "ssim_mean": mean_ssim,
        "ssim_std": std_ssim,
        "lpips_mean": mean_lpips,
        "lpips_std": std_lpips,
        "infer_ms": average_inference_ms,
        "baseline_checkpoint": BASELINE_CHECKPOINT,
    }

    with open(os.path.join(EVAL_DIR, "summary_test.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(EVAL_DIR, "summary_test.txt"), "w", encoding="utf-8") as f:
        f.write("Model: Hybrid Pix2Pix + Conditional Diffusion\n")
        f.write("Fusion: frozen Pix2Pix coarse output used as extra diffusion condition\n")
        f.write(f"Training epochs: {TRAIN_EPOCHS}\n")
        f.write(f"Hint rate: {HINT_RATE}\n")
        f.write("Fusion condition: sketch(3) + hints(3) + mask(1) + Pix2Pix RGB(3) = 10\n")
        f.write("Diffusion U-Net total input: noisy RGB(3) + fusion condition(10) = 13\n")
        f.write(f"Train diffusion steps: {DIFFUSION_STEPS}\n")
        f.write(f"Test DDIM steps: {TEST_DDIM_STEPS}\n")
        f.write(f"Mean PSNR: {mean_psnr:.4f} dB\n")
        f.write(f"PSNR Std: {std_psnr:.4f} dB\n")
        f.write(f"Mean SSIM: {mean_ssim:.4f}\n")
        f.write(f"SSIM Std: {std_ssim:.4f}\n")
        if mean_lpips is not None:
            f.write(f"Mean LPIPS: {mean_lpips:.4f}\n")
            f.write(f"LPIPS Std: {std_lpips:.4f}\n")
        f.write(f"End-to-end fusion inference: {average_inference_ms:.2f} ms/image\n")

    return (
        mean_psnr,
        std_psnr,
        mean_ssim,
        std_ssim,
        mean_lpips,
        std_lpips,
        average_inference_ms,
        visual_results,
    )


# ================================================================
# 18. RESULT GRIDS
# ================================================================

def save_result_grid(results, filename, title, number=8):
    results = results[:min(number, len(results))]
    if not results:
        return None

    rows = len(results)
    fig, axes = plt.subplots(rows, 5, figsize=(20, rows * 4))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = [
        "Sketch",
        "Colour Hints",
        "Pix2Pix Coarse",
        "Fusion Output",
        "Ground Truth",
    ]
    for col, text in enumerate(titles):
        axes[0, col].set_title(text, fontweight="bold")

    for row, result in enumerate(results):
        images = [
            result["sketch"],
            result["hint"],
            result["coarse"],
            result["generated"],
            result["target"],
        ]
        for col, image in enumerate(images):
            axes[row, col].imshow(image)
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(
            f"Fusion PSNR {result['psnr']:.2f}\nSSIM {result['ssim']:.3f}"
        )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ================================================================
# 19. DATASET SANITY CHECK FIGURE
# ================================================================

def save_dataset_check(dataset):
    sample = dataset[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(tensor_to_uint8(sample["sketch"]))
    axes[0].set_title("INPUT SKETCH")
    axes[0].axis("off")

    hint_disp = torch.ones_like(sample["hint_rgb"])
    hint_disp = hint_disp * (1.0 - sample["hint_mask"]) + sample["hint_rgb"] * sample["hint_mask"]
    axes[1].imshow(tensor_to_uint8(hint_disp))
    axes[1].set_title("5% COLOUR HINTS")
    axes[1].axis("off")

    axes[2].imshow(tensor_to_uint8(sample["target"]))
    axes[2].set_title("GROUND TRUTH COLOUR")
    axes[2].axis("off")

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "dataset_pair_check.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ================================================================
# 20. MAIN
# ================================================================

def main():
    print("\n" + "=" * 80)
    print("EXPERIMENT 3 - HYBRID PIX2PIX + CONDITIONAL DIFFUSION")
    print("Frozen Pix2Pix coarse colourisation + Palette-style diffusion refinement")
    print("=" * 80)
    print("Device              :", DEVICE)
    if torch.cuda.is_available():
        print("GPU                 :", torch.cuda.get_device_name(0))
        mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU memory          : {mem:.2f} GB")
        print("CUDA                :", torch.version.cuda)
    print("Image size          :", IMAGE_SIZE)
    print("Hint rate           :", HINT_RATE)
    print("Epochs              :", TRAIN_EPOCHS)
    print("Diffusion steps     :", DIFFUSION_STEPS)
    print("Test DDIM steps     :", TEST_DDIM_STEPS)
    print("Batch size          :", BATCH_SIZE)
    print("Grad accumulation   :", GRAD_ACCUM_STEPS)
    print("Effective batch     :", BATCH_SIZE * GRAD_ACCUM_STEPS)
    print("AMP                 :", USE_AMP)
    print("Output              :", OUTPUT_ROOT)
    print("=" * 80)

    train_files, val_source_files = check_dataset()

    # EXACT same idea as baseline: split original val folder 50/50 into val/test.
    shuffled_val = val_source_files.copy()
    random.Random(SEED).shuffle(shuffled_val)
    midpoint = len(shuffled_val) // 2
    val_files = shuffled_val[:midpoint]
    test_files = shuffled_val[midpoint:]

    print("\nDATASET SPLIT")
    print("Training   :", len(train_files))
    print("Validation :", len(val_files))
    print("Test       :", len(test_files))

    train_dataset = AnimeSketchColourDataset(
        train_files, IMAGE_SIZE, HINT_RATE, augment=True, deterministic_hints=False
    )
    val_dataset = AnimeSketchColourDataset(
        val_files, IMAGE_SIZE, HINT_RATE, augment=False, deterministic_hints=True
    )
    test_dataset = AnimeSketchColourDataset(
        test_files, IMAGE_SIZE, HINT_RATE, augment=False, deterministic_hints=True
    )

    pair_check = save_dataset_check(train_dataset)
    print("Dataset check saved:", pair_check)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        persistent_workers=NUM_WORKERS > 0,
        drop_last=True,
    )

    # Fixed deterministic validation subset for affordable model selection.
    rng = np.random.default_rng(SEED)
    val_indices = np.arange(len(val_dataset))
    rng.shuffle(val_indices)
    val_indices = val_indices[:min(VAL_MAX_IMAGES, len(val_indices))]
    val_subset = Subset(val_dataset, val_indices.tolist())

    val_loader = DataLoader(
        val_subset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        persistent_workers=NUM_WORKERS > 0,
    )

    # Separate loader used only for preview images.
    sample_loader = DataLoader(
        val_dataset,
        batch_size=max(NUM_SAMPLE_IMAGES, 4),
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        persistent_workers=NUM_WORKERS > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        persistent_workers=NUM_WORKERS > 0,
    )

    # Load Experiment-1 generator first and keep it frozen.
    gan_generator = load_frozen_pix2pix()

    model = StrongConditionalUNet(
        base=BASE_CHANNELS,
        cond_ch=10,
        out_ch=3,
        dropout=DROPOUT,
        attention_heads=ATTENTION_HEADS,
    ).to(DEVICE)

    ema_model = copy.deepcopy(model).to(DEVICE).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    schedule = DiffusionSchedule(DIFFUSION_STEPS, DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )

    optimizer_steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    total_optimizer_steps = optimizer_steps_per_epoch * TRAIN_EPOCHS
    lr_scheduler = make_lr_scheduler(optimizer, total_optimizer_steps)
    scaler = make_grad_scaler()

    print(f"\nModel parameters     : {sum(p.numel() for p in model.parameters()):,}")
    print(f"AMP precision        : {'BF16' if USE_BF16 else ('FP16' if USE_AMP else 'FP32')}")
    print(f"Learning rate        : {LEARNING_RATE:.2e}")
    print(f"Auxiliary min SNR    : {AUX_MIN_SNR}")
    print(f"Optimizer steps/epoch: {optimizer_steps_per_epoch:,}")
    print(f"Total optimizer steps: {total_optimizer_steps:,}")

    # ------------------------------------------------------------
    # AUTO RESUME
    # ------------------------------------------------------------
    checkpoint_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pth")))

    start_epoch = 1
    global_step = 0
    history = []
    best_psnr = -float("inf")
    best_ssim = -float("inf")
    best_epoch = 0

    if checkpoint_files:
        latest = checkpoint_files[-1]
        print("\nCheckpoint found:", latest)
        ckpt = torch.load(latest, map_location=DEVICE)

        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        history = ckpt.get("history", [])
        best_psnr = ckpt.get("best_psnr", -float("inf"))
        best_ssim = ckpt.get("best_ssim", -float("inf"))
        best_epoch = ckpt.get("best_epoch", 0)

        print("Resuming from epoch:", start_epoch)
    else:
        print("\nNo checkpoint found. Starting fresh.")

    # ------------------------------------------------------------
    # CSV LOG
    # ------------------------------------------------------------
    csv_path = os.path.join(LOG_DIR, "training_log.csv")
    if start_epoch == 1:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "noise_loss",
                "x0_l1",
                "hint_loss",
                "val_psnr",
                "val_ssim",
                "lr",
                "epoch_seconds",
            ])

    total_training_start = time.time()

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------
    for epoch in range(start_epoch, TRAIN_EPOCHS + 1):
        print("\n" + "=" * 80)
        print(f"EPOCH [{epoch}/{TRAIN_EPOCHS}]")
        print("=" * 80)

        epoch_start = time.time()

        train_stats = train_one_epoch(
            model,
            ema_model,
            gan_generator,
            schedule,
            train_loader,
            optimizer,
            lr_scheduler,
            scaler,
            epoch,
            global_step,
        )
        global_step = train_stats["global_step"]

        val_psnr = None
        val_ssim = None

        if (
            epoch <= MONITOR_FIRST_N_EPOCHS
            or epoch % VAL_EVERY == 0
            or epoch == TRAIN_EPOCHS
        ):
            print(f"\nRunning deterministic {VAL_DDIM_STEPS}-step DDIM validation...")
            val_psnr, val_ssim = validate_model(
                ema_model, gan_generator, schedule, val_loader, VAL_DDIM_STEPS
            )

        epoch_seconds = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train total loss : {train_stats['train_loss']:.6f}")
        print(f"Noise loss       : {train_stats['noise_loss']:.6f}")
        print(f"x0 L1            : {train_stats['x0_l1']:.6f}")
        print(f"Hint loss        : {train_stats['hint_loss']:.6f}")
        print(f"Skipped steps    : {train_stats.get('skipped_steps', 0)}")
        if val_psnr is not None:
            print(f"Val PSNR         : {val_psnr:.4f} dB")
            print(f"Val SSIM         : {val_ssim:.4f}")
        print(f"Learning rate    : {current_lr:.3e}")
        print(f"Epoch time       : {epoch_seconds:.1f} sec")

        record = {
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            "noise_loss": train_stats["noise_loss"],
            "x0_l1": train_stats["x0_l1"],
            "hint_loss": train_stats["hint_loss"],
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
        }
        history.append(record)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_stats["train_loss"],
                train_stats["noise_loss"],
                train_stats["x0_l1"],
                train_stats["hint_loss"],
                "" if val_psnr is None else val_psnr,
                "" if val_ssim is None else val_ssim,
                current_lr,
                epoch_seconds,
            ])

        # Best model is selected by validation PSNR.
        if val_psnr is not None and val_psnr > best_psnr:
            best_psnr = val_psnr
            best_ssim = val_ssim
            best_epoch = epoch
            best_path = save_checkpoint(
                model,
                ema_model,
                optimizer,
                lr_scheduler,
                scaler,
                epoch,
                global_step,
                history,
                best_psnr,
                best_ssim,
                best_epoch,
                "best_model.pth",
            )
            print("\nNEW BEST DIFFUSION MODEL")
            print(f"Best validation PSNR: {best_psnr:.4f} dB")
            print(f"Best validation SSIM: {best_ssim:.4f}")
            print("Saved:", best_path)

        if epoch <= MONITOR_FIRST_N_EPOCHS or epoch % SAMPLE_EVERY == 0:
            sample_path = save_sample_grid(ema_model, gan_generator, schedule, sample_loader, epoch)
            print("Sample saved:", sample_path)

        if epoch % SAVE_EVERY == 0 or epoch == TRAIN_EPOCHS:
            checkpoint_path = save_checkpoint(
                model,
                ema_model,
                optimizer,
                lr_scheduler,
                scaler,
                epoch,
                global_step,
                history,
                best_psnr,
                best_ssim,
                best_epoch,
                f"epoch_{epoch:04d}.pth",
            )
            print("Checkpoint:", checkpoint_path)

        # Refresh curves after each validation point.
        if val_psnr is not None:
            save_training_curves(history)

    # ------------------------------------------------------------
    # LOAD BEST MODEL
    # ------------------------------------------------------------
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.isfile(best_model_path):
        raise RuntimeError("best_model.pth was not created.")

    print("\n" + "=" * 80)
    print("LOADING BEST EMA MODEL")
    print("=" * 80)

    best_checkpoint = torch.load(best_model_path, map_location=DEVICE)
    ema_model.load_state_dict(best_checkpoint["ema_model"])
    best_epoch = best_checkpoint.get("best_epoch", best_checkpoint["epoch"])
    best_psnr = best_checkpoint.get("best_psnr", best_psnr)
    best_ssim = best_checkpoint.get("best_ssim", best_ssim)

    print("Best epoch          :", best_epoch)
    print(f"Best val PSNR      : {best_psnr:.4f} dB")
    print(f"Best val SSIM      : {best_ssim:.4f}")

    save_training_curves(history)

    if not RUN_FINAL_TEST:
        print("RUN_FINAL_TEST=False, stopping before final test.")
        return

    # ------------------------------------------------------------
    # FINAL FULL TEST
    # ------------------------------------------------------------
    print("\nRunning FULL TEST with best EMA model...")
    (
        mean_psnr,
        std_psnr,
        mean_ssim,
        std_ssim,
        mean_lpips,
        std_lpips,
        inference_ms,
        visual_results,
    ) = evaluate_test(ema_model, gan_generator, schedule, test_loader)

    save_result_grid(
        visual_results,
        "results_grid.png",
        "Hybrid Pix2Pix + Diffusion Test Results",
        8,
    )

    best_cases = sorted(visual_results, key=lambda x: x["psnr"], reverse=True)
    failure_cases = sorted(visual_results, key=lambda x: x["psnr"])

    save_result_grid(best_cases, "best_cases.png", "Best Fusion Cases", 4)
    save_result_grid(failure_cases, "failure_cases.png", "Fusion Failure Cases", 4)

    total_hours = (time.time() - total_training_start) / 3600.0

    print("\n" + "=" * 80)
    print("HYBRID PIX2PIX + DIFFUSION FUSION COMPLETE")
    print("=" * 80)
    print("Best epoch:", best_epoch)
    print(f"Best validation PSNR: {best_psnr:.4f} dB")
    print()
    print(f"FUSION TEST PSNR    : {mean_psnr:.4f} +/- {std_psnr:.4f} dB")
    print()
    print(f"FUSION TEST SSIM    : {mean_ssim:.4f} +/- {std_ssim:.4f}")
    print()
    if mean_lpips is not None:
        print(f"FUSION TEST LPIPS   : {mean_lpips:.4f} +/- {std_lpips:.4f}")
        print()
    print(f"FUSION inference    : {inference_ms:.2f} ms/image")
    print(f"Training runtime    : {total_hours:.2f} hours")
    print("Results folder      :", OUTPUT_ROOT)
    print("=" * 80)


if __name__ == "__main__":
    main()
