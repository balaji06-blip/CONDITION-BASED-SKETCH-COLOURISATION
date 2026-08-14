# ================================================================
# CONDITION-BASED SKETCH COLOURISATION
# 100-EPOCH PIX2PIX cGAN BASELINE
#
# Linux / RTX A4000 16 GB
#
# Project structure:
#
# /scratch/bs/project/
# ├── baseline-model.py
# └── archive/
#     └── data/
#         ├── train/
#         └── val/
#
# Dataset image format:
#
# [ COLOUR IMAGE | SKETCH ]
#       LEFT         RIGHT
#
# ================================================================

import os
import csv
import time
import json
import glob
import random
import numpy as np

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn


# ================================================================
# OPTIONAL LPIPS
# ================================================================

try:
    import lpips
    LPIPS_AVAILABLE = True
except Exception:
    LPIPS_AVAILABLE = False


# ================================================================
# 1. PATHS
# ================================================================

# Automatically finds /scratch/bs/project
# because baseline-model.py is inside that folder.

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "archive",
    "data"
)

TRAIN_DIR = os.path.join(
    DATA_ROOT,
    "train"
)

VAL_SOURCE_DIR = os.path.join(
    DATA_ROOT,
    "val"
)


# ================================================================
# 2. OUTPUT FOLDERS
# ================================================================

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "baseline_100epochs_linux"
)

CHECKPOINT_DIR = os.path.join(
    OUTPUT_ROOT,
    "checkpoints"
)

LOG_DIR = os.path.join(
    OUTPUT_ROOT,
    "logs"
)

SAMPLE_DIR = os.path.join(
    OUTPUT_ROOT,
    "samples"
)

FIGURE_DIR = os.path.join(
    OUTPUT_ROOT,
    "figures"
)

EVAL_DIR = os.path.join(
    OUTPUT_ROOT,
    "evaluation"
)

GENERATED_DIR = os.path.join(
    EVAL_DIR,
    "generated"
)

REAL_DIR = os.path.join(
    EVAL_DIR,
    "ground_truth"
)


for folder in [
    OUTPUT_ROOT,
    CHECKPOINT_DIR,
    LOG_DIR,
    SAMPLE_DIR,
    FIGURE_DIR,
    EVAL_DIR,
    GENERATED_DIR,
    REAL_DIR
]:
    os.makedirs(
        folder,
        exist_ok=True
    )


# ================================================================
# 3. CONFIGURATION
# ================================================================

IMAGE_SIZE = 256

HINT_RATE = 0.05

# RTX A4000 16 GB
BATCH_SIZE = 8

# Linux server
NUM_WORKERS = 4

# ------------------------------------------------
# REQUIRED
# ------------------------------------------------

NUM_EPOCHS = 100


LEARNING_RATE = 2e-4

BETA1 = 0.5
BETA2 = 0.999

LAMBDA_L1 = 100.0

NGF = 64
NDF = 64


# Save every 5 epochs
SAVE_EVERY = 5

# Generate preview every 5 epochs
SAMPLE_EVERY = 5

NUM_SAMPLE_IMAGES = 4


SEED = 42


# ================================================================
# 4. DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = (
    DEVICE.type == "cuda"
)


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ================================================================
# 5. RANDOM SEEDS
# ================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ================================================================
# 6. STARTUP INFORMATION
# ================================================================

print("\n" + "=" * 75)

print(
    "CONDITION-BASED SKETCH COLOURISATION"
)

print(
    "100-EPOCH PIX2PIX cGAN BASELINE"
)

print("=" * 75)

print(
    "Project root      :",
    PROJECT_ROOT
)

print(
    "Data root         :",
    DATA_ROOT
)

print(
    "Training folder   :",
    TRAIN_DIR
)

print(
    "Validation source :",
    VAL_SOURCE_DIR
)

print(
    "Output folder     :",
    OUTPUT_ROOT
)

print()

print(
    "Device            :",
    DEVICE
)


if torch.cuda.is_available():

    print(
        "GPU               :",
        torch.cuda.get_device_name(0)
    )

    memory = (
        torch.cuda.get_device_properties(
            0
        ).total_memory
        / (1024 ** 3)
    )

    print(
        f"GPU memory        : "
        f"{memory:.2f} GB"
    )

    print(
        "CUDA version      :",
        torch.version.cuda
    )

else:

    print(
        "WARNING: CUDA NOT AVAILABLE"
    )


print()

print(
    "Epochs            :",
    NUM_EPOCHS
)

print(
    "Batch size        :",
    BATCH_SIZE
)

print(
    "Image size        :",
    IMAGE_SIZE
)

print(
    "Hint rate         :",
    HINT_RATE
)

print(
    "AMP               :",
    USE_AMP
)

print("=" * 75)


# ================================================================
# 7. IMAGE FILE HELPER
# ================================================================

def get_image_files(folder):

    files = []

    for extension in [
        "*.png",
        "*.PNG",
        "*.jpg",
        "*.JPG",
        "*.jpeg",
        "*.JPEG"
    ]:

        files.extend(
            glob.glob(
                os.path.join(
                    folder,
                    extension
                )
            )
        )

    return sorted(
        list(
            set(files)
        )
    )


# ================================================================
# 8. DATASET CHECK
# ================================================================

def check_dataset():

    print(
        "\nChecking dataset..."
    )

    if not os.path.isdir(TRAIN_DIR):

        raise FileNotFoundError(
            f"Training folder not found:\n"
            f"{TRAIN_DIR}"
        )


    if not os.path.isdir(VAL_SOURCE_DIR):

        raise FileNotFoundError(
            f"Validation folder not found:\n"
            f"{VAL_SOURCE_DIR}"
        )


    train_files = get_image_files(
        TRAIN_DIR
    )

    val_files = get_image_files(
        VAL_SOURCE_DIR
    )


    print(
        "Training images   :",
        len(train_files)
    )

    print(
        "Validation images :",
        len(val_files)
    )


    if len(train_files) == 0:

        raise RuntimeError(
            "No images found in train folder."
        )


    if len(val_files) < 4:

        raise RuntimeError(
            "Not enough images in val folder."
        )


    first_path = train_files[0]


    with Image.open(
        first_path
    ) as image:

        print()

        print(
            "Example file :",
            os.path.basename(
                first_path
            )
        )

        print(
            "Image size   :",
            image.size
        )

        print(
            "Image mode   :",
            image.mode
        )


        width, height = image.size


        if width != height * 2:

            print()

            print(
                "WARNING:"
            )

            print(
                "Expected paired images such as "
                "1024 x 512."
            )


    print(
        "\nDataset detected successfully."
    )


    return (
        train_files,
        val_files
    )


# ================================================================
# 9. NORMALISATION
# ================================================================

def to_tensor_normalised(array):

    array = (
        array / 127.5
        - 1.0
    )


    array = np.transpose(
        array,
        (2, 0, 1)
    )


    return torch.from_numpy(
        array.copy()
    ).float()


def denormalise(tensor):

    image = (
        tensor.detach()
        .cpu()
        .float()
        .numpy()
        + 1.0
    ) * 127.5


    image = np.clip(
        image,
        0,
        255
    ).astype(
        np.uint8
    )


    if image.ndim == 3:

        image = np.transpose(
            image,
            (1, 2, 0)
        )


    elif image.ndim == 4:

        image = np.transpose(
            image,
            (0, 2, 3, 1)
        )


    return image


# ================================================================
# 10. DATASET
# ================================================================

class SketchColourDataset(Dataset):

    def __init__(
        self,
        paths,
        image_size=256,
        hint_rate=0.05,
        augment=False,
        deterministic_hints=False
    ):

        self.paths = paths

        self.image_size = (
            image_size
        )

        self.hint_rate = (
            hint_rate
        )

        self.augment = (
            augment
        )

        self.deterministic_hints = (
            deterministic_hints
        )


    def __len__(self):

        return len(
            self.paths
        )


    def __getitem__(
        self,
        index
    ):

        path = (
            self.paths[index]
        )


        paired = Image.open(
            path
        ).convert(
            "RGB"
        )


        width, height = (
            paired.size
        )


        midpoint = (
            width // 2
        )


        # ========================================================
        # YOUR CONFIRMED DATASET:
        #
        # [ COLOUR IMAGE | SKETCH ]
        #
        # LEFT  = coloured ground truth
        # RIGHT = input sketch
        # ========================================================

        colour = paired.crop(
            (
                0,
                0,
                midpoint,
                height
            )
        )


        sketch = paired.crop(
            (
                midpoint,
                0,
                width,
                height
            )
        )


        paired.close()


        colour = colour.resize(
            (
                self.image_size,
                self.image_size
            ),
            Image.LANCZOS
        )


        sketch = sketch.resize(
            (
                self.image_size,
                self.image_size
            ),
            Image.LANCZOS
        )


        # ========================================================
        # AUGMENTATION
        # ========================================================

        if (
            self.augment
            and
            random.random() > 0.5
        ):

            sketch = TF.hflip(
                sketch
            )

            colour = TF.hflip(
                colour
            )


        sketch_np = np.array(
            sketch,
            dtype=np.float32
        )


        colour_np = np.array(
            colour,
            dtype=np.float32
        )


        # ========================================================
        # HINT MAP
        # ========================================================

        H, W, _ = (
            colour_np.shape
        )


        total_pixels = (
            H * W
        )


        number_hints = max(
            1,
            int(
                total_pixels
                * self.hint_rate
            )
        )


        if self.deterministic_hints:

            rng = np.random.default_rng(
                SEED + index
            )


            indices = rng.choice(
                total_pixels,
                size=number_hints,
                replace=False
            )

        else:

            indices = np.random.choice(
                total_pixels,
                size=number_hints,
                replace=False
            )


        rows = (
            indices // W
        )


        columns = (
            indices % W
        )


        hint_map = np.zeros_like(
            colour_np
        )


        hint_mask = np.zeros(
            (H, W),
            dtype=np.float32
        )


        hint_map[
            rows,
            columns,
            :
        ] = colour_np[
            rows,
            columns,
            :
        ]


        hint_mask[
            rows,
            columns
        ] = 1.0


        # ========================================================
        # CONVERT TO TENSORS
        # ========================================================

        sketch_tensor = (
            to_tensor_normalised(
                sketch_np
            )
        )


        colour_tensor = (
            to_tensor_normalised(
                colour_np
            )
        )


        hint_tensor = (
            to_tensor_normalised(
                hint_map
            )
        )


        mask_tensor = (
            torch.from_numpy(
                hint_mask[
                    np.newaxis,
                    :,
                    :
                ]
            ).float()
        )


        # ========================================================
        # GENERATOR INPUT
        #
        # sketch       : 3
        # colour hints : 3
        # hint mask    : 1
        #
        # TOTAL        : 7 channels
        # ========================================================

        model_input = torch.cat(
            [
                sketch_tensor,
                hint_tensor,
                mask_tensor
            ],
            dim=0
        )


        return {

            "input":
                model_input,

            "target":
                colour_tensor,

            "sketch":
                sketch_tensor,

            "hint_map":
                hint_tensor,

            "hint_mask":
                mask_tensor,

            "name":
                os.path.basename(
                    path
                )

        }


# ================================================================
# 11. ENCODER BLOCK
# ================================================================

class EncoderBlock(nn.Module):

    def __init__(
        self,
        input_channels,
        output_channels,
        use_batchnorm=True
    ):

        super().__init__()


        layers = [

            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not use_batchnorm
            )

        ]


        if use_batchnorm:

            layers.append(
                nn.BatchNorm2d(
                    output_channels
                )
            )


        layers.append(
            nn.LeakyReLU(
                0.2,
                inplace=True
            )
        )


        self.block = nn.Sequential(
            *layers
        )


    def forward(self, x):

        return self.block(
            x
        )


# ================================================================
# 12. DECODER BLOCK
# ================================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        input_channels,
        output_channels,
        dropout=False
    ):

        super().__init__()


        layers = [

            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                output_channels
            )

        ]


        if dropout:

            layers.append(
                nn.Dropout(
                    0.5
                )
            )


        layers.append(
            nn.ReLU(
                inplace=True
            )
        )


        self.block = nn.Sequential(
            *layers
        )


    def forward(self, x):

        return self.block(
            x
        )


# ================================================================
# 13. U-NET GENERATOR
# ================================================================

class UNetGenerator(nn.Module):

    def __init__(
        self,
        in_channels=7,
        out_channels=3,
        ngf=64
    ):

        super().__init__()


        self.enc1 = EncoderBlock(
            in_channels,
            ngf,
            use_batchnorm=False
        )


        self.enc2 = EncoderBlock(
            ngf,
            ngf * 2
        )


        self.enc3 = EncoderBlock(
            ngf * 2,
            ngf * 4
        )


        self.enc4 = EncoderBlock(
            ngf * 4,
            ngf * 8
        )


        self.enc5 = EncoderBlock(
            ngf * 8,
            ngf * 8
        )


        self.enc6 = EncoderBlock(
            ngf * 8,
            ngf * 8
        )


        self.enc7 = EncoderBlock(
            ngf * 8,
            ngf * 8
        )


        self.enc8 = EncoderBlock(
            ngf * 8,
            ngf * 8
        )


        self.dec1 = DecoderBlock(
            ngf * 8,
            ngf * 8,
            dropout=True
        )


        self.dec2 = DecoderBlock(
            ngf * 16,
            ngf * 8,
            dropout=True
        )


        self.dec3 = DecoderBlock(
            ngf * 16,
            ngf * 8,
            dropout=True
        )


        self.dec4 = DecoderBlock(
            ngf * 16,
            ngf * 8
        )


        self.dec5 = DecoderBlock(
            ngf * 16,
            ngf * 4
        )


        self.dec6 = DecoderBlock(
            ngf * 8,
            ngf * 2
        )


        self.dec7 = DecoderBlock(
            ngf * 4,
            ngf
        )


        self.output_layer = nn.Sequential(

            nn.ConvTranspose2d(
                ngf * 2,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Tanh()

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


        d1 = self.dec1(
            e8
        )


        d2 = self.dec2(
            torch.cat(
                [
                    d1,
                    e7
                ],
                dim=1
            )
        )


        d3 = self.dec3(
            torch.cat(
                [
                    d2,
                    e6
                ],
                dim=1
            )
        )


        d4 = self.dec4(
            torch.cat(
                [
                    d3,
                    e5
                ],
                dim=1
            )
        )


        d5 = self.dec5(
            torch.cat(
                [
                    d4,
                    e4
                ],
                dim=1
            )
        )


        d6 = self.dec6(
            torch.cat(
                [
                    d5,
                    e3
                ],
                dim=1
            )
        )


        d7 = self.dec7(
            torch.cat(
                [
                    d6,
                    e2
                ],
                dim=1
            )
        )


        return self.output_layer(
            torch.cat(
                [
                    d7,
                    e1
                ],
                dim=1
            )
        )


# ================================================================
# 14. DISCRIMINATOR BLOCK
# ================================================================

class DiscriminatorBlock(nn.Module):

    def __init__(
        self,
        input_channels,
        output_channels,
        stride=2,
        use_batchnorm=True
    ):

        super().__init__()


        layers = [

            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=not use_batchnorm
            )

        ]


        if use_batchnorm:

            layers.append(
                nn.BatchNorm2d(
                    output_channels
                )
            )


        layers.append(
            nn.LeakyReLU(
                0.2,
                inplace=True
            )
        )


        self.block = nn.Sequential(
            *layers
        )


    def forward(self, x):

        return self.block(
            x
        )


# ================================================================
# 15. PATCHGAN DISCRIMINATOR
# ================================================================

class PatchGANDiscriminator(nn.Module):

    def __init__(
        self,
        input_channels=6,
        ndf=64
    ):

        super().__init__()


        self.layer1 = DiscriminatorBlock(
            input_channels,
            ndf,
            stride=2,
            use_batchnorm=False
        )


        self.layer2 = DiscriminatorBlock(
            ndf,
            ndf * 2,
            stride=2
        )


        self.layer3 = DiscriminatorBlock(
            ndf * 2,
            ndf * 4,
            stride=2
        )


        self.layer4 = DiscriminatorBlock(
            ndf * 4,
            ndf * 8,
            stride=1
        )


        self.output = nn.Conv2d(
            ndf * 8,
            1,
            kernel_size=4,
            stride=1,
            padding=1
        )


    def forward(
        self,
        sketch,
        colour
    ):

        x = torch.cat(
            [
                sketch,
                colour
            ],
            dim=1
        )


        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)


        return self.output(
            x
        )


# ================================================================
# 16. INITIALISE WEIGHTS
# ================================================================

def initialise_weights(model):

    for module in model.modules():

        if isinstance(
            module,
            (
                nn.Conv2d,
                nn.ConvTranspose2d
            )
        ):

            nn.init.normal_(
                module.weight,
                0.0,
                0.02
            )


            if module.bias is not None:

                nn.init.constant_(
                    module.bias,
                    0
                )


        elif isinstance(
            module,
            nn.BatchNorm2d
        ):

            nn.init.normal_(
                module.weight,
                1.0,
                0.02
            )


            nn.init.constant_(
                module.bias,
                0
            )


# ================================================================
# 17. SAVE DATASET CHECK
# ================================================================

def save_dataset_check(dataset):

    sample = dataset[0]


    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            12,
            4
        )
    )


    axes[0].imshow(
        denormalise(
            sample["sketch"]
        )
    )

    axes[0].set_title(
        "INPUT SKETCH"
    )

    axes[0].axis(
        "off"
    )


    axes[1].imshow(
        denormalise(
            sample["hint_map"]
        )
    )

    axes[1].set_title(
        "COLOUR HINTS"
    )

    axes[1].axis(
        "off"
    )


    axes[2].imshow(
        denormalise(
            sample["target"]
        )
    )

    axes[2].set_title(
        "GROUND TRUTH COLOUR"
    )

    axes[2].axis(
        "off"
    )


    plt.tight_layout()


    path = os.path.join(
        FIGURE_DIR,
        "dataset_pair_check.png"
    )


    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()


    return path


# ================================================================
# 18. TRAIN ONE EPOCH
# ================================================================

def train_one_epoch(
    generator,
    discriminator,
    loader,
    optimizer_g,
    optimizer_d,
    gan_loss,
    l1_loss,
    scaler
):

    generator.train()

    discriminator.train()


    total_d = 0.0

    total_g_adv = 0.0

    total_g_l1 = 0.0

    total_g = 0.0


    progress = tqdm(
        loader,
        desc="Training",
        leave=True
    )


    for batch in progress:

        model_input = (
            batch["input"]
            .to(
                DEVICE,
                non_blocking=True
            )
        )


        target = (
            batch["target"]
            .to(
                DEVICE,
                non_blocking=True
            )
        )


        sketch = (
            batch["sketch"]
            .to(
                DEVICE,
                non_blocking=True
            )
        )


        # ========================================================
        # GENERATOR FORWARD
        # ========================================================

        with torch.cuda.amp.autocast(
            enabled=USE_AMP
        ):

            generated = generator(
                model_input
            )


        # ========================================================
        # DISCRIMINATOR
        # ========================================================

        for parameter in discriminator.parameters():

            parameter.requires_grad = True


        optimizer_d.zero_grad(
            set_to_none=True
        )


        with torch.cuda.amp.autocast(
            enabled=USE_AMP
        ):

            real_prediction = discriminator(
                sketch,
                target
            )


            fake_prediction = discriminator(
                sketch,
                generated.detach()
            )


            real_labels = torch.ones_like(
                real_prediction
            )


            fake_labels = torch.zeros_like(
                fake_prediction
            )


            d_real = gan_loss(
                real_prediction,
                real_labels
            )


            d_fake = gan_loss(
                fake_prediction,
                fake_labels
            )


            d_loss = (
                d_real
                +
                d_fake
            ) * 0.5


        scaler.scale(
            d_loss
        ).backward()


        scaler.step(
            optimizer_d
        )


        # ========================================================
        # GENERATOR
        # ========================================================

        for parameter in discriminator.parameters():

            parameter.requires_grad = False


        optimizer_g.zero_grad(
            set_to_none=True
        )


        with torch.cuda.amp.autocast(
            enabled=USE_AMP
        ):

            fake_prediction_for_g = discriminator(
                sketch,
                generated
            )


            generator_labels = torch.ones_like(
                fake_prediction_for_g
            )


            g_adv = gan_loss(
                fake_prediction_for_g,
                generator_labels
            )


            g_l1 = l1_loss(
                generated,
                target
            )


            g_total = (
                g_adv
                +
                LAMBDA_L1
                *
                g_l1
            )


        scaler.scale(
            g_total
        ).backward()


        scaler.step(
            optimizer_g
        )


        scaler.update()


        total_d += (
            d_loss.item()
        )

        total_g_adv += (
            g_adv.item()
        )

        total_g_l1 += (
            g_l1.item()
        )

        total_g += (
            g_total.item()
        )


        progress.set_postfix({

            "D":
                f"{d_loss.item():.3f}",

            "G_ADV":
                f"{g_adv.item():.3f}",

            "G_L1":
                f"{g_l1.item():.4f}"

        })


    number_batches = len(
        loader
    )


    return (

        total_d
        /
        number_batches,

        total_g_adv
        /
        number_batches,

        total_g_l1
        /
        number_batches,

        total_g
        /
        number_batches

    )


# ================================================================
# 19. VALIDATE PSNR + SSIM
# ================================================================

@torch.no_grad()
def validate(
    generator,
    loader
):

    generator.eval()


    psnr_scores = []

    ssim_scores = []


    for batch in tqdm(
        loader,
        desc="Validation",
        leave=False
    ):

        model_input = (
            batch["input"]
            .to(DEVICE)
        )


        target = (
            batch["target"]
            .to(DEVICE)
        )


        with torch.cuda.amp.autocast(
            enabled=USE_AMP
        ):

            generated = generator(
                model_input
            )


        for i in range(
            generated.size(0)
        ):

            generated_np = (
                denormalise(
                    generated[i]
                )
            )


            target_np = (
                denormalise(
                    target[i]
                )
            )


            psnr_value = psnr_fn(
                target_np,
                generated_np,
                data_range=255
            )


            ssim_value = ssim_fn(
                target_np,
                generated_np,
                data_range=255,
                channel_axis=2
            )


            psnr_scores.append(
                psnr_value
            )


            ssim_scores.append(
                ssim_value
            )


    generator.train()


    return (
        float(
            np.mean(
                psnr_scores
            )
        ),
        float(
            np.mean(
                ssim_scores
            )
        )
    )


# ================================================================
# 20. SAVE SAMPLE GRID
# ================================================================

@torch.no_grad()
def save_sample_grid(
    generator,
    loader,
    epoch
):

    generator.eval()


    batch = next(
        iter(loader)
    )


    model_input = (
        batch["input"]
        .to(DEVICE)
    )


    with torch.cuda.amp.autocast(
        enabled=USE_AMP
    ):

        generated = generator(
            model_input
        )


    number_images = min(
        NUM_SAMPLE_IMAGES,
        generated.size(0)
    )


    rows = []


    for i in range(
        number_images
    ):

        row = np.concatenate(
            [
                denormalise(
                    batch["sketch"][i]
                ),

                denormalise(
                    batch["hint_map"][i]
                ),

                denormalise(
                    generated[i]
                ),

                denormalise(
                    batch["target"][i]
                )
            ],
            axis=1
        )


        rows.append(
            row
        )


    grid = np.concatenate(
        rows,
        axis=0
    )


    path = os.path.join(
        SAMPLE_DIR,
        f"epoch_{epoch:03d}.png"
    )


    Image.fromarray(
        grid
    ).save(
        path
    )


    return path


# ================================================================
# 21. CHECKPOINT
# ================================================================

def save_checkpoint(
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    scaler,
    epoch,
    history,
    best_psnr,
    best_epoch,
    filename
):

    path = os.path.join(
        CHECKPOINT_DIR,
        filename
    )


    torch.save(
        {
            "epoch":
                epoch,

            "generator":
                generator.state_dict(),

            "discriminator":
                discriminator.state_dict(),

            "optimizer_g":
                optimizer_g.state_dict(),

            "optimizer_d":
                optimizer_d.state_dict(),

            "scaler":
                scaler.state_dict(),

            "history":
                history,

            "best_psnr":
                best_psnr,

            "best_epoch":
                best_epoch
        },
        path
    )


    return path


# ================================================================
# 22. PLOT TRAINING CURVES
# ================================================================

def save_training_curves(
    history
):

    epochs = [
        item["epoch"]
        for item in history
    ]


    # ------------------------------------------------
    # LOSS CURVE
    # ------------------------------------------------

    plt.figure(
        figsize=(
            10,
            6
        )
    )


    plt.plot(
        epochs,
        [
            item["d_loss"]
            for item in history
        ],
        label="Discriminator"
    )


    plt.plot(
        epochs,
        [
            item["g_l1"]
            for item in history
        ],
        label="Generator L1"
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "cGAN Training Loss"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()


    loss_path = os.path.join(
        FIGURE_DIR,
        "training_losses.png"
    )


    plt.savefig(
        loss_path,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()


    # ------------------------------------------------
    # PSNR
    # ------------------------------------------------

    plt.figure(
        figsize=(
            10,
            6
        )
    )


    plt.plot(
        epochs,
        [
            item["val_psnr"]
            for item in history
        ]
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Validation PSNR (dB)"
    )

    plt.title(
        "Validation PSNR"
    )

    plt.grid(
        alpha=0.3
    )


    psnr_path = os.path.join(
        FIGURE_DIR,
        "validation_psnr.png"
    )


    plt.savefig(
        psnr_path,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()


    # ------------------------------------------------
    # SSIM
    # ------------------------------------------------

    plt.figure(
        figsize=(
            10,
            6
        )
    )


    plt.plot(
        epochs,
        [
            item["val_ssim"]
            for item in history
        ]
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Validation SSIM"
    )

    plt.title(
        "Validation SSIM"
    )

    plt.grid(
        alpha=0.3
    )


    ssim_path = os.path.join(
        FIGURE_DIR,
        "validation_ssim.png"
    )


    plt.savefig(
        ssim_path,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()


    return (
        loss_path,
        psnr_path,
        ssim_path
    )


# ================================================================
# 23. FINAL TEST EVALUATION
# ================================================================

@torch.no_grad()
def evaluate_test(
    generator,
    loader
):

    generator.eval()


    # ------------------------------------------------
    # LPIPS
    # ------------------------------------------------

    perceptual_metric = None


    if LPIPS_AVAILABLE:

        try:

            perceptual_metric = (
                lpips.LPIPS(
                    net="alex"
                )
                .to(DEVICE)
            )


            perceptual_metric.eval()


            print(
                "LPIPS enabled."
            )

        except Exception as error:

            print(
                "LPIPS could not be initialized:"
            )

            print(
                error
            )

            perceptual_metric = None

    else:

        print(
            "LPIPS package not installed."
        )


    psnr_scores = []

    ssim_scores = []

    lpips_scores = []

    inference_times = []

    records = []


    visual_results = []


    for batch in tqdm(
        loader,
        desc="Final test evaluation"
    ):

        model_input = (
            batch["input"]
            .to(DEVICE)
        )


        target = (
            batch["target"]
            .to(DEVICE)
        )


        if DEVICE.type == "cuda":

            torch.cuda.synchronize()


        start = time.time()


        with torch.cuda.amp.autocast(
            enabled=USE_AMP
        ):

            generated = generator(
                model_input
            )


        if DEVICE.type == "cuda":

            torch.cuda.synchronize()


        elapsed = (
            time.time()
            -
            start
        )


        inference_times.append(
            elapsed
            /
            generated.size(0)
        )


        for i in range(
            generated.size(0)
        ):

            generated_np = (
                denormalise(
                    generated[i]
                )
            )


            target_np = (
                denormalise(
                    target[i]
                )
            )


            psnr_value = psnr_fn(
                target_np,
                generated_np,
                data_range=255
            )


            ssim_value = ssim_fn(
                target_np,
                generated_np,
                data_range=255,
                channel_axis=2
            )


            lpips_value = None


            if perceptual_metric is not None:

                lpips_value = float(
                    perceptual_metric(
                        generated[
                            i:i+1
                        ].float(),
                        target[
                            i:i+1
                        ].float()
                    ).item()
                )


                lpips_scores.append(
                    lpips_value
                )


            psnr_scores.append(
                psnr_value
            )


            ssim_scores.append(
                ssim_value
            )


            name = (
                batch["name"][i]
            )


            Image.fromarray(
                generated_np
            ).save(
                os.path.join(
                    GENERATED_DIR,
                    name
                )
            )


            Image.fromarray(
                target_np
            ).save(
                os.path.join(
                    REAL_DIR,
                    name
                )
            )


            record = {
                "name":
                    name,
                "psnr":
                    psnr_value,
                "ssim":
                    ssim_value,
                "lpips":
                    lpips_value
            }


            records.append(
                record
            )


            if len(
                visual_results
            ) < 40:

                visual_results.append(
                    {
                        "name":
                            name,

                        "sketch":
                            denormalise(
                                batch["sketch"][i]
                            ),

                        "hint":
                            denormalise(
                                batch["hint_map"][i]
                            ),

                        "generated":
                            generated_np,

                        "target":
                            target_np,

                        "psnr":
                            psnr_value,

                        "ssim":
                            ssim_value
                    }
                )


    mean_psnr = float(
        np.mean(
            psnr_scores
        )
    )


    std_psnr = float(
        np.std(
            psnr_scores
        )
    )


    mean_ssim = float(
        np.mean(
            ssim_scores
        )
    )


    std_ssim = float(
        np.std(
            ssim_scores
        )
    )


    if lpips_scores:

        mean_lpips = float(
            np.mean(
                lpips_scores
            )
        )

        std_lpips = float(
            np.std(
                lpips_scores
            )
        )

    else:

        mean_lpips = None
        std_lpips = None


    average_inference_ms = (
        float(
            np.mean(
                inference_times
            )
        )
        *
        1000
    )


    # ============================================================
    # CSV
    # ============================================================

    metrics_path = os.path.join(
        EVAL_DIR,
        "metrics_test.csv"
    )


    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow(
            [
                "name",
                "psnr",
                "ssim",
                "lpips"
            ]
        )


        for record in records:

            writer.writerow(
                [
                    record["name"],
                    f"{record['psnr']:.4f}",
                    f"{record['ssim']:.4f}",
                    (
                        f"{record['lpips']:.4f}"
                        if record["lpips"] is not None
                        else "NA"
                    )
                ]
            )


    # ============================================================
    # TEXT SUMMARY
    # ============================================================

    summary_path = os.path.join(
        EVAL_DIR,
        "summary_test.txt"
    )


    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "Model: Pix2Pix cGAN Baseline\n"
        )

        file.write(
            f"Training epochs: {NUM_EPOCHS}\n"
        )

        file.write(
            f"Mean PSNR: {mean_psnr:.4f} dB\n"
        )

        file.write(
            f"PSNR Std: {std_psnr:.4f} dB\n"
        )

        file.write(
            f"Mean SSIM: {mean_ssim:.4f}\n"
        )

        file.write(
            f"SSIM Std: {std_ssim:.4f}\n"
        )


        if mean_lpips is not None:

            file.write(
                f"Mean LPIPS: {mean_lpips:.4f}\n"
            )

            file.write(
                f"LPIPS Std: {std_lpips:.4f}\n"
            )


        file.write(
            f"Inference: "
            f"{average_inference_ms:.2f} ms/image\n"
        )


    # ============================================================
    # JSON SUMMARY
    # ============================================================

    json_path = os.path.join(
        EVAL_DIR,
        "summary_test.json"
    )


    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "model":
                    "GAN",

                "epochs":
                    NUM_EPOCHS,

                "psnr_mean":
                    mean_psnr,

                "psnr_std":
                    std_psnr,

                "ssim_mean":
                    mean_ssim,

                "ssim_std":
                    std_ssim,

                "lpips_mean":
                    mean_lpips,

                "lpips_std":
                    std_lpips,

                "infer_ms":
                    average_inference_ms
            },
            file,
            indent=2
        )


    return (
        mean_psnr,
        std_psnr,
        mean_ssim,
        std_ssim,
        mean_lpips,
        std_lpips,
        average_inference_ms,
        visual_results
    )


# ================================================================
# 24. SAVE RESULT GRID
# ================================================================

def save_result_grid(
    results,
    filename,
    title,
    number=8
):

    results = results[
        :min(
            number,
            len(results)
        )
    ]


    if len(results) == 0:

        return None


    rows = len(
        results
    )


    fig, axes = plt.subplots(
        rows,
        4,
        figsize=(
            16,
            rows * 4
        )
    )


    if rows == 1:

        axes = np.expand_dims(
            axes,
            axis=0
        )


    titles = [
        "Sketch",
        "Colour Hints",
        "Generated",
        "Ground Truth"
    ]


    for column, text in enumerate(
        titles
    ):

        axes[
            0,
            column
        ].set_title(
            text,
            fontweight="bold"
        )


    for row, result in enumerate(
        results
    ):

        images = [
            result["sketch"],
            result["hint"],
            result["generated"],
            result["target"]
        ]


        for column, image in enumerate(
            images
        ):

            axes[
                row,
                column
            ].imshow(
                image
            )

            axes[
                row,
                column
            ].axis(
                "off"
            )


        axes[
            row,
            0
        ].set_ylabel(
            f"PSNR {result['psnr']:.2f}\n"
            f"SSIM {result['ssim']:.3f}"
        )


    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold"
    )


    plt.tight_layout()


    path = os.path.join(
        FIGURE_DIR,
        filename
    )


    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()


    return path


# ================================================================
# 25. MAIN
# ================================================================

def main():

    # ============================================================
    # CHECK DATA
    # ============================================================

    train_files, val_source_files = (
        check_dataset()
    )


    # ============================================================
    # SPLIT EXISTING VAL INTO VALIDATION + TEST
    #
    # Example:
    #
    # original dataset:
    # train ≈ 80%
    # val   ≈ 20%
    #
    # We split val into:
    #
    # validation ≈ 10%
    # test       ≈ 10%
    # ============================================================

    shuffled_val = (
        val_source_files.copy()
    )


    random.Random(
        SEED
    ).shuffle(
        shuffled_val
    )


    midpoint = (
        len(shuffled_val)
        //
        2
    )


    val_files = (
        shuffled_val[
            :midpoint
        ]
    )


    test_files = (
        shuffled_val[
            midpoint:
        ]
    )


    print()

    print("=" * 75)

    print(
        "DATASET SPLIT"
    )

    print("=" * 75)

    print(
        "Training   :",
        len(train_files)
    )

    print(
        "Validation :",
        len(val_files)
    )

    print(
        "Test       :",
        len(test_files)
    )

    print("=" * 75)


    # ============================================================
    # DATASETS
    # ============================================================

    train_dataset = (
        SketchColourDataset(
            train_files,
            image_size=IMAGE_SIZE,
            hint_rate=HINT_RATE,
            augment=True,
            deterministic_hints=False
        )
    )


    val_dataset = (
        SketchColourDataset(
            val_files,
            image_size=IMAGE_SIZE,
            hint_rate=HINT_RATE,
            augment=False,
            deterministic_hints=True
        )
    )


    test_dataset = (
        SketchColourDataset(
            test_files,
            image_size=IMAGE_SIZE,
            hint_rate=HINT_RATE,
            augment=False,
            deterministic_hints=True
        )
    )


    # ============================================================
    # DATASET SANITY CHECK
    # ============================================================

    pair_check = (
        save_dataset_check(
            train_dataset
        )
    )


    print()

    print(
        "Dataset check saved:"
    )

    print(
        pair_check
    )

    print()

    print(
        "It MUST show:"
    )

    print(
        "INPUT SKETCH | COLOUR HINTS | "
        "GROUND TRUTH COLOUR"
    )


    # ============================================================
    # DATALOADERS
    # ============================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        persistent_workers=(
            NUM_WORKERS > 0
        )
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        persistent_workers=(
            NUM_WORKERS > 0
        )
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_AMP,
        persistent_workers=(
            NUM_WORKERS > 0
        )
    )


    # ============================================================
    # MODELS
    # ============================================================

    generator = UNetGenerator(
        in_channels=7,
        out_channels=3,
        ngf=NGF
    ).to(
        DEVICE
    )


    discriminator = PatchGANDiscriminator(
        input_channels=6,
        ndf=NDF
    ).to(
        DEVICE
    )


    # ============================================================
    # OPTIMIZERS
    # ============================================================

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(
            BETA1,
            BETA2
        )
    )


    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=(
            BETA1,
            BETA2
        )
    )


    gan_loss = nn.BCEWithLogitsLoss()

    l1_loss = nn.L1Loss()


    scaler = torch.cuda.amp.GradScaler(
        enabled=USE_AMP
    )


    # ============================================================
    # AUTO RESUME
    # ============================================================

    checkpoint_files = sorted(
        glob.glob(
            os.path.join(
                CHECKPOINT_DIR,
                "epoch_*.pth"
            )
        )
    )


    start_epoch = 1

    history = []

    best_psnr = (
        -float("inf")
    )

    best_epoch = 0


    if checkpoint_files:

        latest = checkpoint_files[
            -1
        ]


        print()

        print(
            "Checkpoint found:"
        )

        print(
            latest
        )


        checkpoint = torch.load(
            latest,
            map_location=DEVICE
        )


        generator.load_state_dict(
            checkpoint[
                "generator"
            ]
        )


        discriminator.load_state_dict(
            checkpoint[
                "discriminator"
            ]
        )


        optimizer_g.load_state_dict(
            checkpoint[
                "optimizer_g"
            ]
        )


        optimizer_d.load_state_dict(
            checkpoint[
                "optimizer_d"
            ]
        )


        if "scaler" in checkpoint:

            scaler.load_state_dict(
                checkpoint[
                    "scaler"
                ]
            )


        history = checkpoint.get(
            "history",
            []
        )


        best_psnr = checkpoint.get(
            "best_psnr",
            -float("inf")
        )


        best_epoch = checkpoint.get(
            "best_epoch",
            0
        )


        start_epoch = (
            checkpoint[
                "epoch"
            ]
            +
            1
        )


        print(
            "Resuming from epoch:",
            start_epoch
        )


    else:

        print()

        print(
            "No checkpoint found."
        )

        print(
            "Starting fresh."
        )


        initialise_weights(
            generator
        )


        initialise_weights(
            discriminator
        )


    print()

    print(
        "Generator parameters:",
        f"{sum(p.numel() for p in generator.parameters()):,}"
    )


    print(
        "Discriminator parameters:",
        f"{sum(p.numel() for p in discriminator.parameters()):,}"
    )


    # ============================================================
    # CSV LOG
    # ============================================================

    csv_path = os.path.join(
        LOG_DIR,
        "training_log.csv"
    )


    if start_epoch == 1:

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(
                file
            )


            writer.writerow(
                [
                    "epoch",
                    "d_loss",
                    "g_adv",
                    "g_l1",
                    "g_total",
                    "val_psnr",
                    "val_ssim",
                    "epoch_seconds"
                ]
            )


    # ============================================================
    # TRAINING
    # ============================================================

    total_training_start = (
        time.time()
    )


    for epoch in range(
        start_epoch,
        NUM_EPOCHS + 1
    ):

        print()

        print("=" * 75)

        print(
            f"EPOCH [{epoch}/{NUM_EPOCHS}]"
        )

        print("=" * 75)


        epoch_start = (
            time.time()
        )


        (
            d_loss,
            g_adv,
            g_l1,
            g_total
        ) = train_one_epoch(
            generator,
            discriminator,
            train_loader,
            optimizer_g,
            optimizer_d,
            gan_loss,
            l1_loss,
            scaler
        )


        # ========================================================
        # VALIDATION
        # ========================================================

        val_psnr, val_ssim = validate(
            generator,
            val_loader
        )


        epoch_seconds = (
            time.time()
            -
            epoch_start
        )


        print()

        print(
            f"D Loss     : "
            f"{d_loss:.4f}"
        )

        print(
            f"G Adv      : "
            f"{g_adv:.4f}"
        )

        print(
            f"G L1       : "
            f"{g_l1:.4f}"
        )

        print(
            f"G Total    : "
            f"{g_total:.4f}"
        )

        print(
            f"Val PSNR   : "
            f"{val_psnr:.4f} dB"
        )

        print(
            f"Val SSIM   : "
            f"{val_ssim:.4f}"
        )

        print(
            f"Epoch time : "
            f"{epoch_seconds:.1f} sec"
        )


        if epoch == 1:

            estimated_hours = (
                epoch_seconds
                *
                NUM_EPOCHS
                /
                3600
            )


            print()

            print(
                f"Approximate 100-epoch "
                f"training time: "
                f"{estimated_hours:.2f} hours"
            )


        record = {
            "epoch":
                epoch,
            "d_loss":
                d_loss,
            "g_adv":
                g_adv,
            "g_l1":
                g_l1,
            "g_total":
                g_total,
            "val_psnr":
                val_psnr,
            "val_ssim":
                val_ssim
        }


        history.append(
            record
        )


        with open(
            csv_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(
                file
            )


            writer.writerow(
                [
                    epoch,
                    d_loss,
                    g_adv,
                    g_l1,
                    g_total,
                    val_psnr,
                    val_ssim,
                    epoch_seconds
                ]
            )


        # ========================================================
        # BEST MODEL
        # ========================================================

        if val_psnr > best_psnr:

            best_psnr = (
                val_psnr
            )

            best_epoch = (
                epoch
            )


            best_path = save_checkpoint(
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                scaler,
                epoch,
                history,
                best_psnr,
                best_epoch,
                "best_model.pth"
            )


            print()

            print(
                "NEW BEST MODEL"
            )

            print(
                f"Best validation PSNR: "
                f"{best_psnr:.4f} dB"
            )

            print(
                "Saved:",
                best_path
            )


        # ========================================================
        # SAMPLE
        # ========================================================

        if (
            epoch == 1
            or
            epoch % SAMPLE_EVERY == 0
        ):

            sample_path = (
                save_sample_grid(
                    generator,
                    val_loader,
                    epoch
                )
            )


            print(
                "Sample saved:",
                sample_path
            )


        # ========================================================
        # CHECKPOINT
        # ========================================================

        if (
            epoch % SAVE_EVERY == 0
            or
            epoch == NUM_EPOCHS
        ):

            checkpoint_path = save_checkpoint(
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                scaler,
                epoch,
                history,
                best_psnr,
                best_epoch,
                f"epoch_{epoch:04d}.pth"
            )


            print(
                "Checkpoint:",
                checkpoint_path
            )


    # ============================================================
    # CURVES
    # ============================================================

    (
        loss_path,
        psnr_curve,
        ssim_curve
    ) = save_training_curves(
        history
    )


    # ============================================================
    # LOAD BEST MODEL
    # ============================================================

    best_model_path = os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth"
    )


    print()

    print("=" * 75)

    print(
        "LOADING BEST MODEL"
    )

    print("=" * 75)


    best_checkpoint = torch.load(
        best_model_path,
        map_location=DEVICE
    )


    generator.load_state_dict(
        best_checkpoint[
            "generator"
        ]
    )


    print(
        "Best epoch:",
        best_epoch
    )

    print(
        f"Best validation PSNR: "
        f"{best_psnr:.4f} dB"
    )


    # ============================================================
    # FINAL TEST
    # ============================================================

    (
        mean_psnr,
        std_psnr,
        mean_ssim,
        std_ssim,
        mean_lpips,
        std_lpips,
        inference_ms,
        visual_results
    ) = evaluate_test(
        generator,
        test_loader
    )


    # ============================================================
    # RESULT GRIDS
    # ============================================================

    save_result_grid(
        visual_results,
        "results_grid.png",
        "Baseline cGAN Test Results",
        8
    )


    best_cases = sorted(
        visual_results,
        key=lambda item: item[
            "psnr"
        ],
        reverse=True
    )


    failure_cases = sorted(
        visual_results,
        key=lambda item: item[
            "psnr"
        ]
    )


    save_result_grid(
        best_cases,
        "best_cases.png",
        "Best Cases",
        4
    )


    save_result_grid(
        failure_cases,
        "failure_cases.png",
        "Failure Cases",
        4
    )


    total_hours = (
        time.time()
        -
        total_training_start
    ) / 3600


    # ============================================================
    # FINAL OUTPUT
    # ============================================================

    print()

    print("=" * 75)

    print(
        "100-EPOCH BASELINE COMPLETE"
    )

    print("=" * 75)


    print(
        "Best epoch:",
        best_epoch
    )


    print(
        f"Best validation PSNR: "
        f"{best_psnr:.4f} dB"
    )


    print()

    print(
        f"TEST PSNR : "
        f"{mean_psnr:.4f} "
        f"+/- "
        f"{std_psnr:.4f} dB"
    )


    print(
        f"TEST SSIM : "
        f"{mean_ssim:.4f} "
        f"+/- "
        f"{std_ssim:.4f}"
    )


    if mean_lpips is not None:

        print(
            f"TEST LPIPS: "
            f"{mean_lpips:.4f} "
            f"+/- "
            f"{std_lpips:.4f}"
        )


    print(
        f"Inference : "
        f"{inference_ms:.2f} ms/image"
    )


    print()

    print(
        f"Total training runtime: "
        f"{total_hours:.2f} hours"
    )


    print()

    print(
        "Best model:"
    )

    print(
        best_model_path
    )


    print()

    print(
        "Results:"
    )

    print(
        OUTPUT_ROOT
    )


    print("=" * 75)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()