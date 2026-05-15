import torch
import os


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_IMAGE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "FLAIR_BRATS2020_split",
    "train",
    "images"
)

TRAIN_MASK_DIR = os.path.join(
    BASE_DIR,
    "data",
    "FLAIR_BRATS2020_split",
    "train",
    "masks"
)

VAL_IMAGE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "FLAIR_BRATS2020_split",
    "val",
    "images"
)

VAL_MASK_DIR = os.path.join(
    BASE_DIR,
    "data",
    "FLAIR_BRATS2020_split",
    "val",
    "masks"
)

TEXT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "TextBraTSData"
)


# =========================================================
# TRAINING CONFIG
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 1

NUM_EPOCHS = 50

LEARNING_RATE = 5e-5

NUM_WORKERS = 0

PIN_MEMORY = False


# =========================================================
# IMAGE CONFIG
# =========================================================

IMAGE_SIZE = (128, 128, 128)

NUM_CLASSES = 4

INPUT_CHANNELS = 1


# =========================================================
# TEXT CONFIG
# =========================================================

MAX_TEXT_LENGTH = 77

TEXT_EMBED_DIM = 512


# =========================================================
# MODEL CONFIG
# =========================================================

FEATURES = [16, 32, 64, 128]

USE_TEXT = True


# =========================================================
# CHECKPOINTS
# =========================================================

CHECKPOINT_DIR = os.path.join(BASE_DIR, "outputs", "checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)