import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset.dataset_loader import BrainTumorDataset
from models.multimodal_unet import MultimodalUNet3D

from utils.config import *


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device: {device}")


# =========================================================
# LOAD DATASET
# =========================================================

val_dataset = BrainTumorDataset(
    image_dir=VAL_IMAGE_DIR,
    mask_dir=VAL_MASK_DIR,
    text_dir=TEXT_DIR
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False
)
batch = next(iter(val_loader))

print(batch.keys())


# =========================================================
# LOAD MODEL
# =========================================================

model = MultimodalUNet3D().to(device)

checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(
    checkpoint,
    strict=False
)

model.eval()


# =========================================================
# OUTPUT DIRECTORY
# =========================================================

save_dir = "outputs/predictions"

os.makedirs(save_dir, exist_ok=True)


# =========================================================
# INFERENCE
# =========================================================

# =========================================================
# INFERENCE
# =========================================================

with torch.no_grad():

    for idx, batch in enumerate(val_loader):

        image = batch["image"].to(device)

        mask = batch["mask"].to(device)

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)


        # =================================================
        # FORWARD
        # =================================================

        output = model(
            image,
            input_ids,
            attention_mask
        )


        # =================================================
        # PREDICTION
        # =================================================

        prediction = torch.sigmoid(output)

        prediction = (prediction > 0.5).float()


        # =================================================
        # NUMPY
        # =================================================

        image_np = image[0,0].cpu().numpy()

        gt_mask_np = mask[0,0].cpu().numpy()

        pred_mask_np = prediction[0,0].cpu().numpy()


        # =================================================
        # MIDDLE SLICE
        # =================================================

        middle_slice = image_np.shape[0] // 2

        image_slice = image_np[middle_slice]

        gt_slice = gt_mask_np[middle_slice]

        pred_slice = pred_mask_np[middle_slice]


        # =================================================
        # VISUALIZATION
        # =================================================

        fig, ax = plt.subplots(1,3, figsize=(15,5))

        ax[0].imshow(
            image_slice,
            cmap='gray'
        )

        ax[0].set_title("Input MRI")


        ax[1].imshow(
            gt_slice,
            cmap='gray'
        )

        ax[1].set_title("Ground Truth")


        ax[2].imshow(
            pred_slice,
            cmap='gray'
        )

        ax[2].set_title("Predicted Mask")


        for a in ax:
            a.axis('off')


        save_path = os.path.join(
            save_dir,
            f"prediction_{idx}.png"
        )

        plt.savefig(save_path)

        plt.close()

        print(f"Saved: {save_path}")


        # SAVE FIRST 10 ONLY
        if idx == 9:
            break