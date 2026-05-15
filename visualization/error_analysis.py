import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset.dataset_loader import BrainTumorDataset
from models.multimodal_unet import MultimodalUNet3D

from training.metrics import dice_score

from utils.config import *


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device: {device}")


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
# LOAD DATASET
# =========================================================

dataset = BrainTumorDataset(
    image_dir=VAL_IMAGE_DIR,
    mask_dir=VAL_MASK_DIR,
    text_dir=TEXT_DIR
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)


# =========================================================
# OUTPUT DIRECTORIES
# =========================================================

os.makedirs(
    "outputs/error_analysis",
    exist_ok=True
)

os.makedirs(
    "outputs/error_analysis/failure_cases",
    exist_ok=True
)


# =========================================================
# STORE RESULTS
# =========================================================

all_dice_scores = []

failure_cases = []


# =========================================================
# INFERENCE LOOP
# =========================================================

with torch.no_grad():

    for idx, batch in enumerate(loader):

        image = batch["image"].to(device)

        mask = batch["mask"].to(device)

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)


        # =================================================
        # PREDICTION
        # =================================================

        output = model(
            image,
            input_ids,
            attention_mask
        )

        prediction = torch.sigmoid(output)

        prediction = (prediction > 0.5).float()


        # =================================================
        # DICE SCORE
        # =================================================

        dice = dice_score(
            prediction,
            mask
        )

        all_dice_scores.append(dice)


        # =================================================
        # STORE FAILURE CASES
        # =================================================

        if dice < 0.90:

            failure_cases.append(
                (
                    idx,
                    dice,
                    image.cpu(),
                    mask.cpu(),
                    prediction.cpu()
                )
            )


# =========================================================
# HISTOGRAM
# =========================================================

plt.figure(figsize=(10,6))

plt.hist(
    all_dice_scores,
    bins=15
)

plt.xlabel("Dice Score")

plt.ylabel("Frequency")

plt.title("Dice Score Distribution")

plt.grid(True)

hist_path = (
    "outputs/error_analysis/"
    "dice_distribution.png"
)

plt.savefig(hist_path)

plt.show()

print(f"Saved Histogram: {hist_path}")


# =========================================================
# FAILURE CASE VISUALIZATION
# =========================================================

print("\nWorst Performing Samples:\n")

failure_cases = sorted(
    failure_cases,
    key=lambda x: x[1]
)

for i, (idx, dice, image, mask, prediction) in enumerate(failure_cases[:5]):

    image_np = image[0,0].numpy()

    mask_np = mask[0,0].numpy()

    pred_np = prediction[0,0].numpy()


    # =====================================================
    # MIDDLE SLICE
    # =====================================================

    middle_slice = image_np.shape[0] // 2

    image_slice = image_np[middle_slice]

    mask_slice = mask_np[middle_slice]

    pred_slice = pred_np[middle_slice]


    # =====================================================
    # PLOT
    # =====================================================

    fig, ax = plt.subplots(1,3, figsize=(15,5))

    ax[0].imshow(image_slice, cmap='gray')
    ax[0].set_title("MRI")

    ax[1].imshow(mask_slice, cmap='gray')
    ax[1].set_title("Ground Truth")

    ax[2].imshow(pred_slice, cmap='gray')
    ax[2].set_title(
        f"Prediction\nDice={dice:.4f}"
    )

    for a in ax:
        a.axis('off')


    save_path = (
        f"outputs/error_analysis/"
        f"failure_cases/failure_{idx}.png"
    )

    plt.savefig(save_path)

    plt.close()

    print(f"Saved Failure Case: {save_path}")


# =========================================================
# FINAL STATISTICS
# =========================================================

print("\n=================================================")

print("ERROR ANALYSIS SUMMARY")

print("=================================================\n")

print(f"Total Samples: {len(all_dice_scores)}")

print(
    f"Mean Dice Score: "
    f"{np.mean(all_dice_scores):.4f}"
)

print(
    f"Minimum Dice Score: "
    f"{np.min(all_dice_scores):.4f}"
)

print(
    f"Maximum Dice Score: "
    f"{np.max(all_dice_scores):.4f}"
)

print(
    f"Standard Deviation: "
    f"{np.std(all_dice_scores):.4f}"
)

print(
    f"Number of Failure Cases (<0.90 Dice): "
    f"{len(failure_cases)}"
)