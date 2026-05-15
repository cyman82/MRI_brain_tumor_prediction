import os
import json

import torch
import torch.optim as optim

from torch.utils.data import (
    DataLoader,
    Subset
)

from tqdm import tqdm

from torch.cuda.amp import (
    autocast,
    GradScaler
)

from dataset.dataset_loader import BrainTumorDataset

from models.multimodal_unet import MultimodalUNet3D

from training.losses import CombinedLoss

from training.metrics import (
    dice_score,
    iou_score,
    precision_score,
    recall_score
)

from utils.config import *


# =========================================================
# TRAIN FUNCTION
# =========================================================

def train_one_epoch(
    loader,
    model,
    optimizer,
    criterion,
    scaler,
    device
):

    model.train()

    epoch_loss = 0.0
    epoch_dice = 0.0
    epoch_iou = 0.0

    loop = tqdm(loader)

    for batch in loop:

        images = batch["image"].to(device)

        masks = batch["mask"].to(device)

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        with autocast():

            outputs = model(
                images,
                input_ids,
                attention_mask
            )

            loss = criterion(
                outputs,
                masks
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        dice = dice_score(
            outputs.detach(),
            masks
        )

        iou = iou_score(
            outputs.detach(),
            masks
        )

        epoch_loss += loss.item()

        epoch_dice += dice

        epoch_iou += iou

        loop.set_postfix(
            loss=loss.item(),
            dice=dice,
            iou=iou
        )

    avg_loss = epoch_loss / len(loader)

    avg_dice = epoch_dice / len(loader)

    avg_iou = epoch_iou / len(loader)

    return avg_loss, avg_dice, avg_iou


# =========================================================
# VALIDATION FUNCTION
# =========================================================

def validate(
    loader,
    model,
    criterion,
    device
):

    model.eval()

    epoch_loss = 0.0
    epoch_dice = 0.0
    epoch_iou = 0.0
    epoch_precision = 0.0
    epoch_recall = 0.0

    with torch.no_grad():

        for batch in tqdm(loader):

            images = batch["image"].to(device)

            masks = batch["mask"].to(device)

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                images,
                input_ids,
                attention_mask
            )

            loss = criterion(
                outputs,
                masks
            )

            dice = dice_score(
                outputs,
                masks
            )

            iou = iou_score(
                outputs,
                masks
            )

            precision = precision_score(
                outputs,
                masks
            )

            recall = recall_score(
                outputs,
                masks
            )

            epoch_loss += loss.item()

            epoch_dice += dice

            epoch_iou += iou

            epoch_precision += precision

            epoch_recall += recall

    avg_loss = epoch_loss / len(loader)

    avg_dice = epoch_dice / len(loader)

    avg_iou = epoch_iou / len(loader)

    avg_precision = epoch_precision / len(loader)

    avg_recall = epoch_recall / len(loader)

    return (
        avg_loss,
        avg_dice,
        avg_iou,
        avg_precision,
        avg_recall
    )


# =========================================================
# MAIN
# =========================================================

def main():

    # =====================================================
    # TRAIN DATASET
    # =====================================================

    full_train_dataset = BrainTumorDataset(
        image_dir=TRAIN_IMAGE_DIR,
        mask_dir=TRAIN_MASK_DIR,
        text_dir=TEXT_DIR
    )

    # =====================================================
    # OPTIONAL DEBUG SUBSET
    # =====================================================

    train_dataset = full_train_dataset
    # =====================================================
    # VALIDATION DATASET
    # =====================================================

    val_dataset = BrainTumorDataset(
        image_dir=VAL_IMAGE_DIR,
        mask_dir=VAL_MASK_DIR,
        text_dir=TEXT_DIR
    )

    # =====================================================
    # DATALOADERS
    # =====================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # =====================================================
    # MODEL
    # =====================================================

    model = MultimodalUNet3D().to(DEVICE)

    # =====================================================
    # LOSS
    # =====================================================

    criterion = CombinedLoss()

    # =====================================================
    # OPTIMIZER
    # =====================================================

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # =====================================================
    # LR SCHEDULER
    # =====================================================

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2
    )

    # =====================================================
    # MIXED PRECISION
    # =====================================================

    scaler = GradScaler()

    # =====================================================
    # EARLY STOPPING
    # =====================================================

    best_dice = 0.0

    patience = 5

    patience_counter = 0

    # =====================================================
    # HISTORY
    # =====================================================

    history = {

        "train_loss": [],
        "train_dice": [],
        "train_iou": [],

        "val_loss": [],
        "val_dice": [],
        "val_iou": [],

        "val_precision": [],
        "val_recall": []
    }

    # =====================================================
    # TRAINING LOOP
    # =====================================================

    for epoch in range(NUM_EPOCHS):

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

        # -------------------------------------------------
        # TRAIN
        # -------------------------------------------------

        train_loss, train_dice, train_iou = (
            train_one_epoch(
                train_loader,
                model,
                optimizer,
                criterion,
                scaler,
                DEVICE
            )
        )

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        (
            val_loss,
            val_dice,
            val_iou,
            val_precision,
            val_recall
        ) = validate(
            val_loader,
            model,
            criterion,
            DEVICE
        )

        # -------------------------------------------------
        # LR SCHEDULER
        # -------------------------------------------------

        scheduler.step(val_dice)

        # -------------------------------------------------
        # STORE HISTORY
        # -------------------------------------------------

        history["train_loss"].append(train_loss)

        history["train_dice"].append(train_dice)

        history["train_iou"].append(train_iou)

        history["val_loss"].append(val_loss)

        history["val_dice"].append(val_dice)

        history["val_iou"].append(val_iou)

        history["val_precision"].append(val_precision)

        history["val_recall"].append(val_recall)

        # -------------------------------------------------
        # PRINT RESULTS
        # -------------------------------------------------

        print("\nTRAIN RESULTS")

        print(f"Loss : {train_loss:.4f}")

        print(f"Dice: {train_dice:.4f}")

        print(f"IoU  : {train_iou:.4f}")

        print("\nVALIDATION RESULTS")

        print(f"Loss      : {val_loss:.4f}")

        print(f"Dice      : {val_dice:.4f}")

        print(f"IoU       : {val_iou:.4f}")

        print(f"Precision : {val_precision:.4f}")

        print(f"Recall    : {val_recall:.4f}")

        # -------------------------------------------------
        # SAVE BEST MODEL
        # -------------------------------------------------

        if val_dice > best_dice:

            best_dice = val_dice

            patience_counter = 0

            best_model_path = os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth"
            )

            torch.save(
                model.state_dict(),
                best_model_path
            )

            print("\nBest Model Saved!")

        else:

            patience_counter += 1

            print(
                f"\nEarly Stopping Counter: "
                f"{patience_counter}/{patience}"
            )

        # -------------------------------------------------
        # EARLY STOPPING
        # -------------------------------------------------

        if patience_counter >= patience:

            print("\nEarly stopping triggered!")

            break

        # -------------------------------------------------
        # SAVE HISTORY
        # -------------------------------------------------

        history_path = os.path.join(
            CHECKPOINT_DIR,
            "training_history.json"
        )

        with open(history_path, "w") as f:

            json.dump(history, f)

    print("\nTraining Complete!")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()