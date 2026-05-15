import os
import json
import matplotlib.pyplot as plt


# =========================================================
# PATHS
# =========================================================

history_path = os.path.join(
    "outputs",
    "checkpoints",
    "training_history.json"
)

save_dir = os.path.join(
    "outputs",
    "figures"
)

os.makedirs(
    save_dir,
    exist_ok=True
)


# =========================================================
# LOAD TRAINING HISTORY
# =========================================================

with open(history_path, "r") as f:

    history = json.load(f)


# =========================================================
# EXTRACT METRICS
# =========================================================

train_loss = history["train_loss"]

val_loss = history["val_loss"]

train_dice = history["train_dice"]

val_dice = history["val_dice"]

train_iou = history["train_iou"]

val_iou = history["val_iou"]


epochs = range(
    1,
    len(train_loss) + 1
)


# =========================================================
# LOSS GRAPH
# =========================================================

plt.figure(figsize=(8,6))

plt.plot(
    epochs,
    train_loss,
    label="Train Loss",
    linewidth=2
)

plt.plot(
    epochs,
    val_loss,
    label="Validation Loss",
    linewidth=2
)

best_epoch_loss = val_loss.index(min(val_loss)) + 1

best_loss = min(val_loss)

plt.scatter(
    best_epoch_loss,
    best_loss,
    s=100,
    label=f"Best Model Epoch ({best_epoch_loss})"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Loss vs Epoch")

plt.legend()

plt.grid(True)

loss_path = os.path.join(
    save_dir,
    "loss_vs_epoch.png"
)

plt.savefig(loss_path)

plt.close()

print(f"Saved: {loss_path}")


# =========================================================
# DICE GRAPH
# =========================================================

plt.figure(figsize=(8,6))

plt.plot(
    epochs,
    train_dice,
    label="Train Dice",
    linewidth=2
)

plt.plot(
    epochs,
    val_dice,
    label="Validation Dice",
    linewidth=2
)

best_epoch_dice = val_dice.index(max(val_dice)) + 1

best_dice = max(val_dice)

plt.scatter(
    best_epoch_dice,
    best_dice,
    s=100,
    label=f"Best Model Epoch ({best_epoch_dice})"
)

plt.xlabel("Epoch")

plt.ylabel("Dice Score")

plt.title("Dice Score vs Epoch")

plt.legend()

plt.grid(True)

dice_path = os.path.join(
    save_dir,
    "dice_vs_epoch.png"
)

plt.savefig(dice_path)

plt.close()

print(f"Saved: {dice_path}")


# =========================================================
# IOU GRAPH
# =========================================================

plt.figure(figsize=(8,6))

plt.plot(
    epochs,
    train_iou,
    label="Train IoU",
    linewidth=2
)

plt.plot(
    epochs,
    val_iou,
    label="Validation IoU",
    linewidth=2
)

best_epoch_iou = val_iou.index(max(val_iou)) + 1

best_iou = max(val_iou)

plt.scatter(
    best_epoch_iou,
    best_iou,
    s=100,
    label=f"Best Model Epoch ({best_epoch_iou})"
)

plt.xlabel("Epoch")

plt.ylabel("IoU")

plt.title("IoU vs Epoch")

plt.legend()

plt.grid(True)

iou_path = os.path.join(
    save_dir,
    "iou_vs_epoch.png"
)

plt.savefig(iou_path)

plt.close()

print(f"Saved: {iou_path}")


# =========================================================
# BEST MODEL SUMMARY
# =========================================================

print("\n=================================================")

print("BEST MODEL PERFORMANCE")

print("=================================================\n")

print(f"Best Validation Dice : {best_dice:.4f}")

print(f"Best Validation IoU  : {best_iou:.4f}")

print(f"Lowest Validation Loss : {best_loss:.4f}")

print(f"Best Model Epoch : {best_epoch_dice}")