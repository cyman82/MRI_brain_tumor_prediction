import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
# LOAD MODEL
# =========================================================

model = MultimodalUNet3D().to(device)

checkpoint = torch.load(
    os.path.join(CHECKPOINT_DIR, "best_model.pth"),
    map_location=device
)

model.load_state_dict(
    checkpoint,
    strict=False
)

model.eval()


# =========================================================
# DATASET
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


# =========================================================
# OUTPUT DIRECTORY
# =========================================================

os.makedirs(
    "outputs/gradcam",
    exist_ok=True
)


# =========================================================
# STORE ACTIVATIONS + GRADIENTS
# =========================================================

activations = []

gradients = []


# =========================================================
# HOOK FUNCTIONS
# =========================================================

def forward_hook(module, input, output):

    activations.append(output)


def backward_hook(module, grad_input, grad_output):

    gradients.append(grad_output[0])


# =========================================================
# TARGET LAYER
# =========================================================

target_layer = model.image_encoder.bottleneck


# =========================================================
# REGISTER HOOKS
# =========================================================

target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)


# =========================================================
# LOAD SAMPLE
# =========================================================

batch = next(iter(val_loader))

image = batch['image'].to(device)

mask = batch['mask'].to(device)

input_ids = batch['input_ids'].to(device)

attention_mask = batch['attention_mask'].to(device)


# =========================================================
# FORWARD PASS
# =========================================================

output = model(
    image,
    input_ids,
    attention_mask
)


# =========================================================
# TARGET
# =========================================================

target = output.mean()


# =========================================================
# BACKWARD PASS
# =========================================================

model.zero_grad()

target.backward()


# =========================================================
# GET ACTIVATIONS + GRADIENTS
# =========================================================

activation = activations[0]

gradient = gradients[0]


# =========================================================
# GLOBAL AVERAGE POOLING
# =========================================================

weights = gradient.mean(
    dim=(2,3,4),
    keepdim=True
)


# =========================================================
# CREATE CAM
# =========================================================

cam = (weights * activation).sum(dim=1)

cam = F.relu(cam)


# =========================================================
# NORMALIZE
# =========================================================

cam = cam[0].detach().cpu().numpy()

cam = (
    cam - cam.min()
) / (
    cam.max() - cam.min() + 1e-8
)


# =========================================================
# ORIGINAL IMAGE
# =========================================================

image_np = image[0,0].detach().cpu().numpy()


# =========================================================
# MIDDLE SLICE
# =========================================================

middle_slice = cam.shape[0] // 2

cam_slice = cam[middle_slice]

image_slice = image_np[middle_slice]


# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(12,6))


# MRI
plt.subplot(1,2,1)

plt.imshow(
    image_slice,
    cmap='gray'
)

plt.title("MRI Slice")

plt.axis('off')


# GRADCAM
plt.subplot(1,2,2)

plt.imshow(
    image_slice,
    cmap='gray'
)

plt.imshow(
    cam_slice,
    cmap='jet',
    alpha=0.5
)

plt.title("Grad-CAM Visualization")

plt.axis('off')


# =========================================================
# SAVE
# =========================================================

save_path = (
    "outputs/gradcam/gradcam.png"
)

plt.savefig(
    save_path,
    bbox_inches='tight'
)

plt.close()

print(f"Saved Grad-CAM: {save_path}")