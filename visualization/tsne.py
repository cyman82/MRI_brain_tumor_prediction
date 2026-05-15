import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
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
# STORE EMBEDDINGS
# =========================================================

image_embeddings = []

text_embeddings = []


# =========================================================
# EXTRACT FEATURES
# =========================================================

with torch.no_grad():

    for idx, batch in enumerate(loader):

        image = batch["image"].to(device)

        mask = batch["mask"].to(device)

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)


        # =================================================
        # IMAGE FEATURES
        # =================================================

        bottleneck, _ = model.image_encoder(image)

        image_feature = bottleneck.mean(
            dim=(2,3,4)
        )


        # =================================================
        # TEXT FEATURES
        # =================================================

        text_feature = model.text_encoder(
            input_ids,
            attention_mask
        )


        # =================================================
        # STORE
        # =================================================

        image_embeddings.append(
            image_feature.cpu().numpy().flatten()
        )

        text_embeddings.append(
            text_feature.cpu().numpy().flatten()
        )


        # LIMIT SAMPLES
        if idx == 49:
            break


# =========================================================
# COMBINE EMBEDDINGS
# =========================================================

all_embeddings = np.vstack(
    image_embeddings + text_embeddings
)

labels = (
    ["Image"] * len(image_embeddings)
    +
    ["Text"] * len(text_embeddings)
)


# =========================================================
# TSNE
# =========================================================

tsne = TSNE(
    n_components=2,
    perplexity=10,
    random_state=42
)

reduced_embeddings = tsne.fit_transform(
    all_embeddings
)


# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs(
    "outputs/tsne",
    exist_ok=True
)


# =========================================================
# VISUALIZATION
# =========================================================

plt.figure(figsize=(10,8))

for i, label in enumerate(labels):

    if label == "Image":

        plt.scatter(
            reduced_embeddings[i,0],
            reduced_embeddings[i,1],
            marker='o',
            s=60,
            label='Image' if i == 0 else ""
        )

    else:

        plt.scatter(
            reduced_embeddings[i,0],
            reduced_embeddings[i,1],
            marker='x',
            s=60,
            label='Text' if i == len(image_embeddings) else ""
        )


plt.title(
    "t-SNE Visualization of Image and Text Embeddings"
)

plt.xlabel("t-SNE Component 1")

plt.ylabel("t-SNE Component 2")

plt.legend()

plt.grid(True)


# =========================================================
# SAVE
# =========================================================

save_path = "outputs/tsne/tsne_plot.png"

plt.savefig(save_path)

plt.show()

print(f"Saved t-SNE plot at: {save_path}")