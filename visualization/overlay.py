import os
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# INPUT PATHS
# =========================================================

prediction_dir = "outputs/predictions"

save_dir = "outputs/overlays"

os.makedirs(save_dir, exist_ok=True)


# =========================================================
# OVERLAY CREATION
# =========================================================

for file in os.listdir(prediction_dir):

    if file.endswith(".png"):

        image_path = os.path.join(prediction_dir, file)

        img = plt.imread(image_path)

        plt.figure(figsize=(8,8))

        plt.imshow(img)

        plt.title("Prediction Overlay Visualization")

        plt.axis('off')

        save_path = os.path.join(
            save_dir,
            file
        )

        plt.savefig(save_path)

        plt.close()

        print(f"Saved Overlay: {save_path}")