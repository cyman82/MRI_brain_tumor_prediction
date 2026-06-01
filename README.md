# MRI Brain Tumor Segmentation with Text Fusion

A multimodal 3D segmentation project that fuses FLAIR MRI volumes with radiology-style text descriptions from TextBraTS. The model uses a 3D U-Net encoder-decoder with cross-attention to inject CLIP text embeddings into the image feature stream.

## Highlights
- 3D U-Net style encoder-decoder
- CLIP text encoder (openai/clip-vit-base-patch32)
- Cross-attention fusion (image queries, text keys/values)
- Dice + BCEWithLogits loss
- Mixed precision training
- Visualization suite (Grad-CAM, overlays, t-SNE, error analysis)

## Repository Structure
```
Brain Tumor/
  Brain_tumor.ipynb
  dataset/
    dataset_loader.py
  inference/
    predict.py
  models/
    decoder.py
    fusion_module.py
    image_encoder.py
    multimodal_unet.py
    text_encoder.py
  outputs/
    error_analysis/
    figures/
    gradcam/
    overlays/
    predictions/
    tsne/
  training/
    losses.py
    metrics.py
    train.py
  utils/
    config.py
  visualization/
    error_analysis.py
    gradcam.py
    overlay.py
    plots.py
    tsne.py
```

## Data
This project expects preprocessed NumPy volumes and masks.

Default paths are defined in utils/config.py:
- data/FLAIR_BRATS2020_split/train/images
- data/FLAIR_BRATS2020_split/train/masks
- data/FLAIR_BRATS2020_split/val/images
- data/FLAIR_BRATS2020_split/val/masks
- data/TextBraTSData

Observed in the current workspace:
- Train volumes: 258
- Val volumes: 86
- Text folders: 369 (dataset loader uses the first N folders to match images)

### Modalities
- FLAIR only

### Volume Shape
- Images: (128, 128, 128)
- Masks: (128, 128, 128, 4) => converted to (4, 128, 128, 128)

### Text Reports
TextBraTS-style descriptions in .txt format (tumor location, edema, necrosis, etc.).

## Model Overview
**Image encoder**
- 3D CNN with DoubleConv blocks + max pooling
- Bottleneck doubles the channel depth

**Text encoder**
- CLIPTextModel (openai/clip-vit-base-patch32)
- CLS/pooler output used as text embedding

**Fusion**
- Multihead cross-attention
- Image features as queries, projected text as keys/values
- Residual connection + LayerNorm

**Decoder**
- 3D transpose conv upsampling with skip connections
- Final 1x1x1 conv to 4 classes

## Input Pipeline
```
FLAIR volume
  -> min-max normalization
  -> 3D image encoder
Text report
  -> CLIP tokenizer
  -> CLIP text encoder
Fusion
  -> cross-attention fusion
Decoder
  -> tumor mask (4 channels)
```

## Training Setup
Configured in utils/config.py and training/train.py:
- Optimizer: AdamW
- Learning rate: 5e-5
- Batch size: 1
- Epochs: 50 (early stopping patience 5)
- Loss: Dice + BCEWithLogits
- Scheduler: ReduceLROnPlateau (monitor val Dice)
- Mixed precision: enabled

## Metrics
Implemented in training/metrics.py:
- Dice
- IoU
- Precision
- Recall

## Setup
Create a Python environment and install dependencies.

Suggested (example):
```
pip install torch torchvision torchaudio
pip install transformers tqdm matplotlib scikit-learn
```

## Training
```
python -m training.train
```
The best model is saved to outputs/checkpoints/best_model.pth and the training history to outputs/checkpoints/training_history.json.

## Inference
```
python -m inference.predict
```
Saves prediction images to outputs/predictions.

## Visualizations and Analysis
- Training curves: visualization/plots.py
- Error analysis: visualization/error_analysis.py
- Grad-CAM: visualization/gradcam.py
- Overlays: visualization/overlay.py
- t-SNE: visualization/tsne.py

Example runs:
```
python -m visualization.plots
python -m visualization.error_analysis
python -m visualization.gradcam
python -m visualization.overlay
python -m visualization.tsne
```

## Notes
- The notebook (Brain_tumor.ipynb) contains Colab-oriented setup commands.
- If you change data locations, update utils/config.py.

## License
Add a license if you plan to open-source the project.
