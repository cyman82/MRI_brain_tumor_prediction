import os
import re
import numpy as np
import torch

from torch.utils.data import Dataset

from transformers import CLIPTokenizer

from utils.config import *


class BrainTumorDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        text_dir,
        transform=None
    ):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.text_dir = text_dir

        self.transform = transform

        # -------------------------------------------------
        # NUMERICAL SORTING
        # -------------------------------------------------

        self.images = sorted(
            os.listdir(image_dir),
            key=self.extract_number
        )

        self.masks = sorted(
            os.listdir(mask_dir),
            key=self.extract_number
        )

        self.text_folders = sorted(os.listdir(text_dir))

        # -------------------------------------------------
        # CLIP TOKENIZER
        # -------------------------------------------------

        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    # =====================================================
    # EXTRACT NUMBER FROM FILENAME
    # image_10.npy -> 10
    # =====================================================

    def extract_number(self, filename):

        numbers = re.findall(r'\d+', filename)

        return int(numbers[0])

    # =====================================================
    # DATASET LENGTH
    # =====================================================

    def __len__(self):

        return len(self.images)

    # =====================================================
    # LOAD TEXT REPORT
    # =====================================================

    def load_text_report(self, patient_idx):

        folder_name = self.text_folders[patient_idx]

        folder_path = os.path.join(
            self.text_dir,
            folder_name
        )

        txt_files = [
            f for f in os.listdir(folder_path)
            if f.endswith(".txt")
        ]

        txt_path = os.path.join(
            folder_path,
            txt_files[0]
        )

        with open(txt_path, "r", encoding="utf-8") as file:

            report = file.read()

        return report

    # =====================================================
    # TOKENIZE TEXT
    # =====================================================

    def tokenize_text(self, text):

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)

        attention_mask = encoding["attention_mask"].squeeze(0)

        return input_ids, attention_mask

    # =====================================================
    # GET ITEM
    # =====================================================

    def __getitem__(self, idx):

        # -------------------------------------------------
        # IMAGE PATH
        # -------------------------------------------------

        image_path = os.path.join(
            self.image_dir,
            self.images[idx]
        )

        # -------------------------------------------------
        # MASK PATH
        # -------------------------------------------------

        mask_path = os.path.join(
            self.mask_dir,
            self.masks[idx]
        )

        # -------------------------------------------------
        # LOAD MRI VOLUME
        # Shape: (128,128,128)
        # -------------------------------------------------

        image = np.load(image_path)

        # -------------------------------------------------
        # LOAD MASK
        # Shape: (128,128,128,4)
        # -------------------------------------------------

        mask = np.load(mask_path)

        # -------------------------------------------------
        # LOAD TEXT REPORT
        # -------------------------------------------------

        text_report = self.load_text_report(idx)

        # -------------------------------------------------
        # TOKENIZE REPORT
        # -------------------------------------------------

        input_ids, attention_mask = self.tokenize_text(
            text_report
        )

        # -------------------------------------------------
        # NORMALIZE IMAGE
        # -------------------------------------------------

        image = (image - image.min()) / (
            image.max() - image.min() + 1e-8
        )

        # -------------------------------------------------
        # CONVERT TO TENSORS
        # -------------------------------------------------

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        )

        # -------------------------------------------------
        # ADD CHANNEL DIMENSION
        # (128,128,128)
        # ->
        # (1,128,128,128)
        # -------------------------------------------------

        image = image.unsqueeze(0)

        # -------------------------------------------------
        # PERMUTE MASK DIMENSIONS
        # (128,128,128,4)
        # ->
        # (4,128,128,128)
        # -------------------------------------------------

        mask = mask.permute(3, 0, 1, 2)

        # -------------------------------------------------
        # OPTIONAL TRANSFORMS
        # -------------------------------------------------

        if self.transform is not None:

            image = self.transform(image)

        return {
            "image": image,
            "mask": mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text": text_report
        }