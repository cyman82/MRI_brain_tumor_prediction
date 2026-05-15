import torch
import torch.nn as nn

from models.image_encoder import ImageEncoder3D

from models.text_encoder import TextEncoder

from models.fusion_module import CrossAttentionFusion

from models.decoder import Decoder3D


# =========================================================
# MULTIMODAL 3D U-NET
# =========================================================

class MultimodalUNet3D(nn.Module):

    def __init__(self):

        super(MultimodalUNet3D, self).__init__()

        # -------------------------------------------------
        # IMAGE ENCODER
        # -------------------------------------------------

        self.image_encoder = ImageEncoder3D()

        # -------------------------------------------------
        # TEXT ENCODER
        # -------------------------------------------------

        self.text_encoder = TextEncoder()

        # -------------------------------------------------
        # FUSION MODULE
        # -------------------------------------------------

        self.fusion = CrossAttentionFusion()

        # -------------------------------------------------
        # DECODER
        # -------------------------------------------------

        self.decoder = Decoder3D()

    # =====================================================
    # FORWARD PASS
    # =====================================================

    def forward(
        self,
        image,
        input_ids,
        attention_mask
    ):

        # -------------------------------------------------
        # IMAGE FEATURES
        # -------------------------------------------------

        image_features, skip_connections = (
            self.image_encoder(image)
        )

        # -------------------------------------------------
        # TEXT EMBEDDING
        # -------------------------------------------------

        text_embedding = self.text_encoder(
            input_ids,
            attention_mask
        )

        # -------------------------------------------------
        # MULTIMODAL FUSION
        # -------------------------------------------------

        fused_features = self.fusion(
            image_features,
            text_embedding
        )

        # -------------------------------------------------
        # DECODER
        # -------------------------------------------------

        segmentation_output = self.decoder(
            fused_features,
            skip_connections
        )

        return segmentation_output