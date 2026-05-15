import torch
import torch.nn as nn


# =========================================================
# CROSS ATTENTION FUSION MODULE
# =========================================================

class CrossAttentionFusion(nn.Module):

    def __init__(
        self,
        image_channels=512,
        text_dim=512,
        num_heads=8
    ):

        super(CrossAttentionFusion, self).__init__()

        # -------------------------------------------------
        # MULTIHEAD ATTENTION
        # -------------------------------------------------

        self.attention = nn.MultiheadAttention(
            embed_dim=image_channels,
            num_heads=num_heads,
            batch_first=True
        )

        # -------------------------------------------------
        # TEXT PROJECTION
        # -------------------------------------------------

        self.text_projection = nn.Linear(
            text_dim,
            image_channels
        )

        # -------------------------------------------------
        # NORMALIZATION
        # -------------------------------------------------

        self.norm = nn.LayerNorm(image_channels)

    # =====================================================
    # FORWARD PASS
    # =====================================================

    def forward(self, image_features, text_embedding):

        """
        image_features:
        (B, C, D, H, W)

        text_embedding:
        (B, 512)
        """

        B, C, D, H, W = image_features.shape

        # -------------------------------------------------
        # FLATTEN SPATIAL DIMENSIONS
        # -------------------------------------------------

        image_features_flat = image_features.view(
            B,
            C,
            D * H * W
        )

        # -------------------------------------------------
        # TRANSPOSE FOR ATTENTION
        # (B,C,N) -> (B,N,C)
        # -------------------------------------------------

        image_features_flat = image_features_flat.permute(
            0,
            2,
            1
        )

        # -------------------------------------------------
        # PROJECT TEXT EMBEDDING
        # -------------------------------------------------

        text_embedding = self.text_projection(
            text_embedding
        )

        # -------------------------------------------------
        # ADD SEQUENCE DIMENSION
        # (B,512) -> (B,1,512)
        # -------------------------------------------------

        text_embedding = text_embedding.unsqueeze(1)

        # -------------------------------------------------
        # CROSS ATTENTION
        # Query = image
        # Key/Value = text
        # -------------------------------------------------

        fused_features, attention_weights = self.attention(
            query=image_features_flat,
            key=text_embedding,
            value=text_embedding
        )

        # -------------------------------------------------
        # RESIDUAL CONNECTION
        # -------------------------------------------------

        fused_features = fused_features + image_features_flat

        # -------------------------------------------------
        # NORMALIZATION
        # -------------------------------------------------

        fused_features = self.norm(fused_features)

        # -------------------------------------------------
        # RESHAPE BACK
        # (B,N,C) -> (B,C,D,H,W)
        # -------------------------------------------------

        fused_features = fused_features.permute(
            0,
            2,
            1
        )

        fused_features = fused_features.view(
            B,
            C,
            D,
            H,
            W
        )

        return fused_features