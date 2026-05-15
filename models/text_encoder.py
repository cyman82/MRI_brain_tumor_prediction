import torch
import torch.nn as nn

from transformers import CLIPTextModel


# =========================================================
# CLIP TEXT ENCODER
# =========================================================

class TextEncoder(nn.Module):

    def __init__(self):

        super(TextEncoder, self).__init__()

        # -------------------------------------------------
        # LOAD PRETRAINED CLIP TEXT MODEL
        # -------------------------------------------------

        self.text_model = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    # =====================================================
    # FORWARD PASS
    # =====================================================

    def forward(self, input_ids, attention_mask):

        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # -------------------------------------------------
        # CLS TOKEN REPRESENTATION
        # -------------------------------------------------

        text_embedding = outputs.pooler_output

        return text_embedding