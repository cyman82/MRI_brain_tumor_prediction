import torch
import torch.nn as nn


# =========================================================
# DICE LOSS
# =========================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-5):

        super(DiceLoss, self).__init__()

        self.smooth = smooth

    def forward(self, predictions, targets):

        # -------------------------------------------------
        # APPLY SIGMOID
        # -------------------------------------------------

        predictions = torch.sigmoid(predictions)

        # -------------------------------------------------
        # FLATTEN
        # -------------------------------------------------

        predictions = predictions.view(-1)

        targets = targets.view(-1)

        # -------------------------------------------------
        # INTERSECTION
        # -------------------------------------------------

        intersection = (predictions * targets).sum()

        # -------------------------------------------------
        # DICE SCORE
        # -------------------------------------------------

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            predictions.sum() +
            targets.sum() +
            self.smooth
        )

        return 1.0 - dice


# =========================================================
# COMBINED LOSS
# =========================================================

class CombinedLoss(nn.Module):

    def __init__(self):

        super(CombinedLoss, self).__init__()

        self.dice_loss = DiceLoss()

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):

        dice = self.dice_loss(
            predictions,
            targets
        )

        bce = self.bce_loss(
            predictions,
            targets
        )

        total_loss = dice + bce

        return total_loss