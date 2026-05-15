import torch


# =========================================================
# DICE SCORE
# =========================================================

def dice_score(predictions, targets, smooth=1e-5):

    predictions = torch.sigmoid(predictions)

    predictions = (predictions > 0.5).float()

    predictions = predictions.view(-1)

    targets = targets.view(-1)

    intersection = (predictions * targets).sum()

    dice = (
        2.0 * intersection + smooth
    ) / (
        predictions.sum() +
        targets.sum() +
        smooth
    )

    return dice.item()


# =========================================================
# INTERSECTION OVER UNION (IoU)
# =========================================================

def iou_score(predictions, targets, smooth=1e-5):

    predictions = torch.sigmoid(predictions)

    predictions = (predictions > 0.5).float()

    predictions = predictions.view(-1)

    targets = targets.view(-1)

    intersection = (predictions * targets).sum()

    union = (
        predictions.sum() +
        targets.sum() -
        intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.item()


# =========================================================
# PRECISION
# =========================================================

def precision_score(predictions, targets, smooth=1e-5):

    predictions = torch.sigmoid(predictions)

    predictions = (predictions > 0.5).float()

    predictions = predictions.view(-1)

    targets = targets.view(-1)

    true_positive = (
        predictions * targets
    ).sum()

    predicted_positive = predictions.sum()

    precision = (
        true_positive + smooth
    ) / (
        predicted_positive + smooth
    )

    return precision.item()


# =========================================================
# RECALL
# =========================================================

def recall_score(predictions, targets, smooth=1e-5):

    predictions = torch.sigmoid(predictions)

    predictions = (predictions > 0.5).float()

    predictions = predictions.view(-1)

    targets = targets.view(-1)

    true_positive = (
        predictions * targets
    ).sum()

    actual_positive = targets.sum()

    recall = (
        true_positive + smooth
    ) / (
        actual_positive + smooth
    )

    return recall.item()