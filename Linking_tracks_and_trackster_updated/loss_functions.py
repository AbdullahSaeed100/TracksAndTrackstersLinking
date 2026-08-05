"""Logits-based loss functions for binary edge classification.

The active GNN returns raw edge logits. Edge-type masking and trk-ts/ts-ts
loss combination belong in the training loop, not in these base loss classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def derive_loss_parameters(training_summary):
    """Derive per-edge-type imbalance parameters from training counts only."""
    parameters = {}

    for edge_type in ("trk_ts", "ts_ts"):
        total = int(training_summary[f"{edge_type}_edges"])
        positive = int(training_summary[f"{edge_type}_positive"])
        negative = total - positive

        if positive <= 0 or negative <= 0:
            raise ValueError(
                f"Training counts for {edge_type} must contain both "
                "positive and negative examples"
            )

        parameters[edge_type] = {
            "positive": positive,
            "negative": negative,
            "pos_weight": negative / positive,
            "alpha": negative / total,
        }

    return parameters


class WeightedBCELoss(nn.Module):
    """Logits-based binary cross-entropy with positive-example weighting."""

    def __init__(self, pos_weight=1.0):
        super().__init__()
        if pos_weight <= 0:
            raise ValueError("pos_weight must be greater than 0")
        self.register_buffer(
            "pos_weight",
            torch.tensor(float(pos_weight), dtype=torch.float32),
        )

    def forward(self, logits, targets):
        positive_weight = self.pos_weight.to(
            device=targets.device,
            dtype=targets.dtype,
        )
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=positive_weight,
        )


class FocalLoss(nn.Module):
    """Logits-based binary focal loss with fixed positive-class alpha."""

    def __init__(self, alpha, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        if not (0 <= alpha <= 1 and gamma >= 0 and 0 <= label_smoothing < 1):
            raise ValueError(
                "FocalLoss requires 0 <= alpha <= 1, gamma >= 0, "
                "and 0 <= label_smoothing < 1"
            )

        self.register_buffer(
            "alpha",
            torch.tensor(float(alpha), dtype=torch.float32),
        )
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets):
        smoothed_targets = (
            targets * (1.0 - self.label_smoothing)
            + 0.5 * self.label_smoothing
        )
        bce = F.binary_cross_entropy_with_logits(
            logits,
            smoothed_targets,
            reduction="none",
        )

        probabilities = torch.sigmoid(logits)
        p_t = torch.where(targets == 1, probabilities, 1.0 - probabilities)
        positive_alpha = self.alpha.to(
            device=targets.device,
            dtype=targets.dtype,
        )
        alpha_t = torch.where(targets == 1, positive_alpha, 1.0 - positive_alpha)
        focal_factor = (1.0 - p_t).pow(self.gamma)
        return (alpha_t * focal_factor * bce).mean()
