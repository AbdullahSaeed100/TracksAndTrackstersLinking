"""Validation utilities for the two edge-linking tasks.

The model returns raw logits. Probabilities are created here only for metric
calculation and threshold selection. This module never accesses the test set.
"""

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from tqdm.auto import tqdm


EDGE_TYPES = {"trk_ts": 0, "ts_ts": 1}


def _as_numpy(values):
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _prepare_inputs(logits, labels):
    logits = _as_numpy(logits)
    labels = _as_numpy(labels)

    if logits.ndim != 1 or labels.ndim != 1 or logits.shape != labels.shape:
        raise ValueError("logits and labels must be non-empty 1D arrays with equal shape")
    if logits.size == 0:
        raise ValueError("logits and labels must not be empty")
    if not np.isfinite(logits).all() or not np.isfinite(labels).all():
        raise ValueError("logits and labels must contain only finite values")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must contain only 0 and 1")

    return logits.astype(np.float64, copy=False), labels.astype(np.int64, copy=False)


def logits_to_probabilities(logits):
    """Convert one-dimensional raw logits to sigmoid probabilities."""
    logits = _as_numpy(logits)
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("logits must be a non-empty 1D array")
    if not np.isfinite(logits).all():
        raise ValueError("logits must contain only finite values")

    tensor = torch.as_tensor(logits, dtype=torch.float64)
    return torch.sigmoid(tensor).numpy()


def _safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def compute_threshold_metrics(logits, labels, threshold=0.5):
    """Calculate binary metrics for one edge type at a probability threshold."""
    logits, labels = _prepare_inputs(logits, labels)
    

    probabilities = logits_to_probabilities(logits)
    predictions = probabilities >= threshold
    positive = labels == 1
    negative = ~positive

    tp = int(np.sum(predictions & positive))
    fp = int(np.sum(predictions & negative))
    tn = int(np.sum(~predictions & negative))
    fn = int(np.sum(~predictions & positive))

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_positive_rate": _safe_divide(fp, fp + tn),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "accuracy": _safe_divide(tp + tn, labels.size),
    }


def compute_ranking_metrics(logits, labels):
    """Calculate threshold-independent ranking metrics for one edge type."""
    logits, labels = _prepare_inputs(logits, labels)
    if np.unique(labels).size != 2:
        raise ValueError("ranking metrics require both positive and negative labels")

    probabilities = logits_to_probabilities(logits)
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def find_best_f1_threshold(logits, labels):
    """Find the validation probability threshold with the highest F1 score."""
    logits, labels = _prepare_inputs(logits, labels)
    if np.unique(labels).size != 2:
        raise ValueError("threshold selection requires both positive and negative labels")

    probabilities = logits_to_probabilities(logits)
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1 = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(thresholds),
        where=(precision[:-1] + recall[:-1]) != 0,
    )

    best_f1 = f1.max()
    # Thresholds are ascending, so the final tied index is the highest one.
    best_index = np.flatnonzero(np.isclose(f1, best_f1))[-1]
    best = compute_threshold_metrics(logits, labels, thresholds[best_index])

    at_half = compute_threshold_metrics(logits, labels, 0.5)
    if (at_half["f1"], at_half["threshold"]) > (best["f1"], best["threshold"]):
        best = at_half
    return best


def _validate_batch(batch):
    if batch is None:
        raise ValueError("validation loader returned a missing batch")
    if batch.x.ndim != 2 or batch.x.shape[1] != 21:
        raise ValueError(f"validation x must have shape [N, 21], got {tuple(batch.x.shape)}")
    if batch.edge_index.ndim != 2 or batch.edge_index.shape[0] != 2:
        raise ValueError(
            f"validation edge_index must have shape [2, E], got {tuple(batch.edge_index.shape)}"
        )
    if batch.edge_attr.ndim != 2 or batch.edge_attr.shape[1] != 9:
        raise ValueError(
            f"validation edge_attr must have shape [E, 9], got {tuple(batch.edge_attr.shape)}"
        )
    if batch.y.ndim != 1:
        raise ValueError(f"validation labels must have shape [E], got {tuple(batch.y.shape)}")

    edge_count = batch.edge_attr.shape[0]
    if batch.edge_index.shape[1] != edge_count or batch.y.shape[0] != edge_count:
        raise ValueError("edge_index, edge_attr, and labels must describe the same edges")
    if not torch.isfinite(batch.x).all() or not torch.isfinite(batch.edge_attr).all():
        raise ValueError("validation features contain non-finite values")
    if not torch.isfinite(batch.y).all() or not torch.all((batch.y == 0) | (batch.y == 1)):
        raise ValueError("validation labels must be finite binary values")

    edge_type = batch.edge_attr[:, 7]
    if not torch.all((edge_type == 0) | (edge_type == 1)):
        raise ValueError("edge type column must contain only 0 for trk-ts or 1 for ts-ts")
    

def validate_epoch(
    model,
    validation_loader,
    trk_ts_loss,
    ts_ts_loss,
    device,
    show_progress=False,
    description="Validation",
):
    """Evaluate one complete validation epoch, separately for both edge types."""
    losses = {"trk_ts": trk_ts_loss, "ts_ts": ts_ts_loss}
    loss_sums = {name: 0.0 for name in EDGE_TYPES}
    edge_counts = {name: 0 for name in EDGE_TYPES}
    collected_logits = {name: [] for name in EDGE_TYPES}
    collected_labels = {name: [] for name in EDGE_TYPES}

    previous_training_state = model.training
    model.eval()

    try:
        with torch.no_grad():
            batches = validation_loader
            if show_progress:
                batches = tqdm(batches, desc=description, leave=False)

            for batch in batches:
                _validate_batch(batch)
                batch = batch.to(device)

                x = batch.x.float()
                edge_index = batch.edge_index
                edge_attr = batch.edge_attr.float()
                labels = batch.y.float()

                logits = model(x, edge_index, edge_attr)
                if logits.ndim != 1 or logits.shape != labels.shape:
                    raise ValueError(
                        f"model logits must have shape {tuple(labels.shape)}, "
                        f"got {tuple(logits.shape)}"
                    )
                if not torch.isfinite(logits).all():
                    raise ValueError("model returned non-finite validation logits")

                for name, type_value in EDGE_TYPES.items():
                    mask = edge_attr[:, 7] == type_value
                    count = int(mask.sum().item())
                    if count == 0:
                        continue

                    type_loss = losses[name](logits[mask], labels[mask])
                    if type_loss.ndim != 0 or not torch.isfinite(type_loss):
                        raise ValueError(f"{name} validation loss must be a finite scalar")

                    loss_sums[name] += type_loss.item() * count
                    edge_counts[name] += count
                    collected_logits[name].append(logits[mask].detach().cpu())
                    collected_labels[name].append(labels[mask].detach().cpu())
    finally:
        model.train(previous_training_state)

    if any(edge_counts[name] == 0 for name in EDGE_TYPES):
        raise ValueError("validation data must contain both trk-ts and ts-ts edges")

    mean_losses = {
        name: loss_sums[name] / edge_counts[name]
        for name in EDGE_TYPES
    }
    results = {
        "loss": {
            **mean_losses,
            "combined": 0.5 * mean_losses["trk_ts"] + 0.5 * mean_losses["ts_ts"],
        }
    }

    for name in EDGE_TYPES:
        logits = torch.cat(collected_logits[name]).numpy()
        labels = torch.cat(collected_labels[name]).numpy()
        results[name] = {
            "edge_count": int(labels.size),
            "positive_count": int(labels.sum()),
            "positive_fraction": float(labels.mean()),
            "ranking": compute_ranking_metrics(logits, labels),
            "threshold_0_5": compute_threshold_metrics(logits, labels, 0.5),
            "best_f1": find_best_f1_threshold(logits, labels),
        }

    return results
