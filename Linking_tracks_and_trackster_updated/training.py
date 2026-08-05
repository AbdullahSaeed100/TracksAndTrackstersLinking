"""Training, checkpoint, resume, and early-stopping utilities.

The structure follows the useful lifecycle of the reference project while
using the active logits interface, separate edge-type losses, and verified
validation implementation.
"""

import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from evaluation import validate_epoch


EDGE_TYPES = {"trk_ts": 0, "ts_ts": 1}
REQUIRED_CHECKPOINT_METADATA = {
    "seed",
    "model_config",
    "loss_parameters",
    "split_fingerprint",
    "preprocessing_fingerprint",
}


def set_random_seed(seed):
    """Seed Python, NumPy, CPU PyTorch, and all available CUDA devices."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_data_loaders(
    training_dataset,
    validation_dataset,
    batch_size,
    seed=42,
    num_workers=0,
):
    """Build reproducible training and deterministic validation loaders."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size and num_workers must be positive ")

    generator = torch.Generator()
    generator.manual_seed(seed)

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if num_workers else None,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if num_workers else None,
    )
    return training_loader, validation_loader


def _validate_training_batch(batch):
    if batch is None:
        raise ValueError("training loader returned a missing batch")
    if batch.x.ndim != 2 or batch.x.shape[1] != 21:
        raise ValueError(f"training x must have shape [N, 21], got {tuple(batch.x.shape)}")
    if batch.edge_index.ndim != 2 or batch.edge_index.shape[0] != 2:
        raise ValueError(
            f"training edge_index must have shape [2, E], got {tuple(batch.edge_index.shape)}"
        )
    if batch.edge_attr.ndim != 2 or batch.edge_attr.shape[1] != 9:
        raise ValueError(
            f"training edge_attr must have shape [E, 9], got {tuple(batch.edge_attr.shape)}"
        )
    if batch.y.ndim != 1:
        raise ValueError(f"training labels must have shape [E], got {tuple(batch.y.shape)}")

    edge_count = batch.edge_attr.shape[0]
    if edge_count == 0:
        raise ValueError("training batches must contain at least one edge")
    if batch.edge_index.shape[1] != edge_count or batch.y.shape[0] != edge_count:
        raise ValueError("edge_index, edge_attr, and labels must describe the same edges")
    if not torch.isfinite(batch.x).all() or not torch.isfinite(batch.edge_attr).all():
        raise ValueError("training features contain non-finite values")
    if not torch.isfinite(batch.y).all() or not torch.all((batch.y == 0) | (batch.y == 1)):
        raise ValueError("training labels must be finite binary values")

    edge_type = batch.edge_attr[:, 7]
    if not torch.all((edge_type == 0) | (edge_type == 1)):
        raise ValueError("edge type column must contain only 0 for trk-ts or 1 for ts-ts")


def train_epoch(
    model,
    optimizer,
    training_loader,
    trk_ts_loss,
    ts_ts_loss,
    device,
    show_progress=False,
    description="Training",
):
    """Train for one epoch with separately normalized edge-type losses."""
    losses = {"trk_ts": trk_ts_loss, "ts_ts": ts_ts_loss}
    loss_sums = {name: 0.0 for name in EDGE_TYPES}
    edge_counts = {name: 0 for name in EDGE_TYPES}
    batch_count = 0

    model.train()
    batches = training_loader
    if show_progress:
        batches = tqdm(batches, desc=description, leave=False)

    for batch in batches:
        # _validate_training_batch(batch)
        batch = batch.to(device)

        x = batch.x.float()
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr.float()
        labels = batch.y.float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, edge_index, edge_attr)
        if logits.ndim != 1 or logits.shape != labels.shape:
            raise ValueError(
                f"model logits must have shape {tuple(labels.shape)}, "
                f"got {tuple(logits.shape)}"
            )
        if not torch.isfinite(logits).all():
            raise ValueError("model returned non-finite training logits")

        type_losses = {}
        # Zero loss connected to logits, used when a batch has no edges of one type.
        # It contributes nothing but remains compatible with backward().
        graph_connected_zero = logits.sum() * 0.0
        for name, type_value in EDGE_TYPES.items():
            mask = edge_attr[:, 7] == type_value
            count = int(mask.sum().item())
            if count == 0:
                type_losses[name] = graph_connected_zero
                continue

            type_loss = losses[name](logits[mask], labels[mask])# the mean weighted loss for this type in this batch(scalar number)
            if type_loss.ndim != 0 or not torch.isfinite(type_loss):
                raise ValueError(f"{name} training loss must be a finite scalar")

            type_losses[name] = type_loss # just the two type losses for this batch used with combined
            loss_sums[name] += type_loss.item() * count # accumulated loss for each type calculated accross the whole epoch
            edge_counts[name] += count # edge counts for each type calculated accross the whole epoch

        combined_loss = 0.5 * type_losses["trk_ts"] + 0.5 * type_losses["ts_ts"] # used for back probagation
        if not torch.isfinite(combined_loss):
            raise ValueError("combined training loss must be finite")

        combined_loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError(f"model gradient is non-finite: {name}")
        optimizer.step()
        batch_count += 1
        if show_progress:
            batches.set_postfix(loss=f"{combined_loss.item():.6f}")

    if batch_count == 0:
        raise ValueError("training loader produced no batches")
    if any(edge_counts[name] == 0 for name in EDGE_TYPES):
        raise ValueError("training epoch must contain both trk-ts and ts-ts edges")

    mean_losses = {
        name: loss_sums[name] / edge_counts[name] for name in EDGE_TYPES
    }
    return {
        "loss": {
            "trk_ts": mean_losses["trk_ts"],
            "ts_ts": mean_losses["ts_ts"],
            "combined": 0.5 * mean_losses["trk_ts"] + 0.5 * mean_losses["ts_ts"],
        },
        "edge_count": dict(edge_counts),
        "batch_count": batch_count,
    }


def _copy_model_state(model):
    """creates an independent copy of all current model parameters.
    It is used to preserve the best model weights found during training."""
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


class EarlyStopping:
    """Track the lowest combined validation loss and stop after no improvement."""

    def __init__(self, patience=15, min_delta=1e-4):
        if patience <= 0 or min_delta < 0:
            raise ValueError("patience must be positive and min_delta non-negative")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_score = None
        self.best_epoch = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None
        self.best_validation_results = None

    def __call__(self, model, validation_results, epoch):
        score = float(validation_results["loss"]["combined"])
        improved = self.best_score is None or score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = int(epoch)
            self.counter = 0
            self.best_model_state = _copy_model_state(model)
            self.best_validation_results = copy.deepcopy(validation_results)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return improved

    def load_best_model(self, model):
        if self.best_model_state is None:
            raise ValueError("no best model state has been recorded")
        model.load_state_dict(self.best_model_state)

    def state_dict(self):
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "early_stop": self.early_stop,
        }

    def load_state_dict(self, state):
        self.patience = int(state["patience"])
        self.min_delta = float(state["min_delta"])
        self.best_score = state["best_score"]
        self.best_epoch = state["best_epoch"]
        self.counter = int(state["counter"])
        self.early_stop = bool(state["early_stop"])


def _capture_rng_state(training_loader):
    """Save all random-generator states so resumed training can reproduce
    the same dropout patterns, random operations, and DataLoader batch order."""
    
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "training_loader_generator": None,
    }
    if getattr(training_loader, "generator", None) is not None:
        state["training_loader_generator"] = training_loader.generator.get_state()
    return state


def _restore_rng_state(state, training_loader):
    """Restore the saved random-generator states so training continues
    reproducibly from the checkpoint. """
    
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Loading a checkpoint with map_location="cuda" also moves RNG-state
    # tensors to CUDA, but PyTorch generator setters require CPU ByteTensors.
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    loader_state = state.get("training_loader_generator")
    if loader_state is not None and getattr(training_loader, "generator", None) is not None:
        training_loader.generator.set_state(loader_state.cpu())


def _validate_checkpoint_metadata(metadata):
    missing = REQUIRED_CHECKPOINT_METADATA - set(metadata)
    if missing:
        raise ValueError(f"checkpoint metadata is missing: {sorted(missing)}")


def save_model(
    model,
    epoch,
    optimizer,
    training_results,
    validation_results,
    checkpoint_path,
    metadata,
    early_stopping,
    training_loader,
):
    """Save the current best model using a structured checkpoint."""
    
    _validate_checkpoint_metadata(metadata)
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f">>> Saving model to {path}")
    checkpoint = {
        "format_version": 1,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_results": copy.deepcopy(training_results),
        "validation_results": copy.deepcopy(validation_results),
        "metadata": copy.deepcopy(metadata),
        "early_stopping": early_stopping.state_dict(),
        "rng_state": _capture_rng_state(training_loader),
    }

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    device,
    training_loader,
    expected_metadata=None,
):
    """Restore a structured checkpoint and return its saved state."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    if expected_metadata is not None and checkpoint["metadata"] != expected_metadata:
        raise ValueError("resume checkpoint metadata does not match this run")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _restore_rng_state(checkpoint["rng_state"], training_loader)
    return checkpoint


def train_model(
    model,
    optimizer,
    training_loader,
    validation_loader,
    trk_ts_loss,
    ts_ts_loss,
    device,
    epochs,
    checkpoint_path,
    checkpoint_metadata,
    patience=15,
    min_delta=1e-4,
    resume_path=None,
    show_progress=False,
):
    """Train, validate, checkpoint the best epoch, and apply early stopping."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    _validate_checkpoint_metadata(checkpoint_metadata)

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    start_epoch = 0

    if resume_path is not None:
        checkpoint = load_checkpoint(
            resume_path,
            model,
            optimizer,
            device,
            training_loader,
            expected_metadata=checkpoint_metadata,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        early_stopping.load_state_dict(checkpoint["early_stopping"])
        early_stopping.early_stop = False
        early_stopping.best_model_state = _copy_model_state(model)
        early_stopping.best_validation_results = copy.deepcopy(
            checkpoint["validation_results"]
        )

    history = []
    epoch_iterator = range(start_epoch, epochs)
    if show_progress:
        epoch_iterator = tqdm(
            epoch_iterator,
            desc="Epochs",
            total=epochs,
            initial=start_epoch,
        )

    for epoch in epoch_iterator:
        training_results = train_epoch(
            model,
            optimizer,
            training_loader,
            trk_ts_loss,
            ts_ts_loss,
            device,
            show_progress=show_progress,
            description=f"Training {epoch + 1}/{epochs}",
        )
        validation_results = validate_epoch(
            model,
            validation_loader,
            trk_ts_loss,
            ts_ts_loss,
            device,
            show_progress=show_progress,
            description=f"Validation {epoch + 1}/{epochs}",
        )

        improved = early_stopping(model, validation_results, epoch)
        history.append({
            "epoch": epoch,
            "training": training_results,
            "validation": validation_results,
            "improved": improved,
        })
        if show_progress:
            epoch_iterator.set_postfix(
                train=f"{training_results['loss']['combined']:.6f}",
                validation=f"{validation_results['loss']['combined']:.6f}",
                best=f"{early_stopping.best_score:.6f}",
            )

        if improved:
            save_model(
                model,
                epoch,
                optimizer,
                training_results,
                validation_results,
                checkpoint_path,
                checkpoint_metadata,
                early_stopping,
                training_loader,
            )
        if early_stopping.early_stop:
            break

    early_stopping.load_best_model(model)
    return {
        "history": history,
        "best_epoch": early_stopping.best_epoch,
        "best_validation": early_stopping.best_validation_results,
        "stopped_early": early_stopping.early_stop,
    }
