import os
import hashlib
import json
from functools import partial
from glob import glob
from pathlib import Path

import torch
from torch.utils.data import Subset
import uproot

import training

from ClusterDataset_trk_trkst import ClusterDataset
from GNN_TrackLinkingNet import GNN_TrackLinkingNet
from loss_functions import WeightedBCELoss, derive_loss_parameters
from feature_preprocessing import (
    fit_track_stats,
    fit_trkst_stats,
    fit_trk_ts_edge_stats,
    fit_ts_ts_edge_stats,
    save_stats,
    load_stats,
    preprocess_track_features,
    preprocess_trkst_features,
    preprocess_trk_ts_edge_features,
    preprocess_ts_ts_edge_features,
)
from dataset_split import (
    indices_from_manifest,
    inspect_raw_dataset,
    load_or_create_stratified_manifest,
    print_split_summary,
    split_summary,
    training_split_fingerprint,
)

PROJECT_DIR = Path(__file__).resolve().parent
STATS_PATH = PROJECT_DIR / "all_stats_train_stratified.json"
SPLIT_MANIFEST_PATH = PROJECT_DIR / "dataset_split_manifest_stratified.json"
HIST_FOLDER_ENV = "TRACKLINK_HIST_FOLDER"
DATA_FOLDER_ENV = "TRACKLINK_DATA_FOLDER"
SPLIT_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)
STRATIFICATION_CANDIDATES = 5000
BASELINE_BATCH_SIZE = 4
BASELINE_LEARNING_RATE = 1e-4
BASELINE_WEIGHT_DECAY = 1e-5
BASELINE_EPOCHS = 100
BASELINE_PATIENCE = 30
BASELINE_MIN_DELTA = 1e-4
BASELINE_TASK_WEIGHTS = {"trk_ts": 0.5, "ts_ts": 0.5}


def required_directory(environment_variable):
    """Return a configured external directory without hardcoding local paths."""
    value = os.environ.get(environment_variable)
    if not value:
        raise RuntimeError(
            f"Environment variable {environment_variable} is not set. "
            "See .env.example and README.md for configuration instructions."
        )

    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Directory configured by {environment_variable} does not exist: {path}"
        )
    return path


def build_raw_dataset():
    return ClusterDataset(
        str(required_directory(DATA_FOLDER_ENV)),
        str(required_directory(HIST_FOLDER_ENV)),
    )


def discover_single_root_file():
    hist_folder = required_directory(HIST_FOLDER_ENV)
    files = sorted(glob(str(hist_folder / "*.root")))
    if len(files) != 1:
        raise ValueError(
            "The approved event-level split currently expects exactly one ROOT file; "
            f"found {len(files)}. Revisit the grouping policy before continuing."
        )
    return files[0]


def load_event_ids(root_path, raw_dataset):
    with uproot.open(root_path) as root_file:
        all_associations = raw_dataset.load_branch_with_highest_cycle(
            root_file, "ticlDumper/associations"
        )
        runs = all_associations["event"]["run_"].array(library="np")
        lumis = all_associations["event"]["luminosityBlock_"].array(library="np")
        events = all_associations["event"]["event_"].array(library="np")

    
    identities = [
        (int(run), int(lumi), int(event))
        for run, lumi, event in zip(runs, lumis, events)
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate (run, luminosityBlock, event) identity found")
    return dict(enumerate(identities))


def get_or_fit_stats(training_dataset, training_fingerprint, force_refit=False):
    if os.path.exists(STATS_PATH) and not force_refit:
        stats = load_stats(STATS_PATH)
        if stats.get("training_split_fingerprint") != training_fingerprint:
            raise ValueError(
                "Cached preprocessing statistics do not match the current training split. "
                "Use force_refit=True after confirming the dataset change."
            )
        return stats
    stats = fit_track_stats(training_dataset)
    stats.update(fit_trkst_stats(training_dataset))
    stats.update(fit_trk_ts_edge_stats(training_dataset))
    stats.update(fit_ts_ts_edge_stats(training_dataset))
    stats["training_split_fingerprint"] = training_fingerprint
    stats["split_seed"] = SPLIT_SEED
    save_stats(stats, STATS_PATH)
    return stats


def full_transform(data, stats):
    raw_edge_feature_count = 8
    actual_edge_feature_count = data.edge_attr.shape[1]
    if actual_edge_feature_count != raw_edge_feature_count:
        raise ValueError(
            "full_transform expects raw edge_attr with 8 columns; "
            f"received {actual_edge_feature_count}. "
            "The graph may already be preprocessed."
        )

    data = preprocess_track_features(data, stats)
    data = preprocess_trkst_features(data, stats)
    data = preprocess_trk_ts_edge_features(data, stats)
    data = preprocess_ts_ts_edge_features(data, stats)
    return data


def build_transformed_dataset(stats):
    transform_fn = partial(full_transform, stats=stats)
    return ClusterDataset(
        str(required_directory(DATA_FOLDER_ENV)),
        str(required_directory(HIST_FOLDER_ENV)),
        transform=transform_fn,
    )


def run_training(
    training_dataset,
    validation_dataset,
    training_summary,
    split_fingerprint,
    stats,
    epochs=BASELINE_EPOCHS,
):
    """Construct and train a fresh weighted-BCE model."""
    training.set_random_seed(SPLIT_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    training_loader, validation_loader = training.build_data_loaders(
        training_dataset,
        validation_dataset,
        batch_size=BASELINE_BATCH_SIZE,
        seed=SPLIT_SEED,
        num_workers=0,
    )

    model_config = {
        "input_dim": 21,
        "hidden_dim": 64,
        "output_dim": 1,
        "num_layers": 3,
        "edge_feature_dim": 9,
        "heads": 4,
        "dropout": 0.2,
    }
    model = GNN_TrackLinkingNet(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASELINE_LEARNING_RATE,
        weight_decay=BASELINE_WEIGHT_DECAY,
    )

    loss_parameters = derive_loss_parameters(training_summary)
    trk_ts_loss = WeightedBCELoss(
        loss_parameters["trk_ts"]["pos_weight"]
    ).to(device)
    ts_ts_loss = WeightedBCELoss(
        loss_parameters["ts_ts"]["pos_weight"]
    ).to(device)

    preprocessing_fingerprint = hashlib.sha256(
        json.dumps(
            stats,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    checkpoint_metadata = {
        "seed": SPLIT_SEED,
        "model_config": model_config,
        "loss_parameters": loss_parameters,
        "split_fingerprint": split_fingerprint,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "training_config": {
            "batch_size": BASELINE_BATCH_SIZE,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": BASELINE_LEARNING_RATE,
                "weight_decay": BASELINE_WEIGHT_DECAY,
            },
            "epoch_budget": int(epochs),
            "early_stopping": {
                "patience": BASELINE_PATIENCE,
                "min_delta": BASELINE_MIN_DELTA,
            },
            "task_weights": dict(BASELINE_TASK_WEIGHTS),
            "loss_type": "separate_weighted_bce",
            "num_workers": 0,
        },
    }

    checkpoint_path = (
        PROJECT_DIR
        / "checkpoints"
        / (
            "weighted_bce_stratified_bs4_"
            f"{split_fingerprint[:12]}_best.pt"
        )
    )
    print(f"Best checkpoint will be saved to: {checkpoint_path}")

    output = training.train_model(
        model=model,
        optimizer=optimizer,
        training_loader=training_loader,
        validation_loader=validation_loader,
        trk_ts_loss=trk_ts_loss,
        ts_ts_loss=ts_ts_loss,
        device=device,
        epochs=epochs,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        patience=BASELINE_PATIENCE,
        min_delta=BASELINE_MIN_DELTA,
        resume_path=None,
        show_progress=True,
    )
    return model, output


def main(force_refit=False):
    raw_dataset = build_raw_dataset()
    root_path = discover_single_root_file()
    source_file = os.path.basename(root_path)#return the file name ex: "histoSinglePi.root"
    event_ids = load_event_ids(root_path, raw_dataset)
    records = inspect_raw_dataset(raw_dataset, source_file, event_ids)
    manifest = load_or_create_stratified_manifest(
        records,
        source_file,
        SPLIT_MANIFEST_PATH,
        seed=SPLIT_SEED,
        ratios=SPLIT_RATIOS,
        candidate_count=STRATIFICATION_CANDIDATES,
    )
    indices = indices_from_manifest(manifest, records)
    summary = split_summary(manifest, records)
    print_split_summary(summary)

    split_fingerprint = training_split_fingerprint(manifest)

    # Preprocessing is fitted only on raw training graphs.
    raw_training_dataset = Subset(raw_dataset, indices["train"])
    stats = get_or_fit_stats(
        raw_training_dataset,
        split_fingerprint,
        force_refit=force_refit,
    )

    transformed_dataset = build_transformed_dataset(stats)
    training_dataset = Subset(transformed_dataset, indices["train"])
    validation_dataset = Subset(transformed_dataset, indices["validation"])

    print(
        f"Training graphs: {len(training_dataset)}, "
        f"validation graphs: {len(validation_dataset)}"
    )
    print("No test subset or test DataLoader was constructed.")

    model, output = run_training(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        training_summary=summary["train"],
        split_fingerprint=split_fingerprint,
        stats=stats,
        epochs=BASELINE_EPOCHS,
    )

    best_epoch = output["best_epoch"] + 1
    best_validation = output["best_validation"]
    print(f"Best epoch: {best_epoch}")
    print(
        "Best combined validation loss:",
        best_validation["loss"]["combined"],
    )
    print(
        "Best trk-ts validation metrics:",
        best_validation["trk_ts"]["ranking"],
    )
    print(
        "Best ts-ts validation metrics:",
        best_validation["ts_ts"]["ranking"],
    )
    return model, output


if __name__ == "__main__":
    main()
