"""Display metrics saved directly in a trusted training checkpoint.

The report does not load any dataset, construct a DataLoader, or run a model.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


EDGE_TYPES = (("trk_ts", "trk-ts"), ("ts_ts", "ts-ts"))


def load_trusted_checkpoint(path):
    """Load this project's structured checkpoint on the CPU."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    required = {
        "format_version",
        "epoch",
        "training_results",
        "validation_results",
        "metadata",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
    if checkpoint["format_version"] != 1:
        raise ValueError(
            f"Unsupported checkpoint format: {checkpoint['format_version']}"
        )
    return path, checkpoint


def print_table(title, table, decimals=6):
    print(f"\n{title}")
    print("-" * len(title))
    print(table.round(decimals).to_string(index=False))


def configuration_table(metadata):
    training = metadata["training_config"]
    optimizer = training["optimizer"]
    stopping = training["early_stopping"]
    return pd.DataFrame(
        {
            "Parameter": [
                "Random seed",
                "Batch size",
                "Optimizer",
                "Learning rate",
                "Weight decay",
                "Epoch budget",
                "Early-stopping patience",
                "Early-stopping minimum delta",
                "Loss type",
                "trk-ts task weight",
                "ts-ts task weight",
                "Number of workers",
            ],
            "Value": [
                metadata["seed"],
                training["batch_size"],
                optimizer["name"],
                optimizer["learning_rate"],
                optimizer["weight_decay"],
                training["epoch_budget"],
                stopping["patience"],
                stopping["min_delta"],
                training["loss_type"],
                training["task_weights"]["trk_ts"],
                training["task_weights"]["ts_ts"],
                training["num_workers"],
            ],
        }
    )


def loss_table(training_results, validation_results):
    return pd.DataFrame(
        {
            "Dataset": ["Training", "Validation"],
            "trk-ts loss": [
                training_results["loss"]["trk_ts"],
                validation_results["loss"]["trk_ts"],
            ],
            "ts-ts loss": [
                training_results["loss"]["ts_ts"],
                validation_results["loss"]["ts_ts"],
            ],
            "Combined loss": [
                training_results["loss"]["combined"],
                validation_results["loss"]["combined"],
            ],
        }
    )


def validation_composition_table(validation_results):
    rows = []
    for key, label in EDGE_TYPES:
        result = validation_results[key]
        rows.append(
            {
                "Edge type": label,
                "Edges": result["edge_count"],
                "Positive edges": result["positive_count"],
                "Positive fraction": result["positive_fraction"],
                "Positive percentage": 100.0 * result["positive_fraction"],
            }
        )
    return pd.DataFrame(rows)


def ranking_table(validation_results):
    rows = []
    for key, label in EDGE_TYPES:
        result = validation_results[key]
        ranking = result["ranking"]
        prevalence = result["positive_fraction"]
        rows.append(
            {
                "Edge type": label,
                "Average precision": ranking["average_precision"],
                "ROC-AUC": ranking["roc_auc"],
                "Random AP baseline": prevalence,
                "AP lift over random": ranking["average_precision"] / prevalence,
            }
        )
    return pd.DataFrame(rows)


def diagnostic_table(validation_results):
    rows = []
    for key, label in EDGE_TYPES:
        result = validation_results[key]["best_f1"]
        rows.append(
            {
                "Edge type": label,
                "Threshold": result["threshold"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1": result["f1"],
                "Specificity": result["specificity"],
                "False-positive rate": result["false_positive_rate"],
                "Balanced accuracy": result["balanced_accuracy"],
                "Accuracy": result["accuracy"],
            }
        )
    return pd.DataFrame(rows)


def plot_confusion_matrices(validation_results, best_epoch):
    """Plot conventional matrices with actual rows and predicted columns."""
    sns.set_theme(style="white")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    for axis, (key, label) in zip(axes, EDGE_TYPES):
        result = validation_results[key]["best_f1"]
        matrix = [
            [result["tn"], result["fp"]],
            [result["fn"], result["tp"]],
        ]
        sns.heatmap(
            matrix,
            annot=True,
            fmt=",",
            cmap="Blues",
            cbar=False,
            square=True,
            linewidths=1,
            linecolor="white",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            annot_kws={"fontsize": 13},
            ax=axis,
        )
        axis.set_title(
            f"{label} confusion matrix\n"
            f"Best-F1 threshold = {result['threshold']:.4f}"
        )
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
        axis.tick_params(axis="x", rotation=0)
        axis.tick_params(axis="y", rotation=0)

    figure.suptitle(
        f"Validation Confusion Matrices - Best Epoch {best_epoch}",
        fontsize=15,
    )
    figure.tight_layout()
    plt.show()


def render_checkpoint_report(checkpoint_path):
    path, checkpoint = load_trusted_checkpoint(checkpoint_path)
    metadata = checkpoint["metadata"]
    validation = checkpoint["validation_results"]
    best_epoch = int(checkpoint["epoch"]) + 1

    print("=" * 72)
    print("FRESH STRATIFIED BASELINE CHECKPOINT REPORT")
    print("=" * 72)
    print(f"Checkpoint: {path}")
    print(f"Best epoch: {best_epoch}")
    print(f"Split fingerprint: {metadata['split_fingerprint']}")
    print(f"Preprocessing fingerprint: {metadata['preprocessing_fingerprint']}")

    print_table("TRAINING CONFIGURATION", configuration_table(metadata))
    print_table(
        f"LOSSES AT BEST EPOCH {best_epoch}",
        loss_table(checkpoint["training_results"], validation),
    )
    print_table(
        "VALIDATION DATA COMPOSITION",
        validation_composition_table(validation),
    )
    print_table("VALIDATION RANKING METRICS", ranking_table(validation))
    print_table("BEST-F1 VALIDATION DIAGNOSTICS", diagnostic_table(validation))
    plot_confusion_matrices(validation, best_epoch)

    print("\nThe report used saved checkpoint values only.")
    print("No dataset, DataLoader, model inference, or test evaluation was used.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Display metrics saved in a trusted project checkpoint."
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to the structured .pt checkpoint produced by training.py.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    render_checkpoint_report(arguments.checkpoint)
