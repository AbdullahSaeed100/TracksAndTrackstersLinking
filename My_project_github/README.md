# Track-to-Trackster and Trackster-to-Trackster Edge Classification

This repository trains a graph neural network to classify two edge types:

- track-to-trackster (`trk-ts`)
- trackster-to-trackster (`ts-ts`)

The model returns raw logits. Loss functions receive those logits directly;
sigmoid is applied externally only when probabilities are required for
evaluation.

## Scientific safeguards

- Splits contain complete events rather than individual edges.
- Preprocessing statistics are fitted using the training subset only.
- Weighted-BCE parameters are derived using the training subset only.
- Validation results remain separate for `trk-ts` and `ts-ts` edges.
- Training constructs only training and validation subsets.
- The test subset must remain untouched until the final approved evaluation.

## Repository structure

- `run_pipeline.py`: baseline orchestration and configuration
- `training.py`: training, checkpointing, resume support, and early stopping
- `evaluation.py`: validation loss and classification metrics
- `loss_functions.py`: weighted BCE and focal-loss implementations
- `GNN_TrackLinkingNet.py`: graph neural network
- `EdgeConvBlock.py`: message-passing block
- `dataset_split.py`: deterministic event-level stratified split
- `feature_preprocessing.py`: training-only feature statistics and transforms
- `ClusterDataset_trk_trkst.py`: dataset construction and loading
- `report_checkpoint.py`: report saved checkpoint values and confusion matrices

## Environment

The dataset is intentionally not stored in this repository. Configure its
locations with environment variables before running the pipeline.

In a CERN SWAN Python notebook:

```python
import os

os.environ["TRACKLINK_HIST_FOLDER"] = "/your/path/to/root/files"
os.environ["TRACKLINK_DATA_FOLDER"] = "/your/path/to/dataset"

import run_pipeline
```

If `run_pipeline` was imported before setting the variables:

```python
import importlib
import run_pipeline

importlib.reload(run_pipeline)
```

The file `.env.example` documents the required variable names. A real `.env`
file is private and ignored by Git. The code does not automatically load it.

## Dependencies

Install packages from `requirements.txt` in an environment with mutually
compatible PyTorch, CUDA, and PyTorch Geometric versions. CERN SWAN may already
provide many of these packages.

```bash
python -m pip install -r requirements.txt
```

## Frozen stratified manifest

The approved baseline uses:

```text
dataset_split_manifest_stratified.json
```

For exact scientific reproducibility, copy the approved frozen manifest from
the training environment into the repository before publishing it. Do not
regenerate or replace it after observing validation results. The desktop source
copy used to prepare this repository did not contain the real manifest, so no
placeholder assignment file has been invented.

The generated preprocessing statistics file is not committed. On the first
run, the pipeline fits it from the frozen training subset and saves it as:

```text
all_stats_train_stratified.json
```

## Run the approved baseline

After configuring the external data directories and placing the frozen
manifest in the project directory:

```python
import run_pipeline

model, output = run_pipeline.main(force_refit=False)
```

This starts fresh training. It does not resume from another checkpoint and does
not construct a test DataLoader.

## Display a checkpoint report

From a terminal:

```bash
python report_checkpoint.py checkpoints/name_of_checkpoint.pt
```

From a notebook:

```python
from report_checkpoint import render_checkpoint_report

render_checkpoint_report("checkpoints/name_of_checkpoint.pt")
```

The report reads configuration and validation values directly from the
checkpoint. It also draws one conventional confusion matrix for each edge type,
using actual classes as rows and predicted classes as columns.

The best checkpoint is saved only when validation loss improves. Consequently,
it contains the state at the best epoch, but not necessarily the total number of
epochs executed after that checkpoint or the final early-stopping state.

## Files intentionally excluded from Git

- raw and processed datasets
- ROOT files
- model checkpoints
- private environment configuration
- generated preprocessing statistics
- Python and notebook caches

The original executed notebooks were not copied because their saved outputs
contained user-specific filesystem paths. Add clean notebooks later only after
clearing outputs and replacing local paths with environment-based configuration.
