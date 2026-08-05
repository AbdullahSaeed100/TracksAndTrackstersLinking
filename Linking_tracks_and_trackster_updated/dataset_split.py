"""Reproducible, event-level train/validation/test splitting.

The current dataset is produced from one ROOT file, so the indivisible split
unit is one complete event graph.  The saved manifest uses the original event
entry stored in ``graph.event`` rather than the compacted ``data_*.pt`` index.
"""

import hashlib
import json
import math
import random
from pathlib import Path


SPLIT_NAMES = ("train", "validation", "test")
STRATIFICATION_FEATURES = (
    "trk_ts_edges",
    "trk_ts_positive",
    "ts_ts_edges",
    "ts_ts_positive",
    "trk_ts_positive_at_or_above_p90",
    "trk_ts_positive_at_or_above_p95",
    "ts_ts_positive_at_or_above_p90",
    "ts_ts_positive_at_or_above_p95",
)
STRATIFICATION_WEIGHTS = (0.5, 2.0, 0.5, 2.0, 1.0, 2.0, 1.0, 2.0)


def _event_key(graph, source_file, event_ids):
    run, lumi, event = event_ids[int(graph.event)]
    return f"{source_file}::run_{run}::lumi_{lumi}::event_{event}"


def inspect_raw_dataset(raw_dataset, source_file, event_ids):
    """Return stable identities and compact graph summaries for all graphs."""
    records = []
    seen_keys = set()

    for dataset_index in range(len(raw_dataset)):
        graph = raw_dataset.get(dataset_index)
        if graph is None:
            raise ValueError(f"Dataset index {dataset_index} returned None")
        if graph.edge_attr.ndim != 2 or graph.edge_attr.shape[1] != 8:
            raise ValueError(
                f"Raw graph {dataset_index} must have edge_attr [E, 8]; "
                f"received {tuple(graph.edge_attr.shape)}"
            )

        key = _event_key(graph, source_file, event_ids)
        if key in seen_keys:
            raise ValueError(f"Duplicate event identity found: {key}")
        seen_keys.add(key)

        edge_type = graph.edge_attr[:, 7]
        label = graph.y
        trk_ts = edge_type == 0
        ts_ts = edge_type == 1
        records.append({
            "event_key": key, # Ex: "histoSinglePi.root::run_1::lumi_42::event_90125"
            "dataset_index": dataset_index,# event index in the raw_dataset out of process() funct
            "edges": int(label.numel()),
            "trk_ts_edges": int(trk_ts.sum().item()),
            "trk_ts_positive": int(((label == 1) & trk_ts).sum().item()),
            "ts_ts_edges": int(ts_ts.sum().item()),
            "ts_ts_positive": int(((label == 1) & ts_ts).sum().item()),
        })

    if not records:
        raise ValueError("Cannot split an empty dataset")
    return records


def dataset_fingerprint(records):
    """Fingerprint stable identities and graph/label structure."""
    stable_records = [
        {key: record[key] for key in (
            "event_key", "edges", "trk_ts_edges",
            "trk_ts_positive", "ts_ts_edges", "ts_ts_positive",
        )}
        for record in sorted(records, key=lambda item: item["event_key"])
    ]
    payload = json.dumps(stable_records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_sizes(total, ratios):
    if len(ratios) != 3:
        raise ValueError("ratios must contain train, validation, and test values")
    ratios = tuple(float(value) for value in ratios)
    if not all(math.isfinite(value) and value > 0 for value in ratios):
        raise ValueError("split ratios must be finite and greater than zero")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("split ratios must sum to 1")

    train_size = int(total * ratios[0])
    validation_size = int(total * ratios[1])
    test_size = total - train_size - validation_size
    sizes = [train_size, validation_size, test_size]
    if any(size <= 0 for size in sizes):
        raise ValueError("train, validation, and test splits must all be non-empty")
    return sizes


def create_manifest(records, source_file, seed=42, ratios=(0.70, 0.15, 0.15)):
    keys = sorted(record["event_key"] for record in records)
    random.Random(seed).shuffle(keys)# sorted then shuffled with same seed. both actions to guarntee reproducabilty.
    sizes = _split_sizes(len(keys), ratios)
    boundaries = (sizes[0], sizes[0] + sizes[1])
    assignments = {
        "train": keys[:boundaries[0]],
        "validation": keys[boundaries[0]:boundaries[1]],
        "test": keys[boundaries[1]:],
    }
    manifest = {
        # "format_version": 1,
        "split_unit": "complete_event_graph",
        "source_file": source_file,
        "seed": seed,
        "ratios": {
            "train": ratios[0],
            "validation": ratios[1],
            "test": ratios[2],
        },
        "dataset_fingerprint": dataset_fingerprint(records),
        "assignments": assignments,
    }
    validate_manifest(manifest, records)
    return manifest


def _percentile_threshold(values, percentile):
    """Return a deterministic nearest-rank percentile threshold."""
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _standardized_stratification_features(records):
    """Build event-level features used only to balance complete graph splits."""
    trk_positive = [record["trk_ts_positive"] for record in records]
    ts_positive = [record["ts_ts_positive"] for record in records]
    thresholds = {
        "trk_ts_positive_p90": _percentile_threshold(trk_positive, 0.90),
        "trk_ts_positive_p95": _percentile_threshold(trk_positive, 0.95),
        "ts_ts_positive_p90": _percentile_threshold(ts_positive, 0.90),
        "ts_ts_positive_p95": _percentile_threshold(ts_positive, 0.95),
    }

    raw_by_key = {}
    for record in records:
        raw_by_key[record["event_key"]] = (
            float(record["trk_ts_edges"]),
            float(record["trk_ts_positive"]),
            float(record["ts_ts_edges"]),
            float(record["ts_ts_positive"]),
            float(record["trk_ts_positive"] >= thresholds["trk_ts_positive_p90"]),
            float(record["trk_ts_positive"] >= thresholds["trk_ts_positive_p95"]),
            float(record["ts_ts_positive"] >= thresholds["ts_ts_positive_p90"]),
            float(record["ts_ts_positive"] >= thresholds["ts_ts_positive_p95"]),
        )

    feature_count = len(STRATIFICATION_FEATURES)
    means = []
    standard_deviations = []
    for feature_index in range(feature_count):
        values = [features[feature_index] for features in raw_by_key.values()]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means.append(mean)
        standard_deviations.append(math.sqrt(variance))

    standardized_by_key = {}
    for event_key, features in raw_by_key.items():
        standardized_by_key[event_key] = tuple(
            (features[index] - means[index]) / standard_deviations[index]
            if standard_deviations[index] > 0
            else 0.0
            for index in range(feature_count)
        )
    return standardized_by_key, thresholds


def _stratification_score(assignments, standardized_by_key):
    """Measure how far each split's event-level feature means are from global."""
    score = 0.0
    maximum_deviation = 0.0
    feature_count = len(STRATIFICATION_FEATURES)

    for name in SPLIT_NAMES:
        keys = assignments[name]
        split_size = len(keys)
        for feature_index in range(feature_count):
            split_mean = sum(
                standardized_by_key[key][feature_index] for key in keys
            ) / split_size
            deviation = abs(split_mean)
            score += STRATIFICATION_WEIGHTS[feature_index] * split_mean**2
            maximum_deviation = max(maximum_deviation, deviation)
    return score, maximum_deviation


def create_stratified_manifest(
    records,
    source_file,
    seed=42,
    ratios=(0.70, 0.15, 0.15),
    candidate_count=5000,
):
    """Choose one deterministic complete-event split with balanced composition.

    Candidate assignments all have the exact requested graph counts. The search
    balances edge totals, positive totals, and high-positive event tails for both
    edge types. It never splits an event or uses node/edge feature values.
    """
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")

    keys = sorted(record["event_key"] for record in records)
    sizes = _split_sizes(len(keys), ratios)
    boundaries = (sizes[0], sizes[0] + sizes[1])
    standardized_by_key, thresholds = _standardized_stratification_features(records)
    generator = random.Random(seed)

    best_assignments = None
    best_objective = None
    for _ in range(candidate_count):
        candidate_keys = list(keys)
        generator.shuffle(candidate_keys)
        assignments = {
            "train": candidate_keys[:boundaries[0]],
            "validation": candidate_keys[boundaries[0]:boundaries[1]],
            "test": candidate_keys[boundaries[1]:],
        }
        objective = _stratification_score(assignments, standardized_by_key)
        if best_objective is None or objective < best_objective:
            best_objective = objective
            best_assignments = {
                name: list(assignments[name]) for name in SPLIT_NAMES
            }

    manifest = {
        "split_unit": "complete_event_graph",
        "strategy": "deterministic_multitarget_event_stratification",
        "source_file": source_file,
        "seed": seed,
        "ratios": {
            "train": ratios[0],
            "validation": ratios[1],
            "test": ratios[2],
        },
        "dataset_fingerprint": dataset_fingerprint(records),
        "stratification": {
            "candidate_count": candidate_count,
            "features": list(STRATIFICATION_FEATURES),
            "weights": list(STRATIFICATION_WEIGHTS),
            "tail_thresholds": thresholds,
            "score": best_objective[0],
            "maximum_standardized_mean_deviation": best_objective[1],
        },
        "assignments": best_assignments,
    }
    validate_manifest(manifest, records)
    return manifest


def load_or_create_stratified_manifest(
    records,
    source_file,
    manifest_path,
    seed=42,
    ratios=(0.70, 0.15, 0.15),
    candidate_count=5000,
):
    """Load the frozen stratified manifest or create it exactly once."""
    path = Path(manifest_path)
    if path.exists():
        with path.open() as handle:
            manifest = json.load(handle)
        if manifest.get("strategy") != "deterministic_multitarget_event_stratification":
            raise ValueError("Existing manifest is not the approved stratified design")
        if manifest.get("source_file") != source_file:
            raise ValueError("Split manifest source file does not match the current ROOT file")
        if manifest.get("seed") != seed:
            raise ValueError("Existing split manifest uses a different seed")
        expected_ratios = {
            "train": ratios[0],
            "validation": ratios[1],
            "test": ratios[2],
        }
        if manifest.get("ratios") != expected_ratios:
            raise ValueError("Existing split manifest uses different split ratios")
        validate_manifest(manifest, records)
        return manifest

    manifest = create_stratified_manifest(
        records,
        source_file,
        seed=seed,
        ratios=ratios,
        candidate_count=candidate_count,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def validate_manifest(manifest, records):
    current_fingerprint = dataset_fingerprint(records)
    if manifest.get("dataset_fingerprint") != current_fingerprint:
        raise ValueError(
            "The split manifest does not match the current dataset. "
            "Refusing to silently reuse or regenerate a split."
        )

    assignments = manifest.get("assignments", {})
    
    split_lists = [assignments[name] for name in SPLIT_NAMES]
    
    if any(not values for values in split_lists):
        raise ValueError("Train, validation, and test splits must all be non-empty")

    all_assigned = [key for values in split_lists for key in values]# This flattens the three lists into one list.
    # this to check the event(graph) appears once in one of the the three lists. any duplicate will be detected.
    if len(all_assigned) != len(set(all_assigned)):
        raise ValueError("Duplicate or overlapping event assignment")

    all_current = {record["event_key"] for record in records}
    assigned_set = set(all_assigned)
    missing = all_current - assigned_set
    unexpected = assigned_set - all_current
    #checking if the current manifest event keys == records event keys(retrived from raw_dataset) 
    if missing or unexpected:
        raise ValueError(
            f"Split coverage failure: {len(missing)} missing and "
            f"{len(unexpected)} unexpected event identities"
        )


def load_or_create_manifest(
    records,
    source_file,
    manifest_path,
    seed=42,
    ratios=(0.70, 0.15, 0.15),
):
    path = Path(manifest_path)
    if path.exists():# loading same split manifest.
        with path.open() as handle:
            manifest = json.load(handle)
        # Check that the manifest's source file and seed match the current source file and requested seed.
        if manifest.get("source_file") != source_file:
            raise ValueError("Split manifest source file does not match the current ROOT file")
        if manifest.get("seed") != seed:
            raise ValueError("Existing split manifest uses a different seed")
        expected_ratios = {
            "train": ratios[0],
            "validation": ratios[1],
            "test": ratios[2],
        }
        if manifest.get("ratios") != expected_ratios:
            raise ValueError("Existing split manifest uses different split ratios")
        validate_manifest(manifest, records)
        return manifest
    
    # creating new split manifest
    manifest = create_manifest(records, source_file, seed=seed, ratios=ratios)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def indices_from_manifest(manifest, records):
    """Remap stable event identities to the dataset's current indices."""
    index_by_key = {record["event_key"]: record["dataset_index"] for record in records}
    return {
        name: [index_by_key[key] for key in manifest["assignments"][name]]
        for name in SPLIT_NAMES
    }


def training_split_fingerprint(manifest):
    """Identify the dataset version and the exact training assignment."""
    payload = {
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "train_event_keys": sorted(manifest["assignments"]["train"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def split_summary(manifest, records):
    record_by_key = {record["event_key"]: record for record in records}
    summary = {}
    for name in SPLIT_NAMES:
        selected = [record_by_key[key] for key in manifest["assignments"][name]]
        values = {
            "graphs": len(selected),
            "edges": sum(item["edges"] for item in selected),
            "trk_ts_edges": sum(item["trk_ts_edges"] for item in selected),
            "trk_ts_positive": sum(item["trk_ts_positive"] for item in selected),
            "ts_ts_edges": sum(item["ts_ts_edges"] for item in selected),
            "ts_ts_positive": sum(item["ts_ts_positive"] for item in selected),
        }
        values["trk_ts_positive_fraction"] = (
            values["trk_ts_positive"] / values["trk_ts_edges"]
            if values["trk_ts_edges"] else None
        )
        values["ts_ts_positive_fraction"] = (
            values["ts_ts_positive"] / values["ts_ts_edges"]
            if values["ts_ts_edges"] else None
        )
        summary[name] = values

        if values["trk_ts_edges"] == 0 or values["ts_ts_edges"] == 0:
            raise ValueError(f"The {name} split does not contain both edge types")
        if values["trk_ts_positive"] == 0 or values["ts_ts_positive"] == 0:
            raise ValueError(
                f"The {name} split lacks positive examples for an edge type"
            )
    return summary


def print_split_summary(summary):
    for name in SPLIT_NAMES:
        item = summary[name]
        print(
            f"{name:10s}: graphs={item['graphs']}, edges={item['edges']}, "
            f"trk-ts={item['trk_ts_edges']} "
            f"(positive={item['trk_ts_positive']}, "
            f"fraction={item['trk_ts_positive_fraction']:.4%}), "
            f"ts-ts={item['ts_ts_edges']} "
            f"(positive={item['ts_ts_positive']}, "
            f"fraction={item['ts_ts_positive_fraction']:.4%})"
        )
    print("Split integrity: overlap=0, missing=0")
